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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_aa4bcb291b491239d72c(const __grid_constant__ CUtensorMap qn_tma, const __grid_constant__ CUtensorMap kn_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap a_tma, float* __restrict__ g_cs, float* __restrict__ beta, float* __restrict__ h0, float* __restrict__ seg_start, const __grid_constant__ CUtensorMap h_tma, __nv_bfloat16* __restrict__ h_out, const __grid_constant__ CUtensorMap v_new_tma, __nv_bfloat16* __restrict__ v_new, const __grid_constant__ CUtensorMap o_tma, __nv_bfloat16* __restrict__ o_out, float* __restrict__ final_state, const __grid_constant__ CUtensorMap w_tma, __nv_bfloat16* __restrict__ w_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int* __restrict__ walk, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int use_seg_start, float scale, int tape)
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

    const int mbar_base = smem + 227328;
    #define kq_full_addr (mbar_base + 0)
    #define kq_fixed_addr (mbar_base + 24)
    #define kq_free_addr (mbar_base + 48)
    #define h_read_addr (mbar_base + 72)
    #define va_full_addr (mbar_base + 96)
    #define v_fixed_addr (mbar_base + 112)
    #define va_free_addr (mbar_base + 128)
    #define bk_ready_addr (mbar_base + 144)
    #define bk_free_addr (mbar_base + 152)
    #define gates_ready_addr (mbar_base + 160)
    #define gates_free_addr (mbar_base + 192)
    #define hs_ready_addr (mbar_base + 224)
    #define h_staged_addr (mbar_base + 232)
    #define w_full_addr (mbar_base + 256)
    #define w_drained_addr (mbar_base + 264)
    #define wt_ready_addr (mbar_base + 272)
    #define wn_free_addr (mbar_base + 280)
    #define u_full_addr (mbar_base + 288)
    #define u_rounded_addr (mbar_base + 296)
    #define ws_full_addr (mbar_base + 304)
    #define vp_ready_addr (mbar_base + 312)
    #define s_full_addr (mbar_base + 320)
    #define vn_staged_addr (mbar_base + 328)
    #define vn_free_addr (mbar_base + 336)
    #define o_staged_addr (mbar_base + 344)
    #define ost_free_addr (mbar_base + 352)
    #define g_full_addr (mbar_base + 360)
    #define pg_ready_addr (mbar_base + 368)
    #define o1_full_addr (mbar_base + 376)
    #define o_scaled_addr (mbar_base + 384)
    #define o_full_addr (mbar_base + 392)
    #define o_free_addr (mbar_base + 400)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* s_kq = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kq_addr = smem + 0;
    __nv_bfloat16* s_qb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int s_qb_addr = smem + 8192;
    __nv_bfloat16* s_kT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kT_addr = smem + 0;
    __nv_bfloat16* s_vT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_vT_addr = smem + 98304;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 114688);
    const int s_a_addr = smem + 114688;
    __nv_bfloat16* s_bk = reinterpret_cast<__nv_bfloat16*>(smem_raw + 147456);
    const int s_bk_addr = smem + 147456;
    float* s_gt = reinterpret_cast<float*>(smem_raw + 221184);
    const int s_gt_addr = smem + 221184;
    __nv_bfloat16* s_pg = reinterpret_cast<__nv_bfloat16*>(smem_raw + 163840);
    const int s_pg_addr = smem + 163840;
    __nv_bfloat16* s_wn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 172032);
    const int s_wn_addr = smem + 172032;
    __nv_bfloat16* s_vst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 204800);
    const int s_vst_addr = smem + 204800;
    __nv_bfloat16* s_ost = reinterpret_cast<__nv_bfloat16*>(smem_raw + 188416);
    const int s_ost_addr = smem + 188416;

    // Mbarrier init (51 barriers at smem_raw[227328..227736))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'kqring' ---
            // kq_full: 3 barriers, init_count=1
            mbarrier_init(smem + 227328, 1);
            mbarrier_init(smem + 227336, 1);
            mbarrier_init(smem + 227344, 1);
            // kq_fixed: 3 barriers, init_count=128
            mbarrier_init(smem + 227352, 128);
            mbarrier_init(smem + 227360, 128);
            mbarrier_init(smem + 227368, 128);
            // kq_free: 3 barriers, init_count=1
            mbarrier_init(smem + 227376, 1);
            mbarrier_init(smem + 227384, 1);
            mbarrier_init(smem + 227392, 1);
            // h_read: 3 barriers, init_count=1
            mbarrier_init(smem + 227400, 1);
            mbarrier_init(smem + 227408, 1);
            mbarrier_init(smem + 227416, 1);
            // --- pipeline 'varing' ---
            // va_full: 2 barriers, init_count=1
            mbarrier_init(smem + 227424, 1);
            mbarrier_init(smem + 227432, 1);
            // v_fixed: 2 barriers, init_count=128
            mbarrier_init(smem + 227440, 128);
            mbarrier_init(smem + 227448, 128);
            // va_free: 2 barriers, init_count=1
            mbarrier_init(smem + 227456, 1);
            mbarrier_init(smem + 227464, 1);
            // --- pipeline 'one' ---
            // bk_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 227472, 128);
            // bk_free: 1 barriers, init_count=1
            mbarrier_init(smem + 227480, 1);
            // --- pipeline 'gring' ---
            // gates_ready: 4 barriers, init_count=32
            mbarrier_init(smem + 227488, 32);
            mbarrier_init(smem + 227496, 32);
            mbarrier_init(smem + 227504, 32);
            mbarrier_init(smem + 227512, 32);
            // gates_free: 4 barriers, init_count=384
            mbarrier_init(smem + 227520, 384);
            mbarrier_init(smem + 227528, 384);
            mbarrier_init(smem + 227536, 384);
            mbarrier_init(smem + 227544, 384);
            // --- pipeline 'one' ---
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 227552, 128);
            // --- pipeline 'kqring' ---
            // h_staged: 3 barriers, init_count=128
            mbarrier_init(smem + 227560, 128);
            mbarrier_init(smem + 227568, 128);
            mbarrier_init(smem + 227576, 128);
            // --- pipeline 'one' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227584, 1);
            // w_drained: 1 barriers, init_count=128
            mbarrier_init(smem + 227592, 128);
            // wt_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 227600, 128);
            // wn_free: 1 barriers, init_count=1
            mbarrier_init(smem + 227608, 1);
            // u_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227616, 1);
            // u_rounded: 1 barriers, init_count=128
            mbarrier_init(smem + 227624, 128);
            // ws_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227632, 1);
            // vp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 227640, 128);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227648, 1);
            // vn_staged: 1 barriers, init_count=128
            mbarrier_init(smem + 227656, 128);
            // vn_free: 1 barriers, init_count=1
            mbarrier_init(smem + 227664, 1);
            // o_staged: 1 barriers, init_count=128
            mbarrier_init(smem + 227672, 128);
            // ost_free: 1 barriers, init_count=1
            mbarrier_init(smem + 227680, 1);
            // g_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227688, 1);
            // pg_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 227696, 128);
            // o1_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227704, 1);
            // o_scaled: 1 barriers, init_count=128
            mbarrier_init(smem + 227712, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 227720, 1);
            // o_free: 1 barriers, init_count=128
            mbarrier_init(smem + 227728, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 227736);
    if (warp == 0) {
        int _tmem_hold = smem + 227736;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_s = taddr;
    const unsigned int tmem_hs = taddr + 128;
    const unsigned int tmem_vn = taddr + 192;
    const unsigned int tmem_vp = taddr + 256;
    const unsigned int tmem_vnb = taddr + 288;
    const unsigned int tmem_o = taddr + 320;
    const unsigned int tmem_g = taddr + 384;
    const unsigned int tmem_w = taddr + 448;

    // ---- Role: wga ----
    if (warp <= 3) {
        { // wga_main
            int blk = blockIdx.x;
            int seg = walk[blk / num_v_heads];
            int hv = blk % num_v_heads;
            int c_begin = seg_chunk_start[seg];
            int nc = seg_chunk_start[seg + 1] - c_begin;
            int seq = seg_seq[seg];
            int vrow = warp % 4 * 32 + lane;
            int row_off = warp % 4 * 32 << 16;
            long long head_base = (long long)hv * 16384ll + (long long)vrow;
            int is_last = 0;
            if (seq >= 0) {
                if (c_begin + nc == seq_chunk_start[seq + 1]) {
                    is_last = 1;
                }
            }
            int start_src = 0;
            long long start_base = 0ll;
            if (use_seg_start != 0) {
                start_src = 2;
                start_base = (long long)seg * (long long)num_v_heads * 16384ll + head_base;
            }
            if (use_seg_start == 0) {
                if (use_initial_state != 0) {
                    if (seq >= 0) {
                        start_src = 1;
                        start_base = (long long)seq * (long long)num_v_heads * 16384ll + head_base;
                    }
                }
            }
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
            if (start_src == 2) {
                blk_0[0] = seg_start[start_base + 0ll];
                blk_0[1] = seg_start[start_base + 128ll];
                blk_0[2] = seg_start[start_base + 256ll];
                blk_0[3] = seg_start[start_base + 384ll];
                blk_0[4] = seg_start[start_base + 512ll];
                blk_0[5] = seg_start[start_base + 640ll];
                blk_0[6] = seg_start[start_base + 768ll];
                blk_0[7] = seg_start[start_base + 896ll];
                blk_0[8] = seg_start[start_base + 1024ll];
                blk_0[9] = seg_start[start_base + 1152ll];
                blk_0[10] = seg_start[start_base + 1280ll];
                blk_0[11] = seg_start[start_base + 1408ll];
                blk_0[12] = seg_start[start_base + 1536ll];
                blk_0[13] = seg_start[start_base + 1664ll];
                blk_0[14] = seg_start[start_base + 1792ll];
                blk_0[15] = seg_start[start_base + 1920ll];
                blk_0[16] = seg_start[start_base + 2048ll];
                blk_0[17] = seg_start[start_base + 2176ll];
                blk_0[18] = seg_start[start_base + 2304ll];
                blk_0[19] = seg_start[start_base + 2432ll];
                blk_0[20] = seg_start[start_base + 2560ll];
                blk_0[21] = seg_start[start_base + 2688ll];
                blk_0[22] = seg_start[start_base + 2816ll];
                blk_0[23] = seg_start[start_base + 2944ll];
                blk_0[24] = seg_start[start_base + 3072ll];
                blk_0[25] = seg_start[start_base + 3200ll];
                blk_0[26] = seg_start[start_base + 3328ll];
                blk_0[27] = seg_start[start_base + 3456ll];
                blk_0[28] = seg_start[start_base + 3584ll];
                blk_0[29] = seg_start[start_base + 3712ll];
                blk_0[30] = seg_start[start_base + 3840ll];
                blk_0[31] = seg_start[start_base + 3968ll];
            }
            if (start_src == 1) {
                blk_0[0] = h0[start_base + 0ll];
                blk_0[1] = h0[start_base + 128ll];
                blk_0[2] = h0[start_base + 256ll];
                blk_0[3] = h0[start_base + 384ll];
                blk_0[4] = h0[start_base + 512ll];
                blk_0[5] = h0[start_base + 640ll];
                blk_0[6] = h0[start_base + 768ll];
                blk_0[7] = h0[start_base + 896ll];
                blk_0[8] = h0[start_base + 1024ll];
                blk_0[9] = h0[start_base + 1152ll];
                blk_0[10] = h0[start_base + 1280ll];
                blk_0[11] = h0[start_base + 1408ll];
                blk_0[12] = h0[start_base + 1536ll];
                blk_0[13] = h0[start_base + 1664ll];
                blk_0[14] = h0[start_base + 1792ll];
                blk_0[15] = h0[start_base + 1920ll];
                blk_0[16] = h0[start_base + 2048ll];
                blk_0[17] = h0[start_base + 2176ll];
                blk_0[18] = h0[start_base + 2304ll];
                blk_0[19] = h0[start_base + 2432ll];
                blk_0[20] = h0[start_base + 2560ll];
                blk_0[21] = h0[start_base + 2688ll];
                blk_0[22] = h0[start_base + 2816ll];
                blk_0[23] = h0[start_base + 2944ll];
                blk_0[24] = h0[start_base + 3072ll];
                blk_0[25] = h0[start_base + 3200ll];
                blk_0[26] = h0[start_base + 3328ll];
                blk_0[27] = h0[start_base + 3456ll];
                blk_0[28] = h0[start_base + 3584ll];
                blk_0[29] = h0[start_base + 3712ll];
                blk_0[30] = h0[start_base + 3840ll];
                blk_0[31] = h0[start_base + 3968ll];
            }
            tmem_st_x32_f32(tmem_s + row_off + 0, blk_0);
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
            if (start_src == 2) {
                blk_1[0] = seg_start[start_base + 4096ll];
                blk_1[1] = seg_start[start_base + 4224ll];
                blk_1[2] = seg_start[start_base + 4352ll];
                blk_1[3] = seg_start[start_base + 4480ll];
                blk_1[4] = seg_start[start_base + 4608ll];
                blk_1[5] = seg_start[start_base + 4736ll];
                blk_1[6] = seg_start[start_base + 4864ll];
                blk_1[7] = seg_start[start_base + 4992ll];
                blk_1[8] = seg_start[start_base + 5120ll];
                blk_1[9] = seg_start[start_base + 5248ll];
                blk_1[10] = seg_start[start_base + 5376ll];
                blk_1[11] = seg_start[start_base + 5504ll];
                blk_1[12] = seg_start[start_base + 5632ll];
                blk_1[13] = seg_start[start_base + 5760ll];
                blk_1[14] = seg_start[start_base + 5888ll];
                blk_1[15] = seg_start[start_base + 6016ll];
                blk_1[16] = seg_start[start_base + 6144ll];
                blk_1[17] = seg_start[start_base + 6272ll];
                blk_1[18] = seg_start[start_base + 6400ll];
                blk_1[19] = seg_start[start_base + 6528ll];
                blk_1[20] = seg_start[start_base + 6656ll];
                blk_1[21] = seg_start[start_base + 6784ll];
                blk_1[22] = seg_start[start_base + 6912ll];
                blk_1[23] = seg_start[start_base + 7040ll];
                blk_1[24] = seg_start[start_base + 7168ll];
                blk_1[25] = seg_start[start_base + 7296ll];
                blk_1[26] = seg_start[start_base + 7424ll];
                blk_1[27] = seg_start[start_base + 7552ll];
                blk_1[28] = seg_start[start_base + 7680ll];
                blk_1[29] = seg_start[start_base + 7808ll];
                blk_1[30] = seg_start[start_base + 7936ll];
                blk_1[31] = seg_start[start_base + 8064ll];
            }
            if (start_src == 1) {
                blk_1[0] = h0[start_base + 4096ll];
                blk_1[1] = h0[start_base + 4224ll];
                blk_1[2] = h0[start_base + 4352ll];
                blk_1[3] = h0[start_base + 4480ll];
                blk_1[4] = h0[start_base + 4608ll];
                blk_1[5] = h0[start_base + 4736ll];
                blk_1[6] = h0[start_base + 4864ll];
                blk_1[7] = h0[start_base + 4992ll];
                blk_1[8] = h0[start_base + 5120ll];
                blk_1[9] = h0[start_base + 5248ll];
                blk_1[10] = h0[start_base + 5376ll];
                blk_1[11] = h0[start_base + 5504ll];
                blk_1[12] = h0[start_base + 5632ll];
                blk_1[13] = h0[start_base + 5760ll];
                blk_1[14] = h0[start_base + 5888ll];
                blk_1[15] = h0[start_base + 6016ll];
                blk_1[16] = h0[start_base + 6144ll];
                blk_1[17] = h0[start_base + 6272ll];
                blk_1[18] = h0[start_base + 6400ll];
                blk_1[19] = h0[start_base + 6528ll];
                blk_1[20] = h0[start_base + 6656ll];
                blk_1[21] = h0[start_base + 6784ll];
                blk_1[22] = h0[start_base + 6912ll];
                blk_1[23] = h0[start_base + 7040ll];
                blk_1[24] = h0[start_base + 7168ll];
                blk_1[25] = h0[start_base + 7296ll];
                blk_1[26] = h0[start_base + 7424ll];
                blk_1[27] = h0[start_base + 7552ll];
                blk_1[28] = h0[start_base + 7680ll];
                blk_1[29] = h0[start_base + 7808ll];
                blk_1[30] = h0[start_base + 7936ll];
                blk_1[31] = h0[start_base + 8064ll];
            }
            tmem_st_x32_f32(tmem_s + row_off + 32, blk_1);
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
            if (start_src == 2) {
                blk_2[0] = seg_start[start_base + 8192ll];
                blk_2[1] = seg_start[start_base + 8320ll];
                blk_2[2] = seg_start[start_base + 8448ll];
                blk_2[3] = seg_start[start_base + 8576ll];
                blk_2[4] = seg_start[start_base + 8704ll];
                blk_2[5] = seg_start[start_base + 8832ll];
                blk_2[6] = seg_start[start_base + 8960ll];
                blk_2[7] = seg_start[start_base + 9088ll];
                blk_2[8] = seg_start[start_base + 9216ll];
                blk_2[9] = seg_start[start_base + 9344ll];
                blk_2[10] = seg_start[start_base + 9472ll];
                blk_2[11] = seg_start[start_base + 9600ll];
                blk_2[12] = seg_start[start_base + 9728ll];
                blk_2[13] = seg_start[start_base + 9856ll];
                blk_2[14] = seg_start[start_base + 9984ll];
                blk_2[15] = seg_start[start_base + 10112ll];
                blk_2[16] = seg_start[start_base + 10240ll];
                blk_2[17] = seg_start[start_base + 10368ll];
                blk_2[18] = seg_start[start_base + 10496ll];
                blk_2[19] = seg_start[start_base + 10624ll];
                blk_2[20] = seg_start[start_base + 10752ll];
                blk_2[21] = seg_start[start_base + 10880ll];
                blk_2[22] = seg_start[start_base + 11008ll];
                blk_2[23] = seg_start[start_base + 11136ll];
                blk_2[24] = seg_start[start_base + 11264ll];
                blk_2[25] = seg_start[start_base + 11392ll];
                blk_2[26] = seg_start[start_base + 11520ll];
                blk_2[27] = seg_start[start_base + 11648ll];
                blk_2[28] = seg_start[start_base + 11776ll];
                blk_2[29] = seg_start[start_base + 11904ll];
                blk_2[30] = seg_start[start_base + 12032ll];
                blk_2[31] = seg_start[start_base + 12160ll];
            }
            if (start_src == 1) {
                blk_2[0] = h0[start_base + 8192ll];
                blk_2[1] = h0[start_base + 8320ll];
                blk_2[2] = h0[start_base + 8448ll];
                blk_2[3] = h0[start_base + 8576ll];
                blk_2[4] = h0[start_base + 8704ll];
                blk_2[5] = h0[start_base + 8832ll];
                blk_2[6] = h0[start_base + 8960ll];
                blk_2[7] = h0[start_base + 9088ll];
                blk_2[8] = h0[start_base + 9216ll];
                blk_2[9] = h0[start_base + 9344ll];
                blk_2[10] = h0[start_base + 9472ll];
                blk_2[11] = h0[start_base + 9600ll];
                blk_2[12] = h0[start_base + 9728ll];
                blk_2[13] = h0[start_base + 9856ll];
                blk_2[14] = h0[start_base + 9984ll];
                blk_2[15] = h0[start_base + 10112ll];
                blk_2[16] = h0[start_base + 10240ll];
                blk_2[17] = h0[start_base + 10368ll];
                blk_2[18] = h0[start_base + 10496ll];
                blk_2[19] = h0[start_base + 10624ll];
                blk_2[20] = h0[start_base + 10752ll];
                blk_2[21] = h0[start_base + 10880ll];
                blk_2[22] = h0[start_base + 11008ll];
                blk_2[23] = h0[start_base + 11136ll];
                blk_2[24] = h0[start_base + 11264ll];
                blk_2[25] = h0[start_base + 11392ll];
                blk_2[26] = h0[start_base + 11520ll];
                blk_2[27] = h0[start_base + 11648ll];
                blk_2[28] = h0[start_base + 11776ll];
                blk_2[29] = h0[start_base + 11904ll];
                blk_2[30] = h0[start_base + 12032ll];
                blk_2[31] = h0[start_base + 12160ll];
            }
            tmem_st_x32_f32(tmem_s + row_off + 64, blk_2);
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
            if (start_src == 2) {
                blk_3[0] = seg_start[start_base + 12288ll];
                blk_3[1] = seg_start[start_base + 12416ll];
                blk_3[2] = seg_start[start_base + 12544ll];
                blk_3[3] = seg_start[start_base + 12672ll];
                blk_3[4] = seg_start[start_base + 12800ll];
                blk_3[5] = seg_start[start_base + 12928ll];
                blk_3[6] = seg_start[start_base + 13056ll];
                blk_3[7] = seg_start[start_base + 13184ll];
                blk_3[8] = seg_start[start_base + 13312ll];
                blk_3[9] = seg_start[start_base + 13440ll];
                blk_3[10] = seg_start[start_base + 13568ll];
                blk_3[11] = seg_start[start_base + 13696ll];
                blk_3[12] = seg_start[start_base + 13824ll];
                blk_3[13] = seg_start[start_base + 13952ll];
                blk_3[14] = seg_start[start_base + 14080ll];
                blk_3[15] = seg_start[start_base + 14208ll];
                blk_3[16] = seg_start[start_base + 14336ll];
                blk_3[17] = seg_start[start_base + 14464ll];
                blk_3[18] = seg_start[start_base + 14592ll];
                blk_3[19] = seg_start[start_base + 14720ll];
                blk_3[20] = seg_start[start_base + 14848ll];
                blk_3[21] = seg_start[start_base + 14976ll];
                blk_3[22] = seg_start[start_base + 15104ll];
                blk_3[23] = seg_start[start_base + 15232ll];
                blk_3[24] = seg_start[start_base + 15360ll];
                blk_3[25] = seg_start[start_base + 15488ll];
                blk_3[26] = seg_start[start_base + 15616ll];
                blk_3[27] = seg_start[start_base + 15744ll];
                blk_3[28] = seg_start[start_base + 15872ll];
                blk_3[29] = seg_start[start_base + 16000ll];
                blk_3[30] = seg_start[start_base + 16128ll];
                blk_3[31] = seg_start[start_base + 16256ll];
            }
            if (start_src == 1) {
                blk_3[0] = h0[start_base + 12288ll];
                blk_3[1] = h0[start_base + 12416ll];
                blk_3[2] = h0[start_base + 12544ll];
                blk_3[3] = h0[start_base + 12672ll];
                blk_3[4] = h0[start_base + 12800ll];
                blk_3[5] = h0[start_base + 12928ll];
                blk_3[6] = h0[start_base + 13056ll];
                blk_3[7] = h0[start_base + 13184ll];
                blk_3[8] = h0[start_base + 13312ll];
                blk_3[9] = h0[start_base + 13440ll];
                blk_3[10] = h0[start_base + 13568ll];
                blk_3[11] = h0[start_base + 13696ll];
                blk_3[12] = h0[start_base + 13824ll];
                blk_3[13] = h0[start_base + 13952ll];
                blk_3[14] = h0[start_base + 14080ll];
                blk_3[15] = h0[start_base + 14208ll];
                blk_3[16] = h0[start_base + 14336ll];
                blk_3[17] = h0[start_base + 14464ll];
                blk_3[18] = h0[start_base + 14592ll];
                blk_3[19] = h0[start_base + 14720ll];
                blk_3[20] = h0[start_base + 14848ll];
                blk_3[21] = h0[start_base + 14976ll];
                blk_3[22] = h0[start_base + 15104ll];
                blk_3[23] = h0[start_base + 15232ll];
                blk_3[24] = h0[start_base + 15360ll];
                blk_3[25] = h0[start_base + 15488ll];
                blk_3[26] = h0[start_base + 15616ll];
                blk_3[27] = h0[start_base + 15744ll];
                blk_3[28] = h0[start_base + 15872ll];
                blk_3[29] = h0[start_base + 16000ll];
                blk_3[30] = h0[start_base + 16128ll];
                blk_3[31] = h0[start_base + 16256ll];
            }
            tmem_st_x32_f32(tmem_s + row_off + 96, blk_3);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int store_h = tape & 1;
            int store_v = tape >> 1 & 1;
            unsigned int hst_c = 0;
            unsigned int gs_c = 0;
            int tok_begin = 0;
            int n_last = 0;
            if (nc > 0) {
                tok_begin = chunk_start[c_begin];
                n_last = chunk_len[c_begin + nc - 1];
            }
            unsigned int _phase_gates_ready = 0;
            unsigned int _phase_ws_full_0 = 0;
            unsigned int _phase_vn_free_0 = 1;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int n = ((c + 1 == nc) ? n_last : 64);
                mbarrier_wait(gates_ready_addr + (gs_c) * 8, _phase_gates_ready);
                int g_off = c % 4 * 384;
                float e2gl = s_gt[g_off + 320];
                long long h_base = ((long long)(c_begin + c) * (long long)num_v_heads + (long long)hv) * 16384ll + (long long)(vrow * 128);
                int h_mode = store_h;
                if (c > 0) {
                    h_mode = 2 * store_h;
                }
                unsigned int h_stage = s_kq_addr + (unsigned int)((c + 2) % 3 * 32768);
                #pragma unroll 1
                for (int kb = 0; kb < 4; kb++) {
                    float sv[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                        : "r"(tmem_s + row_off + 32 * kb));
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
                        :: "r"(tmem_hs + row_off + 16 * kb), "r"(*reinterpret_cast<const uint32_t*>(&hp[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp[15])));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                if (h_mode == 0) {
                    #pragma unroll 1
                    for (int kb_1 = 0; kb_1 < 4; kb_1++) {
                        float sv_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31])
                            : "r"(tmem_s + row_off + 32 * kb_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        const float2 _scale2_0 = {e2gl, e2gl};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_1)[_ls], _scale2_0);
                        tmem_st_x32_f32(tmem_s + row_off + 32 * kb_1, sv_1);
                    }
                }
                if (h_mode != 0) {
                    #pragma unroll 1
                    for (int kb_2 = 0; kb_2 < 4; kb_2++) {
                        float sv_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv_2[0]), "=f"(sv_2[1]), "=f"(sv_2[2]), "=f"(sv_2[3]), "=f"(sv_2[4]), "=f"(sv_2[5]), "=f"(sv_2[6]), "=f"(sv_2[7]), "=f"(sv_2[8]), "=f"(sv_2[9]), "=f"(sv_2[10]), "=f"(sv_2[11]), "=f"(sv_2[12]), "=f"(sv_2[13]), "=f"(sv_2[14]), "=f"(sv_2[15]), "=f"(sv_2[16]), "=f"(sv_2[17]), "=f"(sv_2[18]), "=f"(sv_2[19]), "=f"(sv_2[20]), "=f"(sv_2[21]), "=f"(sv_2[22]), "=f"(sv_2[23]), "=f"(sv_2[24]), "=f"(sv_2[25]), "=f"(sv_2[26]), "=f"(sv_2[27]), "=f"(sv_2[28]), "=f"(sv_2[29]), "=f"(sv_2[30]), "=f"(sv_2[31])
                            : "r"(tmem_s + row_off + 32 * kb_2));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        unsigned int hp_1[16];
                        if (h_mode != 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_2[_lp*2 + 0], sv_2[_lp*2+1 + 0]));
                                hp_1[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        if (h_mode == 1) {
                            {
                                const unsigned* _raw_stv8_1 = reinterpret_cast<const unsigned*>(hp_1);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(h_out + (h_base + (long long)(32 * kb_2)))), "r"(_raw_stv8_1[0]), "r"(_raw_stv8_1[1]), "r"(_raw_stv8_1[2]), "r"(_raw_stv8_1[3]), "r"(_raw_stv8_1[4]), "r"(_raw_stv8_1[5]), "r"(_raw_stv8_1[6]), "r"(_raw_stv8_1[7]) : "memory");
                            }
                            {
                                const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(hp_1 + (8));
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(h_out + (h_base + (long long)(32 * kb_2 + 16)))), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                            }
                        }
                        if (h_mode == 2) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(h_stage + (unsigned int)(vrow / 64 * 16384 + 32 * kb_2 / 64 * 8192 + vrow % 64 * 128 + (32 * kb_2 % 64 / 8 ^ vrow % 8) * 16 + 32 * kb_2 % 8 * 2)), "r"(hp_1[0]), "r"(hp_1[1]), "r"(hp_1[2]), "r"(hp_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(h_stage + (unsigned int)(vrow / 64 * 16384 + (32 * kb_2 + 8) / 64 * 8192 + vrow % 64 * 128 + ((32 * kb_2 + 8) % 64 / 8 ^ vrow % 8) * 16 + (32 * kb_2 + 8) % 8 * 2)), "r"(hp_1[4]), "r"(hp_1[5]), "r"(hp_1[6]), "r"(hp_1[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(h_stage + (unsigned int)(vrow / 64 * 16384 + (32 * kb_2 + 16) / 64 * 8192 + vrow % 64 * 128 + ((32 * kb_2 + 16) % 64 / 8 ^ vrow % 8) * 16 + (32 * kb_2 + 16) % 8 * 2)), "r"(hp_1[8]), "r"(hp_1[9]), "r"(hp_1[10]), "r"(hp_1[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(h_stage + (unsigned int)(vrow / 64 * 16384 + (32 * kb_2 + 24) / 64 * 8192 + vrow % 64 * 128 + ((32 * kb_2 + 24) % 64 / 8 ^ vrow % 8) * 16 + (32 * kb_2 + 24) % 8 * 2)), "r"(hp_1[12]), "r"(hp_1[13]), "r"(hp_1[14]), "r"(hp_1[15]) : "memory");
                        }
                        const float2 _scale2_3 = {e2gl, e2gl};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_2)[_ls], _scale2_3);
                        tmem_st_x32_f32(tmem_s + row_off + 32 * kb_2, sv_2);
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(h_staged_addr + (hst_c) * 8);
                hst_c += 1;
                if (hst_c == 3) { hst_c = 0; }
                mbarrier_wait(ws_full_addr, _phase_ws_full_0);
                _phase_ws_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll 1
                for (int hh = 0; hh < 2; hh++) {
                    float vv[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(vv[0]), "=f"(vv[1]), "=f"(vv[2]), "=f"(vv[3]), "=f"(vv[4]), "=f"(vv[5]), "=f"(vv[6]), "=f"(vv[7]), "=f"(vv[8]), "=f"(vv[9]), "=f"(vv[10]), "=f"(vv[11]), "=f"(vv[12]), "=f"(vv[13]), "=f"(vv[14]), "=f"(vv[15]), "=f"(vv[16]), "=f"(vv[17]), "=f"(vv[18]), "=f"(vv[19]), "=f"(vv[20]), "=f"(vv[21]), "=f"(vv[22]), "=f"(vv[23]), "=f"(vv[24]), "=f"(vv[25]), "=f"(vv[26]), "=f"(vv[27]), "=f"(vv[28]), "=f"(vv[29]), "=f"(vv[30]), "=f"(vv[31])
                        : "r"(tmem_vn + row_off + 32 * hh));
                    float dks[32];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[0])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(0) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[4])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(4) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 4))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[8])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(8) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 8))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[12])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(12) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 12))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[16])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(16) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 16))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[20])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(20) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 20))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[24])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(24) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 24))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dks[28])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dks[(28) + 3]))
                        : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 * hh + 28))));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (n < 64) {
                        vv[0] = ((32 * hh < n) ? vv[0] : 0.0f);
                        vv[1] = ((32 * hh + 1 < n) ? vv[1] : 0.0f);
                        vv[2] = ((32 * hh + 2 < n) ? vv[2] : 0.0f);
                        vv[3] = ((32 * hh + 3 < n) ? vv[3] : 0.0f);
                        vv[4] = ((32 * hh + 4 < n) ? vv[4] : 0.0f);
                        vv[5] = ((32 * hh + 5 < n) ? vv[5] : 0.0f);
                        vv[6] = ((32 * hh + 6 < n) ? vv[6] : 0.0f);
                        vv[7] = ((32 * hh + 7 < n) ? vv[7] : 0.0f);
                        vv[8] = ((32 * hh + 8 < n) ? vv[8] : 0.0f);
                        vv[9] = ((32 * hh + 9 < n) ? vv[9] : 0.0f);
                        vv[10] = ((32 * hh + 10 < n) ? vv[10] : 0.0f);
                        vv[11] = ((32 * hh + 11 < n) ? vv[11] : 0.0f);
                        vv[12] = ((32 * hh + 12 < n) ? vv[12] : 0.0f);
                        vv[13] = ((32 * hh + 13 < n) ? vv[13] : 0.0f);
                        vv[14] = ((32 * hh + 14 < n) ? vv[14] : 0.0f);
                        vv[15] = ((32 * hh + 15 < n) ? vv[15] : 0.0f);
                        vv[16] = ((32 * hh + 16 < n) ? vv[16] : 0.0f);
                        vv[17] = ((32 * hh + 17 < n) ? vv[17] : 0.0f);
                        vv[18] = ((32 * hh + 18 < n) ? vv[18] : 0.0f);
                        vv[19] = ((32 * hh + 19 < n) ? vv[19] : 0.0f);
                        vv[20] = ((32 * hh + 20 < n) ? vv[20] : 0.0f);
                        vv[21] = ((32 * hh + 21 < n) ? vv[21] : 0.0f);
                        vv[22] = ((32 * hh + 22 < n) ? vv[22] : 0.0f);
                        vv[23] = ((32 * hh + 23 < n) ? vv[23] : 0.0f);
                        vv[24] = ((32 * hh + 24 < n) ? vv[24] : 0.0f);
                        vv[25] = ((32 * hh + 25 < n) ? vv[25] : 0.0f);
                        vv[26] = ((32 * hh + 26 < n) ? vv[26] : 0.0f);
                        vv[27] = ((32 * hh + 27 < n) ? vv[27] : 0.0f);
                        vv[28] = ((32 * hh + 28 < n) ? vv[28] : 0.0f);
                        vv[29] = ((32 * hh + 29 < n) ? vv[29] : 0.0f);
                        vv[30] = ((32 * hh + 30 < n) ? vv[30] : 0.0f);
                        vv[31] = ((32 * hh + 31 < n) ? vv[31] : 0.0f);
                    }
                    unsigned int vb[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv[_lp*2 + 0], vv[_lp*2+1 + 0]));
                        vb[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(tmem_vnb + row_off + 16 * hh), "r"(*reinterpret_cast<const uint32_t*>(&vb[0])), "r"(*reinterpret_cast<const uint32_t*>(&vb[1])), "r"(*reinterpret_cast<const uint32_t*>(&vb[2])), "r"(*reinterpret_cast<const uint32_t*>(&vb[3])), "r"(*reinterpret_cast<const uint32_t*>(&vb[4])), "r"(*reinterpret_cast<const uint32_t*>(&vb[5])), "r"(*reinterpret_cast<const uint32_t*>(&vb[6])), "r"(*reinterpret_cast<const uint32_t*>(&vb[7])), "r"(*reinterpret_cast<const uint32_t*>(&vb[8])), "r"(*reinterpret_cast<const uint32_t*>(&vb[9])), "r"(*reinterpret_cast<const uint32_t*>(&vb[10])), "r"(*reinterpret_cast<const uint32_t*>(&vb[11])), "r"(*reinterpret_cast<const uint32_t*>(&vb[12])), "r"(*reinterpret_cast<const uint32_t*>(&vb[13])), "r"(*reinterpret_cast<const uint32_t*>(&vb[14])), "r"(*reinterpret_cast<const uint32_t*>(&vb[15])));
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
                        :: "r"(tmem_vp + row_off + 16 * hh), "r"(*reinterpret_cast<const uint32_t*>(&vp[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp[7])), "r"(*reinterpret_cast<const uint32_t*>(&vp[8])), "r"(*reinterpret_cast<const uint32_t*>(&vp[9])), "r"(*reinterpret_cast<const uint32_t*>(&vp[10])), "r"(*reinterpret_cast<const uint32_t*>(&vp[11])), "r"(*reinterpret_cast<const uint32_t*>(&vp[12])), "r"(*reinterpret_cast<const uint32_t*>(&vp[13])), "r"(*reinterpret_cast<const uint32_t*>(&vp[14])), "r"(*reinterpret_cast<const uint32_t*>(&vp[15])));
                }
                mbarrier_arrive(gates_free_addr + (gs_c) * 8);
                gs_c += 1;
                if (gs_c == 4) { gs_c = 0; _phase_gates_ready ^= 1; }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(vp_ready_addr);
                mbarrier_wait(vn_free_addr, _phase_vn_free_0);
                _phase_vn_free_0 ^= 1;
                if (store_v != 0) {
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    unsigned int vt[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vt[0])), "=r"(*reinterpret_cast<uint32_t*>(&vt[1])), "=r"(*reinterpret_cast<uint32_t*>(&vt[2])), "=r"(*reinterpret_cast<uint32_t*>(&vt[3])), "=r"(*reinterpret_cast<uint32_t*>(&vt[4])), "=r"(*reinterpret_cast<uint32_t*>(&vt[5])), "=r"(*reinterpret_cast<uint32_t*>(&vt[6])), "=r"(*reinterpret_cast<uint32_t*>(&vt[7])), "=r"(*reinterpret_cast<uint32_t*>(&vt[8])), "=r"(*reinterpret_cast<uint32_t*>(&vt[9])), "=r"(*reinterpret_cast<uint32_t*>(&vt[10])), "=r"(*reinterpret_cast<uint32_t*>(&vt[11])), "=r"(*reinterpret_cast<uint32_t*>(&vt[12])), "=r"(*reinterpret_cast<uint32_t*>(&vt[13])), "=r"(*reinterpret_cast<uint32_t*>(&vt[14])), "=r"(*reinterpret_cast<uint32_t*>(&vt[15]))
                        : "r"(tmem_vnb + row_off + 0));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vt[16])), "=r"(*reinterpret_cast<uint32_t*>(&vt[17])), "=r"(*reinterpret_cast<uint32_t*>(&vt[18])), "=r"(*reinterpret_cast<uint32_t*>(&vt[19])), "=r"(*reinterpret_cast<uint32_t*>(&vt[20])), "=r"(*reinterpret_cast<uint32_t*>(&vt[21])), "=r"(*reinterpret_cast<uint32_t*>(&vt[22])), "=r"(*reinterpret_cast<uint32_t*>(&vt[23])), "=r"(*reinterpret_cast<uint32_t*>(&vt[24])), "=r"(*reinterpret_cast<uint32_t*>(&vt[25])), "=r"(*reinterpret_cast<uint32_t*>(&vt[26])), "=r"(*reinterpret_cast<uint32_t*>(&vt[27])), "=r"(*reinterpret_cast<uint32_t*>(&vt[28])), "=r"(*reinterpret_cast<uint32_t*>(&vt[29])), "=r"(*reinterpret_cast<uint32_t*>(&vt[30])), "=r"(*reinterpret_cast<uint32_t*>(&vt[31]))
                        : "r"(tmem_vnb + row_off + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    int mi = lane / 8;
                    int jr = lane % 8;
                    uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 8 * (mi / 2)) / 64 * 8192 + (4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 8 * (mi / 2)) % 64 / 8 ^ (4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&vt[0])), "r"(*reinterpret_cast<const uint32_t*>(&vt[1])), "r"(*reinterpret_cast<const uint32_t*>(&vt[2])), "r"(*reinterpret_cast<const uint32_t*>(&vt[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((16 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 8 * (mi / 2)) / 64 * 8192 + (16 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 8 * (mi / 2)) % 64 / 8 ^ (16 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&vt[4])), "r"(*reinterpret_cast<const uint32_t*>(&vt[5])), "r"(*reinterpret_cast<const uint32_t*>(&vt[6])), "r"(*reinterpret_cast<const uint32_t*>(&vt[7]))
                        : "memory");
                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((32 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 8 * (mi / 2)) / 64 * 8192 + (32 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 8 * (mi / 2)) % 64 / 8 ^ (32 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&vt[8])), "r"(*reinterpret_cast<const uint32_t*>(&vt[9])), "r"(*reinterpret_cast<const uint32_t*>(&vt[10])), "r"(*reinterpret_cast<const uint32_t*>(&vt[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((48 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 8 * (mi / 2)) / 64 * 8192 + (48 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 8 * (mi / 2)) % 64 / 8 ^ (48 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&vt[12])), "r"(*reinterpret_cast<const uint32_t*>(&vt[13])), "r"(*reinterpret_cast<const uint32_t*>(&vt[14])), "r"(*reinterpret_cast<const uint32_t*>(&vt[15]))
                        : "memory");
                    uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) / 64 * 8192 + (4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 16 + 8 * (mi / 2)) % 64 / 8 ^ (4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&vt[16])), "r"(*reinterpret_cast<const uint32_t*>(&vt[17])), "r"(*reinterpret_cast<const uint32_t*>(&vt[18])), "r"(*reinterpret_cast<const uint32_t*>(&vt[19]))
                        : "memory");
                    uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((16 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) / 64 * 8192 + (16 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 16 + 8 * (mi / 2)) % 64 / 8 ^ (16 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&vt[20])), "r"(*reinterpret_cast<const uint32_t*>(&vt[21])), "r"(*reinterpret_cast<const uint32_t*>(&vt[22])), "r"(*reinterpret_cast<const uint32_t*>(&vt[23]))
                        : "memory");
                    uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((32 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) / 64 * 8192 + (32 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 16 + 8 * (mi / 2)) % 64 / 8 ^ (32 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&vt[24])), "r"(*reinterpret_cast<const uint32_t*>(&vt[25])), "r"(*reinterpret_cast<const uint32_t*>(&vt[26])), "r"(*reinterpret_cast<const uint32_t*>(&vt[27]))
                        : "memory");
                    uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(s_vst_addr + (unsigned int)((48 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) / 64 * 16384 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) / 64 * 8192 + (48 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 64 * 128 + ((32 * (warp % 4) + 16 + 8 * (mi / 2)) % 64 / 8 ^ (48 + 4 * (jr / 2) + 2 * (mi % 2) + jr % 2) % 8) * 16 + (32 * (warp % 4) + 16 + 8 * (mi / 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&vt[28])), "r"(*reinterpret_cast<const uint32_t*>(&vt[29])), "r"(*reinterpret_cast<const uint32_t*>(&vt[30])), "r"(*reinterpret_cast<const uint32_t*>(&vt[31]))
                        : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vn_staged_addr);
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (store_final_state != 0) {
                if (is_last != 0) {
                    float blk_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(blk_4[0]), "=f"(blk_4[1]), "=f"(blk_4[2]), "=f"(blk_4[3]), "=f"(blk_4[4]), "=f"(blk_4[5]), "=f"(blk_4[6]), "=f"(blk_4[7]), "=f"(blk_4[8]), "=f"(blk_4[9]), "=f"(blk_4[10]), "=f"(blk_4[11]), "=f"(blk_4[12]), "=f"(blk_4[13]), "=f"(blk_4[14]), "=f"(blk_4[15]), "=f"(blk_4[16]), "=f"(blk_4[17]), "=f"(blk_4[18]), "=f"(blk_4[19]), "=f"(blk_4[20]), "=f"(blk_4[21]), "=f"(blk_4[22]), "=f"(blk_4[23]), "=f"(blk_4[24]), "=f"(blk_4[25]), "=f"(blk_4[26]), "=f"(blk_4[27]), "=f"(blk_4[28]), "=f"(blk_4[29]), "=f"(blk_4[30]), "=f"(blk_4[31])
                        : "r"(tmem_s + row_off + 0));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 0ll] = blk_4[0];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 128ll] = blk_4[1];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 256ll] = blk_4[2];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 384ll] = blk_4[3];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 512ll] = blk_4[4];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 640ll] = blk_4[5];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 768ll] = blk_4[6];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 896ll] = blk_4[7];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1024ll] = blk_4[8];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1152ll] = blk_4[9];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1280ll] = blk_4[10];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1408ll] = blk_4[11];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1536ll] = blk_4[12];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1664ll] = blk_4[13];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1792ll] = blk_4[14];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 1920ll] = blk_4[15];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2048ll] = blk_4[16];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2176ll] = blk_4[17];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2304ll] = blk_4[18];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2432ll] = blk_4[19];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2560ll] = blk_4[20];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2688ll] = blk_4[21];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2816ll] = blk_4[22];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 2944ll] = blk_4[23];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3072ll] = blk_4[24];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3200ll] = blk_4[25];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3328ll] = blk_4[26];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3456ll] = blk_4[27];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3584ll] = blk_4[28];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3712ll] = blk_4[29];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3840ll] = blk_4[30];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 3968ll] = blk_4[31];
                    float blk_5[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(blk_5[0]), "=f"(blk_5[1]), "=f"(blk_5[2]), "=f"(blk_5[3]), "=f"(blk_5[4]), "=f"(blk_5[5]), "=f"(blk_5[6]), "=f"(blk_5[7]), "=f"(blk_5[8]), "=f"(blk_5[9]), "=f"(blk_5[10]), "=f"(blk_5[11]), "=f"(blk_5[12]), "=f"(blk_5[13]), "=f"(blk_5[14]), "=f"(blk_5[15]), "=f"(blk_5[16]), "=f"(blk_5[17]), "=f"(blk_5[18]), "=f"(blk_5[19]), "=f"(blk_5[20]), "=f"(blk_5[21]), "=f"(blk_5[22]), "=f"(blk_5[23]), "=f"(blk_5[24]), "=f"(blk_5[25]), "=f"(blk_5[26]), "=f"(blk_5[27]), "=f"(blk_5[28]), "=f"(blk_5[29]), "=f"(blk_5[30]), "=f"(blk_5[31])
                        : "r"(tmem_s + row_off + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4096ll] = blk_5[0];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4224ll] = blk_5[1];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4352ll] = blk_5[2];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4480ll] = blk_5[3];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4608ll] = blk_5[4];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4736ll] = blk_5[5];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4864ll] = blk_5[6];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 4992ll] = blk_5[7];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5120ll] = blk_5[8];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5248ll] = blk_5[9];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5376ll] = blk_5[10];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5504ll] = blk_5[11];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5632ll] = blk_5[12];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5760ll] = blk_5[13];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 5888ll] = blk_5[14];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6016ll] = blk_5[15];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6144ll] = blk_5[16];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6272ll] = blk_5[17];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6400ll] = blk_5[18];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6528ll] = blk_5[19];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6656ll] = blk_5[20];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6784ll] = blk_5[21];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 6912ll] = blk_5[22];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7040ll] = blk_5[23];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7168ll] = blk_5[24];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7296ll] = blk_5[25];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7424ll] = blk_5[26];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7552ll] = blk_5[27];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7680ll] = blk_5[28];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7808ll] = blk_5[29];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 7936ll] = blk_5[30];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8064ll] = blk_5[31];
                    float blk_6[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(blk_6[0]), "=f"(blk_6[1]), "=f"(blk_6[2]), "=f"(blk_6[3]), "=f"(blk_6[4]), "=f"(blk_6[5]), "=f"(blk_6[6]), "=f"(blk_6[7]), "=f"(blk_6[8]), "=f"(blk_6[9]), "=f"(blk_6[10]), "=f"(blk_6[11]), "=f"(blk_6[12]), "=f"(blk_6[13]), "=f"(blk_6[14]), "=f"(blk_6[15]), "=f"(blk_6[16]), "=f"(blk_6[17]), "=f"(blk_6[18]), "=f"(blk_6[19]), "=f"(blk_6[20]), "=f"(blk_6[21]), "=f"(blk_6[22]), "=f"(blk_6[23]), "=f"(blk_6[24]), "=f"(blk_6[25]), "=f"(blk_6[26]), "=f"(blk_6[27]), "=f"(blk_6[28]), "=f"(blk_6[29]), "=f"(blk_6[30]), "=f"(blk_6[31])
                        : "r"(tmem_s + row_off + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8192ll] = blk_6[0];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8320ll] = blk_6[1];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8448ll] = blk_6[2];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8576ll] = blk_6[3];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8704ll] = blk_6[4];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8832ll] = blk_6[5];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 8960ll] = blk_6[6];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9088ll] = blk_6[7];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9216ll] = blk_6[8];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9344ll] = blk_6[9];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9472ll] = blk_6[10];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9600ll] = blk_6[11];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9728ll] = blk_6[12];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9856ll] = blk_6[13];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 9984ll] = blk_6[14];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10112ll] = blk_6[15];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10240ll] = blk_6[16];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10368ll] = blk_6[17];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10496ll] = blk_6[18];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10624ll] = blk_6[19];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10752ll] = blk_6[20];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 10880ll] = blk_6[21];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11008ll] = blk_6[22];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11136ll] = blk_6[23];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11264ll] = blk_6[24];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11392ll] = blk_6[25];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11520ll] = blk_6[26];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11648ll] = blk_6[27];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11776ll] = blk_6[28];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 11904ll] = blk_6[29];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12032ll] = blk_6[30];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12160ll] = blk_6[31];
                    float blk_7[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(blk_7[0]), "=f"(blk_7[1]), "=f"(blk_7[2]), "=f"(blk_7[3]), "=f"(blk_7[4]), "=f"(blk_7[5]), "=f"(blk_7[6]), "=f"(blk_7[7]), "=f"(blk_7[8]), "=f"(blk_7[9]), "=f"(blk_7[10]), "=f"(blk_7[11]), "=f"(blk_7[12]), "=f"(blk_7[13]), "=f"(blk_7[14]), "=f"(blk_7[15]), "=f"(blk_7[16]), "=f"(blk_7[17]), "=f"(blk_7[18]), "=f"(blk_7[19]), "=f"(blk_7[20]), "=f"(blk_7[21]), "=f"(blk_7[22]), "=f"(blk_7[23]), "=f"(blk_7[24]), "=f"(blk_7[25]), "=f"(blk_7[26]), "=f"(blk_7[27]), "=f"(blk_7[28]), "=f"(blk_7[29]), "=f"(blk_7[30]), "=f"(blk_7[31])
                        : "r"(tmem_s + row_off + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12288ll] = blk_7[0];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12416ll] = blk_7[1];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12544ll] = blk_7[2];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12672ll] = blk_7[3];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12800ll] = blk_7[4];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 12928ll] = blk_7[5];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13056ll] = blk_7[6];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13184ll] = blk_7[7];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13312ll] = blk_7[8];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13440ll] = blk_7[9];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13568ll] = blk_7[10];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13696ll] = blk_7[11];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13824ll] = blk_7[12];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 13952ll] = blk_7[13];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14080ll] = blk_7[14];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14208ll] = blk_7[15];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14336ll] = blk_7[16];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14464ll] = blk_7[17];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14592ll] = blk_7[18];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14720ll] = blk_7[19];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14848ll] = blk_7[20];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 14976ll] = blk_7[21];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15104ll] = blk_7[22];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15232ll] = blk_7[23];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15360ll] = blk_7[24];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15488ll] = blk_7[25];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15616ll] = blk_7[26];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15744ll] = blk_7[27];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 15872ll] = blk_7[28];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 16000ll] = blk_7[29];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 16128ll] = blk_7[30];
                    final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + 16256ll] = blk_7[31];
                }
            }
        }
    }
    // ---- Role: outr ----
    if (warp >= 4 && warp <= 7) {
        { // outr_main
            int blk_8 = blockIdx.x;
            int seg_1 = walk[blk_8 / num_v_heads];
            int hv_1 = blk_8 % num_v_heads;
            int c_begin_1 = seg_chunk_start[seg_1];
            int nc_1 = seg_chunk_start[seg_1 + 1] - c_begin_1;
            int seq_1 = seg_seq[seg_1];
            int lane_1 = lane;
            int q4 = warp % 4;
            int row_off_1 = q4 * 32 << 16;
            int i_row = 16 * q4 + lane_1 % 16;
            int j0 = 32 * (lane_1 / 16);
            unsigned int gw_c = 0;
            int tok_begin_1 = 0;
            int n_last_1 = 0;
            if (nc_1 > 0) {
                tok_begin_1 = chunk_start[c_begin_1];
                n_last_1 = chunk_len[c_begin_1 + nc_1 - 1];
            }
            unsigned int _phase_w_full_0 = 0;
            unsigned int _phase_ws_full_0_1 = 1;
            unsigned int _phase_wn_free_0 = 1;
            if (nc_1 > 0) {
                mbarrier_wait(w_full_addr, _phase_w_full_0);
                _phase_w_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float wv[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])), "=r"(*reinterpret_cast<uint32_t*>(&wv[1])), "=r"(*reinterpret_cast<uint32_t*>(&wv[2])), "=r"(*reinterpret_cast<uint32_t*>(&wv[3])), "=r"(*reinterpret_cast<uint32_t*>(&wv[4])), "=r"(*reinterpret_cast<uint32_t*>(&wv[5])), "=r"(*reinterpret_cast<uint32_t*>(&wv[6])), "=r"(*reinterpret_cast<uint32_t*>(&wv[7])), "=r"(*reinterpret_cast<uint32_t*>(&wv[8])), "=r"(*reinterpret_cast<uint32_t*>(&wv[9])), "=r"(*reinterpret_cast<uint32_t*>(&wv[10])), "=r"(*reinterpret_cast<uint32_t*>(&wv[11])), "=r"(*reinterpret_cast<uint32_t*>(&wv[12])), "=r"(*reinterpret_cast<uint32_t*>(&wv[13])), "=r"(*reinterpret_cast<uint32_t*>(&wv[14])), "=r"(*reinterpret_cast<uint32_t*>(&wv[15])), "=r"(*reinterpret_cast<uint32_t*>(&wv[16])), "=r"(*reinterpret_cast<uint32_t*>(&wv[17])), "=r"(*reinterpret_cast<uint32_t*>(&wv[18])), "=r"(*reinterpret_cast<uint32_t*>(&wv[19])), "=r"(*reinterpret_cast<uint32_t*>(&wv[20])), "=r"(*reinterpret_cast<uint32_t*>(&wv[21])), "=r"(*reinterpret_cast<uint32_t*>(&wv[22])), "=r"(*reinterpret_cast<uint32_t*>(&wv[23])), "=r"(*reinterpret_cast<uint32_t*>(&wv[24])), "=r"(*reinterpret_cast<uint32_t*>(&wv[25])), "=r"(*reinterpret_cast<uint32_t*>(&wv[26])), "=r"(*reinterpret_cast<uint32_t*>(&wv[27])), "=r"(*reinterpret_cast<uint32_t*>(&wv[28])), "=r"(*reinterpret_cast<uint32_t*>(&wv[29])), "=r"(*reinterpret_cast<uint32_t*>(&wv[30])), "=r"(*reinterpret_cast<uint32_t*>(&wv[31]))
                    : "r"(tmem_w + row_off_1 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wv[32])), "=r"(*reinterpret_cast<uint32_t*>(&wv[33])), "=r"(*reinterpret_cast<uint32_t*>(&wv[34])), "=r"(*reinterpret_cast<uint32_t*>(&wv[35])), "=r"(*reinterpret_cast<uint32_t*>(&wv[36])), "=r"(*reinterpret_cast<uint32_t*>(&wv[37])), "=r"(*reinterpret_cast<uint32_t*>(&wv[38])), "=r"(*reinterpret_cast<uint32_t*>(&wv[39])), "=r"(*reinterpret_cast<uint32_t*>(&wv[40])), "=r"(*reinterpret_cast<uint32_t*>(&wv[41])), "=r"(*reinterpret_cast<uint32_t*>(&wv[42])), "=r"(*reinterpret_cast<uint32_t*>(&wv[43])), "=r"(*reinterpret_cast<uint32_t*>(&wv[44])), "=r"(*reinterpret_cast<uint32_t*>(&wv[45])), "=r"(*reinterpret_cast<uint32_t*>(&wv[46])), "=r"(*reinterpret_cast<uint32_t*>(&wv[47])), "=r"(*reinterpret_cast<uint32_t*>(&wv[48])), "=r"(*reinterpret_cast<uint32_t*>(&wv[49])), "=r"(*reinterpret_cast<uint32_t*>(&wv[50])), "=r"(*reinterpret_cast<uint32_t*>(&wv[51])), "=r"(*reinterpret_cast<uint32_t*>(&wv[52])), "=r"(*reinterpret_cast<uint32_t*>(&wv[53])), "=r"(*reinterpret_cast<uint32_t*>(&wv[54])), "=r"(*reinterpret_cast<uint32_t*>(&wv[55])), "=r"(*reinterpret_cast<uint32_t*>(&wv[56])), "=r"(*reinterpret_cast<uint32_t*>(&wv[57])), "=r"(*reinterpret_cast<uint32_t*>(&wv[58])), "=r"(*reinterpret_cast<uint32_t*>(&wv[59])), "=r"(*reinterpret_cast<uint32_t*>(&wv[60])), "=r"(*reinterpret_cast<uint32_t*>(&wv[61])), "=r"(*reinterpret_cast<uint32_t*>(&wv[62])), "=r"(*reinterpret_cast<uint32_t*>(&wv[63]))
                    : "r"(tmem_w + row_off_1 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(w_drained_addr);
                unsigned int wp[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wv[_lp*2 + 0], wv[_lp*2+1 + 0]));
                    wp[_lp] = *(uint32_t*)&_bf2;
                }
                mbarrier_wait(ws_full_addr, _phase_ws_full_0_1);
                _phase_ws_full_0_1 ^= 1;
                mbarrier_wait(wn_free_addr, _phase_wn_free_0);
                _phase_wn_free_0 ^= 1;
                int mi_1 = lane / 8;
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_1 % 2)) / 64 * 8192 + (8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_1 % 2)) % 64 / 8 ^ (8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&wp[0])), "r"(*reinterpret_cast<const uint32_t*>(&wp[1])), "r"(*reinterpret_cast<const uint32_t*>(&wp[2])), "r"(*reinterpret_cast<const uint32_t*>(&wp[3]))
                    : "memory");
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_1 % 2)) / 64 * 8192 + (16 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_1 % 2)) % 64 / 8 ^ (16 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&wp[4])), "r"(*reinterpret_cast<const uint32_t*>(&wp[5])), "r"(*reinterpret_cast<const uint32_t*>(&wp[6])), "r"(*reinterpret_cast<const uint32_t*>(&wp[7]))
                    : "memory");
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_1 % 2)) / 64 * 8192 + (32 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_1 % 2)) % 64 / 8 ^ (32 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&wp[8])), "r"(*reinterpret_cast<const uint32_t*>(&wp[9])), "r"(*reinterpret_cast<const uint32_t*>(&wp[10])), "r"(*reinterpret_cast<const uint32_t*>(&wp[11]))
                    : "memory");
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_1 % 2)) / 64 * 8192 + (48 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_1 % 2)) % 64 / 8 ^ (48 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&wp[12])), "r"(*reinterpret_cast<const uint32_t*>(&wp[13])), "r"(*reinterpret_cast<const uint32_t*>(&wp[14])), "r"(*reinterpret_cast<const uint32_t*>(&wp[15]))
                    : "memory");
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&wp[16])), "r"(*reinterpret_cast<const uint32_t*>(&wp[17])), "r"(*reinterpret_cast<const uint32_t*>(&wp[18])), "r"(*reinterpret_cast<const uint32_t*>(&wp[19]))
                    : "memory");
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (16 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (16 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&wp[20])), "r"(*reinterpret_cast<const uint32_t*>(&wp[21])), "r"(*reinterpret_cast<const uint32_t*>(&wp[22])), "r"(*reinterpret_cast<const uint32_t*>(&wp[23]))
                    : "memory");
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (32 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (32 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&wp[24])), "r"(*reinterpret_cast<const uint32_t*>(&wp[25])), "r"(*reinterpret_cast<const uint32_t*>(&wp[26])), "r"(*reinterpret_cast<const uint32_t*>(&wp[27]))
                    : "memory");
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi_1 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_1 % 2)) / 64 * 8192 + (48 + 8 * (mi_1 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_1 % 2)) % 64 / 8 ^ (48 + 8 * (mi_1 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_1 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&wp[28])), "r"(*reinterpret_cast<const uint32_t*>(&wp[29])), "r"(*reinterpret_cast<const uint32_t*>(&wp[30])), "r"(*reinterpret_cast<const uint32_t*>(&wp[31]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(wt_ready_addr);
            }
            unsigned int _phase_gates_ready_1 = 0;
            unsigned int _phase_g_full_0 = 0;
            unsigned int _phase_o1_full_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_ost_free_0 = 1;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                int n_1 = ((c_1 + 1 == nc_1) ? n_last_1 : 64);
                mbarrier_wait(gates_ready_addr + (gw_c) * 8, _phase_gates_ready_1);
                int g_off_1 = c_1 % 4 * 384;
                mbarrier_wait(g_full_addr, _phase_g_full_0);
                _phase_g_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float gv[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(gv[0]), "=f"(gv[1]), "=f"(gv[2]), "=f"(gv[3]), "=f"(gv[4]), "=f"(gv[5]), "=f"(gv[6]), "=f"(gv[7]), "=f"(gv[8]), "=f"(gv[9]), "=f"(gv[10]), "=f"(gv[11]), "=f"(gv[12]), "=f"(gv[13]), "=f"(gv[14]), "=f"(gv[15]), "=f"(gv[16]), "=f"(gv[17]), "=f"(gv[18]), "=f"(gv[19]), "=f"(gv[20]), "=f"(gv[21]), "=f"(gv[22]), "=f"(gv[23]), "=f"(gv[24]), "=f"(gv[25]), "=f"(gv[26]), "=f"(gv[27]), "=f"(gv[28]), "=f"(gv[29]), "=f"(gv[30]), "=f"(gv[31])
                    : "r"(tmem_g + row_off_1));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float gi = s_gt[g_off_1 + i_row];
                float pv8_0[8];
                float gj8_0[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_0[(4) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 4))));
                float _exp2_8 = approx_exp2(gi - gj8_0[0]);
                float pij_0 = gv[0] * _exp2_8 * scale;
                pv8_0[0] = ((j0 <= i_row) ? ((i_row < n_1) ? pij_0 : 0.0f) : 0.0f);
                float _exp2_9 = approx_exp2(gi - gj8_0[1]);
                float pij_1 = gv[1] * _exp2_9 * scale;
                pv8_0[1] = ((j0 + 1 <= i_row) ? ((i_row < n_1) ? pij_1 : 0.0f) : 0.0f);
                float _exp2_10 = approx_exp2(gi - gj8_0[2]);
                float pij_2 = gv[2] * _exp2_10 * scale;
                pv8_0[2] = ((j0 + 2 <= i_row) ? ((i_row < n_1) ? pij_2 : 0.0f) : 0.0f);
                float _exp2_11 = approx_exp2(gi - gj8_0[3]);
                float pij_3 = gv[3] * _exp2_11 * scale;
                pv8_0[3] = ((j0 + 3 <= i_row) ? ((i_row < n_1) ? pij_3 : 0.0f) : 0.0f);
                float _exp2_12 = approx_exp2(gi - gj8_0[4]);
                float pij_4 = gv[4] * _exp2_12 * scale;
                pv8_0[4] = ((j0 + 4 <= i_row) ? ((i_row < n_1) ? pij_4 : 0.0f) : 0.0f);
                float _exp2_13 = approx_exp2(gi - gj8_0[5]);
                float pij_5 = gv[5] * _exp2_13 * scale;
                pv8_0[5] = ((j0 + 5 <= i_row) ? ((i_row < n_1) ? pij_5 : 0.0f) : 0.0f);
                float _exp2_14 = approx_exp2(gi - gj8_0[6]);
                float pij_6 = gv[6] * _exp2_14 * scale;
                pv8_0[6] = ((j0 + 6 <= i_row) ? ((i_row < n_1) ? pij_6 : 0.0f) : 0.0f);
                float _exp2_15 = approx_exp2(gi - gj8_0[7]);
                float pij_7 = gv[7] * _exp2_15 * scale;
                pv8_0[7] = ((j0 + 7 <= i_row) ? ((i_row < n_1) ? pij_7 : 0.0f) : 0.0f);
                unsigned int pk_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_0[_lp*2 + 0], pv8_0[_lp*2+1 + 0]));
                    pk_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (j0 / 8 ^ i_row % 8) * 16)), "r"(pk_0[0]), "r"(pk_0[1]), "r"(pk_0[2]), "r"(pk_0[3]) : "memory");
                float pv8_1[8];
                float gj8_1[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 8))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_1[(4) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 8 + 4))));
                float _exp2_16 = approx_exp2(gi - gj8_1[0]);
                float pij_8 = gv[8] * _exp2_16 * scale;
                pv8_1[0] = ((j0 + 8 <= i_row) ? ((i_row < n_1) ? pij_8 : 0.0f) : 0.0f);
                float _exp2_17 = approx_exp2(gi - gj8_1[1]);
                float pij_9 = gv[9] * _exp2_17 * scale;
                pv8_1[1] = ((j0 + 8 + 1 <= i_row) ? ((i_row < n_1) ? pij_9 : 0.0f) : 0.0f);
                float _exp2_18 = approx_exp2(gi - gj8_1[2]);
                float pij_10 = gv[10] * _exp2_18 * scale;
                pv8_1[2] = ((j0 + 8 + 2 <= i_row) ? ((i_row < n_1) ? pij_10 : 0.0f) : 0.0f);
                float _exp2_19 = approx_exp2(gi - gj8_1[3]);
                float pij_11 = gv[11] * _exp2_19 * scale;
                pv8_1[3] = ((j0 + 8 + 3 <= i_row) ? ((i_row < n_1) ? pij_11 : 0.0f) : 0.0f);
                float _exp2_20 = approx_exp2(gi - gj8_1[4]);
                float pij_12 = gv[12] * _exp2_20 * scale;
                pv8_1[4] = ((j0 + 8 + 4 <= i_row) ? ((i_row < n_1) ? pij_12 : 0.0f) : 0.0f);
                float _exp2_21 = approx_exp2(gi - gj8_1[5]);
                float pij_13 = gv[13] * _exp2_21 * scale;
                pv8_1[5] = ((j0 + 8 + 5 <= i_row) ? ((i_row < n_1) ? pij_13 : 0.0f) : 0.0f);
                float _exp2_22 = approx_exp2(gi - gj8_1[6]);
                float pij_14 = gv[14] * _exp2_22 * scale;
                pv8_1[6] = ((j0 + 8 + 6 <= i_row) ? ((i_row < n_1) ? pij_14 : 0.0f) : 0.0f);
                float _exp2_23 = approx_exp2(gi - gj8_1[7]);
                float pij_15 = gv[15] * _exp2_23 * scale;
                pv8_1[7] = ((j0 + 8 + 7 <= i_row) ? ((i_row < n_1) ? pij_15 : 0.0f) : 0.0f);
                unsigned int pk_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_1[_lp*2 + 0], pv8_1[_lp*2+1 + 0]));
                    pk_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (j0 / 8 + 1 ^ i_row % 8) * 16)), "r"(pk_1[0]), "r"(pk_1[1]), "r"(pk_1[2]), "r"(pk_1[3]) : "memory");
                float pv8_2[8];
                float gj8_2[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 16))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_2[(4) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 16 + 4))));
                float _exp2_24 = approx_exp2(gi - gj8_2[0]);
                float pij_16 = gv[16] * _exp2_24 * scale;
                pv8_2[0] = ((j0 + 16 <= i_row) ? ((i_row < n_1) ? pij_16 : 0.0f) : 0.0f);
                float _exp2_25 = approx_exp2(gi - gj8_2[1]);
                float pij_17 = gv[17] * _exp2_25 * scale;
                pv8_2[1] = ((j0 + 16 + 1 <= i_row) ? ((i_row < n_1) ? pij_17 : 0.0f) : 0.0f);
                float _exp2_26 = approx_exp2(gi - gj8_2[2]);
                float pij_18 = gv[18] * _exp2_26 * scale;
                pv8_2[2] = ((j0 + 16 + 2 <= i_row) ? ((i_row < n_1) ? pij_18 : 0.0f) : 0.0f);
                float _exp2_27 = approx_exp2(gi - gj8_2[3]);
                float pij_19 = gv[19] * _exp2_27 * scale;
                pv8_2[3] = ((j0 + 16 + 3 <= i_row) ? ((i_row < n_1) ? pij_19 : 0.0f) : 0.0f);
                float _exp2_28 = approx_exp2(gi - gj8_2[4]);
                float pij_20 = gv[20] * _exp2_28 * scale;
                pv8_2[4] = ((j0 + 16 + 4 <= i_row) ? ((i_row < n_1) ? pij_20 : 0.0f) : 0.0f);
                float _exp2_29 = approx_exp2(gi - gj8_2[5]);
                float pij_21 = gv[21] * _exp2_29 * scale;
                pv8_2[5] = ((j0 + 16 + 5 <= i_row) ? ((i_row < n_1) ? pij_21 : 0.0f) : 0.0f);
                float _exp2_30 = approx_exp2(gi - gj8_2[6]);
                float pij_22 = gv[22] * _exp2_30 * scale;
                pv8_2[6] = ((j0 + 16 + 6 <= i_row) ? ((i_row < n_1) ? pij_22 : 0.0f) : 0.0f);
                float _exp2_31 = approx_exp2(gi - gj8_2[7]);
                float pij_23 = gv[23] * _exp2_31 * scale;
                pv8_2[7] = ((j0 + 16 + 7 <= i_row) ? ((i_row < n_1) ? pij_23 : 0.0f) : 0.0f);
                unsigned int pk_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_2[_lp*2 + 0], pv8_2[_lp*2+1 + 0]));
                    pk_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (j0 / 8 + 2 ^ i_row % 8) * 16)), "r"(pk_2[0]), "r"(pk_2[1]), "r"(pk_2[2]), "r"(pk_2[3]) : "memory");
                float pv8_3[8];
                float gj8_3[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 24))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gj8_3[(4) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + j0 + 24 + 4))));
                float _exp2_32 = approx_exp2(gi - gj8_3[0]);
                float pij_24 = gv[24] * _exp2_32 * scale;
                pv8_3[0] = ((j0 + 24 <= i_row) ? ((i_row < n_1) ? pij_24 : 0.0f) : 0.0f);
                float _exp2_33 = approx_exp2(gi - gj8_3[1]);
                float pij_25 = gv[25] * _exp2_33 * scale;
                pv8_3[1] = ((j0 + 24 + 1 <= i_row) ? ((i_row < n_1) ? pij_25 : 0.0f) : 0.0f);
                float _exp2_34 = approx_exp2(gi - gj8_3[2]);
                float pij_26 = gv[26] * _exp2_34 * scale;
                pv8_3[2] = ((j0 + 24 + 2 <= i_row) ? ((i_row < n_1) ? pij_26 : 0.0f) : 0.0f);
                float _exp2_35 = approx_exp2(gi - gj8_3[3]);
                float pij_27 = gv[27] * _exp2_35 * scale;
                pv8_3[3] = ((j0 + 24 + 3 <= i_row) ? ((i_row < n_1) ? pij_27 : 0.0f) : 0.0f);
                float _exp2_36 = approx_exp2(gi - gj8_3[4]);
                float pij_28 = gv[28] * _exp2_36 * scale;
                pv8_3[4] = ((j0 + 24 + 4 <= i_row) ? ((i_row < n_1) ? pij_28 : 0.0f) : 0.0f);
                float _exp2_37 = approx_exp2(gi - gj8_3[5]);
                float pij_29 = gv[29] * _exp2_37 * scale;
                pv8_3[5] = ((j0 + 24 + 5 <= i_row) ? ((i_row < n_1) ? pij_29 : 0.0f) : 0.0f);
                float _exp2_38 = approx_exp2(gi - gj8_3[6]);
                float pij_30 = gv[30] * _exp2_38 * scale;
                pv8_3[6] = ((j0 + 24 + 6 <= i_row) ? ((i_row < n_1) ? pij_30 : 0.0f) : 0.0f);
                float _exp2_39 = approx_exp2(gi - gj8_3[7]);
                float pij_31 = gv[31] * _exp2_39 * scale;
                pv8_3[7] = ((j0 + 24 + 7 <= i_row) ? ((i_row < n_1) ? pij_31 : 0.0f) : 0.0f);
                unsigned int pk_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_3[_lp*2 + 0], pv8_3[_lp*2+1 + 0]));
                    pk_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (j0 / 8 + 3 ^ i_row % 8) * 16)), "r"(pk_3[0]), "r"(pk_3[1]), "r"(pk_3[2]), "r"(pk_3[3]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(pg_ready_addr);
                mbarrier_wait(o1_full_addr, _phase_o1_full_0);
                _phase_o1_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float ov1[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(ov1[0]), "=f"(ov1[1]), "=f"(ov1[2]), "=f"(ov1[3]), "=f"(ov1[4]), "=f"(ov1[5]), "=f"(ov1[6]), "=f"(ov1[7]), "=f"(ov1[8]), "=f"(ov1[9]), "=f"(ov1[10]), "=f"(ov1[11]), "=f"(ov1[12]), "=f"(ov1[13]), "=f"(ov1[14]), "=f"(ov1[15]), "=f"(ov1[16]), "=f"(ov1[17]), "=f"(ov1[18]), "=f"(ov1[19]), "=f"(ov1[20]), "=f"(ov1[21]), "=f"(ov1[22]), "=f"(ov1[23]), "=f"(ov1[24]), "=f"(ov1[25]), "=f"(ov1[26]), "=f"(ov1[27]), "=f"(ov1[28]), "=f"(ov1[29]), "=f"(ov1[30]), "=f"(ov1[31])
                    : "r"(tmem_o + row_off_1));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(ov1[32]), "=f"(ov1[33]), "=f"(ov1[34]), "=f"(ov1[35]), "=f"(ov1[36]), "=f"(ov1[37]), "=f"(ov1[38]), "=f"(ov1[39]), "=f"(ov1[40]), "=f"(ov1[41]), "=f"(ov1[42]), "=f"(ov1[43]), "=f"(ov1[44]), "=f"(ov1[45]), "=f"(ov1[46]), "=f"(ov1[47]), "=f"(ov1[48]), "=f"(ov1[49]), "=f"(ov1[50]), "=f"(ov1[51]), "=f"(ov1[52]), "=f"(ov1[53]), "=f"(ov1[54]), "=f"(ov1[55]), "=f"(ov1[56]), "=f"(ov1[57]), "=f"(ov1[58]), "=f"(ov1[59]), "=f"(ov1[60]), "=f"(ov1[61]), "=f"(ov1[62]), "=f"(ov1[63])
                    : "r"(tmem_o + row_off_1 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float eg4_0[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_0[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128))));
                ov1[0] = ov1[0] * eg4_0[0];
                ov1[1] = ov1[1] * eg4_0[1];
                ov1[2] = ov1[2] * eg4_0[2];
                ov1[3] = ov1[3] * eg4_0[3];
                float eg4_1[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_1[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 4))));
                ov1[4] = ov1[4] * eg4_1[0];
                ov1[5] = ov1[5] * eg4_1[1];
                ov1[6] = ov1[6] * eg4_1[2];
                ov1[7] = ov1[7] * eg4_1[3];
                float eg4_2[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_2[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 8))));
                ov1[8] = ov1[8] * eg4_2[0];
                ov1[9] = ov1[9] * eg4_2[1];
                ov1[10] = ov1[10] * eg4_2[2];
                ov1[11] = ov1[11] * eg4_2[3];
                float eg4_3[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_3[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 12))));
                ov1[12] = ov1[12] * eg4_3[0];
                ov1[13] = ov1[13] * eg4_3[1];
                ov1[14] = ov1[14] * eg4_3[2];
                ov1[15] = ov1[15] * eg4_3[3];
                float eg4_4[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_4[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 16))));
                ov1[16] = ov1[16] * eg4_4[0];
                ov1[17] = ov1[17] * eg4_4[1];
                ov1[18] = ov1[18] * eg4_4[2];
                ov1[19] = ov1[19] * eg4_4[3];
                float eg4_5[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_5[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 20))));
                ov1[20] = ov1[20] * eg4_5[0];
                ov1[21] = ov1[21] * eg4_5[1];
                ov1[22] = ov1[22] * eg4_5[2];
                ov1[23] = ov1[23] * eg4_5[3];
                float eg4_6[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_6[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 24))));
                ov1[24] = ov1[24] * eg4_6[0];
                ov1[25] = ov1[25] * eg4_6[1];
                ov1[26] = ov1[26] * eg4_6[2];
                ov1[27] = ov1[27] * eg4_6[3];
                float eg4_7[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_7[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 28))));
                ov1[28] = ov1[28] * eg4_7[0];
                ov1[29] = ov1[29] * eg4_7[1];
                ov1[30] = ov1[30] * eg4_7[2];
                ov1[31] = ov1[31] * eg4_7[3];
                float eg4_8[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_8[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 32))));
                ov1[32] = ov1[32] * eg4_8[0];
                ov1[33] = ov1[33] * eg4_8[1];
                ov1[34] = ov1[34] * eg4_8[2];
                ov1[35] = ov1[35] * eg4_8[3];
                float eg4_9[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_9[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 36))));
                ov1[36] = ov1[36] * eg4_9[0];
                ov1[37] = ov1[37] * eg4_9[1];
                ov1[38] = ov1[38] * eg4_9[2];
                ov1[39] = ov1[39] * eg4_9[3];
                float eg4_10[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_10[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 40))));
                ov1[40] = ov1[40] * eg4_10[0];
                ov1[41] = ov1[41] * eg4_10[1];
                ov1[42] = ov1[42] * eg4_10[2];
                ov1[43] = ov1[43] * eg4_10[3];
                float eg4_11[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_11[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 44))));
                ov1[44] = ov1[44] * eg4_11[0];
                ov1[45] = ov1[45] * eg4_11[1];
                ov1[46] = ov1[46] * eg4_11[2];
                ov1[47] = ov1[47] * eg4_11[3];
                float eg4_12[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_12[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 48))));
                ov1[48] = ov1[48] * eg4_12[0];
                ov1[49] = ov1[49] * eg4_12[1];
                ov1[50] = ov1[50] * eg4_12[2];
                ov1[51] = ov1[51] * eg4_12[3];
                float eg4_13[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_13[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 52))));
                ov1[52] = ov1[52] * eg4_13[0];
                ov1[53] = ov1[53] * eg4_13[1];
                ov1[54] = ov1[54] * eg4_13[2];
                ov1[55] = ov1[55] * eg4_13[3];
                float eg4_14[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_14[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 56))));
                ov1[56] = ov1[56] * eg4_14[0];
                ov1[57] = ov1[57] * eg4_14[1];
                ov1[58] = ov1[58] * eg4_14[2];
                ov1[59] = ov1[59] * eg4_14[3];
                float eg4_15[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&eg4_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&eg4_15[(0) + 3]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off_1 + 128 + 60))));
                ov1[60] = ov1[60] * eg4_15[0];
                ov1[61] = ov1[61] * eg4_15[1];
                ov1[62] = ov1[62] * eg4_15[2];
                ov1[63] = ov1[63] * eg4_15[3];
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x64.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                    :: "r"(tmem_o + row_off_1), "f"(ov1[0]), "f"(ov1[1]), "f"(ov1[2]), "f"(ov1[3]), "f"(ov1[4]), "f"(ov1[5]), "f"(ov1[6]), "f"(ov1[7]), "f"(ov1[8]), "f"(ov1[9]), "f"(ov1[10]), "f"(ov1[11]), "f"(ov1[12]), "f"(ov1[13]), "f"(ov1[14]), "f"(ov1[15]), "f"(ov1[16]), "f"(ov1[17]), "f"(ov1[18]), "f"(ov1[19]), "f"(ov1[20]), "f"(ov1[21]), "f"(ov1[22]), "f"(ov1[23]), "f"(ov1[24]), "f"(ov1[25]), "f"(ov1[26]), "f"(ov1[27]), "f"(ov1[28]), "f"(ov1[29]), "f"(ov1[30]), "f"(ov1[31]), "f"(ov1[32]), "f"(ov1[33]), "f"(ov1[34]), "f"(ov1[35]), "f"(ov1[36]), "f"(ov1[37]), "f"(ov1[38]), "f"(ov1[39]), "f"(ov1[40]), "f"(ov1[41]), "f"(ov1[42]), "f"(ov1[43]), "f"(ov1[44]), "f"(ov1[45]), "f"(ov1[46]), "f"(ov1[47]), "f"(ov1[48]), "f"(ov1[49]), "f"(ov1[50]), "f"(ov1[51]), "f"(ov1[52]), "f"(ov1[53]), "f"(ov1[54]), "f"(ov1[55]), "f"(ov1[56]), "f"(ov1[57]), "f"(ov1[58]), "f"(ov1[59]), "f"(ov1[60]), "f"(ov1[61]), "f"(ov1[62]), "f"(ov1[63]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_scaled_addr);
                mbarrier_arrive(gates_free_addr + (gw_c) * 8);
                gw_c += 1;
                if (gw_c == 4) { gw_c = 0; _phase_gates_ready_1 ^= 1; }
                if (c_1 + 1 < nc_1) {
                    mbarrier_wait(w_full_addr, _phase_w_full_0);
                    _phase_w_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float wv_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[31]))
                        : "r"(tmem_w + row_off_1 + 0));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&wv_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&wv_1[63]))
                        : "r"(tmem_w + row_off_1 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(w_drained_addr);
                    unsigned int wp_1[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wv_1[_lp*2 + 0], wv_1[_lp*2+1 + 0]));
                        wp_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    mbarrier_wait(ws_full_addr, _phase_ws_full_0_1);
                    _phase_ws_full_0_1 ^= 1;
                    mbarrier_wait(wn_free_addr, _phase_wn_free_0);
                    _phase_wn_free_0 ^= 1;
                    int mi_2 = lane / 8;
                    uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_2 % 2)) / 64 * 8192 + (8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_2 % 2)) % 64 / 8 ^ (8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_2 % 2)) / 64 * 8192 + (16 + 8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_2 % 2)) % 64 / 8 ^ (16 + 8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[7]))
                        : "memory");
                    uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_2 % 2)) / 64 * 8192 + (32 + 8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_2 % 2)) % 64 / 8 ^ (32 + 8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_2 % 2)) / 64 * 8192 + (48 + 8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_2 % 2)) % 64 / 8 ^ (48 + 8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[15]))
                        : "memory");
                    uint32_t _stmatrix_addr_44 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_2 % 2)) / 64 * 8192 + (8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_2 % 2)) % 64 / 8 ^ (8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_44), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[19]))
                        : "memory");
                    uint32_t _stmatrix_addr_45 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((16 + 8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_2 % 2)) / 64 * 8192 + (16 + 8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_2 % 2)) % 64 / 8 ^ (16 + 8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_45), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[23]))
                        : "memory");
                    uint32_t _stmatrix_addr_46 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((32 + 8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_2 % 2)) / 64 * 8192 + (32 + 8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_2 % 2)) % 64 / 8 ^ (32 + 8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_46), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[27]))
                        : "memory");
                    uint32_t _stmatrix_addr_47 = static_cast<uint32_t>(s_wn_addr + (unsigned int)((48 + 8 * (mi_2 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_2 % 2)) / 64 * 8192 + (48 + 8 * (mi_2 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_2 % 2)) % 64 / 8 ^ (48 + 8 * (mi_2 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_2 % 2)) % 8 * 2));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_47), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&wp_1[31]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(wt_ready_addr);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float od_v[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&od_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[1])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[2])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[3])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[4])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[5])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[6])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[7])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[8])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[9])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[10])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[11])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[12])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[13])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[14])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[15])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[16])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[17])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[18])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[19])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[20])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[21])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[22])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[23])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[24])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[25])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[26])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[27])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[28])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[29])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[30])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[31]))
                    : "r"(tmem_o + row_off_1 + 0));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&od_v[32])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[33])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[34])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[35])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[36])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[37])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[38])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[39])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[40])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[41])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[42])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[43])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[44])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[45])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[46])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[47])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[48])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[49])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[50])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[51])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[52])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[53])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[54])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[55])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[56])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[57])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[58])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[59])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[60])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[61])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[62])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[63]))
                    : "r"(tmem_o + row_off_1 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_free_addr);
                mbarrier_wait(ost_free_addr, _phase_ost_free_0);
                _phase_ost_free_0 ^= 1;
                unsigned int od_p[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(od_v[_lp*2 + 0], od_v[_lp*2+1 + 0]));
                    od_p[_lp] = *(uint32_t*)&_bf2;
                }
                int mi_3 = lane / 8;
                uint32_t _stmatrix_addr_48 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_3 % 2)) / 64 * 8192 + (8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_3 % 2)) % 64 / 8 ^ (8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_48), "r"(*reinterpret_cast<const uint32_t*>(&od_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[3]))
                    : "memory");
                uint32_t _stmatrix_addr_49 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((16 + 8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_3 % 2)) / 64 * 8192 + (16 + 8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_3 % 2)) % 64 / 8 ^ (16 + 8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_49), "r"(*reinterpret_cast<const uint32_t*>(&od_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[7]))
                    : "memory");
                uint32_t _stmatrix_addr_50 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((32 + 8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_3 % 2)) / 64 * 8192 + (32 + 8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_3 % 2)) % 64 / 8 ^ (32 + 8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_50), "r"(*reinterpret_cast<const uint32_t*>(&od_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[11]))
                    : "memory");
                uint32_t _stmatrix_addr_51 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((48 + 8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 8 * (mi_3 % 2)) / 64 * 8192 + (48 + 8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 8 * (mi_3 % 2)) % 64 / 8 ^ (48 + 8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_51), "r"(*reinterpret_cast<const uint32_t*>(&od_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[15]))
                    : "memory");
                uint32_t _stmatrix_addr_52 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_3 % 2)) / 64 * 8192 + (8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_3 % 2)) % 64 / 8 ^ (8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_52), "r"(*reinterpret_cast<const uint32_t*>(&od_p[16])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[17])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[18])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[19]))
                    : "memory");
                uint32_t _stmatrix_addr_53 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((16 + 8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_3 % 2)) / 64 * 8192 + (16 + 8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_3 % 2)) % 64 / 8 ^ (16 + 8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_53), "r"(*reinterpret_cast<const uint32_t*>(&od_p[20])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[21])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[22])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[23]))
                    : "memory");
                uint32_t _stmatrix_addr_54 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((32 + 8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_3 % 2)) / 64 * 8192 + (32 + 8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_3 % 2)) % 64 / 8 ^ (32 + 8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_54), "r"(*reinterpret_cast<const uint32_t*>(&od_p[24])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[25])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[26])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[27]))
                    : "memory");
                uint32_t _stmatrix_addr_55 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((48 + 8 * (mi_3 / 2) + lane % 8) / 64 * 16384 + (32 * q4 + 16 + 8 * (mi_3 % 2)) / 64 * 8192 + (48 + 8 * (mi_3 / 2) + lane % 8) % 64 * 128 + ((32 * q4 + 16 + 8 * (mi_3 % 2)) % 64 / 8 ^ (48 + 8 * (mi_3 / 2) + lane % 8) % 8) * 16 + (32 * q4 + 16 + 8 * (mi_3 % 2)) % 8 * 2));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_55), "r"(*reinterpret_cast<const uint32_t*>(&od_p[28])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[29])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[30])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[31]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(o_staged_addr);
            }
        }
    }
    // ---- Role: prep ----
    if (warp >= 8 && warp <= 11) {
        { // prep_main
            int blk_9 = blockIdx.x;
            int seg_2 = walk[blk_9 / num_v_heads];
            int hv_2 = blk_9 % num_v_heads;
            int c_begin_2 = seg_chunk_start[seg_2];
            int nc_2 = seg_chunk_start[seg_2 + 1] - c_begin_2;
            int seq_2 = seg_seq[seg_2];
            int ptid = tid - 256;
            int row_off_2 = warp % 4 * 32 << 16;
            int tok_begin_2 = 0;
            int n_last_2 = 0;
            if (nc_2 > 0) {
                tok_begin_2 = chunk_start[c_begin_2];
                n_last_2 = chunk_len[c_begin_2 + nc_2 - 1];
            }
            unsigned int gp_c = 0;
            unsigned int kq_c = 0;
            unsigned int va_c = 0;
            unsigned int _phase_gates_ready_2 = 0;
            unsigned int _phase_va_full = 0;
            unsigned int _phase_kq_full = 0;
            unsigned int _phase_bk_free_0 = 1;
            #pragma unroll 1
            for (int c_2 = 0; c_2 < nc_2; c_2++) {
                int n_2 = ((c_2 + 1 == nc_2) ? n_last_2 : 64);
                mbarrier_wait(gates_ready_addr + (gp_c) * 8, _phase_gates_ready_2);
                int g_off_2 = c_2 % 4 * 384;
                mbarrier_wait(va_full_addr + (va_c) * 8, _phase_va_full);
                unsigned int zero = 0;
                #pragma unroll 4
                for (int it = 0; it < 8; it++) {
                    int tr = ptid / 8 + 16 * (it % 4);
                    unsigned int v_at = s_vT_addr + (unsigned int)(c_2 % 2 * 24576 + 8192 * (it / 4) + tr * 128 + ptid % 8 * 16);
                    if (tr < n_2) {
                        float bt = s_gt[g_off_2 + 192 + tr];
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
                    unsigned int k_at = s_kq_addr + (unsigned int)(c_2 % 3 * 32768 + 16384 * (it_1 / 4) + tr_1 * 128 + ptid % 8 * 16);
                    unsigned int bk_at = s_bk_addr + (unsigned int)(8192 * (it_1 / 4) + tr_1 * 128 + ptid % 8 * 16);
                    if (tr_1 < n_2) {
                        float bg = s_gt[g_off_2 + 256 + tr_1];
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
                int _parity_u_full = (int)((unsigned int)c_2 % 2);
                mbarrier_wait(u_full_addr, _parity_u_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float uv[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(uv[0]), "=f"(uv[1]), "=f"(uv[2]), "=f"(uv[3]), "=f"(uv[4]), "=f"(uv[5]), "=f"(uv[6]), "=f"(uv[7]), "=f"(uv[8]), "=f"(uv[9]), "=f"(uv[10]), "=f"(uv[11]), "=f"(uv[12]), "=f"(uv[13]), "=f"(uv[14]), "=f"(uv[15]), "=f"(uv[16]), "=f"(uv[17]), "=f"(uv[18]), "=f"(uv[19]), "=f"(uv[20]), "=f"(uv[21]), "=f"(uv[22]), "=f"(uv[23]), "=f"(uv[24]), "=f"(uv[25]), "=f"(uv[26]), "=f"(uv[27]), "=f"(uv[28]), "=f"(uv[29]), "=f"(uv[30]), "=f"(uv[31])
                    : "r"(tmem_vn + row_off_2));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(uv[32]), "=f"(uv[33]), "=f"(uv[34]), "=f"(uv[35]), "=f"(uv[36]), "=f"(uv[37]), "=f"(uv[38]), "=f"(uv[39]), "=f"(uv[40]), "=f"(uv[41]), "=f"(uv[42]), "=f"(uv[43]), "=f"(uv[44]), "=f"(uv[45]), "=f"(uv[46]), "=f"(uv[47]), "=f"(uv[48]), "=f"(uv[49]), "=f"(uv[50]), "=f"(uv[51]), "=f"(uv[52]), "=f"(uv[53]), "=f"(uv[54]), "=f"(uv[55]), "=f"(uv[56]), "=f"(uv[57]), "=f"(uv[58]), "=f"(uv[59]), "=f"(uv[60]), "=f"(uv[61]), "=f"(uv[62]), "=f"(uv[63])
                    : "r"(tmem_vn + row_off_2 + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int up[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uv[_lp*2 + 0], uv[_lp*2+1 + 0]));
                    up[_lp] = *(uint32_t*)&_bf2;
                }
                uv[0] = __uint_as_float(up[0] << 16);
                uv[1] = __uint_as_float(up[0] & 4294901760u);
                uv[2] = __uint_as_float(up[1] << 16);
                uv[3] = __uint_as_float(up[1] & 4294901760u);
                uv[4] = __uint_as_float(up[2] << 16);
                uv[5] = __uint_as_float(up[2] & 4294901760u);
                uv[6] = __uint_as_float(up[3] << 16);
                uv[7] = __uint_as_float(up[3] & 4294901760u);
                uv[8] = __uint_as_float(up[4] << 16);
                uv[9] = __uint_as_float(up[4] & 4294901760u);
                uv[10] = __uint_as_float(up[5] << 16);
                uv[11] = __uint_as_float(up[5] & 4294901760u);
                uv[12] = __uint_as_float(up[6] << 16);
                uv[13] = __uint_as_float(up[6] & 4294901760u);
                uv[14] = __uint_as_float(up[7] << 16);
                uv[15] = __uint_as_float(up[7] & 4294901760u);
                uv[16] = __uint_as_float(up[8] << 16);
                uv[17] = __uint_as_float(up[8] & 4294901760u);
                uv[18] = __uint_as_float(up[9] << 16);
                uv[19] = __uint_as_float(up[9] & 4294901760u);
                uv[20] = __uint_as_float(up[10] << 16);
                uv[21] = __uint_as_float(up[10] & 4294901760u);
                uv[22] = __uint_as_float(up[11] << 16);
                uv[23] = __uint_as_float(up[11] & 4294901760u);
                uv[24] = __uint_as_float(up[12] << 16);
                uv[25] = __uint_as_float(up[12] & 4294901760u);
                uv[26] = __uint_as_float(up[13] << 16);
                uv[27] = __uint_as_float(up[13] & 4294901760u);
                uv[28] = __uint_as_float(up[14] << 16);
                uv[29] = __uint_as_float(up[14] & 4294901760u);
                uv[30] = __uint_as_float(up[15] << 16);
                uv[31] = __uint_as_float(up[15] & 4294901760u);
                uv[32] = __uint_as_float(up[16] << 16);
                uv[33] = __uint_as_float(up[16] & 4294901760u);
                uv[34] = __uint_as_float(up[17] << 16);
                uv[35] = __uint_as_float(up[17] & 4294901760u);
                uv[36] = __uint_as_float(up[18] << 16);
                uv[37] = __uint_as_float(up[18] & 4294901760u);
                uv[38] = __uint_as_float(up[19] << 16);
                uv[39] = __uint_as_float(up[19] & 4294901760u);
                uv[40] = __uint_as_float(up[20] << 16);
                uv[41] = __uint_as_float(up[20] & 4294901760u);
                uv[42] = __uint_as_float(up[21] << 16);
                uv[43] = __uint_as_float(up[21] & 4294901760u);
                uv[44] = __uint_as_float(up[22] << 16);
                uv[45] = __uint_as_float(up[22] & 4294901760u);
                uv[46] = __uint_as_float(up[23] << 16);
                uv[47] = __uint_as_float(up[23] & 4294901760u);
                uv[48] = __uint_as_float(up[24] << 16);
                uv[49] = __uint_as_float(up[24] & 4294901760u);
                uv[50] = __uint_as_float(up[25] << 16);
                uv[51] = __uint_as_float(up[25] & 4294901760u);
                uv[52] = __uint_as_float(up[26] << 16);
                uv[53] = __uint_as_float(up[26] & 4294901760u);
                uv[54] = __uint_as_float(up[27] << 16);
                uv[55] = __uint_as_float(up[27] & 4294901760u);
                uv[56] = __uint_as_float(up[28] << 16);
                uv[57] = __uint_as_float(up[28] & 4294901760u);
                uv[58] = __uint_as_float(up[29] << 16);
                uv[59] = __uint_as_float(up[29] & 4294901760u);
                uv[60] = __uint_as_float(up[30] << 16);
                uv[61] = __uint_as_float(up[30] & 4294901760u);
                uv[62] = __uint_as_float(up[31] << 16);
                uv[63] = __uint_as_float(up[31] & 4294901760u);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x64.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                    :: "r"(tmem_vn + row_off_2), "f"(uv[0]), "f"(uv[1]), "f"(uv[2]), "f"(uv[3]), "f"(uv[4]), "f"(uv[5]), "f"(uv[6]), "f"(uv[7]), "f"(uv[8]), "f"(uv[9]), "f"(uv[10]), "f"(uv[11]), "f"(uv[12]), "f"(uv[13]), "f"(uv[14]), "f"(uv[15]), "f"(uv[16]), "f"(uv[17]), "f"(uv[18]), "f"(uv[19]), "f"(uv[20]), "f"(uv[21]), "f"(uv[22]), "f"(uv[23]), "f"(uv[24]), "f"(uv[25]), "f"(uv[26]), "f"(uv[27]), "f"(uv[28]), "f"(uv[29]), "f"(uv[30]), "f"(uv[31]), "f"(uv[32]), "f"(uv[33]), "f"(uv[34]), "f"(uv[35]), "f"(uv[36]), "f"(uv[37]), "f"(uv[38]), "f"(uv[39]), "f"(uv[40]), "f"(uv[41]), "f"(uv[42]), "f"(uv[43]), "f"(uv[44]), "f"(uv[45]), "f"(uv[46]), "f"(uv[47]), "f"(uv[48]), "f"(uv[49]), "f"(uv[50]), "f"(uv[51]), "f"(uv[52]), "f"(uv[53]), "f"(uv[54]), "f"(uv[55]), "f"(uv[56]), "f"(uv[57]), "f"(uv[58]), "f"(uv[59]), "f"(uv[60]), "f"(uv[61]), "f"(uv[62]), "f"(uv[63]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(u_rounded_addr);
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int blk_10 = blockIdx.x;
            int seg_3 = walk[blk_10 / num_v_heads];
            int hv_3 = blk_10 % num_v_heads;
            int c_begin_3 = seg_chunk_start[seg_3];
            int nc_3 = seg_chunk_start[seg_3 + 1] - c_begin_3;
            int seq_3 = seg_seq[seg_3];
            unsigned int kq_c_1 = 0;
            unsigned int va_c_1 = 0;
            unsigned int _phase_va_full_1 = 0;
            unsigned int _phase_bk_ready_0 = 0;
            unsigned int _phase_w_drained_0 = 1;
            unsigned int _phase_kq_full_1 = 0;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_wt_ready_0 = 0;
            unsigned int _phase_o_free_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_pg_ready_0 = 0;
            unsigned int _phase_o_scaled_0 = 0;
            if (elect_sync()) {
                if (nc_3 > 0) {
                    mbarrier_wait(va_full_addr + (va_c_1) * 8, _phase_va_full_1);
                    mbarrier_wait(v_fixed_addr + (va_c_1) * 8, _phase_va_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((((s_vT_addr) >> 4) & 0x7FFF) | 0x2000000) + (va_c_1) * 1536;
                    int _smem_desc_lo_1 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                    int _smem_desc_lo_0 = ((((s_vT_addr) >> 4) & 0x3FFF) | 0x2000000) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_vn), "r"(0) : "memory");
                    tcgen05_commit(u_full_addr);
                    mbarrier_wait(bk_ready_addr, _phase_bk_ready_0);
                    _phase_bk_ready_0 ^= 1;
                    mbarrier_wait(w_drained_addr, _phase_w_drained_0);
                    _phase_w_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_2 = (((s_bk_addr) >> 4) & 0x7FFF) | 0x2000000;
                    int _smem_desc_lo_3 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                    int _smem_desc_lo_2 = (((s_bk_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_w), "r"(0) : "memory");
                    tcgen05_commit(w_full_addr);
                    tcgen05_commit(bk_free_addr);
                    tcgen05_commit(va_free_addr + (va_c_1) * 8);
                    va_c_1 += 1;
                    if (va_c_1 == 2) { va_c_1 = 0; _phase_va_full_1 ^= 1; }
                }
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(kq_full_addr + (kq_c_1) * 8, _phase_kq_full_1);
                    mbarrier_wait(kq_fixed_addr + (kq_c_1) * 8, _phase_kq_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((s_qb_addr) >> 4) & 0x7FFF) + (kq_c_1) * 2048;
                    int _smem_desc_lo_5 = (((s_kq_addr) >> 4) & 0x7FFF) + (kq_c_1) * 2048;
#else
                    int _smem_desc_lo_4 = (((s_qb_addr) >> 4) & 0x3FFF) + (kq_c_1) * 2048;
                    int _smem_desc_lo_5 = (((s_kq_addr) >> 4) & 0x3FFF) + (kq_c_1) * 2048;
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
                    "mov.b32 id, 67634320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_g), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = (((s_qb_addr) >> 4) & 0x7FFF) + (kq_c_1) * 2048;
                    int _smem_desc_lo_7 = (((s_kq_addr + 4096) >> 4) & 0x7FFF) + (kq_c_1) * 2048;
#else
                    int _smem_desc_lo_6 = (((s_qb_addr) >> 4) & 0x3FFF) + (kq_c_1) * 2048;
                    int _smem_desc_lo_7 = (((s_kq_addr + 4096) >> 4) & 0x3FFF) + (kq_c_1) * 2048;
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
                    "mov.b32 id, 67634320;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_g + (1048576))), "r"(0) : "memory");
                    tcgen05_commit(g_full_addr);
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                    _phase_hs_ready_0 ^= 1;
                    mbarrier_wait(wt_ready_addr, _phase_wt_ready_0);
                    _phase_wt_ready_0 ^= 1;
                    int _parity_u_rounded = (int)((unsigned int)c_3 % 2);
                    mbarrier_wait(u_rounded_addr, _parity_u_rounded);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = ((s_wn_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_8 = ((s_wn_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, %4;\n\t"
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
                    :: "r"(tmem_vn), "r"(_smem_desc_lo_8), "r"(tmem_hs), "r"(1), "r"((((uint32_t)(135267472)) | ((uint32_t)(0) << 13) | ((uint32_t)(1) << 14))) : "memory");
                    tcgen05_commit(ws_full_addr);
                    mbarrier_wait(o_free_addr, _phase_o_free_0);
                    _phase_o_free_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_9 = (((s_qb_addr) >> 4) & 0x7FFF) + (kq_c_1) * 2048;
#else
                    int _smem_desc_lo_9 = (((s_qb_addr) >> 4) & 0x3FFF) + (kq_c_1) * 2048;
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
                    "add.u32 blo, blo, 1018;\n\t"
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
                    :: "r"(tmem_o), "r"(_smem_desc_lo_9), "r"(tmem_hs), "r"(0) : "memory");
                    tcgen05_commit(o1_full_addr);
                    mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                    _phase_vp_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((((s_kT_addr) >> 4) & 0x7FFF) | 0x4000000) + (kq_c_1) * 2048;
#else
                    int _smem_desc_lo_10 = ((((s_kT_addr) >> 4) & 0x3FFF) | 0x4000000) + (kq_c_1) * 2048;
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
                    :: "r"(tmem_s), "r"(_smem_desc_lo_10), "r"(tmem_vp), "r"(1) : "memory");
                    tcgen05_commit(s_full_addr);
                    tcgen05_commit(kq_free_addr + (kq_c_1) * 8);
                    kq_c_1 += 1;
                    if (kq_c_1 == 3) { kq_c_1 = 0; _phase_kq_full_1 ^= 1; }
                    if (c_3 + 1 < nc_3) {
                        mbarrier_wait(va_full_addr + (va_c_1) * 8, _phase_va_full_1);
                        mbarrier_wait(v_fixed_addr + (va_c_1) * 8, _phase_va_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_11 = ((((s_vT_addr) >> 4) & 0x7FFF) | 0x2000000) + (va_c_1) * 1536;
                        int _smem_desc_lo_12 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                        int _smem_desc_lo_11 = ((((s_vT_addr) >> 4) & 0x3FFF) | 0x2000000) + (va_c_1) * 1536;
                        int _smem_desc_lo_12 = (((s_a_addr) >> 4) & 0x3FFF) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_11), "r"(_smem_desc_lo_12), "r"(tmem_vn), "r"(0) : "memory");
                        tcgen05_commit(u_full_addr);
                        mbarrier_wait(bk_ready_addr, _phase_bk_ready_0);
                        _phase_bk_ready_0 ^= 1;
                        mbarrier_wait(w_drained_addr, _phase_w_drained_0);
                        _phase_w_drained_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_13 = (((s_bk_addr) >> 4) & 0x7FFF) | 0x2000000;
                        int _smem_desc_lo_14 = (((s_a_addr) >> 4) & 0x7FFF) + (va_c_1) * 1536;
#else
                        int _smem_desc_lo_13 = (((s_bk_addr) >> 4) & 0x3FFF) | 0x2000000;
                        int _smem_desc_lo_14 = (((s_a_addr) >> 4) & 0x3FFF) + (va_c_1) * 1536;
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
                    :: "r"(_smem_desc_lo_13), "r"(_smem_desc_lo_14), "r"(tmem_w), "r"(0) : "memory");
                        tcgen05_commit(w_full_addr);
                        tcgen05_commit(bk_free_addr);
                        tcgen05_commit(va_free_addr + (va_c_1) * 8);
                        va_c_1 += 1;
                        if (va_c_1 == 2) { va_c_1 = 0; _phase_va_full_1 ^= 1; }
                    }
                    mbarrier_wait(pg_ready_addr, _phase_pg_ready_0);
                    mbarrier_wait(o_scaled_addr, _phase_o_scaled_0);
                    _phase_pg_ready_0 ^= 1;
                    _phase_o_scaled_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_15 = ((s_pg_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_15 = ((s_pg_addr) >> 4) & 0x3FFF;
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
                    "}\n"
                    :: "r"(tmem_o), "r"(_smem_desc_lo_15), "r"(tmem_vnb), "r"(1) : "memory");
                    tcgen05_commit(o_full_addr);
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            int blk_11 = blockIdx.x;
            int seg_4 = walk[blk_11 / num_v_heads];
            int hv_4 = blk_11 % num_v_heads;
            int c_begin_4 = seg_chunk_start[seg_4];
            int nc_4 = seg_chunk_start[seg_4 + 1] - c_begin_4;
            int seq_4 = seg_seq[seg_4];
            int hq = hv_4 / (num_v_heads / num_heads);
            int store_h_1 = tape & 1;
            int tok_begin_3 = 0;
            int n_last_3 = 0;
            if (nc_4 > 0) {
                tok_begin_3 = chunk_start[c_begin_4];
                n_last_3 = chunk_len[c_begin_4 + nc_4 - 1];
            }
            unsigned int kq_c_2 = 0;
            unsigned int va_c_2 = 0;
            unsigned int _phase_va_free = 1;
            unsigned int _phase_kq_free = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_4 = 0; c_4 < nc_4; c_4++) {
                    int tok0 = tok_begin_3 + 64 * c_4;
                    mbarrier_wait(va_free_addr + (va_c_2) * 8, _phase_va_free);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(va_full_addr + (va_c_2) * 8, 24576);
                    tma_3d_gmem2smem(s_vT_addr + va_c_2 * 24576 + 0, (&v_tma), 0, hv_4, tok0, va_full_addr + (va_c_2) * 8);
                    tma_3d_gmem2smem(s_vT_addr + va_c_2 * 24576 + 8192, (&v_tma), 64, hv_4, tok0, va_full_addr + (va_c_2) * 8);
                    tma_3d_gmem2smem(s_a_addr + va_c_2 * 24576, (&a_tma), 0, hv_4, tok0, va_full_addr + (va_c_2) * 8);
                    va_c_2 += 1;
                    if (va_c_2 == 2) { va_c_2 = 0; _phase_va_free ^= 1; }
                    mbarrier_wait(kq_free_addr + (kq_c_2) * 8, _phase_kq_free);
                    if (store_h_1 != 0) {
                        mbarrier_wait(h_read_addr + (kq_c_2) * 8, _phase_kq_free);
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(kq_full_addr + (kq_c_2) * 8, 32768);
                    tma_3d_gmem2smem(s_kq_addr + kq_c_2 * 32768 + 0, (&kn_tma), 0, hq, tok0, kq_full_addr + (kq_c_2) * 8);
                    tma_3d_gmem2smem(s_kq_addr + kq_c_2 * 32768 + 8192, (&qn_tma), 0, hq, tok0, kq_full_addr + (kq_c_2) * 8);
                    tma_3d_gmem2smem(s_kq_addr + kq_c_2 * 32768 + 16384, (&kn_tma), 64, hq, tok0, kq_full_addr + (kq_c_2) * 8);
                    tma_3d_gmem2smem(s_kq_addr + kq_c_2 * 32768 + 24576, (&qn_tma), 64, hq, tok0, kq_full_addr + (kq_c_2) * 8);
                    kq_c_2 += 1;
                    if (kq_c_2 == 3) { kq_c_2 = 0; _phase_kq_free ^= 1; }
                }
            }
        }
    }
    // ---- Role: gate ----
    if (warp == 14) {
        { // gate_main
            int blk_12 = blockIdx.x;
            int seg_5 = walk[blk_12 / num_v_heads];
            int hv_5 = blk_12 % num_v_heads;
            int c_begin_5 = seg_chunk_start[seg_5];
            int nc_5 = seg_chunk_start[seg_5 + 1] - c_begin_5;
            int seq_5 = seg_seq[seg_5];
            int lane_2 = lane;
            int tok_begin_4 = 0;
            int n_last_4 = 0;
            if (nc_5 > 0) {
                tok_begin_4 = chunk_start[c_begin_5];
                n_last_4 = chunk_len[c_begin_5 + nc_5 - 1];
            }
            unsigned int gt_c = 0;
            unsigned int _phase_gates_free = 1;
            #pragma unroll 1
            for (int c_5 = 0; c_5 < nc_5; c_5++) {
                mbarrier_wait(gates_free_addr + (gt_c) * 8, _phase_gates_free);
                int tok0_1 = tok_begin_4 + 64 * c_5;
                int n_3 = ((c_5 + 1 == nc_5) ? n_last_4 : 64);
                int g_off_3 = c_5 % 4 * 384;
                int _max_0 = ((n_3) > (1) ? (n_3) : (1));
                float g_last = g_cs[(long long)(tok0_1 + _max_0 - 1) * (long long)num_v_heads + (long long)hv_5];
                float gv_0 = 0.0f;
                float btv_0 = 0.0f;
                if (lane_2 < n_3) {
                    long long tok_0 = (long long)(tok0_1 + lane_2);
                    gv_0 = g_cs[tok_0 * (long long)num_v_heads + (long long)hv_5];
                    btv_0 = beta[tok_0 * (long long)num_v_heads + (long long)hv_5];
                }
                float dv_0 = 0.0f;
                float ev_0 = 0.0f;
                float bgv_0 = 0.0f;
                if (lane_2 < n_3) {
                    float _exp2_1 = approx_exp2(g_last - gv_0);
                    dv_0 = _exp2_1;
                    float _exp2_2 = approx_exp2(gv_0);
                    ev_0 = _exp2_2 * scale;
                    float _exp2_3 = approx_exp2(gv_0);
                    bgv_0 = btv_0 * _exp2_3;
                }
                s_gt[g_off_3 + lane_2] = gv_0;
                s_gt[g_off_3 + 64 + lane_2] = dv_0;
                s_gt[g_off_3 + 128 + lane_2] = ev_0;
                s_gt[g_off_3 + 192 + lane_2] = btv_0;
                s_gt[g_off_3 + 256 + lane_2] = bgv_0;
                float gv_1 = 0.0f;
                float btv_1 = 0.0f;
                if (lane_2 + 32 < n_3) {
                    long long tok_1 = (long long)(tok0_1 + lane_2 + 32);
                    gv_1 = g_cs[tok_1 * (long long)num_v_heads + (long long)hv_5];
                    btv_1 = beta[tok_1 * (long long)num_v_heads + (long long)hv_5];
                }
                float dv_1 = 0.0f;
                float ev_1 = 0.0f;
                float bgv_1 = 0.0f;
                if (lane_2 + 32 < n_3) {
                    float _exp2_4 = approx_exp2(g_last - gv_1);
                    dv_1 = _exp2_4;
                    float _exp2_5 = approx_exp2(gv_1);
                    ev_1 = _exp2_5 * scale;
                    float _exp2_6 = approx_exp2(gv_1);
                    bgv_1 = btv_1 * _exp2_6;
                }
                s_gt[g_off_3 + lane_2 + 32] = gv_1;
                s_gt[g_off_3 + 64 + lane_2 + 32] = dv_1;
                s_gt[g_off_3 + 128 + lane_2 + 32] = ev_1;
                s_gt[g_off_3 + 192 + lane_2 + 32] = btv_1;
                s_gt[g_off_3 + 256 + lane_2 + 32] = bgv_1;
                if (lane_2 == 0) {
                    float _exp2_7 = approx_exp2(g_last);
                    s_gt[g_off_3 + 320] = _exp2_7;
                }
                mbarrier_arrive(gates_ready_addr + (gt_c) * 8);
                gt_c += 1;
                if (gt_c == 4) { gt_c = 0; _phase_gates_free ^= 1; }
            }
        }
    }
    // ---- Role: store ----
    if (warp == 15) {
        { // store_main
            int blk_13 = blockIdx.x;
            int seg_6 = walk[blk_13 / num_v_heads];
            int hv_6 = blk_13 % num_v_heads;
            int c_begin_6 = seg_chunk_start[seg_6];
            int nc_6 = seg_chunk_start[seg_6 + 1] - c_begin_6;
            int seq_6 = seg_seq[seg_6];
            int lane_3 = lane;
            int store_h_2 = tape & 1;
            int store_v_1 = tape >> 1 & 1;
            int store_w = tape >> 3 & 1;
            unsigned int hk_c = 0;
            int tok_begin_5 = 0;
            int n_last_5 = 0;
            if (nc_6 > 0) {
                tok_begin_5 = chunk_start[c_begin_6];
                n_last_5 = chunk_len[c_begin_6 + nc_6 - 1];
            }
            unsigned int _phase_h_staged = 0;
            unsigned int _phase_wt_ready_0_1 = 0;
            unsigned int _phase_vn_staged_0 = 0;
            unsigned int _phase_o_staged_0 = 0;
            #pragma unroll 1
            for (int c_6 = 0; c_6 < nc_6; c_6++) {
                int n_4 = ((c_6 + 1 == nc_6) ? n_last_5 : 64);
                int tok0_2 = tok_begin_5 + 64 * c_6;
                mbarrier_wait(h_staged_addr + (hk_c) * 8, _phase_h_staged);
                hk_c += 1;
                if (hk_c == 3) { hk_c = 0; _phase_h_staged ^= 1; }
                if (c_6 > 0) {
                    if (elect_sync()) {
                        if (store_h_2 != 0) {
                            int hrow = (c_begin_6 + c_6) * num_v_heads + hv_6;
                            tma_store_3d((&h_tma), 0, 0, hrow, s_kq_addr + (unsigned int)((c_6 + 2) % 3 * 32768));
                            tma_store_3d((&h_tma), 64, 0, hrow, s_kq_addr + (unsigned int)((c_6 + 2) % 3 * 32768 + 8192));
                            tma_store_3d((&h_tma), 0, 64, hrow, s_kq_addr + (unsigned int)((c_6 + 2) % 3 * 32768 + 16384));
                            tma_store_3d((&h_tma), 64, 64, hrow, s_kq_addr + (unsigned int)((c_6 + 2) % 3 * 32768 + 16384 + 8192));
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        mbarrier_arrive(h_read_addr + ((int)((unsigned int)(c_6 - 1) % 3)) * 8);
                    }
                }
                mbarrier_wait(wt_ready_addr, _phase_wt_ready_0_1);
                _phase_wt_ready_0_1 ^= 1;
                if (store_w != 0) {
                    if (n_4 == 64) {
                        if (elect_sync()) {
                            tma_store_3d((&w_tma), 0, hv_6, tok0_2, s_wn_addr + 0);
                            tma_store_3d((&w_tma), 64, hv_6, tok0_2, s_wn_addr + 8192);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    } else {
                        #pragma unroll 1
                        for (int it_2 = 0; it_2 < 32; it_2++) {
                            int te = lane_3 + 32 * it_2;
                            int trow = te / 16;
                            if (trow < n_4) {
                                unsigned int tw[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&tw[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw[(0) + 3]))
                                    : "r"(s_wn_addr + (unsigned int)(trow / 64 * 16384 + te % 16 * 8 / 64 * 8192 + trow % 64 * 128 + (te % 16 * 8 % 64 / 8 ^ trow % 8) * 16 + te % 16 * 8 % 8 * 2)));
                                reinterpret_cast<int4*>(w_out + (((long long)(tok0_2 + trow) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te % 16 * 8)))[0] = reinterpret_cast<int4*>(tw)[0];
                            }
                        }
                        __syncwarp();
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(wn_free_addr);
                }
                mbarrier_wait(vn_staged_addr, _phase_vn_staged_0);
                _phase_vn_staged_0 ^= 1;
                if (store_v_1 != 0) {
                    if (n_4 == 64) {
                        if (elect_sync()) {
                            tma_store_3d((&v_new_tma), 0, hv_6, tok0_2, s_vst_addr + 0);
                            tma_store_3d((&v_new_tma), 64, hv_6, tok0_2, s_vst_addr + 8192);
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                    } else {
                        #pragma unroll 1
                        for (int it_3 = 0; it_3 < 32; it_3++) {
                            int te_1 = lane_3 + 32 * it_3;
                            int trow_1 = te_1 / 16;
                            if (trow_1 < n_4) {
                                unsigned int tw_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&tw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_1[(0) + 3]))
                                    : "r"(s_vst_addr + (unsigned int)(trow_1 / 64 * 16384 + te_1 % 16 * 8 / 64 * 8192 + trow_1 % 64 * 128 + (te_1 % 16 * 8 % 64 / 8 ^ trow_1 % 8) * 16 + te_1 % 16 * 8 % 8 * 2)));
                                reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_1) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_1 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_1)[0];
                            }
                        }
                        __syncwarp();
                    }
                }
                if (elect_sync()) {
                    mbarrier_arrive(vn_free_addr);
                }
                mbarrier_wait(o_staged_addr, _phase_o_staged_0);
                _phase_o_staged_0 ^= 1;
                if (n_4 == 64) {
                    if (elect_sync()) {
                        tma_store_3d((&o_tma), 0, hv_6, tok0_2, s_ost_addr + 0);
                        tma_store_3d((&o_tma), 64, hv_6, tok0_2, s_ost_addr + 8192);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                } else {
                    #pragma unroll 1
                    for (int it_4 = 0; it_4 < 32; it_4++) {
                        int te_2 = lane_3 + 32 * it_4;
                        int trow_2 = te_2 / 16;
                        if (trow_2 < n_4) {
                            unsigned int tw_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_2[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_2 / 64 * 16384 + te_2 % 16 * 8 / 64 * 8192 + trow_2 % 64 * 128 + (te_2 % 16 * 8 % 64 / 8 ^ trow_2 % 8) * 16 + te_2 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok0_2 + trow_2) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_2 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_2)[0];
                        }
                    }
                    __syncwarp();
                }
                if (elect_sync()) {
                    mbarrier_arrive(ost_free_addr);
                }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group 0;");
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
