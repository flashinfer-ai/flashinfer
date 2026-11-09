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

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_4b170e3482d035f65cd8(const __grid_constant__ CUtensorMap qn_tma, const __grid_constant__ CUtensorMap kn_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap u_tma, float* __restrict__ g_cs, float* __restrict__ h0, float* __restrict__ seg_start, const __grid_constant__ CUtensorMap h_tma, const __grid_constant__ CUtensorMap v_new_tma, __nv_bfloat16* __restrict__ v_new, const __grid_constant__ CUtensorMap o_tma, __nv_bfloat16* __restrict__ o_out, float* __restrict__ final_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int* __restrict__ walk, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int use_seg_start, float scale, int tape, float* __restrict__ seg_state, int* __restrict__ seg_flag, int split_out)
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

    const int mbar_base = smem + 206848;
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_free_addr (mbar_base + 16)
    #define gates_ready_addr (mbar_base + 32)
    #define gates_free_addr (mbar_base + 48)
    #define hs_half_addr (mbar_base + 64)
    #define hs_ready_addr (mbar_base + 72)
    #define hs_read_addr (mbar_base + 80)
    #define ws_full_addr (mbar_base + 88)
    #define vp_half_addr (mbar_base + 96)
    #define vp_ready_addr (mbar_base + 104)
    #define vn_read_addr (mbar_base + 112)
    #define s_full_addr (mbar_base + 120)
    #define h_staged_addr (mbar_base + 128)
    #define h_free_addr (mbar_base + 136)
    #define vn_staged_addr (mbar_base + 144)
    #define vn_free_addr (mbar_base + 152)
    #define o_staged_addr (mbar_base + 160)
    #define ost_free_addr (mbar_base + 168)
    #define g_full_addr (mbar_base + 176)
    #define pg_ready_addr (mbar_base + 184)
    #define o1_full_addr (mbar_base + 192)
    #define o_scaled_addr (mbar_base + 200)
    #define o_full_addr (mbar_base + 208)
    #define o_free_addr (mbar_base + 216)

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
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_w_addr = smem + 32768;
    __nv_bfloat16* s_u = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_u_addr = smem + 49152;
    float* s_gt = reinterpret_cast<float*>(smem_raw + 131072);
    const int s_gt_addr = smem + 131072;
    __nv_bfloat16* s_pg = reinterpret_cast<__nv_bfloat16*>(smem_raw + 133120);
    const int s_pg_addr = smem + 133120;
    __nv_bfloat16* s_hst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 141312);
    const int s_hst_addr = smem + 141312;
    __nv_bfloat16* s_vst = reinterpret_cast<__nv_bfloat16*>(smem_raw + 174080);
    const int s_vst_addr = smem + 174080;
    __nv_bfloat16* s_ost = reinterpret_cast<__nv_bfloat16*>(smem_raw + 190464);
    const int s_ost_addr = smem + 190464;

    // Mbarrier init (28 barriers at smem_raw[206848..207072))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 206848, 1);
            mbarrier_init(smem + 206856, 1);
            // tiles_free: 2 barriers, init_count=1
            mbarrier_init(smem + 206864, 1);
            mbarrier_init(smem + 206872, 1);
            // --- pipeline 'gring' ---
            // gates_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 206880, 32);
            mbarrier_init(smem + 206888, 32);
            // gates_free: 2 barriers, init_count=256
            mbarrier_init(smem + 206896, 256);
            mbarrier_init(smem + 206904, 256);
            // --- pipeline 'one' ---
            // hs_half: 1 barriers, init_count=128
            mbarrier_init(smem + 206912, 128);
            // hs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 206920, 128);
            // hs_read: 1 barriers, init_count=128
            mbarrier_init(smem + 206928, 128);
            // ws_full: 1 barriers, init_count=1
            mbarrier_init(smem + 206936, 1);
            // vp_half: 1 barriers, init_count=128
            mbarrier_init(smem + 206944, 128);
            // vp_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 206952, 128);
            // vn_read: 1 barriers, init_count=128
            mbarrier_init(smem + 206960, 128);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 206968, 1);
            // h_staged: 1 barriers, init_count=128
            mbarrier_init(smem + 206976, 128);
            // h_free: 1 barriers, init_count=1
            mbarrier_init(smem + 206984, 1);
            // vn_staged: 1 barriers, init_count=128
            mbarrier_init(smem + 206992, 128);
            // vn_free: 1 barriers, init_count=1
            mbarrier_init(smem + 207000, 1);
            // o_staged: 1 barriers, init_count=128
            mbarrier_init(smem + 207008, 128);
            // ost_free: 1 barriers, init_count=1
            mbarrier_init(smem + 207016, 1);
            // g_full: 1 barriers, init_count=1
            mbarrier_init(smem + 207024, 1);
            // pg_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 207032, 128);
            // o1_full: 1 barriers, init_count=1
            mbarrier_init(smem + 207040, 1);
            // o_scaled: 1 barriers, init_count=128
            mbarrier_init(smem + 207048, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 207056, 1);
            // o_free: 1 barriers, init_count=128
            mbarrier_init(smem + 207064, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 207072);
    if (warp == 0) {
        int _tmem_hold = smem + 207072;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_s = taddr;
    const unsigned int tmem_hs = taddr + 128;
    const unsigned int tmem_ws = taddr + 192;
    const unsigned int tmem_vp = taddr + 256;
    const unsigned int tmem_vnb = taddr + 288;
    const unsigned int tmem_o = taddr + 320;
    const unsigned int tmem_g = taddr + 384;

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
            int tok_begin = 0;
            int n_last = 0;
            if (nc > 0) {
                tok_begin = chunk_start[c_begin];
                n_last = chunk_len[c_begin + nc - 1];
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
            unsigned int gs_c = 0;
            unsigned int tf_c = 0;
            unsigned int _phase_hs_read_0 = 1;
            unsigned int _phase_gates_ready = 0;
            unsigned int _phase_ws_full_0 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_vn_read_0 = 1;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                int n = ((c + 1 == nc) ? n_last : 64);
                mbarrier_wait(hs_read_addr, _phase_hs_read_0);
                _phase_hs_read_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float sa[32];
                float sb[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sa[0]), "=f"(sa[1]), "=f"(sa[2]), "=f"(sa[3]), "=f"(sa[4]), "=f"(sa[5]), "=f"(sa[6]), "=f"(sa[7]), "=f"(sa[8]), "=f"(sa[9]), "=f"(sa[10]), "=f"(sa[11]), "=f"(sa[12]), "=f"(sa[13]), "=f"(sa[14]), "=f"(sa[15]), "=f"(sa[16]), "=f"(sa[17]), "=f"(sa[18]), "=f"(sa[19]), "=f"(sa[20]), "=f"(sa[21]), "=f"(sa[22]), "=f"(sa[23]), "=f"(sa[24]), "=f"(sa[25]), "=f"(sa[26]), "=f"(sa[27]), "=f"(sa[28]), "=f"(sa[29]), "=f"(sa[30]), "=f"(sa[31])
                    : "r"(tmem_s + row_off));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sb[0]), "=f"(sb[1]), "=f"(sb[2]), "=f"(sb[3]), "=f"(sb[4]), "=f"(sb[5]), "=f"(sb[6]), "=f"(sb[7]), "=f"(sb[8]), "=f"(sb[9]), "=f"(sb[10]), "=f"(sb[11]), "=f"(sb[12]), "=f"(sb[13]), "=f"(sb[14]), "=f"(sb[15]), "=f"(sb[16]), "=f"(sb[17]), "=f"(sb[18]), "=f"(sb[19]), "=f"(sb[20]), "=f"(sb[21]), "=f"(sb[22]), "=f"(sb[23]), "=f"(sb[24]), "=f"(sb[25]), "=f"(sb[26]), "=f"(sb[27]), "=f"(sb[28]), "=f"(sb[29]), "=f"(sb[30]), "=f"(sb[31])
                    : "r"(tmem_s + row_off + 32));
                unsigned int hp[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sa[_lp*2 + 0], sa[_lp*2+1 + 0]));
                    hp[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hs + row_off + 0), "r"(*reinterpret_cast<const uint32_t*>(&hp[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp[15])));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sa[0]), "=f"(sa[1]), "=f"(sa[2]), "=f"(sa[3]), "=f"(sa[4]), "=f"(sa[5]), "=f"(sa[6]), "=f"(sa[7]), "=f"(sa[8]), "=f"(sa[9]), "=f"(sa[10]), "=f"(sa[11]), "=f"(sa[12]), "=f"(sa[13]), "=f"(sa[14]), "=f"(sa[15]), "=f"(sa[16]), "=f"(sa[17]), "=f"(sa[18]), "=f"(sa[19]), "=f"(sa[20]), "=f"(sa[21]), "=f"(sa[22]), "=f"(sa[23]), "=f"(sa[24]), "=f"(sa[25]), "=f"(sa[26]), "=f"(sa[27]), "=f"(sa[28]), "=f"(sa[29]), "=f"(sa[30]), "=f"(sa[31])
                    : "r"(tmem_s + row_off + 64));
                unsigned int hp_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sb[_lp*2 + 0], sb[_lp*2+1 + 0]));
                    hp_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hs + row_off + 16), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_0[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_half_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sb[0]), "=f"(sb[1]), "=f"(sb[2]), "=f"(sb[3]), "=f"(sb[4]), "=f"(sb[5]), "=f"(sb[6]), "=f"(sb[7]), "=f"(sb[8]), "=f"(sb[9]), "=f"(sb[10]), "=f"(sb[11]), "=f"(sb[12]), "=f"(sb[13]), "=f"(sb[14]), "=f"(sb[15]), "=f"(sb[16]), "=f"(sb[17]), "=f"(sb[18]), "=f"(sb[19]), "=f"(sb[20]), "=f"(sb[21]), "=f"(sb[22]), "=f"(sb[23]), "=f"(sb[24]), "=f"(sb[25]), "=f"(sb[26]), "=f"(sb[27]), "=f"(sb[28]), "=f"(sb[29]), "=f"(sb[30]), "=f"(sb[31])
                    : "r"(tmem_s + row_off + 96));
                unsigned int hp_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sa[_lp*2 + 0], sa[_lp*2+1 + 0]));
                    hp_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hs + row_off + 32), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_1[15])));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int hp_2[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sb[_lp*2 + 0], sb[_lp*2+1 + 0]));
                    hp_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(tmem_hs + row_off + 48), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&hp_2[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_ready_addr);
                mbarrier_wait(gates_ready_addr + (gs_c) * 8, _phase_gates_ready);
                int g_off = c % 2 * 256;
                float e2gl = s_gt[g_off + 192];
                float sv[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                    : "r"(tmem_s + row_off + 0));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_0 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv)[_ls], _scale2_0);
                tmem_st_x32_f32(tmem_s + row_off + 0, sv);
                float sv_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_0[0]), "=f"(sv_0[1]), "=f"(sv_0[2]), "=f"(sv_0[3]), "=f"(sv_0[4]), "=f"(sv_0[5]), "=f"(sv_0[6]), "=f"(sv_0[7]), "=f"(sv_0[8]), "=f"(sv_0[9]), "=f"(sv_0[10]), "=f"(sv_0[11]), "=f"(sv_0[12]), "=f"(sv_0[13]), "=f"(sv_0[14]), "=f"(sv_0[15]), "=f"(sv_0[16]), "=f"(sv_0[17]), "=f"(sv_0[18]), "=f"(sv_0[19]), "=f"(sv_0[20]), "=f"(sv_0[21]), "=f"(sv_0[22]), "=f"(sv_0[23]), "=f"(sv_0[24]), "=f"(sv_0[25]), "=f"(sv_0[26]), "=f"(sv_0[27]), "=f"(sv_0[28]), "=f"(sv_0[29]), "=f"(sv_0[30]), "=f"(sv_0[31])
                    : "r"(tmem_s + row_off + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_1 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_0)[_ls], _scale2_1);
                tmem_st_x32_f32(tmem_s + row_off + 32, sv_0);
                float sv_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31])
                    : "r"(tmem_s + row_off + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_2 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_1)[_ls], _scale2_2);
                tmem_st_x32_f32(tmem_s + row_off + 64, sv_1);
                float sv_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(sv_2[0]), "=f"(sv_2[1]), "=f"(sv_2[2]), "=f"(sv_2[3]), "=f"(sv_2[4]), "=f"(sv_2[5]), "=f"(sv_2[6]), "=f"(sv_2[7]), "=f"(sv_2[8]), "=f"(sv_2[9]), "=f"(sv_2[10]), "=f"(sv_2[11]), "=f"(sv_2[12]), "=f"(sv_2[13]), "=f"(sv_2[14]), "=f"(sv_2[15]), "=f"(sv_2[16]), "=f"(sv_2[17]), "=f"(sv_2[18]), "=f"(sv_2[19]), "=f"(sv_2[20]), "=f"(sv_2[21]), "=f"(sv_2[22]), "=f"(sv_2[23]), "=f"(sv_2[24]), "=f"(sv_2[25]), "=f"(sv_2[26]), "=f"(sv_2[27]), "=f"(sv_2[28]), "=f"(sv_2[29]), "=f"(sv_2[30]), "=f"(sv_2[31])
                    : "r"(tmem_s + row_off + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                const float2 _scale2_3 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sv_2)[_ls], _scale2_3);
                tmem_st_x32_f32(tmem_s + row_off + 96, sv_2);
                mbarrier_wait(ws_full_addr, _phase_ws_full_0);
                _phase_ws_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(tiles_full_addr + (tf_c) * 8, _phase_tiles_full);
                if (n < 64) {
                    int e_0 = tid;
                    int kr_0 = e_0 % 512 / 8;
                    if (kr_0 >= n) {
                        unsigned int zero_0 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_0 / 512 * 16384 + kr_0 * 128 + e_0 % 8 * 16)), "r"(zero_0), "r"(zero_0), "r"(zero_0), "r"(zero_0) : "memory");
                    }
                    int e_1 = tid + 128;
                    int kr_1 = e_1 % 512 / 8;
                    if (kr_1 >= n) {
                        unsigned int zero_1 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_1 / 512 * 16384 + kr_1 * 128 + e_1 % 8 * 16)), "r"(zero_1), "r"(zero_1), "r"(zero_1), "r"(zero_1) : "memory");
                    }
                    int e_2 = tid + 256;
                    int kr_2 = e_2 % 512 / 8;
                    if (kr_2 >= n) {
                        unsigned int zero_2 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_2 / 512 * 16384 + kr_2 * 128 + e_2 % 8 * 16)), "r"(zero_2), "r"(zero_2), "r"(zero_2), "r"(zero_2) : "memory");
                    }
                    int e_3 = tid + 384;
                    int kr_3 = e_3 % 512 / 8;
                    if (kr_3 >= n) {
                        unsigned int zero_3 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_3 / 512 * 16384 + kr_3 * 128 + e_3 % 8 * 16)), "r"(zero_3), "r"(zero_3), "r"(zero_3), "r"(zero_3) : "memory");
                    }
                    int e_4 = tid + 512;
                    int kr_4 = e_4 % 512 / 8;
                    if (kr_4 >= n) {
                        unsigned int zero_4 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_4 / 512 * 16384 + kr_4 * 128 + e_4 % 8 * 16)), "r"(zero_4), "r"(zero_4), "r"(zero_4), "r"(zero_4) : "memory");
                    }
                    int e_5 = tid + 640;
                    int kr_5 = e_5 % 512 / 8;
                    if (kr_5 >= n) {
                        unsigned int zero_5 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_5 / 512 * 16384 + kr_5 * 128 + e_5 % 8 * 16)), "r"(zero_5), "r"(zero_5), "r"(zero_5), "r"(zero_5) : "memory");
                    }
                    int e_6 = tid + 768;
                    int kr_6 = e_6 % 512 / 8;
                    if (kr_6 >= n) {
                        unsigned int zero_6 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_6 / 512 * 16384 + kr_6 * 128 + e_6 % 8 * 16)), "r"(zero_6), "r"(zero_6), "r"(zero_6), "r"(zero_6) : "memory");
                    }
                    int e_7 = tid + 896;
                    int kr_7 = e_7 % 512 / 8;
                    if (kr_7 >= n) {
                        unsigned int zero_7 = 0;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kq_addr + (unsigned int)(tf_c * 65536) + (unsigned int)(e_7 / 512 * 16384 + kr_7 * 128 + e_7 % 8 * 16)), "r"(zero_7), "r"(zero_7), "r"(zero_7), "r"(zero_7) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
                mbarrier_wait(vn_read_addr, _phase_vn_read_0);
                _phase_vn_read_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int m = lane / 8;
                int r = lane % 8;
                int cq = lane % 4;
                float dp[8];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp[(0) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 2 * cq))));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp[(2) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 8 + 2 * cq))));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[4])), "=r"(*reinterpret_cast<uint32_t*>(&dp[(4) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 16 + 2 * cq))));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[6])), "=r"(*reinterpret_cast<uint32_t*>(&dp[(6) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 24 + 2 * cq))));
                float dks[16];
                dks[0] = dp[0];
                dks[1] = dp[1];
                dks[2] = dp[0];
                dks[3] = dp[1];
                dks[4] = dp[2];
                dks[5] = dp[3];
                dks[6] = dp[2];
                dks[7] = dp[3];
                dks[8] = dp[4];
                dks[9] = dp[5];
                dks[10] = dp[4];
                dks[11] = dp[5];
                dks[12] = dp[6];
                dks[13] = dp[7];
                dks[14] = dp[6];
                dks[15] = dp[7];
                int lane_off_0 = row_off;
                int vbase_0 = warp % 4 * 32;
                float vv_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vv_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&vv_0[15]))
                    : "r"(tmem_ws + lane_off_0 + 0));
                unsigned int ub_0[8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_0[0]), "=r"(ub_0[1]), "=r"(ub_0[2]), "=r"(ub_0[3])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((m / 2 * 8 + r) / 64 * 16384 + (vbase_0 + m % 2 * 8) / 64 * 8192 + (m / 2 * 8 + r) % 64 * 128 + ((vbase_0 + m % 2 * 8) % 64 / 8 ^ (m / 2 * 8 + r) % 8) * 16 + (vbase_0 + m % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_0[4]), "=r"(ub_0[5]), "=r"(ub_0[6]), "=r"(ub_0[7])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((16 + m / 2 * 8 + r) / 64 * 16384 + (vbase_0 + m % 2 * 8) / 64 * 8192 + (16 + m / 2 * 8 + r) % 64 * 128 + ((vbase_0 + m % 2 * 8) % 64 / 8 ^ (16 + m / 2 * 8 + r) % 8) * 16 + (vbase_0 + m % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                vv_0[0] = __uint_as_float(ub_0[0] << 16) - vv_0[0];
                vv_0[1] = __uint_as_float(ub_0[0] >> 16 << 16) - vv_0[1];
                vv_0[2] = __uint_as_float(ub_0[1] << 16) - vv_0[2];
                vv_0[3] = __uint_as_float(ub_0[1] >> 16 << 16) - vv_0[3];
                vv_0[4] = __uint_as_float(ub_0[2] << 16) - vv_0[4];
                vv_0[5] = __uint_as_float(ub_0[2] >> 16 << 16) - vv_0[5];
                vv_0[6] = __uint_as_float(ub_0[3] << 16) - vv_0[6];
                vv_0[7] = __uint_as_float(ub_0[3] >> 16 << 16) - vv_0[7];
                vv_0[8] = __uint_as_float(ub_0[4] << 16) - vv_0[8];
                vv_0[9] = __uint_as_float(ub_0[4] >> 16 << 16) - vv_0[9];
                vv_0[10] = __uint_as_float(ub_0[5] << 16) - vv_0[10];
                vv_0[11] = __uint_as_float(ub_0[5] >> 16 << 16) - vv_0[11];
                vv_0[12] = __uint_as_float(ub_0[6] << 16) - vv_0[12];
                vv_0[13] = __uint_as_float(ub_0[6] >> 16 << 16) - vv_0[13];
                vv_0[14] = __uint_as_float(ub_0[7] << 16) - vv_0[14];
                vv_0[15] = __uint_as_float(ub_0[7] >> 16 << 16) - vv_0[15];
                if (n < 64) {
                    vv_0[0] = ((2 * cq < n) ? vv_0[0] : 0.0f);
                    vv_0[1] = ((2 * cq + 1 < n) ? vv_0[1] : 0.0f);
                    vv_0[2] = ((2 * cq < n) ? vv_0[2] : 0.0f);
                    vv_0[3] = ((2 * cq + 1 < n) ? vv_0[3] : 0.0f);
                    vv_0[4] = ((8 + 2 * cq < n) ? vv_0[4] : 0.0f);
                    vv_0[5] = ((8 + 2 * cq + 1 < n) ? vv_0[5] : 0.0f);
                    vv_0[6] = ((8 + 2 * cq < n) ? vv_0[6] : 0.0f);
                    vv_0[7] = ((8 + 2 * cq + 1 < n) ? vv_0[7] : 0.0f);
                    vv_0[8] = ((16 + 2 * cq < n) ? vv_0[8] : 0.0f);
                    vv_0[9] = ((16 + 2 * cq + 1 < n) ? vv_0[9] : 0.0f);
                    vv_0[10] = ((16 + 2 * cq < n) ? vv_0[10] : 0.0f);
                    vv_0[11] = ((16 + 2 * cq + 1 < n) ? vv_0[11] : 0.0f);
                    vv_0[12] = ((24 + 2 * cq < n) ? vv_0[12] : 0.0f);
                    vv_0[13] = ((24 + 2 * cq + 1 < n) ? vv_0[13] : 0.0f);
                    vv_0[14] = ((24 + 2 * cq < n) ? vv_0[14] : 0.0f);
                    vv_0[15] = ((24 + 2 * cq + 1 < n) ? vv_0[15] : 0.0f);
                }
                unsigned int vb_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_0[_lp*2 + 0], vv_0[_lp*2+1 + 0]));
                    vb_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vnb + lane_off_0 + 0), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&vb_0[7])));
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_0)[_ls], reinterpret_cast<const float2*>(dks)[_ls]);
                unsigned int vp_0[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_0[_lp*2 + 0], vv_0[_lp*2+1 + 0]));
                    vp_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vp + lane_off_0 + 0), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_0[7])));
                int lane_off_1 = row_off + 1048576;
                int vbase_1 = warp % 4 * 32 + 16;
                float vv_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vv_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&vv_1[15]))
                    : "r"(tmem_ws + lane_off_1 + 0));
                unsigned int ub_1[8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_1[0]), "=r"(ub_1[1]), "=r"(ub_1[2]), "=r"(ub_1[3])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((m / 2 * 8 + r) / 64 * 16384 + (vbase_1 + m % 2 * 8) / 64 * 8192 + (m / 2 * 8 + r) % 64 * 128 + ((vbase_1 + m % 2 * 8) % 64 / 8 ^ (m / 2 * 8 + r) % 8) * 16 + (vbase_1 + m % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_1[4]), "=r"(ub_1[5]), "=r"(ub_1[6]), "=r"(ub_1[7])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((16 + m / 2 * 8 + r) / 64 * 16384 + (vbase_1 + m % 2 * 8) / 64 * 8192 + (16 + m / 2 * 8 + r) % 64 * 128 + ((vbase_1 + m % 2 * 8) % 64 / 8 ^ (16 + m / 2 * 8 + r) % 8) * 16 + (vbase_1 + m % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                vv_1[0] = __uint_as_float(ub_1[0] << 16) - vv_1[0];
                vv_1[1] = __uint_as_float(ub_1[0] >> 16 << 16) - vv_1[1];
                vv_1[2] = __uint_as_float(ub_1[1] << 16) - vv_1[2];
                vv_1[3] = __uint_as_float(ub_1[1] >> 16 << 16) - vv_1[3];
                vv_1[4] = __uint_as_float(ub_1[2] << 16) - vv_1[4];
                vv_1[5] = __uint_as_float(ub_1[2] >> 16 << 16) - vv_1[5];
                vv_1[6] = __uint_as_float(ub_1[3] << 16) - vv_1[6];
                vv_1[7] = __uint_as_float(ub_1[3] >> 16 << 16) - vv_1[7];
                vv_1[8] = __uint_as_float(ub_1[4] << 16) - vv_1[8];
                vv_1[9] = __uint_as_float(ub_1[4] >> 16 << 16) - vv_1[9];
                vv_1[10] = __uint_as_float(ub_1[5] << 16) - vv_1[10];
                vv_1[11] = __uint_as_float(ub_1[5] >> 16 << 16) - vv_1[11];
                vv_1[12] = __uint_as_float(ub_1[6] << 16) - vv_1[12];
                vv_1[13] = __uint_as_float(ub_1[6] >> 16 << 16) - vv_1[13];
                vv_1[14] = __uint_as_float(ub_1[7] << 16) - vv_1[14];
                vv_1[15] = __uint_as_float(ub_1[7] >> 16 << 16) - vv_1[15];
                if (n < 64) {
                    vv_1[0] = ((2 * cq < n) ? vv_1[0] : 0.0f);
                    vv_1[1] = ((2 * cq + 1 < n) ? vv_1[1] : 0.0f);
                    vv_1[2] = ((2 * cq < n) ? vv_1[2] : 0.0f);
                    vv_1[3] = ((2 * cq + 1 < n) ? vv_1[3] : 0.0f);
                    vv_1[4] = ((8 + 2 * cq < n) ? vv_1[4] : 0.0f);
                    vv_1[5] = ((8 + 2 * cq + 1 < n) ? vv_1[5] : 0.0f);
                    vv_1[6] = ((8 + 2 * cq < n) ? vv_1[6] : 0.0f);
                    vv_1[7] = ((8 + 2 * cq + 1 < n) ? vv_1[7] : 0.0f);
                    vv_1[8] = ((16 + 2 * cq < n) ? vv_1[8] : 0.0f);
                    vv_1[9] = ((16 + 2 * cq + 1 < n) ? vv_1[9] : 0.0f);
                    vv_1[10] = ((16 + 2 * cq < n) ? vv_1[10] : 0.0f);
                    vv_1[11] = ((16 + 2 * cq + 1 < n) ? vv_1[11] : 0.0f);
                    vv_1[12] = ((24 + 2 * cq < n) ? vv_1[12] : 0.0f);
                    vv_1[13] = ((24 + 2 * cq + 1 < n) ? vv_1[13] : 0.0f);
                    vv_1[14] = ((24 + 2 * cq < n) ? vv_1[14] : 0.0f);
                    vv_1[15] = ((24 + 2 * cq + 1 < n) ? vv_1[15] : 0.0f);
                }
                unsigned int vb_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_1[_lp*2 + 0], vv_1[_lp*2+1 + 0]));
                    vb_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vnb + lane_off_1 + 0), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vb_1[7])));
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_1)[_ls], reinterpret_cast<const float2*>(dks)[_ls]);
                unsigned int vp_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_1[_lp*2 + 0], vv_1[_lp*2+1 + 0]));
                    vp_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vp + lane_off_1 + 0), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_1[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(vp_half_addr);
                int m_0 = lane / 8;
                int r_0 = lane % 8;
                int cq_0 = lane % 4;
                float dp_0[8];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp_0[(0) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 + 2 * cq_0))));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp_0[(2) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 + 8 + 2 * cq_0))));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&dp_0[(4) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 + 16 + 2 * cq_0))));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&dp_0[(6) + 1]))
                    : "r"(s_gt_addr + (unsigned int)(4 * (g_off + 64 + 32 + 24 + 2 * cq_0))));
                float dks_0[16];
                dks_0[0] = dp_0[0];
                dks_0[1] = dp_0[1];
                dks_0[2] = dp_0[0];
                dks_0[3] = dp_0[1];
                dks_0[4] = dp_0[2];
                dks_0[5] = dp_0[3];
                dks_0[6] = dp_0[2];
                dks_0[7] = dp_0[3];
                dks_0[8] = dp_0[4];
                dks_0[9] = dp_0[5];
                dks_0[10] = dp_0[4];
                dks_0[11] = dp_0[5];
                dks_0[12] = dp_0[6];
                dks_0[13] = dp_0[7];
                dks_0[14] = dp_0[6];
                dks_0[15] = dp_0[7];
                int lane_off_2 = row_off;
                int vbase_2 = warp % 4 * 32;
                float vv_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vv_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&vv_2[15]))
                    : "r"(tmem_ws + lane_off_2 + 32));
                unsigned int ub_2[8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_2[0]), "=r"(ub_2[1]), "=r"(ub_2[2]), "=r"(ub_2[3])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((32 + m_0 / 2 * 8 + r_0) / 64 * 16384 + (vbase_2 + m_0 % 2 * 8) / 64 * 8192 + (32 + m_0 / 2 * 8 + r_0) % 64 * 128 + ((vbase_2 + m_0 % 2 * 8) % 64 / 8 ^ (32 + m_0 / 2 * 8 + r_0) % 8) * 16 + (vbase_2 + m_0 % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_2[4]), "=r"(ub_2[5]), "=r"(ub_2[6]), "=r"(ub_2[7])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((48 + m_0 / 2 * 8 + r_0) / 64 * 16384 + (vbase_2 + m_0 % 2 * 8) / 64 * 8192 + (48 + m_0 / 2 * 8 + r_0) % 64 * 128 + ((vbase_2 + m_0 % 2 * 8) % 64 / 8 ^ (48 + m_0 / 2 * 8 + r_0) % 8) * 16 + (vbase_2 + m_0 % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                vv_2[0] = __uint_as_float(ub_2[0] << 16) - vv_2[0];
                vv_2[1] = __uint_as_float(ub_2[0] >> 16 << 16) - vv_2[1];
                vv_2[2] = __uint_as_float(ub_2[1] << 16) - vv_2[2];
                vv_2[3] = __uint_as_float(ub_2[1] >> 16 << 16) - vv_2[3];
                vv_2[4] = __uint_as_float(ub_2[2] << 16) - vv_2[4];
                vv_2[5] = __uint_as_float(ub_2[2] >> 16 << 16) - vv_2[5];
                vv_2[6] = __uint_as_float(ub_2[3] << 16) - vv_2[6];
                vv_2[7] = __uint_as_float(ub_2[3] >> 16 << 16) - vv_2[7];
                vv_2[8] = __uint_as_float(ub_2[4] << 16) - vv_2[8];
                vv_2[9] = __uint_as_float(ub_2[4] >> 16 << 16) - vv_2[9];
                vv_2[10] = __uint_as_float(ub_2[5] << 16) - vv_2[10];
                vv_2[11] = __uint_as_float(ub_2[5] >> 16 << 16) - vv_2[11];
                vv_2[12] = __uint_as_float(ub_2[6] << 16) - vv_2[12];
                vv_2[13] = __uint_as_float(ub_2[6] >> 16 << 16) - vv_2[13];
                vv_2[14] = __uint_as_float(ub_2[7] << 16) - vv_2[14];
                vv_2[15] = __uint_as_float(ub_2[7] >> 16 << 16) - vv_2[15];
                if (n < 64) {
                    vv_2[0] = ((32 + 2 * cq_0 < n) ? vv_2[0] : 0.0f);
                    vv_2[1] = ((32 + 2 * cq_0 + 1 < n) ? vv_2[1] : 0.0f);
                    vv_2[2] = ((32 + 2 * cq_0 < n) ? vv_2[2] : 0.0f);
                    vv_2[3] = ((32 + 2 * cq_0 + 1 < n) ? vv_2[3] : 0.0f);
                    vv_2[4] = ((40 + 2 * cq_0 < n) ? vv_2[4] : 0.0f);
                    vv_2[5] = ((40 + 2 * cq_0 + 1 < n) ? vv_2[5] : 0.0f);
                    vv_2[6] = ((40 + 2 * cq_0 < n) ? vv_2[6] : 0.0f);
                    vv_2[7] = ((40 + 2 * cq_0 + 1 < n) ? vv_2[7] : 0.0f);
                    vv_2[8] = ((48 + 2 * cq_0 < n) ? vv_2[8] : 0.0f);
                    vv_2[9] = ((48 + 2 * cq_0 + 1 < n) ? vv_2[9] : 0.0f);
                    vv_2[10] = ((48 + 2 * cq_0 < n) ? vv_2[10] : 0.0f);
                    vv_2[11] = ((48 + 2 * cq_0 + 1 < n) ? vv_2[11] : 0.0f);
                    vv_2[12] = ((56 + 2 * cq_0 < n) ? vv_2[12] : 0.0f);
                    vv_2[13] = ((56 + 2 * cq_0 + 1 < n) ? vv_2[13] : 0.0f);
                    vv_2[14] = ((56 + 2 * cq_0 < n) ? vv_2[14] : 0.0f);
                    vv_2[15] = ((56 + 2 * cq_0 + 1 < n) ? vv_2[15] : 0.0f);
                }
                unsigned int vb_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_2[_lp*2 + 0], vv_2[_lp*2+1 + 0]));
                    vb_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vnb + lane_off_2 + 16), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&vb_2[7])));
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_2)[_ls], reinterpret_cast<const float2*>(dks_0)[_ls]);
                unsigned int vp_2[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_2[_lp*2 + 0], vv_2[_lp*2+1 + 0]));
                    vp_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vp + lane_off_2 + 16), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_2[7])));
                int lane_off_3 = row_off + 1048576;
                int vbase_3 = warp % 4 * 32 + 16;
                float vv_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vv_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&vv_3[15]))
                    : "r"(tmem_ws + lane_off_3 + 32));
                unsigned int ub_3[8];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_3[0]), "=r"(ub_3[1]), "=r"(ub_3[2]), "=r"(ub_3[3])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((32 + m_0 / 2 * 8 + r_0) / 64 * 16384 + (vbase_3 + m_0 % 2 * 8) / 64 * 8192 + (32 + m_0 / 2 * 8 + r_0) % 64 * 128 + ((vbase_3 + m_0 % 2 * 8) % 64 / 8 ^ (32 + m_0 / 2 * 8 + r_0) % 8) * 16 + (vbase_3 + m_0 % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(ub_3[4]), "=r"(ub_3[5]), "=r"(ub_3[6]), "=r"(ub_3[7])
                    : "r"(s_u_addr + tf_c * 65536 + (unsigned int)((48 + m_0 / 2 * 8 + r_0) / 64 * 16384 + (vbase_3 + m_0 % 2 * 8) / 64 * 8192 + (48 + m_0 / 2 * 8 + r_0) % 64 * 128 + ((vbase_3 + m_0 % 2 * 8) % 64 / 8 ^ (48 + m_0 / 2 * 8 + r_0) % 8) * 16 + (vbase_3 + m_0 % 2 * 8) % 8 * 2))
                    : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                vv_3[0] = __uint_as_float(ub_3[0] << 16) - vv_3[0];
                vv_3[1] = __uint_as_float(ub_3[0] >> 16 << 16) - vv_3[1];
                vv_3[2] = __uint_as_float(ub_3[1] << 16) - vv_3[2];
                vv_3[3] = __uint_as_float(ub_3[1] >> 16 << 16) - vv_3[3];
                vv_3[4] = __uint_as_float(ub_3[2] << 16) - vv_3[4];
                vv_3[5] = __uint_as_float(ub_3[2] >> 16 << 16) - vv_3[5];
                vv_3[6] = __uint_as_float(ub_3[3] << 16) - vv_3[6];
                vv_3[7] = __uint_as_float(ub_3[3] >> 16 << 16) - vv_3[7];
                vv_3[8] = __uint_as_float(ub_3[4] << 16) - vv_3[8];
                vv_3[9] = __uint_as_float(ub_3[4] >> 16 << 16) - vv_3[9];
                vv_3[10] = __uint_as_float(ub_3[5] << 16) - vv_3[10];
                vv_3[11] = __uint_as_float(ub_3[5] >> 16 << 16) - vv_3[11];
                vv_3[12] = __uint_as_float(ub_3[6] << 16) - vv_3[12];
                vv_3[13] = __uint_as_float(ub_3[6] >> 16 << 16) - vv_3[13];
                vv_3[14] = __uint_as_float(ub_3[7] << 16) - vv_3[14];
                vv_3[15] = __uint_as_float(ub_3[7] >> 16 << 16) - vv_3[15];
                if (n < 64) {
                    vv_3[0] = ((32 + 2 * cq_0 < n) ? vv_3[0] : 0.0f);
                    vv_3[1] = ((32 + 2 * cq_0 + 1 < n) ? vv_3[1] : 0.0f);
                    vv_3[2] = ((32 + 2 * cq_0 < n) ? vv_3[2] : 0.0f);
                    vv_3[3] = ((32 + 2 * cq_0 + 1 < n) ? vv_3[3] : 0.0f);
                    vv_3[4] = ((40 + 2 * cq_0 < n) ? vv_3[4] : 0.0f);
                    vv_3[5] = ((40 + 2 * cq_0 + 1 < n) ? vv_3[5] : 0.0f);
                    vv_3[6] = ((40 + 2 * cq_0 < n) ? vv_3[6] : 0.0f);
                    vv_3[7] = ((40 + 2 * cq_0 + 1 < n) ? vv_3[7] : 0.0f);
                    vv_3[8] = ((48 + 2 * cq_0 < n) ? vv_3[8] : 0.0f);
                    vv_3[9] = ((48 + 2 * cq_0 + 1 < n) ? vv_3[9] : 0.0f);
                    vv_3[10] = ((48 + 2 * cq_0 < n) ? vv_3[10] : 0.0f);
                    vv_3[11] = ((48 + 2 * cq_0 + 1 < n) ? vv_3[11] : 0.0f);
                    vv_3[12] = ((56 + 2 * cq_0 < n) ? vv_3[12] : 0.0f);
                    vv_3[13] = ((56 + 2 * cq_0 + 1 < n) ? vv_3[13] : 0.0f);
                    vv_3[14] = ((56 + 2 * cq_0 < n) ? vv_3[14] : 0.0f);
                    vv_3[15] = ((56 + 2 * cq_0 + 1 < n) ? vv_3[15] : 0.0f);
                }
                unsigned int vb_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_3[_lp*2 + 0], vv_3[_lp*2+1 + 0]));
                    vb_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vnb + lane_off_3 + 16), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&vb_3[7])));
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vv_3)[_ls], reinterpret_cast<const float2*>(dks_0)[_ls]);
                unsigned int vp_3[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_3[_lp*2 + 0], vv_3[_lp*2+1 + 0]));
                    vp_3[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.16x128b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(tmem_vp + lane_off_3 + 16), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&vp_3[7])));
                mbarrier_arrive(gates_free_addr + (gs_c) * 8);
                gs_c += 1;
                if (gs_c == 2) { gs_c = 0; _phase_gates_ready ^= 1; }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(vp_ready_addr);
                tf_c += 1;
                if (tf_c == 2) { tf_c = 0; _phase_tiles_full ^= 1; }
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (store_final_state != 0) {
                if (is_last != 0) {
                    if (split_out == 0) {
                        #pragma unroll 1
                        for (int kb = 0; kb < 4; kb++) {
                            float blk_4[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(blk_4[0]), "=f"(blk_4[1]), "=f"(blk_4[2]), "=f"(blk_4[3]), "=f"(blk_4[4]), "=f"(blk_4[5]), "=f"(blk_4[6]), "=f"(blk_4[7]), "=f"(blk_4[8]), "=f"(blk_4[9]), "=f"(blk_4[10]), "=f"(blk_4[11]), "=f"(blk_4[12]), "=f"(blk_4[13]), "=f"(blk_4[14]), "=f"(blk_4[15]), "=f"(blk_4[16]), "=f"(blk_4[17]), "=f"(blk_4[18]), "=f"(blk_4[19]), "=f"(blk_4[20]), "=f"(blk_4[21]), "=f"(blk_4[22]), "=f"(blk_4[23]), "=f"(blk_4[24]), "=f"(blk_4[25]), "=f"(blk_4[26]), "=f"(blk_4[27]), "=f"(blk_4[28]), "=f"(blk_4[29]), "=f"(blk_4[30]), "=f"(blk_4[31])
                                : "r"(tmem_s + row_off + 32 * kb));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb))] = blk_4[0];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 1))] = blk_4[1];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 2))] = blk_4[2];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 3))] = blk_4[3];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 4))] = blk_4[4];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 5))] = blk_4[5];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 6))] = blk_4[6];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 7))] = blk_4[7];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 8))] = blk_4[8];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 9))] = blk_4[9];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 10))] = blk_4[10];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 11))] = blk_4[11];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 12))] = blk_4[12];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 13))] = blk_4[13];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 14))] = blk_4[14];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 15))] = blk_4[15];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 16))] = blk_4[16];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 17))] = blk_4[17];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 18))] = blk_4[18];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 19))] = blk_4[19];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 20))] = blk_4[20];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 21))] = blk_4[21];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 22))] = blk_4[22];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 23))] = blk_4[23];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 24))] = blk_4[24];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 25))] = blk_4[25];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 26))] = blk_4[26];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 27))] = blk_4[27];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 28))] = blk_4[28];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 29))] = blk_4[29];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 30))] = blk_4[30];
                            final_state[(long long)seq * (long long)num_v_heads * 16384ll + head_base + (long long)(128 * (32 * kb + 31))] = blk_4[31];
                        }
                    }
                    if (split_out != 0) {
                        int out_seg = walk[blockIdx.x / num_v_heads];
                        int out_hv = blockIdx.x % num_v_heads;
                        int out_v = warp % 4 * 32 + lane;
                        #pragma unroll 1
                        for (int kb_1 = 0; kb_1 < 4; kb_1++) {
                            float blk_5[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(blk_5[0]), "=f"(blk_5[1]), "=f"(blk_5[2]), "=f"(blk_5[3]), "=f"(blk_5[4]), "=f"(blk_5[5]), "=f"(blk_5[6]), "=f"(blk_5[7]), "=f"(blk_5[8]), "=f"(blk_5[9]), "=f"(blk_5[10]), "=f"(blk_5[11]), "=f"(blk_5[12]), "=f"(blk_5[13]), "=f"(blk_5[14]), "=f"(blk_5[15]), "=f"(blk_5[16]), "=f"(blk_5[17]), "=f"(blk_5[18]), "=f"(blk_5[19]), "=f"(blk_5[20]), "=f"(blk_5[21]), "=f"(blk_5[22]), "=f"(blk_5[23]), "=f"(blk_5[24]), "=f"(blk_5[25]), "=f"(blk_5[26]), "=f"(blk_5[27]), "=f"(blk_5[28]), "=f"(blk_5[29]), "=f"(blk_5[30]), "=f"(blk_5[31])
                                : "r"(tmem_s + row_off + 32 * kb_1));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1))] = blk_5[0];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 1))] = blk_5[1];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 2))] = blk_5[2];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 3))] = blk_5[3];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 4))] = blk_5[4];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 5))] = blk_5[5];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 6))] = blk_5[6];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 7))] = blk_5[7];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 8))] = blk_5[8];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 9))] = blk_5[9];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 10))] = blk_5[10];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 11))] = blk_5[11];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 12))] = blk_5[12];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 13))] = blk_5[13];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 14))] = blk_5[14];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 15))] = blk_5[15];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 16))] = blk_5[16];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 17))] = blk_5[17];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 18))] = blk_5[18];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 19))] = blk_5[19];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 20))] = blk_5[20];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 21))] = blk_5[21];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 22))] = blk_5[22];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 23))] = blk_5[23];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 24))] = blk_5[24];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 25))] = blk_5[25];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 26))] = blk_5[26];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 27))] = blk_5[27];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 28))] = blk_5[28];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 29))] = blk_5[29];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 30))] = blk_5[30];
                            seg_state[((long long)out_seg * (long long)num_v_heads + (long long)out_hv) * 32768ll + (long long)out_v + (long long)(256 * (32 * kb_1 + 31))] = blk_5[31];
                        }
                        if (out_v == 0) {
                            seg_flag[out_seg * num_v_heads + out_hv] = 0;
                        }
                    }
                }
            }
        }
    }
    // ---- Role: tapw ----
    if (warp >= 4 && warp <= 7) {
        { // tapw_main
            int blk_6 = blockIdx.x;
            int seg_1 = walk[blk_6 / num_v_heads];
            int hv_1 = blk_6 % num_v_heads;
            int c_begin_1 = seg_chunk_start[seg_1];
            int nc_1 = seg_chunk_start[seg_1 + 1] - c_begin_1;
            int seq_1 = seg_seq[seg_1];
            int vrow_1 = warp % 4 * 32 + lane;
            int row_off_1 = warp % 4 * 32 << 16;
            int store_h = tape & 1;
            int store_v = tape >> 1 & 1;
            unsigned int _phase_hs_ready_0 = 0;
            unsigned int _phase_h_free_0 = 1;
            unsigned int _phase_vp_ready_0 = 0;
            unsigned int _phase_vn_free_0 = 1;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                mbarrier_wait(hs_ready_addr, _phase_hs_ready_0);
                _phase_hs_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float hb[64];
                if (store_h != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(hb[0]), "=f"(hb[1]), "=f"(hb[2]), "=f"(hb[3]), "=f"(hb[4]), "=f"(hb[5]), "=f"(hb[6]), "=f"(hb[7]), "=f"(hb[8]), "=f"(hb[9]), "=f"(hb[10]), "=f"(hb[11]), "=f"(hb[12]), "=f"(hb[13]), "=f"(hb[14]), "=f"(hb[15]), "=f"(hb[16]), "=f"(hb[17]), "=f"(hb[18]), "=f"(hb[19]), "=f"(hb[20]), "=f"(hb[21]), "=f"(hb[22]), "=f"(hb[23]), "=f"(hb[24]), "=f"(hb[25]), "=f"(hb[26]), "=f"(hb[27]), "=f"(hb[28]), "=f"(hb[29]), "=f"(hb[30]), "=f"(hb[31])
                        : "r"(tmem_hs + row_off_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(hb[32]), "=f"(hb[33]), "=f"(hb[34]), "=f"(hb[35]), "=f"(hb[36]), "=f"(hb[37]), "=f"(hb[38]), "=f"(hb[39]), "=f"(hb[40]), "=f"(hb[41]), "=f"(hb[42]), "=f"(hb[43]), "=f"(hb[44]), "=f"(hb[45]), "=f"(hb[46]), "=f"(hb[47]), "=f"(hb[48]), "=f"(hb[49]), "=f"(hb[50]), "=f"(hb[51]), "=f"(hb[52]), "=f"(hb[53]), "=f"(hb[54]), "=f"(hb[55]), "=f"(hb[56]), "=f"(hb[57]), "=f"(hb[58]), "=f"(hb[59]), "=f"(hb[60]), "=f"(hb[61]), "=f"(hb[62]), "=f"(hb[63])
                        : "r"(tmem_hs + row_off_1 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(hs_read_addr);
                mbarrier_wait(h_free_addr, _phase_h_free_0);
                _phase_h_free_0 ^= 1;
                if (store_h != 0) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (0 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[0])), "r"(__as_u32(hb[1])), "r"(__as_u32(hb[2])), "r"(__as_u32(hb[3])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (1 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[4])), "r"(__as_u32(hb[5])), "r"(__as_u32(hb[6])), "r"(__as_u32(hb[7])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (2 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[8])), "r"(__as_u32(hb[9])), "r"(__as_u32(hb[10])), "r"(__as_u32(hb[11])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (3 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[12])), "r"(__as_u32(hb[13])), "r"(__as_u32(hb[14])), "r"(__as_u32(hb[15])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (4 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[16])), "r"(__as_u32(hb[17])), "r"(__as_u32(hb[18])), "r"(__as_u32(hb[19])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (5 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[20])), "r"(__as_u32(hb[21])), "r"(__as_u32(hb[22])), "r"(__as_u32(hb[23])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (6 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[24])), "r"(__as_u32(hb[25])), "r"(__as_u32(hb[26])), "r"(__as_u32(hb[27])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + vrow_1 % 64 * 128 + (7 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[28])), "r"(__as_u32(hb[29])), "r"(__as_u32(hb[30])), "r"(__as_u32(hb[31])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (0 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[32])), "r"(__as_u32(hb[33])), "r"(__as_u32(hb[34])), "r"(__as_u32(hb[35])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (1 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[36])), "r"(__as_u32(hb[37])), "r"(__as_u32(hb[38])), "r"(__as_u32(hb[39])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (2 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[40])), "r"(__as_u32(hb[41])), "r"(__as_u32(hb[42])), "r"(__as_u32(hb[43])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (3 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[44])), "r"(__as_u32(hb[45])), "r"(__as_u32(hb[46])), "r"(__as_u32(hb[47])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (4 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[48])), "r"(__as_u32(hb[49])), "r"(__as_u32(hb[50])), "r"(__as_u32(hb[51])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (5 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[52])), "r"(__as_u32(hb[53])), "r"(__as_u32(hb[54])), "r"(__as_u32(hb[55])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (6 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[56])), "r"(__as_u32(hb[57])), "r"(__as_u32(hb[58])), "r"(__as_u32(hb[59])) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_hst_addr + (unsigned int)(vrow_1 / 64 * 16384 + 8192 + vrow_1 % 64 * 128 + (7 ^ vrow_1 % 8) * 16)), "r"(__as_u32(hb[60])), "r"(__as_u32(hb[61])), "r"(__as_u32(hb[62])), "r"(__as_u32(hb[63])) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(h_staged_addr);
                mbarrier_wait(vp_ready_addr, _phase_vp_ready_0);
                _phase_vp_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float vb[32];
                if (store_v != 0) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vb[0])), "=r"(*reinterpret_cast<uint32_t*>(&vb[1])), "=r"(*reinterpret_cast<uint32_t*>(&vb[2])), "=r"(*reinterpret_cast<uint32_t*>(&vb[3])), "=r"(*reinterpret_cast<uint32_t*>(&vb[4])), "=r"(*reinterpret_cast<uint32_t*>(&vb[5])), "=r"(*reinterpret_cast<uint32_t*>(&vb[6])), "=r"(*reinterpret_cast<uint32_t*>(&vb[7])), "=r"(*reinterpret_cast<uint32_t*>(&vb[8])), "=r"(*reinterpret_cast<uint32_t*>(&vb[9])), "=r"(*reinterpret_cast<uint32_t*>(&vb[10])), "=r"(*reinterpret_cast<uint32_t*>(&vb[11])), "=r"(*reinterpret_cast<uint32_t*>(&vb[12])), "=r"(*reinterpret_cast<uint32_t*>(&vb[13])), "=r"(*reinterpret_cast<uint32_t*>(&vb[14])), "=r"(*reinterpret_cast<uint32_t*>(&vb[15]))
                        : "r"(tmem_vnb + row_off_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x8.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&vb[16])), "=r"(*reinterpret_cast<uint32_t*>(&vb[17])), "=r"(*reinterpret_cast<uint32_t*>(&vb[18])), "=r"(*reinterpret_cast<uint32_t*>(&vb[19])), "=r"(*reinterpret_cast<uint32_t*>(&vb[20])), "=r"(*reinterpret_cast<uint32_t*>(&vb[21])), "=r"(*reinterpret_cast<uint32_t*>(&vb[22])), "=r"(*reinterpret_cast<uint32_t*>(&vb[23])), "=r"(*reinterpret_cast<uint32_t*>(&vb[24])), "=r"(*reinterpret_cast<uint32_t*>(&vb[25])), "=r"(*reinterpret_cast<uint32_t*>(&vb[26])), "=r"(*reinterpret_cast<uint32_t*>(&vb[27])), "=r"(*reinterpret_cast<uint32_t*>(&vb[28])), "=r"(*reinterpret_cast<uint32_t*>(&vb[29])), "=r"(*reinterpret_cast<uint32_t*>(&vb[30])), "=r"(*reinterpret_cast<uint32_t*>(&vb[31]))
                        : "r"(tmem_vnb + (row_off_1 + 1048576)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(vn_read_addr);
                mbarrier_wait(vn_free_addr, _phase_vn_free_0);
                _phase_vn_free_0 ^= 1;
                if (store_v != 0) {
                    unsigned int vu[32];
                    vu[0] = __as_u32(vb[0]);
                    vu[1] = __as_u32(vb[1]);
                    vu[2] = __as_u32(vb[2]);
                    vu[3] = __as_u32(vb[3]);
                    vu[4] = __as_u32(vb[4]);
                    vu[5] = __as_u32(vb[5]);
                    vu[6] = __as_u32(vb[6]);
                    vu[7] = __as_u32(vb[7]);
                    vu[8] = __as_u32(vb[8]);
                    vu[9] = __as_u32(vb[9]);
                    vu[10] = __as_u32(vb[10]);
                    vu[11] = __as_u32(vb[11]);
                    vu[12] = __as_u32(vb[12]);
                    vu[13] = __as_u32(vb[13]);
                    vu[14] = __as_u32(vb[14]);
                    vu[15] = __as_u32(vb[15]);
                    vu[16] = __as_u32(vb[16]);
                    vu[17] = __as_u32(vb[17]);
                    vu[18] = __as_u32(vb[18]);
                    vu[19] = __as_u32(vb[19]);
                    vu[20] = __as_u32(vb[20]);
                    vu[21] = __as_u32(vb[21]);
                    vu[22] = __as_u32(vb[22]);
                    vu[23] = __as_u32(vb[23]);
                    vu[24] = __as_u32(vb[24]);
                    vu[25] = __as_u32(vb[25]);
                    vu[26] = __as_u32(vb[26]);
                    vu[27] = __as_u32(vb[27]);
                    vu[28] = __as_u32(vb[28]);
                    vu[29] = __as_u32(vb[29]);
                    vu[30] = __as_u32(vb[30]);
                    vu[31] = __as_u32(vb[31]);
                    int m_1 = lane / 8;
                    int r_1 = lane % 8;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + m_1 % 2 * 8) / 64 * 8192 + (m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + m_1 % 2 * 8) % 64 / 8 ^ (m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&vu[0])), "r"(*reinterpret_cast<const uint32_t*>(&vu[1])), "r"(*reinterpret_cast<const uint32_t*>(&vu[2])), "r"(*reinterpret_cast<const uint32_t*>(&vu[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((16 + m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + m_1 % 2 * 8) / 64 * 8192 + (16 + m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + m_1 % 2 * 8) % 64 / 8 ^ (16 + m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&vu[4])), "r"(*reinterpret_cast<const uint32_t*>(&vu[5])), "r"(*reinterpret_cast<const uint32_t*>(&vu[6])), "r"(*reinterpret_cast<const uint32_t*>(&vu[7]))
                        : "memory");
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((32 + m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + m_1 % 2 * 8) / 64 * 8192 + (32 + m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + m_1 % 2 * 8) % 64 / 8 ^ (32 + m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&vu[8])), "r"(*reinterpret_cast<const uint32_t*>(&vu[9])), "r"(*reinterpret_cast<const uint32_t*>(&vu[10])), "r"(*reinterpret_cast<const uint32_t*>(&vu[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((48 + m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + m_1 % 2 * 8) / 64 * 8192 + (48 + m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + m_1 % 2 * 8) % 64 / 8 ^ (48 + m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&vu[12])), "r"(*reinterpret_cast<const uint32_t*>(&vu[13])), "r"(*reinterpret_cast<const uint32_t*>(&vu[14])), "r"(*reinterpret_cast<const uint32_t*>(&vu[15]))
                        : "memory");
                    uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) / 64 * 8192 + (m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + 16 + m_1 % 2 * 8) % 64 / 8 ^ (m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&vu[16])), "r"(*reinterpret_cast<const uint32_t*>(&vu[17])), "r"(*reinterpret_cast<const uint32_t*>(&vu[18])), "r"(*reinterpret_cast<const uint32_t*>(&vu[19]))
                        : "memory");
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((16 + m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) / 64 * 8192 + (16 + m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + 16 + m_1 % 2 * 8) % 64 / 8 ^ (16 + m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&vu[20])), "r"(*reinterpret_cast<const uint32_t*>(&vu[21])), "r"(*reinterpret_cast<const uint32_t*>(&vu[22])), "r"(*reinterpret_cast<const uint32_t*>(&vu[23]))
                        : "memory");
                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((32 + m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) / 64 * 8192 + (32 + m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + 16 + m_1 % 2 * 8) % 64 / 8 ^ (32 + m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&vu[24])), "r"(*reinterpret_cast<const uint32_t*>(&vu[25])), "r"(*reinterpret_cast<const uint32_t*>(&vu[26])), "r"(*reinterpret_cast<const uint32_t*>(&vu[27]))
                        : "memory");
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)(s_vst_addr + (unsigned int)((48 + m_1 / 2 * 8 + r_1) / 64 * 16384 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) / 64 * 8192 + (48 + m_1 / 2 * 8 + r_1) % 64 * 128 + ((warp % 4 * 32 + 16 + m_1 % 2 * 8) % 64 / 8 ^ (48 + m_1 / 2 * 8 + r_1) % 8) * 16 + (warp % 4 * 32 + 16 + m_1 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&vu[28])), "r"(*reinterpret_cast<const uint32_t*>(&vu[29])), "r"(*reinterpret_cast<const uint32_t*>(&vu[30])), "r"(*reinterpret_cast<const uint32_t*>(&vu[31]))
                        : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vn_staged_addr);
            }
        }
    }
    // ---- Role: outr ----
    if (warp >= 8 && warp <= 11) {
        { // outr_main
            int blk_7 = blockIdx.x;
            int seg_2 = walk[blk_7 / num_v_heads];
            int hv_2 = blk_7 % num_v_heads;
            int c_begin_2 = seg_chunk_start[seg_2];
            int nc_2 = seg_chunk_start[seg_2 + 1] - c_begin_2;
            int seq_2 = seg_seq[seg_2];
            int lane_1 = lane;
            int q4 = warp % 4;
            int vrow_2 = q4 * 32 + lane_1;
            int row_off_2 = q4 * 32 << 16;
            int i_row = (q4 - 2) * 32 + lane_1;
            int tok_begin_1 = 0;
            int n_last_1 = 0;
            if (nc_2 > 0) {
                tok_begin_1 = chunk_start[c_begin_2];
                n_last_1 = chunk_len[c_begin_2 + nc_2 - 1];
            }
            unsigned int gw_c = 0;
            unsigned int _phase_gates_ready_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_ost_free_0 = 1;
            unsigned int _phase_g_full_0 = 0;
            unsigned int _phase_o1_full_0 = 0;
            #pragma unroll 1
            for (int c_2 = 0; c_2 < nc_2; c_2++) {
                int n_1 = ((c_2 + 1 == nc_2) ? n_last_1 : 64);
                mbarrier_wait(gates_ready_addr + (gw_c) * 8, _phase_gates_ready_1);
                int g_off_1 = c_2 % 2 * 256;
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float od_v[64];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&od_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[1])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[2])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[3])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[4])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[5])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[6])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[7])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[8])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[9])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[10])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[11])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[12])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[13])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[14])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[15])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[16])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[17])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[18])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[19])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[20])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[21])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[22])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[23])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[24])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[25])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[26])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[27])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[28])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[29])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[30])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[31]))
                    : "r"(tmem_o + row_off_2));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&od_v[32])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[33])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[34])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[35])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[36])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[37])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[38])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[39])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[40])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[41])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[42])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[43])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[44])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[45])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[46])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[47])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[48])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[49])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[50])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[51])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[52])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[53])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[54])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[55])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[56])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[57])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[58])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[59])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[60])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[61])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[62])), "=r"(*reinterpret_cast<uint32_t*>(&od_v[63]))
                    : "r"(tmem_o + (row_off_2 + 1048576)));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_free_addr);
                mbarrier_wait(ost_free_addr, _phase_ost_free_0);
                _phase_ost_free_0 ^= 1;
                if (c_2 - 1 >= 0) {
                    unsigned int od_p[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(od_v[_lp*2 + 0], od_v[_lp*2+1 + 0]));
                        od_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    int m_2 = lane / 8;
                    int r_2 = lane % 8;
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + m_2 % 2 * 8) / 64 * 8192 + (m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + m_2 % 2 * 8) % 64 / 8 ^ (m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&od_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[3]))
                        : "memory");
                    uint32_t _stmatrix_addr_1 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((16 + m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + m_2 % 2 * 8) / 64 * 8192 + (16 + m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + m_2 % 2 * 8) % 64 / 8 ^ (16 + m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&od_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[7]))
                        : "memory");
                    uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((32 + m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + m_2 % 2 * 8) / 64 * 8192 + (32 + m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + m_2 % 2 * 8) % 64 / 8 ^ (32 + m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&od_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[11]))
                        : "memory");
                    uint32_t _stmatrix_addr_3 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((48 + m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + m_2 % 2 * 8) / 64 * 8192 + (48 + m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + m_2 % 2 * 8) % 64 / 8 ^ (48 + m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&od_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[15]))
                        : "memory");
                    uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + 16 + m_2 % 2 * 8) / 64 * 8192 + (m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + 16 + m_2 % 2 * 8) % 64 / 8 ^ (m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + 16 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&od_p[16])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[17])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[18])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[19]))
                        : "memory");
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((16 + m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + 16 + m_2 % 2 * 8) / 64 * 8192 + (16 + m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + 16 + m_2 % 2 * 8) % 64 / 8 ^ (16 + m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + 16 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&od_p[20])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[21])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[22])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[23]))
                        : "memory");
                    uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((32 + m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + 16 + m_2 % 2 * 8) / 64 * 8192 + (32 + m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + 16 + m_2 % 2 * 8) % 64 / 8 ^ (32 + m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + 16 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&od_p[24])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[25])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[26])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[27]))
                        : "memory");
                    uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((48 + m_2 / 2 * 8 + r_2) / 64 * 16384 + (q4 * 32 + 16 + m_2 % 2 * 8) / 64 * 8192 + (48 + m_2 / 2 * 8 + r_2) % 64 * 128 + ((q4 * 32 + 16 + m_2 % 2 * 8) % 64 / 8 ^ (48 + m_2 / 2 * 8 + r_2) % 8) * 16 + (q4 * 32 + 16 + m_2 % 2 * 8) % 8 * 2)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&od_p[28])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[29])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[30])), "r"(*reinterpret_cast<const uint32_t*>(&od_p[31]))
                        : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(o_staged_addr);
                mbarrier_wait(g_full_addr, _phase_g_full_0);
                _phase_g_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (q4 >= 2) {
                    float gv[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(gv[0]), "=f"(gv[1]), "=f"(gv[2]), "=f"(gv[3]), "=f"(gv[4]), "=f"(gv[5]), "=f"(gv[6]), "=f"(gv[7]), "=f"(gv[8]), "=f"(gv[9]), "=f"(gv[10]), "=f"(gv[11]), "=f"(gv[12]), "=f"(gv[13]), "=f"(gv[14]), "=f"(gv[15]), "=f"(gv[16]), "=f"(gv[17]), "=f"(gv[18]), "=f"(gv[19]), "=f"(gv[20]), "=f"(gv[21]), "=f"(gv[22]), "=f"(gv[23]), "=f"(gv[24]), "=f"(gv[25]), "=f"(gv[26]), "=f"(gv[27]), "=f"(gv[28]), "=f"(gv[29]), "=f"(gv[30]), "=f"(gv[31])
                        : "r"(tmem_g + row_off_2));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(gv[32]), "=f"(gv[33]), "=f"(gv[34]), "=f"(gv[35]), "=f"(gv[36]), "=f"(gv[37]), "=f"(gv[38]), "=f"(gv[39]), "=f"(gv[40]), "=f"(gv[41]), "=f"(gv[42]), "=f"(gv[43]), "=f"(gv[44]), "=f"(gv[45]), "=f"(gv[46]), "=f"(gv[47]), "=f"(gv[48]), "=f"(gv[49]), "=f"(gv[50]), "=f"(gv[51]), "=f"(gv[52]), "=f"(gv[53]), "=f"(gv[54]), "=f"(gv[55]), "=f"(gv[56]), "=f"(gv[57]), "=f"(gv[58]), "=f"(gv[59]), "=f"(gv[60]), "=f"(gv[61]), "=f"(gv[62]), "=f"(gv[63])
                        : "r"(tmem_g + row_off_2 + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float gi = s_gt[g_off_1 + i_row];
                    float pv8_0[8];
                    float gj_0 = s_gt[g_off_1];
                    float _exp2_8 = approx_exp2(gi - gj_0);
                    float pij_0 = gv[0] * _exp2_8 * scale;
                    pv8_0[0] = ((0 <= i_row) ? ((i_row < n_1) ? pij_0 : 0.0f) : 0.0f);
                    float gj_1 = s_gt[g_off_1 + 1];
                    float _exp2_9 = approx_exp2(gi - gj_1);
                    float pij_1 = gv[1] * _exp2_9 * scale;
                    pv8_0[1] = ((1 <= i_row) ? ((i_row < n_1) ? pij_1 : 0.0f) : 0.0f);
                    float gj_2 = s_gt[g_off_1 + 2];
                    float _exp2_10 = approx_exp2(gi - gj_2);
                    float pij_2 = gv[2] * _exp2_10 * scale;
                    pv8_0[2] = ((2 <= i_row) ? ((i_row < n_1) ? pij_2 : 0.0f) : 0.0f);
                    float gj_3 = s_gt[g_off_1 + 3];
                    float _exp2_11 = approx_exp2(gi - gj_3);
                    float pij_3 = gv[3] * _exp2_11 * scale;
                    pv8_0[3] = ((3 <= i_row) ? ((i_row < n_1) ? pij_3 : 0.0f) : 0.0f);
                    float gj_4 = s_gt[g_off_1 + 4];
                    float _exp2_12 = approx_exp2(gi - gj_4);
                    float pij_4 = gv[4] * _exp2_12 * scale;
                    pv8_0[4] = ((4 <= i_row) ? ((i_row < n_1) ? pij_4 : 0.0f) : 0.0f);
                    float gj_5 = s_gt[g_off_1 + 5];
                    float _exp2_13 = approx_exp2(gi - gj_5);
                    float pij_5 = gv[5] * _exp2_13 * scale;
                    pv8_0[5] = ((5 <= i_row) ? ((i_row < n_1) ? pij_5 : 0.0f) : 0.0f);
                    float gj_6 = s_gt[g_off_1 + 6];
                    float _exp2_14 = approx_exp2(gi - gj_6);
                    float pij_6 = gv[6] * _exp2_14 * scale;
                    pv8_0[6] = ((6 <= i_row) ? ((i_row < n_1) ? pij_6 : 0.0f) : 0.0f);
                    float gj_7 = s_gt[g_off_1 + 7];
                    float _exp2_15 = approx_exp2(gi - gj_7);
                    float pij_7 = gv[7] * _exp2_15 * scale;
                    pv8_0[7] = ((7 <= i_row) ? ((i_row < n_1) ? pij_7 : 0.0f) : 0.0f);
                    unsigned int pk_0[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_0[_lp*2 + 0], pv8_0[_lp*2+1 + 0]));
                        pk_0[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (0 ^ i_row % 8) * 16)), "r"(pk_0[0]), "r"(pk_0[1]), "r"(pk_0[2]), "r"(pk_0[3]) : "memory");
                    float pv8_1[8];
                    float gj_8 = s_gt[g_off_1 + 8];
                    float _exp2_16 = approx_exp2(gi - gj_8);
                    float pij_8 = gv[8] * _exp2_16 * scale;
                    pv8_1[0] = ((8 <= i_row) ? ((i_row < n_1) ? pij_8 : 0.0f) : 0.0f);
                    float gj_9 = s_gt[g_off_1 + 8 + 1];
                    float _exp2_17 = approx_exp2(gi - gj_9);
                    float pij_9 = gv[9] * _exp2_17 * scale;
                    pv8_1[1] = ((9 <= i_row) ? ((i_row < n_1) ? pij_9 : 0.0f) : 0.0f);
                    float gj_10 = s_gt[g_off_1 + 8 + 2];
                    float _exp2_18 = approx_exp2(gi - gj_10);
                    float pij_10 = gv[10] * _exp2_18 * scale;
                    pv8_1[2] = ((10 <= i_row) ? ((i_row < n_1) ? pij_10 : 0.0f) : 0.0f);
                    float gj_11 = s_gt[g_off_1 + 8 + 3];
                    float _exp2_19 = approx_exp2(gi - gj_11);
                    float pij_11 = gv[11] * _exp2_19 * scale;
                    pv8_1[3] = ((11 <= i_row) ? ((i_row < n_1) ? pij_11 : 0.0f) : 0.0f);
                    float gj_12 = s_gt[g_off_1 + 8 + 4];
                    float _exp2_20 = approx_exp2(gi - gj_12);
                    float pij_12 = gv[12] * _exp2_20 * scale;
                    pv8_1[4] = ((12 <= i_row) ? ((i_row < n_1) ? pij_12 : 0.0f) : 0.0f);
                    float gj_13 = s_gt[g_off_1 + 8 + 5];
                    float _exp2_21 = approx_exp2(gi - gj_13);
                    float pij_13 = gv[13] * _exp2_21 * scale;
                    pv8_1[5] = ((13 <= i_row) ? ((i_row < n_1) ? pij_13 : 0.0f) : 0.0f);
                    float gj_14 = s_gt[g_off_1 + 8 + 6];
                    float _exp2_22 = approx_exp2(gi - gj_14);
                    float pij_14 = gv[14] * _exp2_22 * scale;
                    pv8_1[6] = ((14 <= i_row) ? ((i_row < n_1) ? pij_14 : 0.0f) : 0.0f);
                    float gj_15 = s_gt[g_off_1 + 8 + 7];
                    float _exp2_23 = approx_exp2(gi - gj_15);
                    float pij_15 = gv[15] * _exp2_23 * scale;
                    pv8_1[7] = ((15 <= i_row) ? ((i_row < n_1) ? pij_15 : 0.0f) : 0.0f);
                    unsigned int pk_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_1[_lp*2 + 0], pv8_1[_lp*2+1 + 0]));
                        pk_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (1 ^ i_row % 8) * 16)), "r"(pk_1[0]), "r"(pk_1[1]), "r"(pk_1[2]), "r"(pk_1[3]) : "memory");
                    float pv8_2[8];
                    float gj_16 = s_gt[g_off_1 + 16];
                    float _exp2_24 = approx_exp2(gi - gj_16);
                    float pij_16 = gv[16] * _exp2_24 * scale;
                    pv8_2[0] = ((16 <= i_row) ? ((i_row < n_1) ? pij_16 : 0.0f) : 0.0f);
                    float gj_17 = s_gt[g_off_1 + 16 + 1];
                    float _exp2_25 = approx_exp2(gi - gj_17);
                    float pij_17 = gv[17] * _exp2_25 * scale;
                    pv8_2[1] = ((17 <= i_row) ? ((i_row < n_1) ? pij_17 : 0.0f) : 0.0f);
                    float gj_18 = s_gt[g_off_1 + 16 + 2];
                    float _exp2_26 = approx_exp2(gi - gj_18);
                    float pij_18 = gv[18] * _exp2_26 * scale;
                    pv8_2[2] = ((18 <= i_row) ? ((i_row < n_1) ? pij_18 : 0.0f) : 0.0f);
                    float gj_19 = s_gt[g_off_1 + 16 + 3];
                    float _exp2_27 = approx_exp2(gi - gj_19);
                    float pij_19 = gv[19] * _exp2_27 * scale;
                    pv8_2[3] = ((19 <= i_row) ? ((i_row < n_1) ? pij_19 : 0.0f) : 0.0f);
                    float gj_20 = s_gt[g_off_1 + 16 + 4];
                    float _exp2_28 = approx_exp2(gi - gj_20);
                    float pij_20 = gv[20] * _exp2_28 * scale;
                    pv8_2[4] = ((20 <= i_row) ? ((i_row < n_1) ? pij_20 : 0.0f) : 0.0f);
                    float gj_21 = s_gt[g_off_1 + 16 + 5];
                    float _exp2_29 = approx_exp2(gi - gj_21);
                    float pij_21 = gv[21] * _exp2_29 * scale;
                    pv8_2[5] = ((21 <= i_row) ? ((i_row < n_1) ? pij_21 : 0.0f) : 0.0f);
                    float gj_22 = s_gt[g_off_1 + 16 + 6];
                    float _exp2_30 = approx_exp2(gi - gj_22);
                    float pij_22 = gv[22] * _exp2_30 * scale;
                    pv8_2[6] = ((22 <= i_row) ? ((i_row < n_1) ? pij_22 : 0.0f) : 0.0f);
                    float gj_23 = s_gt[g_off_1 + 16 + 7];
                    float _exp2_31 = approx_exp2(gi - gj_23);
                    float pij_23 = gv[23] * _exp2_31 * scale;
                    pv8_2[7] = ((23 <= i_row) ? ((i_row < n_1) ? pij_23 : 0.0f) : 0.0f);
                    unsigned int pk_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_2[_lp*2 + 0], pv8_2[_lp*2+1 + 0]));
                        pk_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (2 ^ i_row % 8) * 16)), "r"(pk_2[0]), "r"(pk_2[1]), "r"(pk_2[2]), "r"(pk_2[3]) : "memory");
                    float pv8_3[8];
                    float gj_24 = s_gt[g_off_1 + 24];
                    float _exp2_32 = approx_exp2(gi - gj_24);
                    float pij_24 = gv[24] * _exp2_32 * scale;
                    pv8_3[0] = ((24 <= i_row) ? ((i_row < n_1) ? pij_24 : 0.0f) : 0.0f);
                    float gj_25 = s_gt[g_off_1 + 24 + 1];
                    float _exp2_33 = approx_exp2(gi - gj_25);
                    float pij_25 = gv[25] * _exp2_33 * scale;
                    pv8_3[1] = ((25 <= i_row) ? ((i_row < n_1) ? pij_25 : 0.0f) : 0.0f);
                    float gj_26 = s_gt[g_off_1 + 24 + 2];
                    float _exp2_34 = approx_exp2(gi - gj_26);
                    float pij_26 = gv[26] * _exp2_34 * scale;
                    pv8_3[2] = ((26 <= i_row) ? ((i_row < n_1) ? pij_26 : 0.0f) : 0.0f);
                    float gj_27 = s_gt[g_off_1 + 24 + 3];
                    float _exp2_35 = approx_exp2(gi - gj_27);
                    float pij_27 = gv[27] * _exp2_35 * scale;
                    pv8_3[3] = ((27 <= i_row) ? ((i_row < n_1) ? pij_27 : 0.0f) : 0.0f);
                    float gj_28 = s_gt[g_off_1 + 24 + 4];
                    float _exp2_36 = approx_exp2(gi - gj_28);
                    float pij_28 = gv[28] * _exp2_36 * scale;
                    pv8_3[4] = ((28 <= i_row) ? ((i_row < n_1) ? pij_28 : 0.0f) : 0.0f);
                    float gj_29 = s_gt[g_off_1 + 24 + 5];
                    float _exp2_37 = approx_exp2(gi - gj_29);
                    float pij_29 = gv[29] * _exp2_37 * scale;
                    pv8_3[5] = ((29 <= i_row) ? ((i_row < n_1) ? pij_29 : 0.0f) : 0.0f);
                    float gj_30 = s_gt[g_off_1 + 24 + 6];
                    float _exp2_38 = approx_exp2(gi - gj_30);
                    float pij_30 = gv[30] * _exp2_38 * scale;
                    pv8_3[6] = ((30 <= i_row) ? ((i_row < n_1) ? pij_30 : 0.0f) : 0.0f);
                    float gj_31 = s_gt[g_off_1 + 24 + 7];
                    float _exp2_39 = approx_exp2(gi - gj_31);
                    float pij_31 = gv[31] * _exp2_39 * scale;
                    pv8_3[7] = ((31 <= i_row) ? ((i_row < n_1) ? pij_31 : 0.0f) : 0.0f);
                    unsigned int pk_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_3[_lp*2 + 0], pv8_3[_lp*2+1 + 0]));
                        pk_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (3 ^ i_row % 8) * 16)), "r"(pk_3[0]), "r"(pk_3[1]), "r"(pk_3[2]), "r"(pk_3[3]) : "memory");
                    float pv8_4[8];
                    float gj_32 = s_gt[g_off_1 + 32];
                    float _exp2_40 = approx_exp2(gi - gj_32);
                    float pij_32 = gv[32] * _exp2_40 * scale;
                    pv8_4[0] = ((32 <= i_row) ? ((i_row < n_1) ? pij_32 : 0.0f) : 0.0f);
                    float gj_33 = s_gt[g_off_1 + 32 + 1];
                    float _exp2_41 = approx_exp2(gi - gj_33);
                    float pij_33 = gv[33] * _exp2_41 * scale;
                    pv8_4[1] = ((33 <= i_row) ? ((i_row < n_1) ? pij_33 : 0.0f) : 0.0f);
                    float gj_34 = s_gt[g_off_1 + 32 + 2];
                    float _exp2_42 = approx_exp2(gi - gj_34);
                    float pij_34 = gv[34] * _exp2_42 * scale;
                    pv8_4[2] = ((34 <= i_row) ? ((i_row < n_1) ? pij_34 : 0.0f) : 0.0f);
                    float gj_35 = s_gt[g_off_1 + 32 + 3];
                    float _exp2_43 = approx_exp2(gi - gj_35);
                    float pij_35 = gv[35] * _exp2_43 * scale;
                    pv8_4[3] = ((35 <= i_row) ? ((i_row < n_1) ? pij_35 : 0.0f) : 0.0f);
                    float gj_36 = s_gt[g_off_1 + 32 + 4];
                    float _exp2_44 = approx_exp2(gi - gj_36);
                    float pij_36 = gv[36] * _exp2_44 * scale;
                    pv8_4[4] = ((36 <= i_row) ? ((i_row < n_1) ? pij_36 : 0.0f) : 0.0f);
                    float gj_37 = s_gt[g_off_1 + 32 + 5];
                    float _exp2_45 = approx_exp2(gi - gj_37);
                    float pij_37 = gv[37] * _exp2_45 * scale;
                    pv8_4[5] = ((37 <= i_row) ? ((i_row < n_1) ? pij_37 : 0.0f) : 0.0f);
                    float gj_38 = s_gt[g_off_1 + 32 + 6];
                    float _exp2_46 = approx_exp2(gi - gj_38);
                    float pij_38 = gv[38] * _exp2_46 * scale;
                    pv8_4[6] = ((38 <= i_row) ? ((i_row < n_1) ? pij_38 : 0.0f) : 0.0f);
                    float gj_39 = s_gt[g_off_1 + 32 + 7];
                    float _exp2_47 = approx_exp2(gi - gj_39);
                    float pij_39 = gv[39] * _exp2_47 * scale;
                    pv8_4[7] = ((39 <= i_row) ? ((i_row < n_1) ? pij_39 : 0.0f) : 0.0f);
                    unsigned int pk_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_4[_lp*2 + 0], pv8_4[_lp*2+1 + 0]));
                        pk_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (4 ^ i_row % 8) * 16)), "r"(pk_4[0]), "r"(pk_4[1]), "r"(pk_4[2]), "r"(pk_4[3]) : "memory");
                    float pv8_5[8];
                    float gj_40 = s_gt[g_off_1 + 40];
                    float _exp2_48 = approx_exp2(gi - gj_40);
                    float pij_40 = gv[40] * _exp2_48 * scale;
                    pv8_5[0] = ((40 <= i_row) ? ((i_row < n_1) ? pij_40 : 0.0f) : 0.0f);
                    float gj_41 = s_gt[g_off_1 + 40 + 1];
                    float _exp2_49 = approx_exp2(gi - gj_41);
                    float pij_41 = gv[41] * _exp2_49 * scale;
                    pv8_5[1] = ((41 <= i_row) ? ((i_row < n_1) ? pij_41 : 0.0f) : 0.0f);
                    float gj_42 = s_gt[g_off_1 + 40 + 2];
                    float _exp2_50 = approx_exp2(gi - gj_42);
                    float pij_42 = gv[42] * _exp2_50 * scale;
                    pv8_5[2] = ((42 <= i_row) ? ((i_row < n_1) ? pij_42 : 0.0f) : 0.0f);
                    float gj_43 = s_gt[g_off_1 + 40 + 3];
                    float _exp2_51 = approx_exp2(gi - gj_43);
                    float pij_43 = gv[43] * _exp2_51 * scale;
                    pv8_5[3] = ((43 <= i_row) ? ((i_row < n_1) ? pij_43 : 0.0f) : 0.0f);
                    float gj_44 = s_gt[g_off_1 + 40 + 4];
                    float _exp2_52 = approx_exp2(gi - gj_44);
                    float pij_44 = gv[44] * _exp2_52 * scale;
                    pv8_5[4] = ((44 <= i_row) ? ((i_row < n_1) ? pij_44 : 0.0f) : 0.0f);
                    float gj_45 = s_gt[g_off_1 + 40 + 5];
                    float _exp2_53 = approx_exp2(gi - gj_45);
                    float pij_45 = gv[45] * _exp2_53 * scale;
                    pv8_5[5] = ((45 <= i_row) ? ((i_row < n_1) ? pij_45 : 0.0f) : 0.0f);
                    float gj_46 = s_gt[g_off_1 + 40 + 6];
                    float _exp2_54 = approx_exp2(gi - gj_46);
                    float pij_46 = gv[46] * _exp2_54 * scale;
                    pv8_5[6] = ((46 <= i_row) ? ((i_row < n_1) ? pij_46 : 0.0f) : 0.0f);
                    float gj_47 = s_gt[g_off_1 + 40 + 7];
                    float _exp2_55 = approx_exp2(gi - gj_47);
                    float pij_47 = gv[47] * _exp2_55 * scale;
                    pv8_5[7] = ((47 <= i_row) ? ((i_row < n_1) ? pij_47 : 0.0f) : 0.0f);
                    unsigned int pk_5[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_5[_lp*2 + 0], pv8_5[_lp*2+1 + 0]));
                        pk_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (5 ^ i_row % 8) * 16)), "r"(pk_5[0]), "r"(pk_5[1]), "r"(pk_5[2]), "r"(pk_5[3]) : "memory");
                    float pv8_6[8];
                    float gj_48 = s_gt[g_off_1 + 48];
                    float _exp2_56 = approx_exp2(gi - gj_48);
                    float pij_48 = gv[48] * _exp2_56 * scale;
                    pv8_6[0] = ((48 <= i_row) ? ((i_row < n_1) ? pij_48 : 0.0f) : 0.0f);
                    float gj_49 = s_gt[g_off_1 + 48 + 1];
                    float _exp2_57 = approx_exp2(gi - gj_49);
                    float pij_49 = gv[49] * _exp2_57 * scale;
                    pv8_6[1] = ((49 <= i_row) ? ((i_row < n_1) ? pij_49 : 0.0f) : 0.0f);
                    float gj_50 = s_gt[g_off_1 + 48 + 2];
                    float _exp2_58 = approx_exp2(gi - gj_50);
                    float pij_50 = gv[50] * _exp2_58 * scale;
                    pv8_6[2] = ((50 <= i_row) ? ((i_row < n_1) ? pij_50 : 0.0f) : 0.0f);
                    float gj_51 = s_gt[g_off_1 + 48 + 3];
                    float _exp2_59 = approx_exp2(gi - gj_51);
                    float pij_51 = gv[51] * _exp2_59 * scale;
                    pv8_6[3] = ((51 <= i_row) ? ((i_row < n_1) ? pij_51 : 0.0f) : 0.0f);
                    float gj_52 = s_gt[g_off_1 + 48 + 4];
                    float _exp2_60 = approx_exp2(gi - gj_52);
                    float pij_52 = gv[52] * _exp2_60 * scale;
                    pv8_6[4] = ((52 <= i_row) ? ((i_row < n_1) ? pij_52 : 0.0f) : 0.0f);
                    float gj_53 = s_gt[g_off_1 + 48 + 5];
                    float _exp2_61 = approx_exp2(gi - gj_53);
                    float pij_53 = gv[53] * _exp2_61 * scale;
                    pv8_6[5] = ((53 <= i_row) ? ((i_row < n_1) ? pij_53 : 0.0f) : 0.0f);
                    float gj_54 = s_gt[g_off_1 + 48 + 6];
                    float _exp2_62 = approx_exp2(gi - gj_54);
                    float pij_54 = gv[54] * _exp2_62 * scale;
                    pv8_6[6] = ((54 <= i_row) ? ((i_row < n_1) ? pij_54 : 0.0f) : 0.0f);
                    float gj_55 = s_gt[g_off_1 + 48 + 7];
                    float _exp2_63 = approx_exp2(gi - gj_55);
                    float pij_55 = gv[55] * _exp2_63 * scale;
                    pv8_6[7] = ((55 <= i_row) ? ((i_row < n_1) ? pij_55 : 0.0f) : 0.0f);
                    unsigned int pk_6[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_6[_lp*2 + 0], pv8_6[_lp*2+1 + 0]));
                        pk_6[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (6 ^ i_row % 8) * 16)), "r"(pk_6[0]), "r"(pk_6[1]), "r"(pk_6[2]), "r"(pk_6[3]) : "memory");
                    float pv8_7[8];
                    float gj_56 = s_gt[g_off_1 + 56];
                    float _exp2_64 = approx_exp2(gi - gj_56);
                    float pij_56 = gv[56] * _exp2_64 * scale;
                    pv8_7[0] = ((56 <= i_row) ? ((i_row < n_1) ? pij_56 : 0.0f) : 0.0f);
                    float gj_57 = s_gt[g_off_1 + 56 + 1];
                    float _exp2_65 = approx_exp2(gi - gj_57);
                    float pij_57 = gv[57] * _exp2_65 * scale;
                    pv8_7[1] = ((57 <= i_row) ? ((i_row < n_1) ? pij_57 : 0.0f) : 0.0f);
                    float gj_58 = s_gt[g_off_1 + 56 + 2];
                    float _exp2_66 = approx_exp2(gi - gj_58);
                    float pij_58 = gv[58] * _exp2_66 * scale;
                    pv8_7[2] = ((58 <= i_row) ? ((i_row < n_1) ? pij_58 : 0.0f) : 0.0f);
                    float gj_59 = s_gt[g_off_1 + 56 + 3];
                    float _exp2_67 = approx_exp2(gi - gj_59);
                    float pij_59 = gv[59] * _exp2_67 * scale;
                    pv8_7[3] = ((59 <= i_row) ? ((i_row < n_1) ? pij_59 : 0.0f) : 0.0f);
                    float gj_60 = s_gt[g_off_1 + 56 + 4];
                    float _exp2_68 = approx_exp2(gi - gj_60);
                    float pij_60 = gv[60] * _exp2_68 * scale;
                    pv8_7[4] = ((60 <= i_row) ? ((i_row < n_1) ? pij_60 : 0.0f) : 0.0f);
                    float gj_61 = s_gt[g_off_1 + 56 + 5];
                    float _exp2_69 = approx_exp2(gi - gj_61);
                    float pij_61 = gv[61] * _exp2_69 * scale;
                    pv8_7[5] = ((61 <= i_row) ? ((i_row < n_1) ? pij_61 : 0.0f) : 0.0f);
                    float gj_62 = s_gt[g_off_1 + 56 + 6];
                    float _exp2_70 = approx_exp2(gi - gj_62);
                    float pij_62 = gv[62] * _exp2_70 * scale;
                    pv8_7[6] = ((62 <= i_row) ? ((i_row < n_1) ? pij_62 : 0.0f) : 0.0f);
                    float gj_63 = s_gt[g_off_1 + 56 + 7];
                    float _exp2_71 = approx_exp2(gi - gj_63);
                    float pij_63 = gv[63] * _exp2_71 * scale;
                    pv8_7[7] = ((63 <= i_row) ? ((i_row < n_1) ? pij_63 : 0.0f) : 0.0f);
                    unsigned int pk_7[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv8_7[_lp*2 + 0], pv8_7[_lp*2+1 + 0]));
                        pk_7[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_pg_addr + (unsigned int)(i_row * 128 + (7 ^ i_row % 8) * 16)), "r"(pk_7[0]), "r"(pk_7[1]), "r"(pk_7[2]), "r"(pk_7[3]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                }
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
                    : "r"(tmem_o + row_off_2));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(ov1[32]), "=f"(ov1[33]), "=f"(ov1[34]), "=f"(ov1[35]), "=f"(ov1[36]), "=f"(ov1[37]), "=f"(ov1[38]), "=f"(ov1[39]), "=f"(ov1[40]), "=f"(ov1[41]), "=f"(ov1[42]), "=f"(ov1[43]), "=f"(ov1[44]), "=f"(ov1[45]), "=f"(ov1[46]), "=f"(ov1[47]), "=f"(ov1[48]), "=f"(ov1[49]), "=f"(ov1[50]), "=f"(ov1[51]), "=f"(ov1[52]), "=f"(ov1[53]), "=f"(ov1[54]), "=f"(ov1[55]), "=f"(ov1[56]), "=f"(ov1[57]), "=f"(ov1[58]), "=f"(ov1[59]), "=f"(ov1[60]), "=f"(ov1[61]), "=f"(ov1[62]), "=f"(ov1[63])
                    : "r"(tmem_o + row_off_2 + 32));
                float eg[64];
                eg[0] = s_gt[g_off_1 + 128];
                eg[1] = s_gt[g_off_1 + 128 + 1];
                eg[2] = s_gt[g_off_1 + 128 + 2];
                eg[3] = s_gt[g_off_1 + 128 + 3];
                eg[4] = s_gt[g_off_1 + 128 + 4];
                eg[5] = s_gt[g_off_1 + 128 + 5];
                eg[6] = s_gt[g_off_1 + 128 + 6];
                eg[7] = s_gt[g_off_1 + 128 + 7];
                eg[8] = s_gt[g_off_1 + 128 + 8];
                eg[9] = s_gt[g_off_1 + 128 + 9];
                eg[10] = s_gt[g_off_1 + 128 + 10];
                eg[11] = s_gt[g_off_1 + 128 + 11];
                eg[12] = s_gt[g_off_1 + 128 + 12];
                eg[13] = s_gt[g_off_1 + 128 + 13];
                eg[14] = s_gt[g_off_1 + 128 + 14];
                eg[15] = s_gt[g_off_1 + 128 + 15];
                eg[16] = s_gt[g_off_1 + 128 + 16];
                eg[17] = s_gt[g_off_1 + 128 + 17];
                eg[18] = s_gt[g_off_1 + 128 + 18];
                eg[19] = s_gt[g_off_1 + 128 + 19];
                eg[20] = s_gt[g_off_1 + 128 + 20];
                eg[21] = s_gt[g_off_1 + 128 + 21];
                eg[22] = s_gt[g_off_1 + 128 + 22];
                eg[23] = s_gt[g_off_1 + 128 + 23];
                eg[24] = s_gt[g_off_1 + 128 + 24];
                eg[25] = s_gt[g_off_1 + 128 + 25];
                eg[26] = s_gt[g_off_1 + 128 + 26];
                eg[27] = s_gt[g_off_1 + 128 + 27];
                eg[28] = s_gt[g_off_1 + 128 + 28];
                eg[29] = s_gt[g_off_1 + 128 + 29];
                eg[30] = s_gt[g_off_1 + 128 + 30];
                eg[31] = s_gt[g_off_1 + 128 + 31];
                eg[32] = s_gt[g_off_1 + 128 + 32];
                eg[33] = s_gt[g_off_1 + 128 + 33];
                eg[34] = s_gt[g_off_1 + 128 + 34];
                eg[35] = s_gt[g_off_1 + 128 + 35];
                eg[36] = s_gt[g_off_1 + 128 + 36];
                eg[37] = s_gt[g_off_1 + 128 + 37];
                eg[38] = s_gt[g_off_1 + 128 + 38];
                eg[39] = s_gt[g_off_1 + 128 + 39];
                eg[40] = s_gt[g_off_1 + 128 + 40];
                eg[41] = s_gt[g_off_1 + 128 + 41];
                eg[42] = s_gt[g_off_1 + 128 + 42];
                eg[43] = s_gt[g_off_1 + 128 + 43];
                eg[44] = s_gt[g_off_1 + 128 + 44];
                eg[45] = s_gt[g_off_1 + 128 + 45];
                eg[46] = s_gt[g_off_1 + 128 + 46];
                eg[47] = s_gt[g_off_1 + 128 + 47];
                eg[48] = s_gt[g_off_1 + 128 + 48];
                eg[49] = s_gt[g_off_1 + 128 + 49];
                eg[50] = s_gt[g_off_1 + 128 + 50];
                eg[51] = s_gt[g_off_1 + 128 + 51];
                eg[52] = s_gt[g_off_1 + 128 + 52];
                eg[53] = s_gt[g_off_1 + 128 + 53];
                eg[54] = s_gt[g_off_1 + 128 + 54];
                eg[55] = s_gt[g_off_1 + 128 + 55];
                eg[56] = s_gt[g_off_1 + 128 + 56];
                eg[57] = s_gt[g_off_1 + 128 + 57];
                eg[58] = s_gt[g_off_1 + 128 + 58];
                eg[59] = s_gt[g_off_1 + 128 + 59];
                eg[60] = s_gt[g_off_1 + 128 + 60];
                eg[61] = s_gt[g_off_1 + 128 + 61];
                eg[62] = s_gt[g_off_1 + 128 + 62];
                eg[63] = s_gt[g_off_1 + 128 + 63];
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(ov1)[_ls], reinterpret_cast<const float2*>(eg)[_ls]);
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x64.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                    :: "r"(tmem_o + row_off_2), "f"(ov1[0]), "f"(ov1[1]), "f"(ov1[2]), "f"(ov1[3]), "f"(ov1[4]), "f"(ov1[5]), "f"(ov1[6]), "f"(ov1[7]), "f"(ov1[8]), "f"(ov1[9]), "f"(ov1[10]), "f"(ov1[11]), "f"(ov1[12]), "f"(ov1[13]), "f"(ov1[14]), "f"(ov1[15]), "f"(ov1[16]), "f"(ov1[17]), "f"(ov1[18]), "f"(ov1[19]), "f"(ov1[20]), "f"(ov1[21]), "f"(ov1[22]), "f"(ov1[23]), "f"(ov1[24]), "f"(ov1[25]), "f"(ov1[26]), "f"(ov1[27]), "f"(ov1[28]), "f"(ov1[29]), "f"(ov1[30]), "f"(ov1[31]), "f"(ov1[32]), "f"(ov1[33]), "f"(ov1[34]), "f"(ov1[35]), "f"(ov1[36]), "f"(ov1[37]), "f"(ov1[38]), "f"(ov1[39]), "f"(ov1[40]), "f"(ov1[41]), "f"(ov1[42]), "f"(ov1[43]), "f"(ov1[44]), "f"(ov1[45]), "f"(ov1[46]), "f"(ov1[47]), "f"(ov1[48]), "f"(ov1[49]), "f"(ov1[50]), "f"(ov1[51]), "f"(ov1[52]), "f"(ov1[53]), "f"(ov1[54]), "f"(ov1[55]), "f"(ov1[56]), "f"(ov1[57]), "f"(ov1[58]), "f"(ov1[59]), "f"(ov1[60]), "f"(ov1[61]), "f"(ov1[62]), "f"(ov1[63]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_scaled_addr);
                mbarrier_arrive(gates_free_addr + (gw_c) * 8);
                gw_c += 1;
                if (gw_c == 2) { gw_c = 0; _phase_gates_ready_1 ^= 1; }
            }
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float od_v_1[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[31]))
                : "r"(tmem_o + row_off_2));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&od_v_1[63]))
                : "r"(tmem_o + (row_off_2 + 1048576)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(o_free_addr);
            mbarrier_wait(ost_free_addr, _phase_ost_free_0);
            _phase_ost_free_0 ^= 1;
            if (nc_2 - 1 >= 0) {
                unsigned int od_p_1[32];
                #pragma unroll
                for (int _lp = 0; _lp < 32; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(od_v_1[_lp*2 + 0], od_v_1[_lp*2+1 + 0]));
                    od_p_1[_lp] = *(uint32_t*)&_bf2;
                }
                int m_3 = lane / 8;
                int r_3 = lane % 8;
                uint32_t _stmatrix_addr_72 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + m_3 % 2 * 8) / 64 * 8192 + (m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + m_3 % 2 * 8) % 64 / 8 ^ (m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[3]))
                    : "memory");
                uint32_t _stmatrix_addr_73 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((16 + m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + m_3 % 2 * 8) / 64 * 8192 + (16 + m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + m_3 % 2 * 8) % 64 / 8 ^ (16 + m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[7]))
                    : "memory");
                uint32_t _stmatrix_addr_74 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((32 + m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + m_3 % 2 * 8) / 64 * 8192 + (32 + m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + m_3 % 2 * 8) % 64 / 8 ^ (32 + m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[11]))
                    : "memory");
                uint32_t _stmatrix_addr_75 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((48 + m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + m_3 % 2 * 8) / 64 * 8192 + (48 + m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + m_3 % 2 * 8) % 64 / 8 ^ (48 + m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[15]))
                    : "memory");
                uint32_t _stmatrix_addr_76 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + 16 + m_3 % 2 * 8) / 64 * 8192 + (m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + 16 + m_3 % 2 * 8) % 64 / 8 ^ (m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + 16 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_76), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[19]))
                    : "memory");
                uint32_t _stmatrix_addr_77 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((16 + m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + 16 + m_3 % 2 * 8) / 64 * 8192 + (16 + m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + 16 + m_3 % 2 * 8) % 64 / 8 ^ (16 + m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + 16 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_77), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[23]))
                    : "memory");
                uint32_t _stmatrix_addr_78 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((32 + m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + 16 + m_3 % 2 * 8) / 64 * 8192 + (32 + m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + 16 + m_3 % 2 * 8) % 64 / 8 ^ (32 + m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + 16 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[27]))
                    : "memory");
                uint32_t _stmatrix_addr_79 = static_cast<uint32_t>((unsigned long long)(s_ost_addr + (unsigned int)((48 + m_3 / 2 * 8 + r_3) / 64 * 16384 + (q4 * 32 + 16 + m_3 % 2 * 8) / 64 * 8192 + (48 + m_3 / 2 * 8 + r_3) % 64 * 128 + ((q4 * 32 + 16 + m_3 % 2 * 8) % 64 / 8 ^ (48 + m_3 / 2 * 8 + r_3) % 8) * 16 + (q4 * 32 + 16 + m_3 % 2 * 8) % 8 * 2)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_79), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&od_p_1[31]))
                    : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(o_staged_addr);
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int blk_8 = blockIdx.x;
            int seg_3 = walk[blk_8 / num_v_heads];
            int hv_3 = blk_8 % num_v_heads;
            int c_begin_3 = seg_chunk_start[seg_3];
            int nc_3 = seg_chunk_start[seg_3 + 1] - c_begin_3;
            int seq_3 = seg_seq[seg_3];
            unsigned int tf_c_1 = 0;
            unsigned int _phase_tiles_full_1 = 0;
            unsigned int _phase_pg_ready_0 = 1;
            unsigned int _phase_o_scaled_0 = 1;
            unsigned int _phase_hs_half_0 = 0;
            unsigned int _phase_hs_ready_0_1 = 0;
            unsigned int _phase_o_free_0 = 0;
            unsigned int _phase_vp_half_0 = 0;
            unsigned int _phase_vp_ready_0_1 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (tf_c_1) * 8, _phase_tiles_full_1);
                    mbarrier_wait(pg_ready_addr, _phase_pg_ready_0);
                    mbarrier_wait(o_scaled_addr, _phase_o_scaled_0);
                    _phase_pg_ready_0 ^= 1;
                    _phase_o_scaled_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = ((s_pg_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_0 = ((s_pg_addr) >> 4) & 0x3FFF;
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
                    :: "r"(tmem_o), "r"(_smem_desc_lo_0), "r"(tmem_vnb), "r"(1) : "memory");
                    tcgen05_commit(o_full_addr);
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_1 = (((s_kq_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_2 = (((s_kq_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_1 = (((s_kq_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_2 = (((s_kq_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_smem_desc_lo_1), "r"(_smem_desc_lo_2), "r"(tmem_g), "r"(0) : "memory");
                    tcgen05_commit(g_full_addr);
                    mbarrier_wait(hs_half_addr, _phase_hs_half_0);
                    _phase_hs_half_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_3 = (((s_w_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_3 = (((s_w_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    :: "r"(tmem_ws), "r"(_smem_desc_lo_3), "r"(tmem_hs), "r"(0) : "memory");
                    mbarrier_wait(hs_ready_addr, _phase_hs_ready_0_1);
                    _phase_hs_ready_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = (((s_w_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_4 = (((s_w_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    :: "r"(tmem_ws), "r"(_smem_desc_lo_4), "r"(tmem_hs), "r"(1) : "memory");
                    tcgen05_commit(ws_full_addr);
                    mbarrier_wait(o_free_addr, _phase_o_free_0);
                    _phase_o_free_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_5 = (((s_qb_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_5 = (((s_qb_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    :: "r"(tmem_o), "r"(_smem_desc_lo_5), "r"(tmem_hs), "r"(0) : "memory");
                    tcgen05_commit(o1_full_addr);
                    mbarrier_wait(vp_half_addr, _phase_vp_half_0);
                    _phase_vp_half_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((((s_kT_addr) >> 4) & 0x7FFF) | 0x4000000) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_6 = ((((s_kT_addr) >> 4) & 0x3FFF) | 0x4000000) + (tf_c_1) * 4096;
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
                    "}\n"
                    :: "r"(tmem_s), "r"(_smem_desc_lo_6), "r"(tmem_vp), "r"(1) : "memory");
                    mbarrier_wait(vp_ready_addr, _phase_vp_ready_0_1);
                    _phase_vp_ready_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_7 = ((((s_kT_addr) >> 4) & 0x7FFF) | 0x4000000) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_7 = ((((s_kT_addr) >> 4) & 0x3FFF) | 0x4000000) + (tf_c_1) * 4096;
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
                    "add.u32 ta, %2, 16;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_s), "r"(_smem_desc_lo_7), "r"(tmem_vp), "r"(1) : "memory");
                    tcgen05_commit(s_full_addr);
                    tcgen05_commit(tiles_free_addr + (tf_c_1) * 8);
                    tf_c_1 += 1;
                    if (tf_c_1 == 2) { tf_c_1 = 0; _phase_tiles_full_1 ^= 1; }
                }
                mbarrier_wait(pg_ready_addr, _phase_pg_ready_0);
                mbarrier_wait(o_scaled_addr, _phase_o_scaled_0);
                _phase_pg_ready_0 ^= 1;
                _phase_o_scaled_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                int _smem_desc_lo_8 = ((s_pg_addr) >> 4) & 0x7FFF;
#else
                int _smem_desc_lo_8 = ((s_pg_addr) >> 4) & 0x3FFF;
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
                    :: "r"(tmem_o), "r"(_smem_desc_lo_8), "r"(tmem_vnb), "r"(1) : "memory");
                tcgen05_commit(o_full_addr);
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            int blk_9 = blockIdx.x;
            int seg_4 = walk[blk_9 / num_v_heads];
            int hv_4 = blk_9 % num_v_heads;
            int c_begin_4 = seg_chunk_start[seg_4];
            int nc_4 = seg_chunk_start[seg_4 + 1] - c_begin_4;
            int seq_4 = seg_seq[seg_4];
            int hq = hv_4 / (num_v_heads / num_heads);
            int tok_begin_2 = 0;
            int n_last_2 = 0;
            if (nc_4 > 0) {
                tok_begin_2 = chunk_start[c_begin_4];
                n_last_2 = chunk_len[c_begin_4 + nc_4 - 1];
            }
            unsigned int ld_c = 0;
            unsigned int _phase_tiles_free = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_4 = 0; c_4 < nc_4; c_4++) {
                    int tok0 = tok_begin_2 + 64 * c_4;
                    unsigned int so = ld_c * 65536;
                    mbarrier_wait(tiles_free_addr + (ld_c) * 8, _phase_tiles_free);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_c) * 8, 65536);
                    tma_3d_gmem2smem(s_kq_addr + so + 0, (&kn_tma), 0, hq, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_kq_addr + so + 8192, (&qn_tma), 0, hq, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_w_addr + so + 0, (&w_tma), 0, hv_4, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_u_addr + so + 0, (&u_tma), 0, hv_4, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_kq_addr + so + 16384, (&kn_tma), 64, hq, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_kq_addr + so + 24576, (&qn_tma), 64, hq, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_w_addr + so + 8192, (&w_tma), 64, hv_4, tok0, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_u_addr + so + 8192, (&u_tma), 64, hv_4, tok0, tiles_full_addr + (ld_c) * 8);
                    ld_c += 1;
                    if (ld_c == 2) { ld_c = 0; _phase_tiles_free ^= 1; }
                }
            }
        }
    }
    // ---- Role: gate ----
    if (warp == 14) {
        { // gate_main
            int blk_10 = blockIdx.x;
            int seg_5 = walk[blk_10 / num_v_heads];
            int hv_5 = blk_10 % num_v_heads;
            int c_begin_5 = seg_chunk_start[seg_5];
            int nc_5 = seg_chunk_start[seg_5 + 1] - c_begin_5;
            int seq_5 = seg_seq[seg_5];
            int lane_2 = lane;
            int tok_begin_3 = 0;
            int n_last_3 = 0;
            if (nc_5 > 0) {
                tok_begin_3 = chunk_start[c_begin_5];
                n_last_3 = chunk_len[c_begin_5 + nc_5 - 1];
            }
            unsigned int gt_c = 0;
            unsigned int _phase_gates_free = 1;
            #pragma unroll 1
            for (int c_5 = 0; c_5 < nc_5; c_5++) {
                mbarrier_wait(gates_free_addr + (gt_c) * 8, _phase_gates_free);
                int tok0_1 = tok_begin_3 + 64 * c_5;
                int n_2 = ((c_5 + 1 == nc_5) ? n_last_3 : 64);
                int g_off_2 = c_5 % 2 * 256;
                int _max_0 = ((n_2) > (1) ? (n_2) : (1));
                float g_last = g_cs[(long long)(tok0_1 + _max_0 - 1) * (long long)num_v_heads + (long long)hv_5];
                float gv_0 = 0.0f;
                float dv_0 = 0.0f;
                float ev_0 = 0.0f;
                if (lane_2 < n_2) {
                    gv_0 = g_cs[(long long)(tok0_1 + lane_2) * (long long)num_v_heads + (long long)hv_5];
                    float _exp2_1 = approx_exp2(g_last - gv_0);
                    dv_0 = _exp2_1;
                    float _exp2_2 = approx_exp2(gv_0);
                    ev_0 = _exp2_2 * scale;
                }
                s_gt[g_off_2 + lane_2] = gv_0;
                s_gt[g_off_2 + 64 + lane_2] = dv_0;
                s_gt[g_off_2 + 128 + lane_2] = ev_0;
                float gv_1 = 0.0f;
                float dv_1 = 0.0f;
                float ev_1 = 0.0f;
                if (lane_2 + 32 < n_2) {
                    gv_1 = g_cs[(long long)(tok0_1 + lane_2 + 32) * (long long)num_v_heads + (long long)hv_5];
                    float _exp2_3 = approx_exp2(g_last - gv_1);
                    dv_1 = _exp2_3;
                    float _exp2_4 = approx_exp2(gv_1);
                    ev_1 = _exp2_4 * scale;
                }
                s_gt[g_off_2 + lane_2 + 32] = gv_1;
                s_gt[g_off_2 + 64 + lane_2 + 32] = dv_1;
                s_gt[g_off_2 + 128 + lane_2 + 32] = ev_1;
                if (lane_2 == 0) {
                    float _exp2_5 = approx_exp2(g_last);
                    s_gt[g_off_2 + 192] = _exp2_5;
                }
                mbarrier_arrive(gates_ready_addr + (gt_c) * 8);
                gt_c += 1;
                if (gt_c == 2) { gt_c = 0; _phase_gates_free ^= 1; }
            }
        }
    }
    // ---- Role: store ----
    if (warp == 15) {
        { // store_main
            int blk_11 = blockIdx.x;
            int seg_6 = walk[blk_11 / num_v_heads];
            int hv_6 = blk_11 % num_v_heads;
            int c_begin_6 = seg_chunk_start[seg_6];
            int nc_6 = seg_chunk_start[seg_6 + 1] - c_begin_6;
            int seq_6 = seg_seq[seg_6];
            int lane_3 = lane;
            int store_h_1 = tape & 1;
            int store_v_1 = tape >> 1 & 1;
            int tok_begin_4 = 0;
            int n_last_4 = 0;
            if (nc_6 > 0) {
                tok_begin_4 = chunk_start[c_begin_6];
                n_last_4 = chunk_len[c_begin_6 + nc_6 - 1];
            }
            unsigned int _phase_o_staged_0 = 0;
            unsigned int _phase_h_staged_0 = 0;
            unsigned int _phase_vn_staged_0 = 0;
            #pragma unroll 1
            for (int c_6 = 0; c_6 < nc_6; c_6++) {
                int n_3 = ((c_6 + 1 == nc_6) ? n_last_4 : 64);
                int tok0_2 = tok_begin_4 + 64 * c_6;
                int hrow = (c_begin_6 + c_6) * num_v_heads + hv_6;
                mbarrier_wait(o_staged_addr, _phase_o_staged_0);
                _phase_o_staged_0 ^= 1;
                if (c_6 - 1 >= 0) {
                    if (((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64) == 64) {
                        if (elect_sync()) {
                            tma_store_3d((&o_tma), 0, hv_6, tok_begin_4 + 64 * (c_6 - 1), s_ost_addr + 0);
                            tma_store_3d((&o_tma), 64, hv_6, tok_begin_4 + 64 * (c_6 - 1), s_ost_addr + 8192);
                        }
                    } else {
                        int te_0 = lane_3;
                        int trow_0 = te_0 / 16;
                        if (trow_0 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_0[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_0[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_0 / 64 * 16384 + te_0 % 16 * 8 / 64 * 8192 + trow_0 % 64 * 128 + (te_0 % 16 * 8 % 64 / 8 ^ trow_0 % 8) * 16 + te_0 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_0) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_0 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_0)[0];
                        }
                        int te_1 = lane_3 + 32;
                        int trow_1 = te_1 / 16;
                        if (trow_1 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_1[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_1 / 64 * 16384 + te_1 % 16 * 8 / 64 * 8192 + trow_1 % 64 * 128 + (te_1 % 16 * 8 % 64 / 8 ^ trow_1 % 8) * 16 + te_1 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_1) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_1 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_1)[0];
                        }
                        int te_2 = lane_3 + 64;
                        int trow_2 = te_2 / 16;
                        if (trow_2 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_2[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_2 / 64 * 16384 + te_2 % 16 * 8 / 64 * 8192 + trow_2 % 64 * 128 + (te_2 % 16 * 8 % 64 / 8 ^ trow_2 % 8) * 16 + te_2 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_2) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_2 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_2)[0];
                        }
                        int te_3 = lane_3 + 96;
                        int trow_3 = te_3 / 16;
                        if (trow_3 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_3[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_3 / 64 * 16384 + te_3 % 16 * 8 / 64 * 8192 + trow_3 % 64 * 128 + (te_3 % 16 * 8 % 64 / 8 ^ trow_3 % 8) * 16 + te_3 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_3) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_3 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_3)[0];
                        }
                        int te_4 = lane_3 + 128;
                        int trow_4 = te_4 / 16;
                        if (trow_4 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_4[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_4 / 64 * 16384 + te_4 % 16 * 8 / 64 * 8192 + trow_4 % 64 * 128 + (te_4 % 16 * 8 % 64 / 8 ^ trow_4 % 8) * 16 + te_4 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_4) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_4 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_4)[0];
                        }
                        int te_5 = lane_3 + 160;
                        int trow_5 = te_5 / 16;
                        if (trow_5 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_5[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_5 / 64 * 16384 + te_5 % 16 * 8 / 64 * 8192 + trow_5 % 64 * 128 + (te_5 % 16 * 8 % 64 / 8 ^ trow_5 % 8) * 16 + te_5 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_5) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_5 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_5)[0];
                        }
                        int te_6 = lane_3 + 192;
                        int trow_6 = te_6 / 16;
                        if (trow_6 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_6[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_6 / 64 * 16384 + te_6 % 16 * 8 / 64 * 8192 + trow_6 % 64 * 128 + (te_6 % 16 * 8 % 64 / 8 ^ trow_6 % 8) * 16 + te_6 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_6) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_6 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_6)[0];
                        }
                        int te_7 = lane_3 + 224;
                        int trow_7 = te_7 / 16;
                        if (trow_7 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_7[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_7[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_7 / 64 * 16384 + te_7 % 16 * 8 / 64 * 8192 + trow_7 % 64 * 128 + (te_7 % 16 * 8 % 64 / 8 ^ trow_7 % 8) * 16 + te_7 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_7) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_7 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_7)[0];
                        }
                        int te_8 = lane_3 + 256;
                        int trow_8 = te_8 / 16;
                        if (trow_8 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_8[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_8[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_8 / 64 * 16384 + te_8 % 16 * 8 / 64 * 8192 + trow_8 % 64 * 128 + (te_8 % 16 * 8 % 64 / 8 ^ trow_8 % 8) * 16 + te_8 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_8) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_8 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_8)[0];
                        }
                        int te_9 = lane_3 + 288;
                        int trow_9 = te_9 / 16;
                        if (trow_9 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_9[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_9[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_9 / 64 * 16384 + te_9 % 16 * 8 / 64 * 8192 + trow_9 % 64 * 128 + (te_9 % 16 * 8 % 64 / 8 ^ trow_9 % 8) * 16 + te_9 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_9) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_9 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_9)[0];
                        }
                        int te_10 = lane_3 + 320;
                        int trow_10 = te_10 / 16;
                        if (trow_10 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_10[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_10[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_10 / 64 * 16384 + te_10 % 16 * 8 / 64 * 8192 + trow_10 % 64 * 128 + (te_10 % 16 * 8 % 64 / 8 ^ trow_10 % 8) * 16 + te_10 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_10) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_10 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_10)[0];
                        }
                        int te_11 = lane_3 + 352;
                        int trow_11 = te_11 / 16;
                        if (trow_11 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_11[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_11[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_11 / 64 * 16384 + te_11 % 16 * 8 / 64 * 8192 + trow_11 % 64 * 128 + (te_11 % 16 * 8 % 64 / 8 ^ trow_11 % 8) * 16 + te_11 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_11) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_11 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_11)[0];
                        }
                        int te_12 = lane_3 + 384;
                        int trow_12 = te_12 / 16;
                        if (trow_12 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_12[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_12[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_12 / 64 * 16384 + te_12 % 16 * 8 / 64 * 8192 + trow_12 % 64 * 128 + (te_12 % 16 * 8 % 64 / 8 ^ trow_12 % 8) * 16 + te_12 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_12) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_12 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_12)[0];
                        }
                        int te_13 = lane_3 + 416;
                        int trow_13 = te_13 / 16;
                        if (trow_13 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_13[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_13[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_13 / 64 * 16384 + te_13 % 16 * 8 / 64 * 8192 + trow_13 % 64 * 128 + (te_13 % 16 * 8 % 64 / 8 ^ trow_13 % 8) * 16 + te_13 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_13) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_13 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_13)[0];
                        }
                        int te_14 = lane_3 + 448;
                        int trow_14 = te_14 / 16;
                        if (trow_14 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_14[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_14[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_14 / 64 * 16384 + te_14 % 16 * 8 / 64 * 8192 + trow_14 % 64 * 128 + (te_14 % 16 * 8 % 64 / 8 ^ trow_14 % 8) * 16 + te_14 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_14) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_14 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_14)[0];
                        }
                        int te_15 = lane_3 + 480;
                        int trow_15 = te_15 / 16;
                        if (trow_15 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_15[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_15[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_15 / 64 * 16384 + te_15 % 16 * 8 / 64 * 8192 + trow_15 % 64 * 128 + (te_15 % 16 * 8 % 64 / 8 ^ trow_15 % 8) * 16 + te_15 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_15) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_15 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_15)[0];
                        }
                        int te_16 = lane_3 + 512;
                        int trow_16 = te_16 / 16;
                        if (trow_16 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_16[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_16[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_16 / 64 * 16384 + te_16 % 16 * 8 / 64 * 8192 + trow_16 % 64 * 128 + (te_16 % 16 * 8 % 64 / 8 ^ trow_16 % 8) * 16 + te_16 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_16) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_16 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_16)[0];
                        }
                        int te_17 = lane_3 + 544;
                        int trow_17 = te_17 / 16;
                        if (trow_17 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_17[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_17[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_17[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_17[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_17[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_17 / 64 * 16384 + te_17 % 16 * 8 / 64 * 8192 + trow_17 % 64 * 128 + (te_17 % 16 * 8 % 64 / 8 ^ trow_17 % 8) * 16 + te_17 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_17) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_17 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_17)[0];
                        }
                        int te_18 = lane_3 + 576;
                        int trow_18 = te_18 / 16;
                        if (trow_18 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_18[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_18[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_18[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_18[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_18[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_18 / 64 * 16384 + te_18 % 16 * 8 / 64 * 8192 + trow_18 % 64 * 128 + (te_18 % 16 * 8 % 64 / 8 ^ trow_18 % 8) * 16 + te_18 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_18) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_18 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_18)[0];
                        }
                        int te_19 = lane_3 + 608;
                        int trow_19 = te_19 / 16;
                        if (trow_19 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_19[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_19[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_19[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_19[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_19[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_19 / 64 * 16384 + te_19 % 16 * 8 / 64 * 8192 + trow_19 % 64 * 128 + (te_19 % 16 * 8 % 64 / 8 ^ trow_19 % 8) * 16 + te_19 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_19) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_19 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_19)[0];
                        }
                        int te_20 = lane_3 + 640;
                        int trow_20 = te_20 / 16;
                        if (trow_20 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_20[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_20[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_20[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_20[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_20[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_20 / 64 * 16384 + te_20 % 16 * 8 / 64 * 8192 + trow_20 % 64 * 128 + (te_20 % 16 * 8 % 64 / 8 ^ trow_20 % 8) * 16 + te_20 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_20) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_20 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_20)[0];
                        }
                        int te_21 = lane_3 + 672;
                        int trow_21 = te_21 / 16;
                        if (trow_21 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_21[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_21[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_21 / 64 * 16384 + te_21 % 16 * 8 / 64 * 8192 + trow_21 % 64 * 128 + (te_21 % 16 * 8 % 64 / 8 ^ trow_21 % 8) * 16 + te_21 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_21) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_21 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_21)[0];
                        }
                        int te_22 = lane_3 + 704;
                        int trow_22 = te_22 / 16;
                        if (trow_22 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_22[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_22[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_22[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_22[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_22 / 64 * 16384 + te_22 % 16 * 8 / 64 * 8192 + trow_22 % 64 * 128 + (te_22 % 16 * 8 % 64 / 8 ^ trow_22 % 8) * 16 + te_22 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_22) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_22 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_22)[0];
                        }
                        int te_23 = lane_3 + 736;
                        int trow_23 = te_23 / 16;
                        if (trow_23 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_23[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_23[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_23[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_23[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_23 / 64 * 16384 + te_23 % 16 * 8 / 64 * 8192 + trow_23 % 64 * 128 + (te_23 % 16 * 8 % 64 / 8 ^ trow_23 % 8) * 16 + te_23 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_23) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_23 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_23)[0];
                        }
                        int te_24 = lane_3 + 768;
                        int trow_24 = te_24 / 16;
                        if (trow_24 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_24[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_24[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_24[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_24[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_24 / 64 * 16384 + te_24 % 16 * 8 / 64 * 8192 + trow_24 % 64 * 128 + (te_24 % 16 * 8 % 64 / 8 ^ trow_24 % 8) * 16 + te_24 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_24) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_24 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_24)[0];
                        }
                        int te_25 = lane_3 + 800;
                        int trow_25 = te_25 / 16;
                        if (trow_25 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_25[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_25[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_25[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_25[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_25 / 64 * 16384 + te_25 % 16 * 8 / 64 * 8192 + trow_25 % 64 * 128 + (te_25 % 16 * 8 % 64 / 8 ^ trow_25 % 8) * 16 + te_25 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_25) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_25 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_25)[0];
                        }
                        int te_26 = lane_3 + 832;
                        int trow_26 = te_26 / 16;
                        if (trow_26 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_26[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_26[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_26[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_26[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_26 / 64 * 16384 + te_26 % 16 * 8 / 64 * 8192 + trow_26 % 64 * 128 + (te_26 % 16 * 8 % 64 / 8 ^ trow_26 % 8) * 16 + te_26 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_26) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_26 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_26)[0];
                        }
                        int te_27 = lane_3 + 864;
                        int trow_27 = te_27 / 16;
                        if (trow_27 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_27[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_27[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_27[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_27[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_27 / 64 * 16384 + te_27 % 16 * 8 / 64 * 8192 + trow_27 % 64 * 128 + (te_27 % 16 * 8 % 64 / 8 ^ trow_27 % 8) * 16 + te_27 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_27) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_27 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_27)[0];
                        }
                        int te_28 = lane_3 + 896;
                        int trow_28 = te_28 / 16;
                        if (trow_28 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_28[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_28[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_28[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_28[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_28 / 64 * 16384 + te_28 % 16 * 8 / 64 * 8192 + trow_28 % 64 * 128 + (te_28 % 16 * 8 % 64 / 8 ^ trow_28 % 8) * 16 + te_28 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_28) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_28 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_28)[0];
                        }
                        int te_29 = lane_3 + 928;
                        int trow_29 = te_29 / 16;
                        if (trow_29 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_29[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_29[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_29[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_29[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_29 / 64 * 16384 + te_29 % 16 * 8 / 64 * 8192 + trow_29 % 64 * 128 + (te_29 % 16 * 8 % 64 / 8 ^ trow_29 % 8) * 16 + te_29 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_29) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_29 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_29)[0];
                        }
                        int te_30 = lane_3 + 960;
                        int trow_30 = te_30 / 16;
                        if (trow_30 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_30[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_30[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_30 / 64 * 16384 + te_30 % 16 * 8 / 64 * 8192 + trow_30 % 64 * 128 + (te_30 % 16 * 8 % 64 / 8 ^ trow_30 % 8) * 16 + te_30 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_30) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_30 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_30)[0];
                        }
                        int te_31 = lane_3 + 992;
                        int trow_31 = te_31 / 16;
                        if (trow_31 < ((c_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                            unsigned int tw_31[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_31[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_31[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_31[(0) + 3]))
                                : "r"(s_ost_addr + (unsigned int)(trow_31 / 64 * 16384 + te_31 % 16 * 8 / 64 * 8192 + trow_31 % 64 * 128 + (te_31 % 16 * 8 % 64 / 8 ^ trow_31 % 8) * 16 + te_31 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (c_6 - 1) + trow_31) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_31 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_31)[0];
                        }
                        __syncwarp();
                    }
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.commit_group;");
                }
                mbarrier_wait(h_staged_addr, _phase_h_staged_0);
                _phase_h_staged_0 ^= 1;
                if (elect_sync()) {
                    if (store_h_1 != 0) {
                        tma_store_3d((&h_tma), 0, 0, hrow, s_hst_addr + 0);
                        tma_store_3d((&h_tma), 64, 0, hrow, s_hst_addr + 8192);
                        tma_store_3d((&h_tma), 0, 64, hrow, s_hst_addr + 16384);
                        tma_store_3d((&h_tma), 64, 64, hrow, s_hst_addr + 24576);
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                    mbarrier_arrive(ost_free_addr);
                }
                mbarrier_wait(vn_staged_addr, _phase_vn_staged_0);
                _phase_vn_staged_0 ^= 1;
                if (store_v_1 != 0) {
                    if (n_3 == 64) {
                        if (elect_sync()) {
                            tma_store_3d((&v_new_tma), 0, hv_6, tok0_2, s_vst_addr + 0);
                            tma_store_3d((&v_new_tma), 64, hv_6, tok0_2, s_vst_addr + 8192);
                        }
                    } else {
                        int te_32 = lane_3;
                        int trow_32 = te_32 / 16;
                        if (trow_32 < n_3) {
                            unsigned int tw_32[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_32[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_32[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_32[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_32[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_32 / 64 * 16384 + te_32 % 16 * 8 / 64 * 8192 + trow_32 % 64 * 128 + (te_32 % 16 * 8 % 64 / 8 ^ trow_32 % 8) * 16 + te_32 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_32) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_32 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_32)[0];
                        }
                        int te_33 = lane_3 + 32;
                        int trow_33 = te_33 / 16;
                        if (trow_33 < n_3) {
                            unsigned int tw_33[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_33[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_33[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_33[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_33[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_33 / 64 * 16384 + te_33 % 16 * 8 / 64 * 8192 + trow_33 % 64 * 128 + (te_33 % 16 * 8 % 64 / 8 ^ trow_33 % 8) * 16 + te_33 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_33) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_33 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_33)[0];
                        }
                        int te_34 = lane_3 + 64;
                        int trow_34 = te_34 / 16;
                        if (trow_34 < n_3) {
                            unsigned int tw_34[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_34[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_34[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_34[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_34[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_34 / 64 * 16384 + te_34 % 16 * 8 / 64 * 8192 + trow_34 % 64 * 128 + (te_34 % 16 * 8 % 64 / 8 ^ trow_34 % 8) * 16 + te_34 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_34) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_34 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_34)[0];
                        }
                        int te_35 = lane_3 + 96;
                        int trow_35 = te_35 / 16;
                        if (trow_35 < n_3) {
                            unsigned int tw_35[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_35[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_35[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_35[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_35 / 64 * 16384 + te_35 % 16 * 8 / 64 * 8192 + trow_35 % 64 * 128 + (te_35 % 16 * 8 % 64 / 8 ^ trow_35 % 8) * 16 + te_35 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_35) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_35 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_35)[0];
                        }
                        int te_36 = lane_3 + 128;
                        int trow_36 = te_36 / 16;
                        if (trow_36 < n_3) {
                            unsigned int tw_36[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_36[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_36[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_36[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_36 / 64 * 16384 + te_36 % 16 * 8 / 64 * 8192 + trow_36 % 64 * 128 + (te_36 % 16 * 8 % 64 / 8 ^ trow_36 % 8) * 16 + te_36 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_36) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_36 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_36)[0];
                        }
                        int te_37 = lane_3 + 160;
                        int trow_37 = te_37 / 16;
                        if (trow_37 < n_3) {
                            unsigned int tw_37[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_37[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_37[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_37[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_37 / 64 * 16384 + te_37 % 16 * 8 / 64 * 8192 + trow_37 % 64 * 128 + (te_37 % 16 * 8 % 64 / 8 ^ trow_37 % 8) * 16 + te_37 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_37) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_37 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_37)[0];
                        }
                        int te_38 = lane_3 + 192;
                        int trow_38 = te_38 / 16;
                        if (trow_38 < n_3) {
                            unsigned int tw_38[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_38[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_38[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_38[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_38 / 64 * 16384 + te_38 % 16 * 8 / 64 * 8192 + trow_38 % 64 * 128 + (te_38 % 16 * 8 % 64 / 8 ^ trow_38 % 8) * 16 + te_38 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_38) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_38 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_38)[0];
                        }
                        int te_39 = lane_3 + 224;
                        int trow_39 = te_39 / 16;
                        if (trow_39 < n_3) {
                            unsigned int tw_39[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_39[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_39 / 64 * 16384 + te_39 % 16 * 8 / 64 * 8192 + trow_39 % 64 * 128 + (te_39 % 16 * 8 % 64 / 8 ^ trow_39 % 8) * 16 + te_39 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_39) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_39 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_39)[0];
                        }
                        int te_40 = lane_3 + 256;
                        int trow_40 = te_40 / 16;
                        if (trow_40 < n_3) {
                            unsigned int tw_40[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_40[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_40[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_40[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_40[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_40 / 64 * 16384 + te_40 % 16 * 8 / 64 * 8192 + trow_40 % 64 * 128 + (te_40 % 16 * 8 % 64 / 8 ^ trow_40 % 8) * 16 + te_40 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_40) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_40 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_40)[0];
                        }
                        int te_41 = lane_3 + 288;
                        int trow_41 = te_41 / 16;
                        if (trow_41 < n_3) {
                            unsigned int tw_41[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_41[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_41[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_41[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_41 / 64 * 16384 + te_41 % 16 * 8 / 64 * 8192 + trow_41 % 64 * 128 + (te_41 % 16 * 8 % 64 / 8 ^ trow_41 % 8) * 16 + te_41 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_41) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_41 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_41)[0];
                        }
                        int te_42 = lane_3 + 320;
                        int trow_42 = te_42 / 16;
                        if (trow_42 < n_3) {
                            unsigned int tw_42[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_42[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_42[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_42[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_42[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_42 / 64 * 16384 + te_42 % 16 * 8 / 64 * 8192 + trow_42 % 64 * 128 + (te_42 % 16 * 8 % 64 / 8 ^ trow_42 % 8) * 16 + te_42 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_42) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_42 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_42)[0];
                        }
                        int te_43 = lane_3 + 352;
                        int trow_43 = te_43 / 16;
                        if (trow_43 < n_3) {
                            unsigned int tw_43[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_43[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_43[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_43[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_43 / 64 * 16384 + te_43 % 16 * 8 / 64 * 8192 + trow_43 % 64 * 128 + (te_43 % 16 * 8 % 64 / 8 ^ trow_43 % 8) * 16 + te_43 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_43) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_43 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_43)[0];
                        }
                        int te_44 = lane_3 + 384;
                        int trow_44 = te_44 / 16;
                        if (trow_44 < n_3) {
                            unsigned int tw_44[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_44[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_44[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_44[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_44 / 64 * 16384 + te_44 % 16 * 8 / 64 * 8192 + trow_44 % 64 * 128 + (te_44 % 16 * 8 % 64 / 8 ^ trow_44 % 8) * 16 + te_44 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_44) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_44 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_44)[0];
                        }
                        int te_45 = lane_3 + 416;
                        int trow_45 = te_45 / 16;
                        if (trow_45 < n_3) {
                            unsigned int tw_45[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_45[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_45[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_45[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_45 / 64 * 16384 + te_45 % 16 * 8 / 64 * 8192 + trow_45 % 64 * 128 + (te_45 % 16 * 8 % 64 / 8 ^ trow_45 % 8) * 16 + te_45 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_45) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_45 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_45)[0];
                        }
                        int te_46 = lane_3 + 448;
                        int trow_46 = te_46 / 16;
                        if (trow_46 < n_3) {
                            unsigned int tw_46[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_46[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_46[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_46[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_46 / 64 * 16384 + te_46 % 16 * 8 / 64 * 8192 + trow_46 % 64 * 128 + (te_46 % 16 * 8 % 64 / 8 ^ trow_46 % 8) * 16 + te_46 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_46) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_46 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_46)[0];
                        }
                        int te_47 = lane_3 + 480;
                        int trow_47 = te_47 / 16;
                        if (trow_47 < n_3) {
                            unsigned int tw_47[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_47[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_47[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_47[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_47 / 64 * 16384 + te_47 % 16 * 8 / 64 * 8192 + trow_47 % 64 * 128 + (te_47 % 16 * 8 % 64 / 8 ^ trow_47 % 8) * 16 + te_47 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_47) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_47 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_47)[0];
                        }
                        int te_48 = lane_3 + 512;
                        int trow_48 = te_48 / 16;
                        if (trow_48 < n_3) {
                            unsigned int tw_48[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_48[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_48 / 64 * 16384 + te_48 % 16 * 8 / 64 * 8192 + trow_48 % 64 * 128 + (te_48 % 16 * 8 % 64 / 8 ^ trow_48 % 8) * 16 + te_48 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_48) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_48 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_48)[0];
                        }
                        int te_49 = lane_3 + 544;
                        int trow_49 = te_49 / 16;
                        if (trow_49 < n_3) {
                            unsigned int tw_49[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_49[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_49[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_49[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_49[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_49 / 64 * 16384 + te_49 % 16 * 8 / 64 * 8192 + trow_49 % 64 * 128 + (te_49 % 16 * 8 % 64 / 8 ^ trow_49 % 8) * 16 + te_49 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_49) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_49 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_49)[0];
                        }
                        int te_50 = lane_3 + 576;
                        int trow_50 = te_50 / 16;
                        if (trow_50 < n_3) {
                            unsigned int tw_50[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_50[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_50[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_50[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_50[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_50 / 64 * 16384 + te_50 % 16 * 8 / 64 * 8192 + trow_50 % 64 * 128 + (te_50 % 16 * 8 % 64 / 8 ^ trow_50 % 8) * 16 + te_50 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_50) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_50 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_50)[0];
                        }
                        int te_51 = lane_3 + 608;
                        int trow_51 = te_51 / 16;
                        if (trow_51 < n_3) {
                            unsigned int tw_51[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_51[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_51[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_51[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_51[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_51 / 64 * 16384 + te_51 % 16 * 8 / 64 * 8192 + trow_51 % 64 * 128 + (te_51 % 16 * 8 % 64 / 8 ^ trow_51 % 8) * 16 + te_51 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_51) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_51 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_51)[0];
                        }
                        int te_52 = lane_3 + 640;
                        int trow_52 = te_52 / 16;
                        if (trow_52 < n_3) {
                            unsigned int tw_52[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_52[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_52[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_52[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_52[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_52 / 64 * 16384 + te_52 % 16 * 8 / 64 * 8192 + trow_52 % 64 * 128 + (te_52 % 16 * 8 % 64 / 8 ^ trow_52 % 8) * 16 + te_52 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_52) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_52 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_52)[0];
                        }
                        int te_53 = lane_3 + 672;
                        int trow_53 = te_53 / 16;
                        if (trow_53 < n_3) {
                            unsigned int tw_53[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_53[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_53[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_53[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_53[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_53 / 64 * 16384 + te_53 % 16 * 8 / 64 * 8192 + trow_53 % 64 * 128 + (te_53 % 16 * 8 % 64 / 8 ^ trow_53 % 8) * 16 + te_53 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_53) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_53 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_53)[0];
                        }
                        int te_54 = lane_3 + 704;
                        int trow_54 = te_54 / 16;
                        if (trow_54 < n_3) {
                            unsigned int tw_54[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_54[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_54[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_54[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_54[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_54 / 64 * 16384 + te_54 % 16 * 8 / 64 * 8192 + trow_54 % 64 * 128 + (te_54 % 16 * 8 % 64 / 8 ^ trow_54 % 8) * 16 + te_54 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_54) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_54 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_54)[0];
                        }
                        int te_55 = lane_3 + 736;
                        int trow_55 = te_55 / 16;
                        if (trow_55 < n_3) {
                            unsigned int tw_55[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_55[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_55[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_55[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_55[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_55 / 64 * 16384 + te_55 % 16 * 8 / 64 * 8192 + trow_55 % 64 * 128 + (te_55 % 16 * 8 % 64 / 8 ^ trow_55 % 8) * 16 + te_55 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_55) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_55 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_55)[0];
                        }
                        int te_56 = lane_3 + 768;
                        int trow_56 = te_56 / 16;
                        if (trow_56 < n_3) {
                            unsigned int tw_56[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_56[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_56[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_56[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_56[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_56 / 64 * 16384 + te_56 % 16 * 8 / 64 * 8192 + trow_56 % 64 * 128 + (te_56 % 16 * 8 % 64 / 8 ^ trow_56 % 8) * 16 + te_56 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_56) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_56 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_56)[0];
                        }
                        int te_57 = lane_3 + 800;
                        int trow_57 = te_57 / 16;
                        if (trow_57 < n_3) {
                            unsigned int tw_57[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_57[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_57 / 64 * 16384 + te_57 % 16 * 8 / 64 * 8192 + trow_57 % 64 * 128 + (te_57 % 16 * 8 % 64 / 8 ^ trow_57 % 8) * 16 + te_57 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_57) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_57 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_57)[0];
                        }
                        int te_58 = lane_3 + 832;
                        int trow_58 = te_58 / 16;
                        if (trow_58 < n_3) {
                            unsigned int tw_58[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_58[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_58[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_58[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_58[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_58 / 64 * 16384 + te_58 % 16 * 8 / 64 * 8192 + trow_58 % 64 * 128 + (te_58 % 16 * 8 % 64 / 8 ^ trow_58 % 8) * 16 + te_58 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_58) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_58 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_58)[0];
                        }
                        int te_59 = lane_3 + 864;
                        int trow_59 = te_59 / 16;
                        if (trow_59 < n_3) {
                            unsigned int tw_59[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_59[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_59[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_59[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_59[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_59 / 64 * 16384 + te_59 % 16 * 8 / 64 * 8192 + trow_59 % 64 * 128 + (te_59 % 16 * 8 % 64 / 8 ^ trow_59 % 8) * 16 + te_59 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_59) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_59 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_59)[0];
                        }
                        int te_60 = lane_3 + 896;
                        int trow_60 = te_60 / 16;
                        if (trow_60 < n_3) {
                            unsigned int tw_60[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_60[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_60[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_60[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_60[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_60 / 64 * 16384 + te_60 % 16 * 8 / 64 * 8192 + trow_60 % 64 * 128 + (te_60 % 16 * 8 % 64 / 8 ^ trow_60 % 8) * 16 + te_60 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_60) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_60 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_60)[0];
                        }
                        int te_61 = lane_3 + 928;
                        int trow_61 = te_61 / 16;
                        if (trow_61 < n_3) {
                            unsigned int tw_61[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_61[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_61[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_61[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_61[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_61 / 64 * 16384 + te_61 % 16 * 8 / 64 * 8192 + trow_61 % 64 * 128 + (te_61 % 16 * 8 % 64 / 8 ^ trow_61 % 8) * 16 + te_61 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_61) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_61 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_61)[0];
                        }
                        int te_62 = lane_3 + 960;
                        int trow_62 = te_62 / 16;
                        if (trow_62 < n_3) {
                            unsigned int tw_62[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_62[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_62[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_62[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_62[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_62 / 64 * 16384 + te_62 % 16 * 8 / 64 * 8192 + trow_62 % 64 * 128 + (te_62 % 16 * 8 % 64 / 8 ^ trow_62 % 8) * 16 + te_62 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_62) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_62 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_62)[0];
                        }
                        int te_63 = lane_3 + 992;
                        int trow_63 = te_63 / 16;
                        if (trow_63 < n_3) {
                            unsigned int tw_63[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&tw_63[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_63[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_63[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_63[(0) + 3]))
                                : "r"(s_vst_addr + (unsigned int)(trow_63 / 64 * 16384 + te_63 % 16 * 8 / 64 * 8192 + trow_63 % 64 * 128 + (te_63 % 16 * 8 % 64 / 8 ^ trow_63 % 8) * 16 + te_63 % 16 * 8 % 8 * 2)));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0_2 + trow_63) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_63 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_63)[0];
                        }
                        __syncwarp();
                    }
                }
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 1;");
                    mbarrier_arrive(h_free_addr);
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(vn_free_addr);
                }
            }
            mbarrier_wait(o_staged_addr, _phase_o_staged_0);
            _phase_o_staged_0 ^= 1;
            if (nc_6 - 1 >= 0) {
                if (((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64) == 64) {
                    if (elect_sync()) {
                        tma_store_3d((&o_tma), 0, hv_6, tok_begin_4 + 64 * (nc_6 - 1), s_ost_addr + 0);
                        tma_store_3d((&o_tma), 64, hv_6, tok_begin_4 + 64 * (nc_6 - 1), s_ost_addr + 8192);
                    }
                } else {
                    int te_64 = lane_3;
                    int trow_64 = te_64 / 16;
                    if (trow_64 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_64[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_64[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_64[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_64[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_64[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_64 / 64 * 16384 + te_64 % 16 * 8 / 64 * 8192 + trow_64 % 64 * 128 + (te_64 % 16 * 8 % 64 / 8 ^ trow_64 % 8) * 16 + te_64 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_64) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_64 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_64)[0];
                    }
                    int te_65 = lane_3 + 32;
                    int trow_65 = te_65 / 16;
                    if (trow_65 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_65[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_65[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_65[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_65[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_65[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_65 / 64 * 16384 + te_65 % 16 * 8 / 64 * 8192 + trow_65 % 64 * 128 + (te_65 % 16 * 8 % 64 / 8 ^ trow_65 % 8) * 16 + te_65 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_65) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_65 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_65)[0];
                    }
                    int te_66 = lane_3 + 64;
                    int trow_66 = te_66 / 16;
                    if (trow_66 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_66[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_66[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_66[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_66[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_66 / 64 * 16384 + te_66 % 16 * 8 / 64 * 8192 + trow_66 % 64 * 128 + (te_66 % 16 * 8 % 64 / 8 ^ trow_66 % 8) * 16 + te_66 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_66) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_66 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_66)[0];
                    }
                    int te_67 = lane_3 + 96;
                    int trow_67 = te_67 / 16;
                    if (trow_67 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_67[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_67[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_67[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_67[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_67[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_67 / 64 * 16384 + te_67 % 16 * 8 / 64 * 8192 + trow_67 % 64 * 128 + (te_67 % 16 * 8 % 64 / 8 ^ trow_67 % 8) * 16 + te_67 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_67) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_67 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_67)[0];
                    }
                    int te_68 = lane_3 + 128;
                    int trow_68 = te_68 / 16;
                    if (trow_68 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_68[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_68[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_68[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_68[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_68[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_68 / 64 * 16384 + te_68 % 16 * 8 / 64 * 8192 + trow_68 % 64 * 128 + (te_68 % 16 * 8 % 64 / 8 ^ trow_68 % 8) * 16 + te_68 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_68) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_68 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_68)[0];
                    }
                    int te_69 = lane_3 + 160;
                    int trow_69 = te_69 / 16;
                    if (trow_69 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_69[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_69[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_69[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_69[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_69[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_69 / 64 * 16384 + te_69 % 16 * 8 / 64 * 8192 + trow_69 % 64 * 128 + (te_69 % 16 * 8 % 64 / 8 ^ trow_69 % 8) * 16 + te_69 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_69) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_69 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_69)[0];
                    }
                    int te_70 = lane_3 + 192;
                    int trow_70 = te_70 / 16;
                    if (trow_70 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_70[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_70[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_70[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_70[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_70[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_70 / 64 * 16384 + te_70 % 16 * 8 / 64 * 8192 + trow_70 % 64 * 128 + (te_70 % 16 * 8 % 64 / 8 ^ trow_70 % 8) * 16 + te_70 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_70) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_70 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_70)[0];
                    }
                    int te_71 = lane_3 + 224;
                    int trow_71 = te_71 / 16;
                    if (trow_71 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_71[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_71[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_71[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_71[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_71[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_71 / 64 * 16384 + te_71 % 16 * 8 / 64 * 8192 + trow_71 % 64 * 128 + (te_71 % 16 * 8 % 64 / 8 ^ trow_71 % 8) * 16 + te_71 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_71) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_71 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_71)[0];
                    }
                    int te_72 = lane_3 + 256;
                    int trow_72 = te_72 / 16;
                    if (trow_72 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_72[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_72[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_72[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_72[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_72[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_72 / 64 * 16384 + te_72 % 16 * 8 / 64 * 8192 + trow_72 % 64 * 128 + (te_72 % 16 * 8 % 64 / 8 ^ trow_72 % 8) * 16 + te_72 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_72) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_72 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_72)[0];
                    }
                    int te_73 = lane_3 + 288;
                    int trow_73 = te_73 / 16;
                    if (trow_73 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_73[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_73[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_73[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_73[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_73[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_73 / 64 * 16384 + te_73 % 16 * 8 / 64 * 8192 + trow_73 % 64 * 128 + (te_73 % 16 * 8 % 64 / 8 ^ trow_73 % 8) * 16 + te_73 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_73) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_73 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_73)[0];
                    }
                    int te_74 = lane_3 + 320;
                    int trow_74 = te_74 / 16;
                    if (trow_74 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_74[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_74[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_74[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_74[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_74[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_74 / 64 * 16384 + te_74 % 16 * 8 / 64 * 8192 + trow_74 % 64 * 128 + (te_74 % 16 * 8 % 64 / 8 ^ trow_74 % 8) * 16 + te_74 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_74) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_74 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_74)[0];
                    }
                    int te_75 = lane_3 + 352;
                    int trow_75 = te_75 / 16;
                    if (trow_75 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_75[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_75[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_75[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_75[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_75[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_75 / 64 * 16384 + te_75 % 16 * 8 / 64 * 8192 + trow_75 % 64 * 128 + (te_75 % 16 * 8 % 64 / 8 ^ trow_75 % 8) * 16 + te_75 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_75) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_75 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_75)[0];
                    }
                    int te_76 = lane_3 + 384;
                    int trow_76 = te_76 / 16;
                    if (trow_76 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_76[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_76[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_76[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_76[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_76[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_76 / 64 * 16384 + te_76 % 16 * 8 / 64 * 8192 + trow_76 % 64 * 128 + (te_76 % 16 * 8 % 64 / 8 ^ trow_76 % 8) * 16 + te_76 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_76) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_76 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_76)[0];
                    }
                    int te_77 = lane_3 + 416;
                    int trow_77 = te_77 / 16;
                    if (trow_77 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_77[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_77[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_77[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_77[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_77[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_77 / 64 * 16384 + te_77 % 16 * 8 / 64 * 8192 + trow_77 % 64 * 128 + (te_77 % 16 * 8 % 64 / 8 ^ trow_77 % 8) * 16 + te_77 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_77) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_77 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_77)[0];
                    }
                    int te_78 = lane_3 + 448;
                    int trow_78 = te_78 / 16;
                    if (trow_78 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_78[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_78[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_78[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_78[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_78[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_78 / 64 * 16384 + te_78 % 16 * 8 / 64 * 8192 + trow_78 % 64 * 128 + (te_78 % 16 * 8 % 64 / 8 ^ trow_78 % 8) * 16 + te_78 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_78) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_78 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_78)[0];
                    }
                    int te_79 = lane_3 + 480;
                    int trow_79 = te_79 / 16;
                    if (trow_79 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_79[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_79[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_79[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_79[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_79[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_79 / 64 * 16384 + te_79 % 16 * 8 / 64 * 8192 + trow_79 % 64 * 128 + (te_79 % 16 * 8 % 64 / 8 ^ trow_79 % 8) * 16 + te_79 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_79) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_79 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_79)[0];
                    }
                    int te_80 = lane_3 + 512;
                    int trow_80 = te_80 / 16;
                    if (trow_80 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_80[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_80[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_80[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_80[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_80[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_80 / 64 * 16384 + te_80 % 16 * 8 / 64 * 8192 + trow_80 % 64 * 128 + (te_80 % 16 * 8 % 64 / 8 ^ trow_80 % 8) * 16 + te_80 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_80) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_80 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_80)[0];
                    }
                    int te_81 = lane_3 + 544;
                    int trow_81 = te_81 / 16;
                    if (trow_81 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_81[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_81[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_81[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_81[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_81[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_81 / 64 * 16384 + te_81 % 16 * 8 / 64 * 8192 + trow_81 % 64 * 128 + (te_81 % 16 * 8 % 64 / 8 ^ trow_81 % 8) * 16 + te_81 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_81) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_81 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_81)[0];
                    }
                    int te_82 = lane_3 + 576;
                    int trow_82 = te_82 / 16;
                    if (trow_82 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_82[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_82[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_82[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_82[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_82[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_82 / 64 * 16384 + te_82 % 16 * 8 / 64 * 8192 + trow_82 % 64 * 128 + (te_82 % 16 * 8 % 64 / 8 ^ trow_82 % 8) * 16 + te_82 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_82) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_82 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_82)[0];
                    }
                    int te_83 = lane_3 + 608;
                    int trow_83 = te_83 / 16;
                    if (trow_83 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_83[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_83[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_83[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_83[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_83[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_83 / 64 * 16384 + te_83 % 16 * 8 / 64 * 8192 + trow_83 % 64 * 128 + (te_83 % 16 * 8 % 64 / 8 ^ trow_83 % 8) * 16 + te_83 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_83) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_83 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_83)[0];
                    }
                    int te_84 = lane_3 + 640;
                    int trow_84 = te_84 / 16;
                    if (trow_84 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_84[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_84[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_84[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_84[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_84[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_84 / 64 * 16384 + te_84 % 16 * 8 / 64 * 8192 + trow_84 % 64 * 128 + (te_84 % 16 * 8 % 64 / 8 ^ trow_84 % 8) * 16 + te_84 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_84) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_84 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_84)[0];
                    }
                    int te_85 = lane_3 + 672;
                    int trow_85 = te_85 / 16;
                    if (trow_85 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_85[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_85[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_85[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_85[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_85[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_85 / 64 * 16384 + te_85 % 16 * 8 / 64 * 8192 + trow_85 % 64 * 128 + (te_85 % 16 * 8 % 64 / 8 ^ trow_85 % 8) * 16 + te_85 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_85) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_85 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_85)[0];
                    }
                    int te_86 = lane_3 + 704;
                    int trow_86 = te_86 / 16;
                    if (trow_86 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_86[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_86[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_86[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_86[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_86[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_86 / 64 * 16384 + te_86 % 16 * 8 / 64 * 8192 + trow_86 % 64 * 128 + (te_86 % 16 * 8 % 64 / 8 ^ trow_86 % 8) * 16 + te_86 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_86) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_86 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_86)[0];
                    }
                    int te_87 = lane_3 + 736;
                    int trow_87 = te_87 / 16;
                    if (trow_87 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_87[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_87[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_87[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_87[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_87[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_87 / 64 * 16384 + te_87 % 16 * 8 / 64 * 8192 + trow_87 % 64 * 128 + (te_87 % 16 * 8 % 64 / 8 ^ trow_87 % 8) * 16 + te_87 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_87) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_87 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_87)[0];
                    }
                    int te_88 = lane_3 + 768;
                    int trow_88 = te_88 / 16;
                    if (trow_88 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_88[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_88[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_88[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_88[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_88[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_88 / 64 * 16384 + te_88 % 16 * 8 / 64 * 8192 + trow_88 % 64 * 128 + (te_88 % 16 * 8 % 64 / 8 ^ trow_88 % 8) * 16 + te_88 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_88) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_88 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_88)[0];
                    }
                    int te_89 = lane_3 + 800;
                    int trow_89 = te_89 / 16;
                    if (trow_89 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_89[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_89[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_89[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_89[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_89[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_89 / 64 * 16384 + te_89 % 16 * 8 / 64 * 8192 + trow_89 % 64 * 128 + (te_89 % 16 * 8 % 64 / 8 ^ trow_89 % 8) * 16 + te_89 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_89) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_89 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_89)[0];
                    }
                    int te_90 = lane_3 + 832;
                    int trow_90 = te_90 / 16;
                    if (trow_90 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_90[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_90[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_90[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_90[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_90[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_90 / 64 * 16384 + te_90 % 16 * 8 / 64 * 8192 + trow_90 % 64 * 128 + (te_90 % 16 * 8 % 64 / 8 ^ trow_90 % 8) * 16 + te_90 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_90) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_90 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_90)[0];
                    }
                    int te_91 = lane_3 + 864;
                    int trow_91 = te_91 / 16;
                    if (trow_91 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_91[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_91[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_91[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_91[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_91[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_91 / 64 * 16384 + te_91 % 16 * 8 / 64 * 8192 + trow_91 % 64 * 128 + (te_91 % 16 * 8 % 64 / 8 ^ trow_91 % 8) * 16 + te_91 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_91) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_91 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_91)[0];
                    }
                    int te_92 = lane_3 + 896;
                    int trow_92 = te_92 / 16;
                    if (trow_92 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_92[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_92[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_92[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_92[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_92[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_92 / 64 * 16384 + te_92 % 16 * 8 / 64 * 8192 + trow_92 % 64 * 128 + (te_92 % 16 * 8 % 64 / 8 ^ trow_92 % 8) * 16 + te_92 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_92) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_92 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_92)[0];
                    }
                    int te_93 = lane_3 + 928;
                    int trow_93 = te_93 / 16;
                    if (trow_93 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_93[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_93[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_93[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_93[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_93[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_93 / 64 * 16384 + te_93 % 16 * 8 / 64 * 8192 + trow_93 % 64 * 128 + (te_93 % 16 * 8 % 64 / 8 ^ trow_93 % 8) * 16 + te_93 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_93) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_93 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_93)[0];
                    }
                    int te_94 = lane_3 + 960;
                    int trow_94 = te_94 / 16;
                    if (trow_94 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_94[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_94[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_94[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_94[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_94[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_94 / 64 * 16384 + te_94 % 16 * 8 / 64 * 8192 + trow_94 % 64 * 128 + (te_94 % 16 * 8 % 64 / 8 ^ trow_94 % 8) * 16 + te_94 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_94) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_94 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_94)[0];
                    }
                    int te_95 = lane_3 + 992;
                    int trow_95 = te_95 / 16;
                    if (trow_95 < ((nc_6 - 1 + 1 == nc_6) ? n_last_4 : 64)) {
                        unsigned int tw_95[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tw_95[0])), "=r"(*reinterpret_cast<uint32_t*>(&tw_95[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tw_95[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tw_95[(0) + 3]))
                            : "r"(s_ost_addr + (unsigned int)(trow_95 / 64 * 16384 + te_95 % 16 * 8 / 64 * 8192 + trow_95 % 64 * 128 + (te_95 % 16 * 8 % 64 / 8 ^ trow_95 % 8) * 16 + te_95 % 16 * 8 % 8 * 2)));
                        reinterpret_cast<int4*>(o_out + (((long long)(tok_begin_4 + 64 * (nc_6 - 1) + trow_95) * (long long)num_v_heads + (long long)hv_6) * 128ll + (long long)(te_95 % 16 * 8)))[0] = reinterpret_cast<int4*>(tw_95)[0];
                    }
                    __syncwarp();
                }
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.commit_group;");
            }
            if (elect_sync()) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
                mbarrier_arrive(ost_free_addr);
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
