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
#define TMEM_ALL_OFFSET 0
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_S_W_OFF 0
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 65536
#define SMEM_S_U_OFF 16384
#define SMEM_S_U_STAGE_BYTES 16384
#define SMEM_S_U_STRIDE 65536
#define SMEM_S_KN_OFF 32768
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 65536
#define SMEM_S_KNT_OFF 32768
#define SMEM_S_KNT_STAGE_BYTES 16384
#define SMEM_S_KNT_STRIDE 65536
#define SMEM_S_QN_OFF 49152
#define SMEM_S_QN_STAGE_BYTES 16384
#define SMEM_S_QN_STRIDE 65536
#define SMEM_S_GT_OFF 131072
#define SMEM_S_GT_STAGE_BYTES 2048
#define SMEM_S_GT_STRIDE 2048
#define SMEM_S_HS0_OFF 133120
#define SMEM_S_HS0_STAGE_BYTES 16384
#define SMEM_S_HS0_STRIDE 16384
#define SMEM_S_HS1_OFF 149504
#define SMEM_S_HS1_STAGE_BYTES 16384
#define SMEM_S_HS1_STRIDE 16384
#define SMEM_S_VP0_OFF 165888
#define SMEM_S_VP0_STAGE_BYTES 8192
#define SMEM_S_VP0_STRIDE 8192
#define SMEM_S_VP1_OFF 174080
#define SMEM_S_VP1_STAGE_BYTES 8192
#define SMEM_S_VP1_STRIDE 8192
#define SMEM_S_VN_OFF 182272
#define SMEM_S_VN_STAGE_BYTES 16384
#define SMEM_S_VN_STRIDE 16384
#define SMEM_S_VNT_OFF 182272
#define SMEM_S_VNT_STAGE_BYTES 16384
#define SMEM_S_VNT_STRIDE 16384
#define SMEM_S_PG_OFF 198656
#define SMEM_S_PG_STAGE_BYTES 8192
#define SMEM_S_PG_STRIDE 8192
#define SMEM_S_OST_OFF 206848
#define SMEM_S_OST_STAGE_BYTES 16384
#define SMEM_S_OST_STRIDE 16384
#define SMEM_TOTAL 223488
#define THREADS 512
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
kernel_cake_gdn_train_446adf6e2aa8ec228bc5(const __grid_constant__ CUtensorMap qn_tma, const __grid_constant__ CUtensorMap kn_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap u_tma, float* __restrict__ g_cs, float* __restrict__ h0, float* __restrict__ seg_start, const __grid_constant__ CUtensorMap h_tma, const __grid_constant__ CUtensorMap v_new_tma, __nv_bfloat16* __restrict__ v_new, const __grid_constant__ CUtensorMap o_tma, __nv_bfloat16* __restrict__ o_out, float* __restrict__ final_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int* __restrict__ walk, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int use_seg_start, float scale)
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

    const int mbar_base = smem + 223232;
    #define tiles_full_addr (mbar_base + 0)
    #define w_free_addr (mbar_base + 16)
    #define u_free_addr (mbar_base + 32)
    #define qk_free_addr (mbar_base + 48)
    #define gates_ready_addr (mbar_base + 64)
    #define gates_free_addr (mbar_base + 80)
    #define hs_k0_addr (mbar_base + 96)
    #define hs_k1_addr (mbar_base + 104)
    #define s_scaled_addr (mbar_base + 112)
    #define ws_full_addr (mbar_base + 120)
    #define vprime_ready_addr (mbar_base + 128)
    #define s_full_addr (mbar_base + 136)
    #define hs_free_addr (mbar_base + 144)
    #define vn_free_addr (mbar_base + 152)
    #define p_full_addr (mbar_base + 160)
    #define p_free_addr (mbar_base + 176)
    #define o1_full_addr (mbar_base + 192)
    #define pv_ready_addr (mbar_base + 208)
    #define o_full_addr (mbar_base + 216)
    #define o_free_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_w_addr = smem + 0;
    __nv_bfloat16* s_u = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_u_addr = smem + 16384;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_kn_addr = smem + 32768;
    __nv_bfloat16* s_knT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_knT_addr = smem + 32768;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_qn_addr = smem + 49152;
    float* s_gt = reinterpret_cast<float*>(smem_raw + 131072);
    const int s_gt_addr = smem + 131072;
    __nv_bfloat16* s_hs0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 133120);
    const int s_hs0_addr = smem + 133120;
    __nv_bfloat16* s_hs1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int s_hs1_addr = smem + 149504;
    __nv_bfloat16* s_vp0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165888);
    const int s_vp0_addr = smem + 165888;
    __nv_bfloat16* s_vp1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 174080);
    const int s_vp1_addr = smem + 174080;
    __nv_bfloat16* s_vn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 182272);
    const int s_vn_addr = smem + 182272;
    __nv_bfloat16* s_vnT = reinterpret_cast<__nv_bfloat16*>(smem_raw + 182272);
    const int s_vnT_addr = smem + 182272;
    __nv_bfloat16* s_pg = reinterpret_cast<__nv_bfloat16*>(smem_raw + 198656);
    const int s_pg_addr = smem + 198656;
    __nv_bfloat16* s_ost = reinterpret_cast<__nv_bfloat16*>(smem_raw + 206848);
    const int s_ost_addr = smem + 206848;

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 31 barriers)
    // Mbarriers at smem_raw[223232..223480)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 223232, 1);
            mbarrier_init(smem + 223240, 1);
            // w_free: 2 barriers, init_count=1
            mbarrier_init(smem + 223248, 1);
            mbarrier_init(smem + 223256, 1);
            // u_free: 2 barriers, init_count=128
            mbarrier_init(smem + 223264, 128);
            mbarrier_init(smem + 223272, 128);
            // qk_free: 2 barriers, init_count=1
            mbarrier_init(smem + 223280, 1);
            mbarrier_init(smem + 223288, 1);
            // gates_ready: 2 barriers, init_count=32
            mbarrier_init(smem + 223296, 32);
            mbarrier_init(smem + 223304, 32);
            // gates_free: 2 barriers, init_count=384
            mbarrier_init(smem + 223312, 384);
            mbarrier_init(smem + 223320, 384);
            // --- pipeline 'one' ---
            // hs_k0: 1 barriers, init_count=128
            mbarrier_init(smem + 223328, 128);
            // hs_k1: 1 barriers, init_count=128
            mbarrier_init(smem + 223336, 128);
            // s_scaled: 1 barriers, init_count=128
            mbarrier_init(smem + 223344, 128);
            // ws_full: 1 barriers, init_count=1
            mbarrier_init(smem + 223352, 1);
            // vprime_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 223360, 128);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 223368, 1);
            // hs_free: 1 barriers, init_count=1
            mbarrier_init(smem + 223376, 1);
            // vn_free: 1 barriers, init_count=2
            mbarrier_init(smem + 223384, 2);
            // --- pipeline 'ring' ---
            // p_full: 2 barriers, init_count=1
            mbarrier_init(smem + 223392, 1);
            mbarrier_init(smem + 223400, 1);
            // p_free: 2 barriers, init_count=128
            mbarrier_init(smem + 223408, 128);
            mbarrier_init(smem + 223416, 128);
            // o1_full: 2 barriers, init_count=1
            mbarrier_init(smem + 223424, 1);
            mbarrier_init(smem + 223432, 1);
            // --- pipeline 'one' ---
            // pv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 223440, 128);
            // --- pipeline 'ring' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 223448, 1);
            mbarrier_init(smem + 223456, 1);
            // o_free: 2 barriers, init_count=128
            mbarrier_init(smem + 223464, 128);
            mbarrier_init(smem + 223472, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 223480);
    if (warp == 0) {
        int _tmem_hold = smem + 223480;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_all = taddr;
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");
    }

    // ---- Role: state ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // state_main
            int blk = blockIdx.x;
            int seg = walk[blk / num_v_heads];
            int hv = blk % num_v_heads;
            int c_begin = seg_chunk_start[seg];
            int nc = seg_chunk_start[seg + 1] - c_begin;
            int seq = seg_seq[seg];
            int lane_1 = lane;
            int p = warp % 4;
            int cq = lane_1 % 4 * 2;
            int va0 = p * 16 + lane_1 / 4;
            int st_row = p * 16 + lane_1 / 8 % 2 * 8 + lane_1 % 8;
            int st_col = lane_1 / 16 * 8;
            int t_lo = tmem_all + (p * 32 << 16);
            int t_hi = t_lo + 1048576;
            int is_last = 0;
            int src_kind = 0;
            if (seq >= 0) {
                if (c_begin + nc == seq_chunk_start[seq + 1]) {
                    is_last = 1;
                }
                if (use_seg_start != 0) {
                    src_kind = 2;
                } else if (use_initial_state != 0) {
                    src_kind = 1;
                }
            }
            long long seg_base = ((long long)seg * (long long)num_v_heads + (long long)hv) * 16384;
            int _max_0 = ((seq) > (0) ? (seq) : (0));
            long long seq_base = ((long long)_max_0 * (long long)num_v_heads + (long long)hv) * 16384;
            if (src_kind == 2) {
                float blk_0[16];
                blk_0[0] = seg_start[seg_base + (long long)cq * 128 + (long long)va0];
                blk_0[1] = seg_start[seg_base + (long long)(cq + 1) * 128 + (long long)va0];
                blk_0[2] = seg_start[seg_base + (long long)cq * 128 + (long long)(va0 + 8)];
                blk_0[3] = seg_start[seg_base + (long long)(cq + 1) * 128 + (long long)(va0 + 8)];
                blk_0[4] = seg_start[seg_base + (long long)(8 + cq) * 128 + (long long)va0];
                blk_0[5] = seg_start[seg_base + (long long)(8 + cq + 1) * 128 + (long long)va0];
                blk_0[6] = seg_start[seg_base + (long long)(8 + cq) * 128 + (long long)(va0 + 8)];
                blk_0[7] = seg_start[seg_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_0[8] = seg_start[seg_base + (long long)(16 + cq) * 128 + (long long)va0];
                blk_0[9] = seg_start[seg_base + (long long)(16 + cq + 1) * 128 + (long long)va0];
                blk_0[10] = seg_start[seg_base + (long long)(16 + cq) * 128 + (long long)(va0 + 8)];
                blk_0[11] = seg_start[seg_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_0[12] = seg_start[seg_base + (long long)(24 + cq) * 128 + (long long)va0];
                blk_0[13] = seg_start[seg_base + (long long)(24 + cq + 1) * 128 + (long long)va0];
                blk_0[14] = seg_start[seg_base + (long long)(24 + cq) * 128 + (long long)(va0 + 8)];
                blk_0[15] = seg_start[seg_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_0[15])));
                float blk_1[16];
                blk_1[0] = seg_start[seg_base + (long long)(32 + cq) * 128 + (long long)va0];
                blk_1[1] = seg_start[seg_base + (long long)(32 + cq + 1) * 128 + (long long)va0];
                blk_1[2] = seg_start[seg_base + (long long)(32 + cq) * 128 + (long long)(va0 + 8)];
                blk_1[3] = seg_start[seg_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_1[4] = seg_start[seg_base + (long long)(40 + cq) * 128 + (long long)va0];
                blk_1[5] = seg_start[seg_base + (long long)(40 + cq + 1) * 128 + (long long)va0];
                blk_1[6] = seg_start[seg_base + (long long)(40 + cq) * 128 + (long long)(va0 + 8)];
                blk_1[7] = seg_start[seg_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_1[8] = seg_start[seg_base + (long long)(48 + cq) * 128 + (long long)va0];
                blk_1[9] = seg_start[seg_base + (long long)(48 + cq + 1) * 128 + (long long)va0];
                blk_1[10] = seg_start[seg_base + (long long)(48 + cq) * 128 + (long long)(va0 + 8)];
                blk_1[11] = seg_start[seg_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_1[12] = seg_start[seg_base + (long long)(56 + cq) * 128 + (long long)va0];
                blk_1[13] = seg_start[seg_base + (long long)(56 + cq + 1) * 128 + (long long)va0];
                blk_1[14] = seg_start[seg_base + (long long)(56 + cq) * 128 + (long long)(va0 + 8)];
                blk_1[15] = seg_start[seg_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 32), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_1[15])));
                float blk_2[16];
                blk_2[0] = seg_start[seg_base + (long long)(64 + cq) * 128 + (long long)va0];
                blk_2[1] = seg_start[seg_base + (long long)(64 + cq + 1) * 128 + (long long)va0];
                blk_2[2] = seg_start[seg_base + (long long)(64 + cq) * 128 + (long long)(va0 + 8)];
                blk_2[3] = seg_start[seg_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_2[4] = seg_start[seg_base + (long long)(72 + cq) * 128 + (long long)va0];
                blk_2[5] = seg_start[seg_base + (long long)(72 + cq + 1) * 128 + (long long)va0];
                blk_2[6] = seg_start[seg_base + (long long)(72 + cq) * 128 + (long long)(va0 + 8)];
                blk_2[7] = seg_start[seg_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_2[8] = seg_start[seg_base + (long long)(80 + cq) * 128 + (long long)va0];
                blk_2[9] = seg_start[seg_base + (long long)(80 + cq + 1) * 128 + (long long)va0];
                blk_2[10] = seg_start[seg_base + (long long)(80 + cq) * 128 + (long long)(va0 + 8)];
                blk_2[11] = seg_start[seg_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_2[12] = seg_start[seg_base + (long long)(88 + cq) * 128 + (long long)va0];
                blk_2[13] = seg_start[seg_base + (long long)(88 + cq + 1) * 128 + (long long)va0];
                blk_2[14] = seg_start[seg_base + (long long)(88 + cq) * 128 + (long long)(va0 + 8)];
                blk_2[15] = seg_start[seg_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 64), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_2[15])));
                float blk_3[16];
                blk_3[0] = seg_start[seg_base + (long long)(96 + cq) * 128 + (long long)va0];
                blk_3[1] = seg_start[seg_base + (long long)(96 + cq + 1) * 128 + (long long)va0];
                blk_3[2] = seg_start[seg_base + (long long)(96 + cq) * 128 + (long long)(va0 + 8)];
                blk_3[3] = seg_start[seg_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_3[4] = seg_start[seg_base + (long long)(104 + cq) * 128 + (long long)va0];
                blk_3[5] = seg_start[seg_base + (long long)(104 + cq + 1) * 128 + (long long)va0];
                blk_3[6] = seg_start[seg_base + (long long)(104 + cq) * 128 + (long long)(va0 + 8)];
                blk_3[7] = seg_start[seg_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_3[8] = seg_start[seg_base + (long long)(112 + cq) * 128 + (long long)va0];
                blk_3[9] = seg_start[seg_base + (long long)(112 + cq + 1) * 128 + (long long)va0];
                blk_3[10] = seg_start[seg_base + (long long)(112 + cq) * 128 + (long long)(va0 + 8)];
                blk_3[11] = seg_start[seg_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_3[12] = seg_start[seg_base + (long long)(120 + cq) * 128 + (long long)va0];
                blk_3[13] = seg_start[seg_base + (long long)(120 + cq + 1) * 128 + (long long)va0];
                blk_3[14] = seg_start[seg_base + (long long)(120 + cq) * 128 + (long long)(va0 + 8)];
                blk_3[15] = seg_start[seg_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 96), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_3[15])));
                float blk_4[16];
                blk_4[0] = seg_start[seg_base + (long long)cq * 128 + (long long)(va0 + 64)];
                blk_4[1] = seg_start[seg_base + (long long)(cq + 1) * 128 + (long long)(va0 + 64)];
                blk_4[2] = seg_start[seg_base + (long long)cq * 128 + (long long)(va0 + 64 + 8)];
                blk_4[3] = seg_start[seg_base + (long long)(cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_4[4] = seg_start[seg_base + (long long)(8 + cq) * 128 + (long long)(va0 + 64)];
                blk_4[5] = seg_start[seg_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_4[6] = seg_start[seg_base + (long long)(8 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_4[7] = seg_start[seg_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_4[8] = seg_start[seg_base + (long long)(16 + cq) * 128 + (long long)(va0 + 64)];
                blk_4[9] = seg_start[seg_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_4[10] = seg_start[seg_base + (long long)(16 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_4[11] = seg_start[seg_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_4[12] = seg_start[seg_base + (long long)(24 + cq) * 128 + (long long)(va0 + 64)];
                blk_4[13] = seg_start[seg_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_4[14] = seg_start[seg_base + (long long)(24 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_4[15] = seg_start[seg_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_4[15])));
                float blk_5[16];
                blk_5[0] = seg_start[seg_base + (long long)(32 + cq) * 128 + (long long)(va0 + 64)];
                blk_5[1] = seg_start[seg_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_5[2] = seg_start[seg_base + (long long)(32 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[3] = seg_start[seg_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[4] = seg_start[seg_base + (long long)(40 + cq) * 128 + (long long)(va0 + 64)];
                blk_5[5] = seg_start[seg_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_5[6] = seg_start[seg_base + (long long)(40 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[7] = seg_start[seg_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[8] = seg_start[seg_base + (long long)(48 + cq) * 128 + (long long)(va0 + 64)];
                blk_5[9] = seg_start[seg_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_5[10] = seg_start[seg_base + (long long)(48 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[11] = seg_start[seg_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[12] = seg_start[seg_base + (long long)(56 + cq) * 128 + (long long)(va0 + 64)];
                blk_5[13] = seg_start[seg_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_5[14] = seg_start[seg_base + (long long)(56 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_5[15] = seg_start[seg_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 32), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_5[15])));
                float blk_6[16];
                blk_6[0] = seg_start[seg_base + (long long)(64 + cq) * 128 + (long long)(va0 + 64)];
                blk_6[1] = seg_start[seg_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_6[2] = seg_start[seg_base + (long long)(64 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[3] = seg_start[seg_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[4] = seg_start[seg_base + (long long)(72 + cq) * 128 + (long long)(va0 + 64)];
                blk_6[5] = seg_start[seg_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_6[6] = seg_start[seg_base + (long long)(72 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[7] = seg_start[seg_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[8] = seg_start[seg_base + (long long)(80 + cq) * 128 + (long long)(va0 + 64)];
                blk_6[9] = seg_start[seg_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_6[10] = seg_start[seg_base + (long long)(80 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[11] = seg_start[seg_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[12] = seg_start[seg_base + (long long)(88 + cq) * 128 + (long long)(va0 + 64)];
                blk_6[13] = seg_start[seg_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_6[14] = seg_start[seg_base + (long long)(88 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_6[15] = seg_start[seg_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 64), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_6[15])));
                float blk_7[16];
                blk_7[0] = seg_start[seg_base + (long long)(96 + cq) * 128 + (long long)(va0 + 64)];
                blk_7[1] = seg_start[seg_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_7[2] = seg_start[seg_base + (long long)(96 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[3] = seg_start[seg_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[4] = seg_start[seg_base + (long long)(104 + cq) * 128 + (long long)(va0 + 64)];
                blk_7[5] = seg_start[seg_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_7[6] = seg_start[seg_base + (long long)(104 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[7] = seg_start[seg_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[8] = seg_start[seg_base + (long long)(112 + cq) * 128 + (long long)(va0 + 64)];
                blk_7[9] = seg_start[seg_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_7[10] = seg_start[seg_base + (long long)(112 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[11] = seg_start[seg_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[12] = seg_start[seg_base + (long long)(120 + cq) * 128 + (long long)(va0 + 64)];
                blk_7[13] = seg_start[seg_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_7[14] = seg_start[seg_base + (long long)(120 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_7[15] = seg_start[seg_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 96), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_7[15])));
            } else if (src_kind == 1) {
                float blk_8[16];
                blk_8[0] = h0[seq_base + (long long)cq * 128 + (long long)va0];
                blk_8[1] = h0[seq_base + (long long)(cq + 1) * 128 + (long long)va0];
                blk_8[2] = h0[seq_base + (long long)cq * 128 + (long long)(va0 + 8)];
                blk_8[3] = h0[seq_base + (long long)(cq + 1) * 128 + (long long)(va0 + 8)];
                blk_8[4] = h0[seq_base + (long long)(8 + cq) * 128 + (long long)va0];
                blk_8[5] = h0[seq_base + (long long)(8 + cq + 1) * 128 + (long long)va0];
                blk_8[6] = h0[seq_base + (long long)(8 + cq) * 128 + (long long)(va0 + 8)];
                blk_8[7] = h0[seq_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_8[8] = h0[seq_base + (long long)(16 + cq) * 128 + (long long)va0];
                blk_8[9] = h0[seq_base + (long long)(16 + cq + 1) * 128 + (long long)va0];
                blk_8[10] = h0[seq_base + (long long)(16 + cq) * 128 + (long long)(va0 + 8)];
                blk_8[11] = h0[seq_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_8[12] = h0[seq_base + (long long)(24 + cq) * 128 + (long long)va0];
                blk_8[13] = h0[seq_base + (long long)(24 + cq + 1) * 128 + (long long)va0];
                blk_8[14] = h0[seq_base + (long long)(24 + cq) * 128 + (long long)(va0 + 8)];
                blk_8[15] = h0[seq_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_8[15])));
                float blk_9[16];
                blk_9[0] = h0[seq_base + (long long)(32 + cq) * 128 + (long long)va0];
                blk_9[1] = h0[seq_base + (long long)(32 + cq + 1) * 128 + (long long)va0];
                blk_9[2] = h0[seq_base + (long long)(32 + cq) * 128 + (long long)(va0 + 8)];
                blk_9[3] = h0[seq_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_9[4] = h0[seq_base + (long long)(40 + cq) * 128 + (long long)va0];
                blk_9[5] = h0[seq_base + (long long)(40 + cq + 1) * 128 + (long long)va0];
                blk_9[6] = h0[seq_base + (long long)(40 + cq) * 128 + (long long)(va0 + 8)];
                blk_9[7] = h0[seq_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_9[8] = h0[seq_base + (long long)(48 + cq) * 128 + (long long)va0];
                blk_9[9] = h0[seq_base + (long long)(48 + cq + 1) * 128 + (long long)va0];
                blk_9[10] = h0[seq_base + (long long)(48 + cq) * 128 + (long long)(va0 + 8)];
                blk_9[11] = h0[seq_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_9[12] = h0[seq_base + (long long)(56 + cq) * 128 + (long long)va0];
                blk_9[13] = h0[seq_base + (long long)(56 + cq + 1) * 128 + (long long)va0];
                blk_9[14] = h0[seq_base + (long long)(56 + cq) * 128 + (long long)(va0 + 8)];
                blk_9[15] = h0[seq_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 32), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_9[15])));
                float blk_10[16];
                blk_10[0] = h0[seq_base + (long long)(64 + cq) * 128 + (long long)va0];
                blk_10[1] = h0[seq_base + (long long)(64 + cq + 1) * 128 + (long long)va0];
                blk_10[2] = h0[seq_base + (long long)(64 + cq) * 128 + (long long)(va0 + 8)];
                blk_10[3] = h0[seq_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_10[4] = h0[seq_base + (long long)(72 + cq) * 128 + (long long)va0];
                blk_10[5] = h0[seq_base + (long long)(72 + cq + 1) * 128 + (long long)va0];
                blk_10[6] = h0[seq_base + (long long)(72 + cq) * 128 + (long long)(va0 + 8)];
                blk_10[7] = h0[seq_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_10[8] = h0[seq_base + (long long)(80 + cq) * 128 + (long long)va0];
                blk_10[9] = h0[seq_base + (long long)(80 + cq + 1) * 128 + (long long)va0];
                blk_10[10] = h0[seq_base + (long long)(80 + cq) * 128 + (long long)(va0 + 8)];
                blk_10[11] = h0[seq_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_10[12] = h0[seq_base + (long long)(88 + cq) * 128 + (long long)va0];
                blk_10[13] = h0[seq_base + (long long)(88 + cq + 1) * 128 + (long long)va0];
                blk_10[14] = h0[seq_base + (long long)(88 + cq) * 128 + (long long)(va0 + 8)];
                blk_10[15] = h0[seq_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 64), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_10[15])));
                float blk_11[16];
                blk_11[0] = h0[seq_base + (long long)(96 + cq) * 128 + (long long)va0];
                blk_11[1] = h0[seq_base + (long long)(96 + cq + 1) * 128 + (long long)va0];
                blk_11[2] = h0[seq_base + (long long)(96 + cq) * 128 + (long long)(va0 + 8)];
                blk_11[3] = h0[seq_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_11[4] = h0[seq_base + (long long)(104 + cq) * 128 + (long long)va0];
                blk_11[5] = h0[seq_base + (long long)(104 + cq + 1) * 128 + (long long)va0];
                blk_11[6] = h0[seq_base + (long long)(104 + cq) * 128 + (long long)(va0 + 8)];
                blk_11[7] = h0[seq_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_11[8] = h0[seq_base + (long long)(112 + cq) * 128 + (long long)va0];
                blk_11[9] = h0[seq_base + (long long)(112 + cq + 1) * 128 + (long long)va0];
                blk_11[10] = h0[seq_base + (long long)(112 + cq) * 128 + (long long)(va0 + 8)];
                blk_11[11] = h0[seq_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 8)];
                blk_11[12] = h0[seq_base + (long long)(120 + cq) * 128 + (long long)va0];
                blk_11[13] = h0[seq_base + (long long)(120 + cq + 1) * 128 + (long long)va0];
                blk_11[14] = h0[seq_base + (long long)(120 + cq) * 128 + (long long)(va0 + 8)];
                blk_11[15] = h0[seq_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 96), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_11[15])));
                float blk_12[16];
                blk_12[0] = h0[seq_base + (long long)cq * 128 + (long long)(va0 + 64)];
                blk_12[1] = h0[seq_base + (long long)(cq + 1) * 128 + (long long)(va0 + 64)];
                blk_12[2] = h0[seq_base + (long long)cq * 128 + (long long)(va0 + 64 + 8)];
                blk_12[3] = h0[seq_base + (long long)(cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_12[4] = h0[seq_base + (long long)(8 + cq) * 128 + (long long)(va0 + 64)];
                blk_12[5] = h0[seq_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_12[6] = h0[seq_base + (long long)(8 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_12[7] = h0[seq_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_12[8] = h0[seq_base + (long long)(16 + cq) * 128 + (long long)(va0 + 64)];
                blk_12[9] = h0[seq_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_12[10] = h0[seq_base + (long long)(16 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_12[11] = h0[seq_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_12[12] = h0[seq_base + (long long)(24 + cq) * 128 + (long long)(va0 + 64)];
                blk_12[13] = h0[seq_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_12[14] = h0[seq_base + (long long)(24 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_12[15] = h0[seq_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_12[15])));
                float blk_13[16];
                blk_13[0] = h0[seq_base + (long long)(32 + cq) * 128 + (long long)(va0 + 64)];
                blk_13[1] = h0[seq_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_13[2] = h0[seq_base + (long long)(32 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[3] = h0[seq_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[4] = h0[seq_base + (long long)(40 + cq) * 128 + (long long)(va0 + 64)];
                blk_13[5] = h0[seq_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_13[6] = h0[seq_base + (long long)(40 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[7] = h0[seq_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[8] = h0[seq_base + (long long)(48 + cq) * 128 + (long long)(va0 + 64)];
                blk_13[9] = h0[seq_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_13[10] = h0[seq_base + (long long)(48 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[11] = h0[seq_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[12] = h0[seq_base + (long long)(56 + cq) * 128 + (long long)(va0 + 64)];
                blk_13[13] = h0[seq_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_13[14] = h0[seq_base + (long long)(56 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_13[15] = h0[seq_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 32), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_13[15])));
                float blk_14[16];
                blk_14[0] = h0[seq_base + (long long)(64 + cq) * 128 + (long long)(va0 + 64)];
                blk_14[1] = h0[seq_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_14[2] = h0[seq_base + (long long)(64 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[3] = h0[seq_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[4] = h0[seq_base + (long long)(72 + cq) * 128 + (long long)(va0 + 64)];
                blk_14[5] = h0[seq_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_14[6] = h0[seq_base + (long long)(72 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[7] = h0[seq_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[8] = h0[seq_base + (long long)(80 + cq) * 128 + (long long)(va0 + 64)];
                blk_14[9] = h0[seq_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_14[10] = h0[seq_base + (long long)(80 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[11] = h0[seq_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[12] = h0[seq_base + (long long)(88 + cq) * 128 + (long long)(va0 + 64)];
                blk_14[13] = h0[seq_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_14[14] = h0[seq_base + (long long)(88 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_14[15] = h0[seq_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 64), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_14[15])));
                float blk_15[16];
                blk_15[0] = h0[seq_base + (long long)(96 + cq) * 128 + (long long)(va0 + 64)];
                blk_15[1] = h0[seq_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_15[2] = h0[seq_base + (long long)(96 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[3] = h0[seq_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[4] = h0[seq_base + (long long)(104 + cq) * 128 + (long long)(va0 + 64)];
                blk_15[5] = h0[seq_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_15[6] = h0[seq_base + (long long)(104 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[7] = h0[seq_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[8] = h0[seq_base + (long long)(112 + cq) * 128 + (long long)(va0 + 64)];
                blk_15[9] = h0[seq_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_15[10] = h0[seq_base + (long long)(112 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[11] = h0[seq_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[12] = h0[seq_base + (long long)(120 + cq) * 128 + (long long)(va0 + 64)];
                blk_15[13] = h0[seq_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 64)];
                blk_15[14] = h0[seq_base + (long long)(120 + cq) * 128 + (long long)(va0 + 64 + 8)];
                blk_15[15] = h0[seq_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 64 + 8)];
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 96), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_15[15])));
            } else {
                float blk_16[16];
                blk_16[0] = 0.0f;
                blk_16[1] = 0.0f;
                blk_16[2] = 0.0f;
                blk_16[3] = 0.0f;
                blk_16[4] = 0.0f;
                blk_16[5] = 0.0f;
                blk_16[6] = 0.0f;
                blk_16[7] = 0.0f;
                blk_16[8] = 0.0f;
                blk_16[9] = 0.0f;
                blk_16[10] = 0.0f;
                blk_16[11] = 0.0f;
                blk_16[12] = 0.0f;
                blk_16[13] = 0.0f;
                blk_16[14] = 0.0f;
                blk_16[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_16[15])));
                float blk_17[16];
                blk_17[0] = 0.0f;
                blk_17[1] = 0.0f;
                blk_17[2] = 0.0f;
                blk_17[3] = 0.0f;
                blk_17[4] = 0.0f;
                blk_17[5] = 0.0f;
                blk_17[6] = 0.0f;
                blk_17[7] = 0.0f;
                blk_17[8] = 0.0f;
                blk_17[9] = 0.0f;
                blk_17[10] = 0.0f;
                blk_17[11] = 0.0f;
                blk_17[12] = 0.0f;
                blk_17[13] = 0.0f;
                blk_17[14] = 0.0f;
                blk_17[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 32), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_17[15])));
                float blk_18[16];
                blk_18[0] = 0.0f;
                blk_18[1] = 0.0f;
                blk_18[2] = 0.0f;
                blk_18[3] = 0.0f;
                blk_18[4] = 0.0f;
                blk_18[5] = 0.0f;
                blk_18[6] = 0.0f;
                blk_18[7] = 0.0f;
                blk_18[8] = 0.0f;
                blk_18[9] = 0.0f;
                blk_18[10] = 0.0f;
                blk_18[11] = 0.0f;
                blk_18[12] = 0.0f;
                blk_18[13] = 0.0f;
                blk_18[14] = 0.0f;
                blk_18[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 64), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_18[15])));
                float blk_19[16];
                blk_19[0] = 0.0f;
                blk_19[1] = 0.0f;
                blk_19[2] = 0.0f;
                blk_19[3] = 0.0f;
                blk_19[4] = 0.0f;
                blk_19[5] = 0.0f;
                blk_19[6] = 0.0f;
                blk_19[7] = 0.0f;
                blk_19[8] = 0.0f;
                blk_19[9] = 0.0f;
                blk_19[10] = 0.0f;
                blk_19[11] = 0.0f;
                blk_19[12] = 0.0f;
                blk_19[13] = 0.0f;
                blk_19[14] = 0.0f;
                blk_19[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 96), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_19[15])));
                float blk_20[16];
                blk_20[0] = 0.0f;
                blk_20[1] = 0.0f;
                blk_20[2] = 0.0f;
                blk_20[3] = 0.0f;
                blk_20[4] = 0.0f;
                blk_20[5] = 0.0f;
                blk_20[6] = 0.0f;
                blk_20[7] = 0.0f;
                blk_20[8] = 0.0f;
                blk_20[9] = 0.0f;
                blk_20[10] = 0.0f;
                blk_20[11] = 0.0f;
                blk_20[12] = 0.0f;
                blk_20[13] = 0.0f;
                blk_20[14] = 0.0f;
                blk_20[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_20[15])));
                float blk_21[16];
                blk_21[0] = 0.0f;
                blk_21[1] = 0.0f;
                blk_21[2] = 0.0f;
                blk_21[3] = 0.0f;
                blk_21[4] = 0.0f;
                blk_21[5] = 0.0f;
                blk_21[6] = 0.0f;
                blk_21[7] = 0.0f;
                blk_21[8] = 0.0f;
                blk_21[9] = 0.0f;
                blk_21[10] = 0.0f;
                blk_21[11] = 0.0f;
                blk_21[12] = 0.0f;
                blk_21[13] = 0.0f;
                blk_21[14] = 0.0f;
                blk_21[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 32), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_21[15])));
                float blk_22[16];
                blk_22[0] = 0.0f;
                blk_22[1] = 0.0f;
                blk_22[2] = 0.0f;
                blk_22[3] = 0.0f;
                blk_22[4] = 0.0f;
                blk_22[5] = 0.0f;
                blk_22[6] = 0.0f;
                blk_22[7] = 0.0f;
                blk_22[8] = 0.0f;
                blk_22[9] = 0.0f;
                blk_22[10] = 0.0f;
                blk_22[11] = 0.0f;
                blk_22[12] = 0.0f;
                blk_22[13] = 0.0f;
                blk_22[14] = 0.0f;
                blk_22[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 64), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_22[15])));
                float blk_23[16];
                blk_23[0] = 0.0f;
                blk_23[1] = 0.0f;
                blk_23[2] = 0.0f;
                blk_23[3] = 0.0f;
                blk_23[4] = 0.0f;
                blk_23[5] = 0.0f;
                blk_23[6] = 0.0f;
                blk_23[7] = 0.0f;
                blk_23[8] = 0.0f;
                blk_23[9] = 0.0f;
                blk_23[10] = 0.0f;
                blk_23[11] = 0.0f;
                blk_23[12] = 0.0f;
                blk_23[13] = 0.0f;
                blk_23[14] = 0.0f;
                blk_23[15] = 0.0f;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 96), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[1])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[2])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[3])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[4])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[5])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[6])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[7])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[8])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[9])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[10])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[11])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[12])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[13])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[14])), "r"(*reinterpret_cast<const uint32_t*>(&blk_23[15])));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            float sa0[16];
            float sa1[16];
            float sa2[16];
            float sa3[16];
            float sb0[16];
            float sb1[16];
            float sb2[16];
            float sb3[16];
            unsigned int gs_c = 0;
            unsigned int _phase_hs_free_0 = 1;
            unsigned int _phase_gates_ready = 0;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int c = 0; c < nc; c++) {
                mbarrier_wait(hs_free_addr, _phase_hs_free_0);
                _phase_hs_free_0 ^= 1;
                mbarrier_wait(gates_ready_addr + (gs_c) * 8, _phase_gates_ready);
                float e2gl = s_gt[c % 2 * 256 + 192];
                mbarrier_arrive(gates_free_addr + (gs_c) * 8);
                gs_c += 1;
                if (gs_c == 2) { gs_c = 0; _phase_gates_ready ^= 1; }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sa0[0])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[1])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[2])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[3])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[4])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[5])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[6])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[7])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[8])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[9])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[10])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[11])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[12])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[13])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[14])), "=r"(*reinterpret_cast<uint32_t*>(&sa0[15]))
                    : "r"(t_lo));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sa1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[1])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[2])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[3])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[5])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[6])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[7])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[8])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[9])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[10])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[11])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[12])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[13])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[14])), "=r"(*reinterpret_cast<uint32_t*>(&sa1[15]))
                    : "r"(t_lo + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sa2[0])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[1])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[2])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[3])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[4])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[5])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[6])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[7])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[8])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[9])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[10])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[11])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[12])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[13])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[14])), "=r"(*reinterpret_cast<uint32_t*>(&sa2[15]))
                    : "r"(t_hi));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sa3[0])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[1])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[2])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[3])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[4])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[5])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[6])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[7])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[8])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[9])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[10])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[11])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[12])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[13])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[14])), "=r"(*reinterpret_cast<uint32_t*>(&sa3[15]))
                    : "r"(t_hi + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int pk_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa0 + 0)[_lp*2 + 0], (sa0 + 0)[_lp*2+1 + 0]));
                    pk_0[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)(st_col / 64 * 8192 + (st_row * 128 + st_col % 64 * 2 ^ (st_row * 128 + st_col % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_0[3]))
                    : "memory");
                unsigned int pk_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa0 + 8)[_lp*2 + 0], (sa0 + 8)[_lp*2+1 + 0]));
                    pk_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((16 + st_col) / 64 * 8192 + (st_row * 128 + (16 + st_col) % 64 * 2 ^ (st_row * 128 + (16 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_1[3]))
                    : "memory");
                unsigned int pk_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa1 + 0)[_lp*2 + 0], (sa1 + 0)[_lp*2+1 + 0]));
                    pk_2[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((32 + st_col) / 64 * 8192 + (st_row * 128 + (32 + st_col) % 64 * 2 ^ (st_row * 128 + (32 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_2[3]))
                    : "memory");
                unsigned int pk_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa1 + 8)[_lp*2 + 0], (sa1 + 8)[_lp*2+1 + 0]));
                    pk_3[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((48 + st_col) / 64 * 8192 + (st_row * 128 + (48 + st_col) % 64 * 2 ^ (st_row * 128 + (48 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_3[3]))
                    : "memory");
                unsigned int pk_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa2 + 0)[_lp*2 + 0], (sa2 + 0)[_lp*2+1 + 0]));
                    pk_4[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)(st_col / 64 * 8192 + (st_row * 128 + st_col % 64 * 2 ^ (st_row * 128 + st_col % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_4[3]))
                    : "memory");
                unsigned int pk_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa2 + 8)[_lp*2 + 0], (sa2 + 8)[_lp*2+1 + 0]));
                    pk_5[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((16 + st_col) / 64 * 8192 + (st_row * 128 + (16 + st_col) % 64 * 2 ^ (st_row * 128 + (16 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_5[3]))
                    : "memory");
                unsigned int pk_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa3 + 0)[_lp*2 + 0], (sa3 + 0)[_lp*2+1 + 0]));
                    pk_6[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((32 + st_col) / 64 * 8192 + (st_row * 128 + (32 + st_col) % 64 * 2 ^ (st_row * 128 + (32 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_6[3]))
                    : "memory");
                unsigned int pk_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sa3 + 8)[_lp*2 + 0], (sa3 + 8)[_lp*2+1 + 0]));
                    pk_7[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((48 + st_col) / 64 * 8192 + (st_row * 128 + (48 + st_col) % 64 * 2 ^ (st_row * 128 + (48 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_7[3]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(hs_k0_addr);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sb0[0])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[1])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[2])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[3])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[4])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[5])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[6])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[7])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[8])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[9])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[10])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[11])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[12])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[13])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[14])), "=r"(*reinterpret_cast<uint32_t*>(&sb0[15]))
                    : "r"(t_lo + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sb1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[1])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[2])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[3])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[5])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[6])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[7])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[8])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[9])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[10])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[11])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[12])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[13])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[14])), "=r"(*reinterpret_cast<uint32_t*>(&sb1[15]))
                    : "r"(t_lo + 96));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sb2[0])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[1])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[2])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[3])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[4])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[5])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[6])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[7])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[8])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[9])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[10])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[11])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[12])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[13])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[14])), "=r"(*reinterpret_cast<uint32_t*>(&sb2[15]))
                    : "r"(t_hi + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sb3[0])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[1])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[2])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[3])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[4])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[5])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[6])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[7])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[8])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[9])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[10])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[11])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[12])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[13])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[14])), "=r"(*reinterpret_cast<uint32_t*>(&sb3[15]))
                    : "r"(t_hi + 96));
                const float2 _scale2_9 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sa0)[_ls], _scale2_9);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo), "r"(*reinterpret_cast<const uint32_t*>(&sa0[0])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[1])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[2])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[3])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[4])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[5])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[6])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[7])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[8])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[9])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[10])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[11])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[12])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[13])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[14])), "r"(*reinterpret_cast<const uint32_t*>(&sa0[15])));
                const float2 _scale2_10 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sa1)[_ls], _scale2_10);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 32), "r"(*reinterpret_cast<const uint32_t*>(&sa1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sa1[15])));
                const float2 _scale2_11 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sa2)[_ls], _scale2_11);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi), "r"(*reinterpret_cast<const uint32_t*>(&sa2[0])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[1])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[2])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[3])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[4])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[5])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[6])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[7])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[8])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[9])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[10])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[11])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[12])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[13])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[14])), "r"(*reinterpret_cast<const uint32_t*>(&sa2[15])));
                const float2 _scale2_12 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sa3)[_ls], _scale2_12);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 32), "r"(*reinterpret_cast<const uint32_t*>(&sa3[0])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[1])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[2])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[3])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[4])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[5])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[6])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[7])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[8])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[9])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[10])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[11])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[12])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[13])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[14])), "r"(*reinterpret_cast<const uint32_t*>(&sa3[15])));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int pk_8[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb0 + 0)[_lp*2 + 0], (sb0 + 0)[_lp*2+1 + 0]));
                    pk_8[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((64 + st_col) / 64 * 8192 + (st_row * 128 + (64 + st_col) % 64 * 2 ^ (st_row * 128 + (64 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&pk_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_8[3]))
                    : "memory");
                unsigned int pk_9[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb0 + 8)[_lp*2 + 0], (sb0 + 8)[_lp*2+1 + 0]));
                    pk_9[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((80 + st_col) / 64 * 8192 + (st_row * 128 + (80 + st_col) % 64 * 2 ^ (st_row * 128 + (80 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&pk_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_9[3]))
                    : "memory");
                unsigned int pk_10[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb1 + 0)[_lp*2 + 0], (sb1 + 0)[_lp*2+1 + 0]));
                    pk_10[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((96 + st_col) / 64 * 8192 + (st_row * 128 + (96 + st_col) % 64 * 2 ^ (st_row * 128 + (96 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&pk_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_10[3]))
                    : "memory");
                unsigned int pk_11[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb1 + 8)[_lp*2 + 0], (sb1 + 8)[_lp*2+1 + 0]));
                    pk_11[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(s_hs0_addr + (unsigned int)((112 + st_col) / 64 * 8192 + (st_row * 128 + (112 + st_col) % 64 * 2 ^ (st_row * 128 + (112 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&pk_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_11[3]))
                    : "memory");
                unsigned int pk_12[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb2 + 0)[_lp*2 + 0], (sb2 + 0)[_lp*2+1 + 0]));
                    pk_12[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((64 + st_col) / 64 * 8192 + (st_row * 128 + (64 + st_col) % 64 * 2 ^ (st_row * 128 + (64 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&pk_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_12[3]))
                    : "memory");
                unsigned int pk_13[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb2 + 8)[_lp*2 + 0], (sb2 + 8)[_lp*2+1 + 0]));
                    pk_13[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((80 + st_col) / 64 * 8192 + (st_row * 128 + (80 + st_col) % 64 * 2 ^ (st_row * 128 + (80 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&pk_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_13[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_13[3]))
                    : "memory");
                unsigned int pk_14[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb3 + 0)[_lp*2 + 0], (sb3 + 0)[_lp*2+1 + 0]));
                    pk_14[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((96 + st_col) / 64 * 8192 + (st_row * 128 + (96 + st_col) % 64 * 2 ^ (st_row * 128 + (96 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&pk_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_14[3]))
                    : "memory");
                unsigned int pk_15[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((sb3 + 8)[_lp*2 + 0], (sb3 + 8)[_lp*2+1 + 0]));
                    pk_15[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(s_hs1_addr + (unsigned int)((112 + st_col) / 64 * 8192 + (st_row * 128 + (112 + st_col) % 64 * 2 ^ (st_row * 128 + (112 + st_col) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&pk_15[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_15[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_15[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_15[3]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(hs_k1_addr);
                const float2 _scale2_21 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sb0)[_ls], _scale2_21);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 64), "r"(*reinterpret_cast<const uint32_t*>(&sb0[0])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[1])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[2])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[3])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[4])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[5])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[6])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[7])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[8])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[9])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[10])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[11])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[12])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[13])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[14])), "r"(*reinterpret_cast<const uint32_t*>(&sb0[15])));
                const float2 _scale2_22 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sb1)[_ls], _scale2_22);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_lo + 96), "r"(*reinterpret_cast<const uint32_t*>(&sb1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sb1[15])));
                const float2 _scale2_23 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sb2)[_ls], _scale2_23);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 64), "r"(*reinterpret_cast<const uint32_t*>(&sb2[0])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[1])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[2])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[3])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[4])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[5])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[6])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[7])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[8])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[9])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[10])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[11])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[12])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[13])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[14])), "r"(*reinterpret_cast<const uint32_t*>(&sb2[15])));
                const float2 _scale2_24 = {e2gl, e2gl};
                #pragma unroll
                for (int _ls = 0; _ls < 8; _ls++)
                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(sb3)[_ls], _scale2_24);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_hi + 96), "r"(*reinterpret_cast<const uint32_t*>(&sb3[0])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[1])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[2])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[3])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[4])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[5])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[6])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[7])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[8])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[9])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[10])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[11])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[12])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[13])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[14])), "r"(*reinterpret_cast<const uint32_t*>(&sb3[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_scaled_addr);
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            if (store_final_state != 0) {
                if (is_last != 0) {
                    float blk_24[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_24[15]))
                        : "r"(t_lo));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)cq * 128 + (long long)va0] = blk_24[0];
                    final_state[seq_base + (long long)(cq + 1) * 128 + (long long)va0] = blk_24[1];
                    final_state[seq_base + (long long)cq * 128 + (long long)(va0 + 8)] = blk_24[2];
                    final_state[seq_base + (long long)(cq + 1) * 128 + (long long)(va0 + 8)] = blk_24[3];
                    final_state[seq_base + (long long)(8 + cq) * 128 + (long long)va0] = blk_24[4];
                    final_state[seq_base + (long long)(8 + cq + 1) * 128 + (long long)va0] = blk_24[5];
                    final_state[seq_base + (long long)(8 + cq) * 128 + (long long)(va0 + 8)] = blk_24[6];
                    final_state[seq_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_24[7];
                    final_state[seq_base + (long long)(16 + cq) * 128 + (long long)va0] = blk_24[8];
                    final_state[seq_base + (long long)(16 + cq + 1) * 128 + (long long)va0] = blk_24[9];
                    final_state[seq_base + (long long)(16 + cq) * 128 + (long long)(va0 + 8)] = blk_24[10];
                    final_state[seq_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_24[11];
                    final_state[seq_base + (long long)(24 + cq) * 128 + (long long)va0] = blk_24[12];
                    final_state[seq_base + (long long)(24 + cq + 1) * 128 + (long long)va0] = blk_24[13];
                    final_state[seq_base + (long long)(24 + cq) * 128 + (long long)(va0 + 8)] = blk_24[14];
                    final_state[seq_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_24[15];
                    float blk_25[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_25[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_25[15]))
                        : "r"(t_lo + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)(32 + cq) * 128 + (long long)va0] = blk_25[0];
                    final_state[seq_base + (long long)(32 + cq + 1) * 128 + (long long)va0] = blk_25[1];
                    final_state[seq_base + (long long)(32 + cq) * 128 + (long long)(va0 + 8)] = blk_25[2];
                    final_state[seq_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_25[3];
                    final_state[seq_base + (long long)(40 + cq) * 128 + (long long)va0] = blk_25[4];
                    final_state[seq_base + (long long)(40 + cq + 1) * 128 + (long long)va0] = blk_25[5];
                    final_state[seq_base + (long long)(40 + cq) * 128 + (long long)(va0 + 8)] = blk_25[6];
                    final_state[seq_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_25[7];
                    final_state[seq_base + (long long)(48 + cq) * 128 + (long long)va0] = blk_25[8];
                    final_state[seq_base + (long long)(48 + cq + 1) * 128 + (long long)va0] = blk_25[9];
                    final_state[seq_base + (long long)(48 + cq) * 128 + (long long)(va0 + 8)] = blk_25[10];
                    final_state[seq_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_25[11];
                    final_state[seq_base + (long long)(56 + cq) * 128 + (long long)va0] = blk_25[12];
                    final_state[seq_base + (long long)(56 + cq + 1) * 128 + (long long)va0] = blk_25[13];
                    final_state[seq_base + (long long)(56 + cq) * 128 + (long long)(va0 + 8)] = blk_25[14];
                    final_state[seq_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_25[15];
                    float blk_26[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_26[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_26[15]))
                        : "r"(t_lo + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)(64 + cq) * 128 + (long long)va0] = blk_26[0];
                    final_state[seq_base + (long long)(64 + cq + 1) * 128 + (long long)va0] = blk_26[1];
                    final_state[seq_base + (long long)(64 + cq) * 128 + (long long)(va0 + 8)] = blk_26[2];
                    final_state[seq_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_26[3];
                    final_state[seq_base + (long long)(72 + cq) * 128 + (long long)va0] = blk_26[4];
                    final_state[seq_base + (long long)(72 + cq + 1) * 128 + (long long)va0] = blk_26[5];
                    final_state[seq_base + (long long)(72 + cq) * 128 + (long long)(va0 + 8)] = blk_26[6];
                    final_state[seq_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_26[7];
                    final_state[seq_base + (long long)(80 + cq) * 128 + (long long)va0] = blk_26[8];
                    final_state[seq_base + (long long)(80 + cq + 1) * 128 + (long long)va0] = blk_26[9];
                    final_state[seq_base + (long long)(80 + cq) * 128 + (long long)(va0 + 8)] = blk_26[10];
                    final_state[seq_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_26[11];
                    final_state[seq_base + (long long)(88 + cq) * 128 + (long long)va0] = blk_26[12];
                    final_state[seq_base + (long long)(88 + cq + 1) * 128 + (long long)va0] = blk_26[13];
                    final_state[seq_base + (long long)(88 + cq) * 128 + (long long)(va0 + 8)] = blk_26[14];
                    final_state[seq_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_26[15];
                    float blk_27[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_27[15]))
                        : "r"(t_lo + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)(96 + cq) * 128 + (long long)va0] = blk_27[0];
                    final_state[seq_base + (long long)(96 + cq + 1) * 128 + (long long)va0] = blk_27[1];
                    final_state[seq_base + (long long)(96 + cq) * 128 + (long long)(va0 + 8)] = blk_27[2];
                    final_state[seq_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_27[3];
                    final_state[seq_base + (long long)(104 + cq) * 128 + (long long)va0] = blk_27[4];
                    final_state[seq_base + (long long)(104 + cq + 1) * 128 + (long long)va0] = blk_27[5];
                    final_state[seq_base + (long long)(104 + cq) * 128 + (long long)(va0 + 8)] = blk_27[6];
                    final_state[seq_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_27[7];
                    final_state[seq_base + (long long)(112 + cq) * 128 + (long long)va0] = blk_27[8];
                    final_state[seq_base + (long long)(112 + cq + 1) * 128 + (long long)va0] = blk_27[9];
                    final_state[seq_base + (long long)(112 + cq) * 128 + (long long)(va0 + 8)] = blk_27[10];
                    final_state[seq_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_27[11];
                    final_state[seq_base + (long long)(120 + cq) * 128 + (long long)va0] = blk_27[12];
                    final_state[seq_base + (long long)(120 + cq + 1) * 128 + (long long)va0] = blk_27[13];
                    final_state[seq_base + (long long)(120 + cq) * 128 + (long long)(va0 + 8)] = blk_27[14];
                    final_state[seq_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 8)] = blk_27[15];
                    float blk_28[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_28[15]))
                        : "r"(t_hi));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)cq * 128 + (long long)(va0 + 64)] = blk_28[0];
                    final_state[seq_base + (long long)(cq + 1) * 128 + (long long)(va0 + 64)] = blk_28[1];
                    final_state[seq_base + (long long)cq * 128 + (long long)(va0 + 64 + 8)] = blk_28[2];
                    final_state[seq_base + (long long)(cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_28[3];
                    final_state[seq_base + (long long)(8 + cq) * 128 + (long long)(va0 + 64)] = blk_28[4];
                    final_state[seq_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_28[5];
                    final_state[seq_base + (long long)(8 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_28[6];
                    final_state[seq_base + (long long)(8 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_28[7];
                    final_state[seq_base + (long long)(16 + cq) * 128 + (long long)(va0 + 64)] = blk_28[8];
                    final_state[seq_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_28[9];
                    final_state[seq_base + (long long)(16 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_28[10];
                    final_state[seq_base + (long long)(16 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_28[11];
                    final_state[seq_base + (long long)(24 + cq) * 128 + (long long)(va0 + 64)] = blk_28[12];
                    final_state[seq_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_28[13];
                    final_state[seq_base + (long long)(24 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_28[14];
                    final_state[seq_base + (long long)(24 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_28[15];
                    float blk_29[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_29[15]))
                        : "r"(t_hi + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)(32 + cq) * 128 + (long long)(va0 + 64)] = blk_29[0];
                    final_state[seq_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_29[1];
                    final_state[seq_base + (long long)(32 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_29[2];
                    final_state[seq_base + (long long)(32 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_29[3];
                    final_state[seq_base + (long long)(40 + cq) * 128 + (long long)(va0 + 64)] = blk_29[4];
                    final_state[seq_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_29[5];
                    final_state[seq_base + (long long)(40 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_29[6];
                    final_state[seq_base + (long long)(40 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_29[7];
                    final_state[seq_base + (long long)(48 + cq) * 128 + (long long)(va0 + 64)] = blk_29[8];
                    final_state[seq_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_29[9];
                    final_state[seq_base + (long long)(48 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_29[10];
                    final_state[seq_base + (long long)(48 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_29[11];
                    final_state[seq_base + (long long)(56 + cq) * 128 + (long long)(va0 + 64)] = blk_29[12];
                    final_state[seq_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_29[13];
                    final_state[seq_base + (long long)(56 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_29[14];
                    final_state[seq_base + (long long)(56 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_29[15];
                    float blk_30[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_30[15]))
                        : "r"(t_hi + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)(64 + cq) * 128 + (long long)(va0 + 64)] = blk_30[0];
                    final_state[seq_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_30[1];
                    final_state[seq_base + (long long)(64 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_30[2];
                    final_state[seq_base + (long long)(64 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_30[3];
                    final_state[seq_base + (long long)(72 + cq) * 128 + (long long)(va0 + 64)] = blk_30[4];
                    final_state[seq_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_30[5];
                    final_state[seq_base + (long long)(72 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_30[6];
                    final_state[seq_base + (long long)(72 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_30[7];
                    final_state[seq_base + (long long)(80 + cq) * 128 + (long long)(va0 + 64)] = blk_30[8];
                    final_state[seq_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_30[9];
                    final_state[seq_base + (long long)(80 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_30[10];
                    final_state[seq_base + (long long)(80 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_30[11];
                    final_state[seq_base + (long long)(88 + cq) * 128 + (long long)(va0 + 64)] = blk_30[12];
                    final_state[seq_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_30[13];
                    final_state[seq_base + (long long)(88 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_30[14];
                    final_state[seq_base + (long long)(88 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_30[15];
                    float blk_31[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&blk_31[0])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[1])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[2])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[3])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[4])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[5])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[6])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[7])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[8])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[9])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[10])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[11])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[12])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[13])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[14])), "=r"(*reinterpret_cast<uint32_t*>(&blk_31[15]))
                        : "r"(t_hi + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    final_state[seq_base + (long long)(96 + cq) * 128 + (long long)(va0 + 64)] = blk_31[0];
                    final_state[seq_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_31[1];
                    final_state[seq_base + (long long)(96 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_31[2];
                    final_state[seq_base + (long long)(96 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_31[3];
                    final_state[seq_base + (long long)(104 + cq) * 128 + (long long)(va0 + 64)] = blk_31[4];
                    final_state[seq_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_31[5];
                    final_state[seq_base + (long long)(104 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_31[6];
                    final_state[seq_base + (long long)(104 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_31[7];
                    final_state[seq_base + (long long)(112 + cq) * 128 + (long long)(va0 + 64)] = blk_31[8];
                    final_state[seq_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_31[9];
                    final_state[seq_base + (long long)(112 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_31[10];
                    final_state[seq_base + (long long)(112 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_31[11];
                    final_state[seq_base + (long long)(120 + cq) * 128 + (long long)(va0 + 64)] = blk_31[12];
                    final_state[seq_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 64)] = blk_31[13];
                    final_state[seq_base + (long long)(120 + cq) * 128 + (long long)(va0 + 64 + 8)] = blk_31[14];
                    final_state[seq_base + (long long)(120 + cq + 1) * 128 + (long long)(va0 + 64 + 8)] = blk_31[15];
                }
            }
        }
    }
    // ---- Role: value ----
    if (warp >= 4 && warp <= 7) {
        { // value_main
            int blk_32 = blockIdx.x;
            int seg_1 = walk[blk_32 / num_v_heads];
            int hv_1 = blk_32 % num_v_heads;
            int c_begin_1 = seg_chunk_start[seg_1];
            int nc_1 = seg_chunk_start[seg_1 + 1] - c_begin_1;
            int seq_1 = seg_seq[seg_1];
            int lane_2 = lane;
            int p_1 = warp % 4;
            int cq_1 = lane_2 % 4 * 2;
            int st_row_1 = p_1 * 16 + lane_2 / 8 % 2 * 8 + lane_2 % 8;
            int st_col_1 = lane_2 / 16 * 8;
            int m_idx = lane_2 / 8;
            int tid_v = tid - 128;
            int t_lo_1 = tmem_all + (p_1 * 32 << 16);
            int t_hi_1 = t_lo_1 + 1048576;
            unsigned int tf_c = 0;
            unsigned int gv_c = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_gates_ready_1 = 0;
            unsigned int _phase_vn_free_0 = 1;
            unsigned int _phase_ws_full_0 = 0;
            #pragma unroll 1
            for (int c_1 = 0; c_1 < nc_1; c_1++) {
                int chunk = c_begin_1 + c_1;
                int n = chunk_len[chunk];
                int g_off = c_1 % 2 * 256;
                unsigned int u_stage = s_u_addr + tf_c * 65536;
                unsigned int kn_stage = s_kn_addr + tf_c * 65536;
                mbarrier_wait(tiles_full_addr + (tf_c) * 8, _phase_tiles_full);
                mbarrier_wait(gates_ready_addr + (gv_c) * 8, _phase_gates_ready_1);
                float dks[16];
                dks[0] = s_gt[g_off + 64 + cq_1];
                dks[1] = s_gt[g_off + 64 + 1 + cq_1];
                dks[2] = s_gt[g_off + 64 + 8 + cq_1];
                dks[3] = s_gt[g_off + 64 + 8 + 1 + cq_1];
                dks[4] = s_gt[g_off + 64 + 16 + cq_1];
                dks[5] = s_gt[g_off + 64 + 16 + 1 + cq_1];
                dks[6] = s_gt[g_off + 64 + 24 + cq_1];
                dks[7] = s_gt[g_off + 64 + 24 + 1 + cq_1];
                dks[8] = s_gt[g_off + 64 + 32 + cq_1];
                dks[9] = s_gt[g_off + 64 + 32 + 1 + cq_1];
                dks[10] = s_gt[g_off + 64 + 40 + cq_1];
                dks[11] = s_gt[g_off + 64 + 40 + 1 + cq_1];
                dks[12] = s_gt[g_off + 64 + 48 + cq_1];
                dks[13] = s_gt[g_off + 64 + 48 + 1 + cq_1];
                dks[14] = s_gt[g_off + 64 + 56 + cq_1];
                dks[15] = s_gt[g_off + 64 + 56 + 1 + cq_1];
                mbarrier_arrive(gates_free_addr + (gv_c) * 8);
                gv_c += 1;
                if (gv_c == 2) { gv_c = 0; _phase_gates_ready_1 ^= 1; }
                mbarrier_wait(vn_free_addr, _phase_vn_free_0);
                _phase_vn_free_0 ^= 1;
                mbarrier_wait(ws_full_addr, _phase_ws_full_0);
                _phase_ws_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float wa_0[16];
                float wb_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wa_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&wa_0[15]))
                    : "r"(t_lo_1 + 128));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wb_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&wb_0[15]))
                    : "r"(t_lo_1 + 160));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int uf_0[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_0[0]), "=r"(uf_0[1]), "=r"(uf_0[2]), "=r"(uf_0[3])
                    : "r"(u_stage + (unsigned int)((p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_0[8];
                float pv_0[8];
                vv_0[0] = __uint_as_float(uf_0[0] << 16) - wa_0[0];
                vv_0[1] = __uint_as_float(uf_0[0] & 4294901760u) - wa_0[1];
                vv_0[2] = __uint_as_float(uf_0[1] << 16) - wa_0[2];
                vv_0[3] = __uint_as_float(uf_0[1] & 4294901760u) - wa_0[3];
                pv_0[0] = ((cq_1 < n) ? vv_0[0] * dks[0] : 0.0f);
                pv_0[2] = ((cq_1 < n) ? vv_0[2] * dks[0] : 0.0f);
                vv_0[0] = ((cq_1 < n) ? vv_0[0] : 0.0f);
                vv_0[2] = ((cq_1 < n) ? vv_0[2] : 0.0f);
                pv_0[1] = ((1 + cq_1 < n) ? vv_0[1] * dks[1] : 0.0f);
                pv_0[3] = ((1 + cq_1 < n) ? vv_0[3] * dks[1] : 0.0f);
                vv_0[1] = ((1 + cq_1 < n) ? vv_0[1] : 0.0f);
                vv_0[3] = ((1 + cq_1 < n) ? vv_0[3] : 0.0f);
                vv_0[4] = __uint_as_float(uf_0[2] << 16) - wa_0[4];
                vv_0[5] = __uint_as_float(uf_0[2] & 4294901760u) - wa_0[5];
                vv_0[6] = __uint_as_float(uf_0[3] << 16) - wa_0[6];
                vv_0[7] = __uint_as_float(uf_0[3] & 4294901760u) - wa_0[7];
                pv_0[4] = ((8 + cq_1 < n) ? vv_0[4] * dks[2] : 0.0f);
                pv_0[6] = ((8 + cq_1 < n) ? vv_0[6] * dks[2] : 0.0f);
                vv_0[4] = ((8 + cq_1 < n) ? vv_0[4] : 0.0f);
                vv_0[6] = ((8 + cq_1 < n) ? vv_0[6] : 0.0f);
                pv_0[5] = ((9 + cq_1 < n) ? vv_0[5] * dks[3] : 0.0f);
                pv_0[7] = ((9 + cq_1 < n) ? vv_0[7] * dks[3] : 0.0f);
                vv_0[5] = ((9 + cq_1 < n) ? vv_0[5] : 0.0f);
                vv_0[7] = ((9 + cq_1 < n) ? vv_0[7] : 0.0f);
                unsigned int pkv_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_0[_lp*2 + 0], pv_0[_lp*2+1 + 0]));
                    pkv_0[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(s_vp0_addr + (unsigned int)(st_col_1 / 64 * 8192 + (st_row_1 * 128 + st_col_1 % 64 * 2 ^ (st_row_1 * 128 + st_col_1 % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&pkv_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_0[3]))
                    : "memory");
                unsigned int pkn_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_0[_lp*2 + 0], vv_0[_lp*2+1 + 0]));
                    pkn_0[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + ((m_idx / 2 * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ ((m_idx / 2 * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&pkn_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_0[3]))
                    : "memory");
                unsigned int uf_1[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_1[0]), "=r"(uf_1[1]), "=r"(uf_1[2]), "=r"(uf_1[3])
                    : "r"(u_stage + (unsigned int)((p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((16 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((16 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_1[8];
                float pv_1[8];
                vv_1[0] = __uint_as_float(uf_1[0] << 16) - wa_0[8];
                vv_1[1] = __uint_as_float(uf_1[0] & 4294901760u) - wa_0[9];
                vv_1[2] = __uint_as_float(uf_1[1] << 16) - wa_0[10];
                vv_1[3] = __uint_as_float(uf_1[1] & 4294901760u) - wa_0[11];
                pv_1[0] = ((16 + cq_1 < n) ? vv_1[0] * dks[4] : 0.0f);
                pv_1[2] = ((16 + cq_1 < n) ? vv_1[2] * dks[4] : 0.0f);
                vv_1[0] = ((16 + cq_1 < n) ? vv_1[0] : 0.0f);
                vv_1[2] = ((16 + cq_1 < n) ? vv_1[2] : 0.0f);
                pv_1[1] = ((17 + cq_1 < n) ? vv_1[1] * dks[5] : 0.0f);
                pv_1[3] = ((17 + cq_1 < n) ? vv_1[3] * dks[5] : 0.0f);
                vv_1[1] = ((17 + cq_1 < n) ? vv_1[1] : 0.0f);
                vv_1[3] = ((17 + cq_1 < n) ? vv_1[3] : 0.0f);
                vv_1[4] = __uint_as_float(uf_1[2] << 16) - wa_0[12];
                vv_1[5] = __uint_as_float(uf_1[2] & 4294901760u) - wa_0[13];
                vv_1[6] = __uint_as_float(uf_1[3] << 16) - wa_0[14];
                vv_1[7] = __uint_as_float(uf_1[3] & 4294901760u) - wa_0[15];
                pv_1[4] = ((24 + cq_1 < n) ? vv_1[4] * dks[6] : 0.0f);
                pv_1[6] = ((24 + cq_1 < n) ? vv_1[6] * dks[6] : 0.0f);
                vv_1[4] = ((24 + cq_1 < n) ? vv_1[4] : 0.0f);
                vv_1[6] = ((24 + cq_1 < n) ? vv_1[6] : 0.0f);
                pv_1[5] = ((25 + cq_1 < n) ? vv_1[5] * dks[7] : 0.0f);
                pv_1[7] = ((25 + cq_1 < n) ? vv_1[7] * dks[7] : 0.0f);
                vv_1[5] = ((25 + cq_1 < n) ? vv_1[5] : 0.0f);
                vv_1[7] = ((25 + cq_1 < n) ? vv_1[7] : 0.0f);
                unsigned int pkv_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_1[_lp*2 + 0], pv_1[_lp*2+1 + 0]));
                    pkv_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_vp0_addr + (unsigned int)((16 + st_col_1) / 64 * 8192 + (st_row_1 * 128 + (16 + st_col_1) % 64 * 2 ^ (st_row_1 * 128 + (16 + st_col_1) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&pkv_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_1[3]))
                    : "memory");
                unsigned int pkn_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_1[_lp*2 + 0], vv_1[_lp*2+1 + 0]));
                    pkn_1[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + (((2 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ (((2 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&pkn_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_1[3]))
                    : "memory");
                unsigned int uf_2[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_2[0]), "=r"(uf_2[1]), "=r"(uf_2[2]), "=r"(uf_2[3])
                    : "r"(u_stage + (unsigned int)((p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((32 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((32 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_2[8];
                float pv_2[8];
                vv_2[0] = __uint_as_float(uf_2[0] << 16) - wb_0[0];
                vv_2[1] = __uint_as_float(uf_2[0] & 4294901760u) - wb_0[1];
                vv_2[2] = __uint_as_float(uf_2[1] << 16) - wb_0[2];
                vv_2[3] = __uint_as_float(uf_2[1] & 4294901760u) - wb_0[3];
                pv_2[0] = ((32 + cq_1 < n) ? vv_2[0] * dks[8] : 0.0f);
                pv_2[2] = ((32 + cq_1 < n) ? vv_2[2] * dks[8] : 0.0f);
                vv_2[0] = ((32 + cq_1 < n) ? vv_2[0] : 0.0f);
                vv_2[2] = ((32 + cq_1 < n) ? vv_2[2] : 0.0f);
                pv_2[1] = ((33 + cq_1 < n) ? vv_2[1] * dks[9] : 0.0f);
                pv_2[3] = ((33 + cq_1 < n) ? vv_2[3] * dks[9] : 0.0f);
                vv_2[1] = ((33 + cq_1 < n) ? vv_2[1] : 0.0f);
                vv_2[3] = ((33 + cq_1 < n) ? vv_2[3] : 0.0f);
                vv_2[4] = __uint_as_float(uf_2[2] << 16) - wb_0[4];
                vv_2[5] = __uint_as_float(uf_2[2] & 4294901760u) - wb_0[5];
                vv_2[6] = __uint_as_float(uf_2[3] << 16) - wb_0[6];
                vv_2[7] = __uint_as_float(uf_2[3] & 4294901760u) - wb_0[7];
                pv_2[4] = ((40 + cq_1 < n) ? vv_2[4] * dks[10] : 0.0f);
                pv_2[6] = ((40 + cq_1 < n) ? vv_2[6] * dks[10] : 0.0f);
                vv_2[4] = ((40 + cq_1 < n) ? vv_2[4] : 0.0f);
                vv_2[6] = ((40 + cq_1 < n) ? vv_2[6] : 0.0f);
                pv_2[5] = ((41 + cq_1 < n) ? vv_2[5] * dks[11] : 0.0f);
                pv_2[7] = ((41 + cq_1 < n) ? vv_2[7] * dks[11] : 0.0f);
                vv_2[5] = ((41 + cq_1 < n) ? vv_2[5] : 0.0f);
                vv_2[7] = ((41 + cq_1 < n) ? vv_2[7] : 0.0f);
                unsigned int pkv_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_2[_lp*2 + 0], pv_2[_lp*2+1 + 0]));
                    pkv_2[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(s_vp0_addr + (unsigned int)((32 + st_col_1) / 64 * 8192 + (st_row_1 * 128 + (32 + st_col_1) % 64 * 2 ^ (st_row_1 * 128 + (32 + st_col_1) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&pkv_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_2[3]))
                    : "memory");
                unsigned int pkn_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_2[_lp*2 + 0], vv_2[_lp*2+1 + 0]));
                    pkn_2[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + (((4 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ (((4 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&pkn_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_2[3]))
                    : "memory");
                unsigned int uf_3[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_3[0]), "=r"(uf_3[1]), "=r"(uf_3[2]), "=r"(uf_3[3])
                    : "r"(u_stage + (unsigned int)((p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((48 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((48 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_3[8];
                float pv_3[8];
                vv_3[0] = __uint_as_float(uf_3[0] << 16) - wb_0[8];
                vv_3[1] = __uint_as_float(uf_3[0] & 4294901760u) - wb_0[9];
                vv_3[2] = __uint_as_float(uf_3[1] << 16) - wb_0[10];
                vv_3[3] = __uint_as_float(uf_3[1] & 4294901760u) - wb_0[11];
                pv_3[0] = ((48 + cq_1 < n) ? vv_3[0] * dks[12] : 0.0f);
                pv_3[2] = ((48 + cq_1 < n) ? vv_3[2] * dks[12] : 0.0f);
                vv_3[0] = ((48 + cq_1 < n) ? vv_3[0] : 0.0f);
                vv_3[2] = ((48 + cq_1 < n) ? vv_3[2] : 0.0f);
                pv_3[1] = ((49 + cq_1 < n) ? vv_3[1] * dks[13] : 0.0f);
                pv_3[3] = ((49 + cq_1 < n) ? vv_3[3] * dks[13] : 0.0f);
                vv_3[1] = ((49 + cq_1 < n) ? vv_3[1] : 0.0f);
                vv_3[3] = ((49 + cq_1 < n) ? vv_3[3] : 0.0f);
                vv_3[4] = __uint_as_float(uf_3[2] << 16) - wb_0[12];
                vv_3[5] = __uint_as_float(uf_3[2] & 4294901760u) - wb_0[13];
                vv_3[6] = __uint_as_float(uf_3[3] << 16) - wb_0[14];
                vv_3[7] = __uint_as_float(uf_3[3] & 4294901760u) - wb_0[15];
                pv_3[4] = ((56 + cq_1 < n) ? vv_3[4] * dks[14] : 0.0f);
                pv_3[6] = ((56 + cq_1 < n) ? vv_3[6] * dks[14] : 0.0f);
                vv_3[4] = ((56 + cq_1 < n) ? vv_3[4] : 0.0f);
                vv_3[6] = ((56 + cq_1 < n) ? vv_3[6] : 0.0f);
                pv_3[5] = ((57 + cq_1 < n) ? vv_3[5] * dks[15] : 0.0f);
                pv_3[7] = ((57 + cq_1 < n) ? vv_3[7] * dks[15] : 0.0f);
                vv_3[5] = ((57 + cq_1 < n) ? vv_3[5] : 0.0f);
                vv_3[7] = ((57 + cq_1 < n) ? vv_3[7] : 0.0f);
                unsigned int pkv_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_3[_lp*2 + 0], pv_3[_lp*2+1 + 0]));
                    pkv_3[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(s_vp0_addr + (unsigned int)((48 + st_col_1) / 64 * 8192 + (st_row_1 * 128 + (48 + st_col_1) % 64 * 2 ^ (st_row_1 * 128 + (48 + st_col_1) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&pkv_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_3[3]))
                    : "memory");
                unsigned int pkn_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_3[_lp*2 + 0], vv_3[_lp*2+1 + 0]));
                    pkn_3[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + (((6 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ (((6 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&pkn_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_3[3]))
                    : "memory");
                float wa_1[16];
                float wb_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wa_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&wa_1[15]))
                    : "r"(t_hi_1 + 128));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&wb_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&wb_1[15]))
                    : "r"(t_hi_1 + 160));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int uf_4[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_4[0]), "=r"(uf_4[1]), "=r"(uf_4[2]), "=r"(uf_4[3])
                    : "r"(u_stage + (unsigned int)((64 + p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_4[8];
                float pv_4[8];
                vv_4[0] = __uint_as_float(uf_4[0] << 16) - wa_1[0];
                vv_4[1] = __uint_as_float(uf_4[0] & 4294901760u) - wa_1[1];
                vv_4[2] = __uint_as_float(uf_4[1] << 16) - wa_1[2];
                vv_4[3] = __uint_as_float(uf_4[1] & 4294901760u) - wa_1[3];
                pv_4[0] = ((cq_1 < n) ? vv_4[0] * dks[0] : 0.0f);
                pv_4[2] = ((cq_1 < n) ? vv_4[2] * dks[0] : 0.0f);
                vv_4[0] = ((cq_1 < n) ? vv_4[0] : 0.0f);
                vv_4[2] = ((cq_1 < n) ? vv_4[2] : 0.0f);
                pv_4[1] = ((1 + cq_1 < n) ? vv_4[1] * dks[1] : 0.0f);
                pv_4[3] = ((1 + cq_1 < n) ? vv_4[3] * dks[1] : 0.0f);
                vv_4[1] = ((1 + cq_1 < n) ? vv_4[1] : 0.0f);
                vv_4[3] = ((1 + cq_1 < n) ? vv_4[3] : 0.0f);
                vv_4[4] = __uint_as_float(uf_4[2] << 16) - wa_1[4];
                vv_4[5] = __uint_as_float(uf_4[2] & 4294901760u) - wa_1[5];
                vv_4[6] = __uint_as_float(uf_4[3] << 16) - wa_1[6];
                vv_4[7] = __uint_as_float(uf_4[3] & 4294901760u) - wa_1[7];
                pv_4[4] = ((8 + cq_1 < n) ? vv_4[4] * dks[2] : 0.0f);
                pv_4[6] = ((8 + cq_1 < n) ? vv_4[6] * dks[2] : 0.0f);
                vv_4[4] = ((8 + cq_1 < n) ? vv_4[4] : 0.0f);
                vv_4[6] = ((8 + cq_1 < n) ? vv_4[6] : 0.0f);
                pv_4[5] = ((9 + cq_1 < n) ? vv_4[5] * dks[3] : 0.0f);
                pv_4[7] = ((9 + cq_1 < n) ? vv_4[7] * dks[3] : 0.0f);
                vv_4[5] = ((9 + cq_1 < n) ? vv_4[5] : 0.0f);
                vv_4[7] = ((9 + cq_1 < n) ? vv_4[7] : 0.0f);
                unsigned int pkv_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_4[_lp*2 + 0], pv_4[_lp*2+1 + 0]));
                    pkv_4[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(s_vp1_addr + (unsigned int)(st_col_1 / 64 * 8192 + (st_row_1 * 128 + st_col_1 % 64 * 2 ^ (st_row_1 * 128 + st_col_1 % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&pkv_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_4[3]))
                    : "memory");
                unsigned int pkn_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_4[_lp*2 + 0], vv_4[_lp*2+1 + 0]));
                    pkn_4[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((64 + p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + ((m_idx / 2 * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ ((m_idx / 2 * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&pkn_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_4[3]))
                    : "memory");
                unsigned int uf_5[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_5[0]), "=r"(uf_5[1]), "=r"(uf_5[2]), "=r"(uf_5[3])
                    : "r"(u_stage + (unsigned int)((64 + p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((16 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((16 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_5[8];
                float pv_5[8];
                vv_5[0] = __uint_as_float(uf_5[0] << 16) - wa_1[8];
                vv_5[1] = __uint_as_float(uf_5[0] & 4294901760u) - wa_1[9];
                vv_5[2] = __uint_as_float(uf_5[1] << 16) - wa_1[10];
                vv_5[3] = __uint_as_float(uf_5[1] & 4294901760u) - wa_1[11];
                pv_5[0] = ((16 + cq_1 < n) ? vv_5[0] * dks[4] : 0.0f);
                pv_5[2] = ((16 + cq_1 < n) ? vv_5[2] * dks[4] : 0.0f);
                vv_5[0] = ((16 + cq_1 < n) ? vv_5[0] : 0.0f);
                vv_5[2] = ((16 + cq_1 < n) ? vv_5[2] : 0.0f);
                pv_5[1] = ((17 + cq_1 < n) ? vv_5[1] * dks[5] : 0.0f);
                pv_5[3] = ((17 + cq_1 < n) ? vv_5[3] * dks[5] : 0.0f);
                vv_5[1] = ((17 + cq_1 < n) ? vv_5[1] : 0.0f);
                vv_5[3] = ((17 + cq_1 < n) ? vv_5[3] : 0.0f);
                vv_5[4] = __uint_as_float(uf_5[2] << 16) - wa_1[12];
                vv_5[5] = __uint_as_float(uf_5[2] & 4294901760u) - wa_1[13];
                vv_5[6] = __uint_as_float(uf_5[3] << 16) - wa_1[14];
                vv_5[7] = __uint_as_float(uf_5[3] & 4294901760u) - wa_1[15];
                pv_5[4] = ((24 + cq_1 < n) ? vv_5[4] * dks[6] : 0.0f);
                pv_5[6] = ((24 + cq_1 < n) ? vv_5[6] * dks[6] : 0.0f);
                vv_5[4] = ((24 + cq_1 < n) ? vv_5[4] : 0.0f);
                vv_5[6] = ((24 + cq_1 < n) ? vv_5[6] : 0.0f);
                pv_5[5] = ((25 + cq_1 < n) ? vv_5[5] * dks[7] : 0.0f);
                pv_5[7] = ((25 + cq_1 < n) ? vv_5[7] * dks[7] : 0.0f);
                vv_5[5] = ((25 + cq_1 < n) ? vv_5[5] : 0.0f);
                vv_5[7] = ((25 + cq_1 < n) ? vv_5[7] : 0.0f);
                unsigned int pkv_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_5[_lp*2 + 0], pv_5[_lp*2+1 + 0]));
                    pkv_5[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(s_vp1_addr + (unsigned int)((16 + st_col_1) / 64 * 8192 + (st_row_1 * 128 + (16 + st_col_1) % 64 * 2 ^ (st_row_1 * 128 + (16 + st_col_1) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&pkv_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_5[3]))
                    : "memory");
                unsigned int pkn_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_5[_lp*2 + 0], vv_5[_lp*2+1 + 0]));
                    pkn_5[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((64 + p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + (((2 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ (((2 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&pkn_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_5[3]))
                    : "memory");
                unsigned int uf_6[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_6[0]), "=r"(uf_6[1]), "=r"(uf_6[2]), "=r"(uf_6[3])
                    : "r"(u_stage + (unsigned int)((64 + p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((32 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((32 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_6[8];
                float pv_6[8];
                vv_6[0] = __uint_as_float(uf_6[0] << 16) - wb_1[0];
                vv_6[1] = __uint_as_float(uf_6[0] & 4294901760u) - wb_1[1];
                vv_6[2] = __uint_as_float(uf_6[1] << 16) - wb_1[2];
                vv_6[3] = __uint_as_float(uf_6[1] & 4294901760u) - wb_1[3];
                pv_6[0] = ((32 + cq_1 < n) ? vv_6[0] * dks[8] : 0.0f);
                pv_6[2] = ((32 + cq_1 < n) ? vv_6[2] * dks[8] : 0.0f);
                vv_6[0] = ((32 + cq_1 < n) ? vv_6[0] : 0.0f);
                vv_6[2] = ((32 + cq_1 < n) ? vv_6[2] : 0.0f);
                pv_6[1] = ((33 + cq_1 < n) ? vv_6[1] * dks[9] : 0.0f);
                pv_6[3] = ((33 + cq_1 < n) ? vv_6[3] * dks[9] : 0.0f);
                vv_6[1] = ((33 + cq_1 < n) ? vv_6[1] : 0.0f);
                vv_6[3] = ((33 + cq_1 < n) ? vv_6[3] : 0.0f);
                vv_6[4] = __uint_as_float(uf_6[2] << 16) - wb_1[4];
                vv_6[5] = __uint_as_float(uf_6[2] & 4294901760u) - wb_1[5];
                vv_6[6] = __uint_as_float(uf_6[3] << 16) - wb_1[6];
                vv_6[7] = __uint_as_float(uf_6[3] & 4294901760u) - wb_1[7];
                pv_6[4] = ((40 + cq_1 < n) ? vv_6[4] * dks[10] : 0.0f);
                pv_6[6] = ((40 + cq_1 < n) ? vv_6[6] * dks[10] : 0.0f);
                vv_6[4] = ((40 + cq_1 < n) ? vv_6[4] : 0.0f);
                vv_6[6] = ((40 + cq_1 < n) ? vv_6[6] : 0.0f);
                pv_6[5] = ((41 + cq_1 < n) ? vv_6[5] * dks[11] : 0.0f);
                pv_6[7] = ((41 + cq_1 < n) ? vv_6[7] * dks[11] : 0.0f);
                vv_6[5] = ((41 + cq_1 < n) ? vv_6[5] : 0.0f);
                vv_6[7] = ((41 + cq_1 < n) ? vv_6[7] : 0.0f);
                unsigned int pkv_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_6[_lp*2 + 0], pv_6[_lp*2+1 + 0]));
                    pkv_6[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(s_vp1_addr + (unsigned int)((32 + st_col_1) / 64 * 8192 + (st_row_1 * 128 + (32 + st_col_1) % 64 * 2 ^ (st_row_1 * 128 + (32 + st_col_1) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&pkv_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_6[3]))
                    : "memory");
                unsigned int pkn_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_6[_lp*2 + 0], vv_6[_lp*2+1 + 0]));
                    pkn_6[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((64 + p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + (((4 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ (((4 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&pkn_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_6[3]))
                    : "memory");
                unsigned int uf_7[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(uf_7[0]), "=r"(uf_7[1]), "=r"(uf_7[2]), "=r"(uf_7[3])
                    : "r"(u_stage + (unsigned int)((64 + p_1 * 16 + lane_2 % 16 / 8 * 8) / 64 * 8192 + ((48 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 ^ ((48 + 8 * (lane_2 / 16) + lane_2 % 8) * 128 + (64 + p_1 * 16 + lane_2 % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float vv_7[8];
                float pv_7[8];
                vv_7[0] = __uint_as_float(uf_7[0] << 16) - wb_1[8];
                vv_7[1] = __uint_as_float(uf_7[0] & 4294901760u) - wb_1[9];
                vv_7[2] = __uint_as_float(uf_7[1] << 16) - wb_1[10];
                vv_7[3] = __uint_as_float(uf_7[1] & 4294901760u) - wb_1[11];
                pv_7[0] = ((48 + cq_1 < n) ? vv_7[0] * dks[12] : 0.0f);
                pv_7[2] = ((48 + cq_1 < n) ? vv_7[2] * dks[12] : 0.0f);
                vv_7[0] = ((48 + cq_1 < n) ? vv_7[0] : 0.0f);
                vv_7[2] = ((48 + cq_1 < n) ? vv_7[2] : 0.0f);
                pv_7[1] = ((49 + cq_1 < n) ? vv_7[1] * dks[13] : 0.0f);
                pv_7[3] = ((49 + cq_1 < n) ? vv_7[3] * dks[13] : 0.0f);
                vv_7[1] = ((49 + cq_1 < n) ? vv_7[1] : 0.0f);
                vv_7[3] = ((49 + cq_1 < n) ? vv_7[3] : 0.0f);
                vv_7[4] = __uint_as_float(uf_7[2] << 16) - wb_1[12];
                vv_7[5] = __uint_as_float(uf_7[2] & 4294901760u) - wb_1[13];
                vv_7[6] = __uint_as_float(uf_7[3] << 16) - wb_1[14];
                vv_7[7] = __uint_as_float(uf_7[3] & 4294901760u) - wb_1[15];
                pv_7[4] = ((56 + cq_1 < n) ? vv_7[4] * dks[14] : 0.0f);
                pv_7[6] = ((56 + cq_1 < n) ? vv_7[6] * dks[14] : 0.0f);
                vv_7[4] = ((56 + cq_1 < n) ? vv_7[4] : 0.0f);
                vv_7[6] = ((56 + cq_1 < n) ? vv_7[6] : 0.0f);
                pv_7[5] = ((57 + cq_1 < n) ? vv_7[5] * dks[15] : 0.0f);
                pv_7[7] = ((57 + cq_1 < n) ? vv_7[7] * dks[15] : 0.0f);
                vv_7[5] = ((57 + cq_1 < n) ? vv_7[5] : 0.0f);
                vv_7[7] = ((57 + cq_1 < n) ? vv_7[7] : 0.0f);
                unsigned int pkv_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pv_7[_lp*2 + 0], pv_7[_lp*2+1 + 0]));
                    pkv_7[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(s_vp1_addr + (unsigned int)((48 + st_col_1) / 64 * 8192 + (st_row_1 * 128 + (48 + st_col_1) % 64 * 2 ^ (st_row_1 * 128 + (48 + st_col_1) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&pkv_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkv_7[3]))
                    : "memory");
                unsigned int pkn_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vv_7[_lp*2 + 0], vv_7[_lp*2+1 + 0]));
                    pkn_7[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(s_vn_addr + (unsigned int)((64 + p_1 * 16 + m_idx % 2 * 8) / 64 * 8192 + (((6 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 ^ (((6 + m_idx / 2) * 8 + lane_2 % 8) * 128 + (64 + p_1 * 16 + m_idx % 2 * 8) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&pkn_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&pkn_7[3]))
                    : "memory");
                mbarrier_arrive(u_free_addr + (tf_c) * 8);
                tf_c += 1;
                if (tf_c == 2) { tf_c = 0; _phase_tiles_full ^= 1; }
                if (n < 64) {
                    #pragma unroll
                    for (int it = 0; it < 8; it++) {
                        int zrow = (it * 128 + tid_v) % 512 / 8;
                        if (zrow >= n) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(kn_stage + (unsigned int)((it * 128 + tid_v) / 512 * 8192 + zrow * 128 + (it * 128 + tid_v) % 8 * 16)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(vprime_ready_addr);
                if (n < 64) {
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int tok0 = chunk_start[chunk];
                    #pragma unroll
                    for (int it2 = 0; it2 < 8; it2++) {
                        int vrow = (it2 * 128 + tid_v) / 16;
                        int vch = (it2 * 128 + tid_v) % 16;
                        if (vrow < n) {
                            unsigned int words[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                                : "r"(s_vn_addr + (unsigned int)(vch / 8 * 8192 + (vrow * 128 + vch % 8 * 16 ^ (vrow & 7) << 4))));
                            reinterpret_cast<int4*>(v_new + (((long long)(tok0 + vrow) * (long long)num_v_heads + (long long)hv_1) * 128 + (long long)(vch * 8)))[0] = reinterpret_cast<int4*>(words)[0];
                        }
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
            }
        }
    }
    // ---- Role: outr ----
    if (warp >= 8 && warp <= 11) {
        { // outr_main
            int blk_33 = blockIdx.x;
            int seg_2 = walk[blk_33 / num_v_heads];
            int hv_2 = blk_33 % num_v_heads;
            int c_begin_2 = seg_chunk_start[seg_2];
            int nc_2 = seg_chunk_start[seg_2 + 1] - c_begin_2;
            int seq_2 = seg_seq[seg_2];
            int lane_3 = lane;
            int p_2 = warp % 4;
            int cq_2 = lane_3 % 4 * 2;
            int row_a = p_2 * 16 + lane_3 / 4;
            int row_b = row_a + 8;
            int st_row_2 = p_2 * 16 + lane_3 / 8 % 2 * 8 + lane_3 % 8;
            int st_col_2 = lane_3 / 16 * 8;
            int t_lo_2 = tmem_all + (p_2 * 32 << 16);
            float pa[16];
            float pb[16];
            float oa[16];
            float ob[16];
            unsigned int go_c = 0;
            unsigned int pf_c = 0;
            unsigned int o1_c = 0;
            unsigned int of_c = 0;
            unsigned int _phase_gates_ready_2 = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_o1_full = 0;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (int c_2 = 0; c_2 < nc_2; c_2++) {
                int chunk_1 = c_begin_2 + c_2;
                int n_1 = chunk_len[chunk_1];
                int tok0_1 = chunk_start[chunk_1];
                int g_off_1 = c_2 % 2 * 256;
                int t_sl = t_lo_2 + c_2 % 2 * 1048576;
                mbarrier_wait(gates_ready_addr + (go_c) * 8, _phase_gates_ready_2);
                float g_a = s_gt[g_off_1 + row_a];
                float g_b = s_gt[g_off_1 + row_b];
                float gain_a = s_gt[g_off_1 + 128 + row_a];
                float gain_b = s_gt[g_off_1 + 128 + row_b];
                mbarrier_wait(p_full_addr + (pf_c) * 8, _phase_p_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pa[0])), "=r"(*reinterpret_cast<uint32_t*>(&pa[1])), "=r"(*reinterpret_cast<uint32_t*>(&pa[2])), "=r"(*reinterpret_cast<uint32_t*>(&pa[3])), "=r"(*reinterpret_cast<uint32_t*>(&pa[4])), "=r"(*reinterpret_cast<uint32_t*>(&pa[5])), "=r"(*reinterpret_cast<uint32_t*>(&pa[6])), "=r"(*reinterpret_cast<uint32_t*>(&pa[7])), "=r"(*reinterpret_cast<uint32_t*>(&pa[8])), "=r"(*reinterpret_cast<uint32_t*>(&pa[9])), "=r"(*reinterpret_cast<uint32_t*>(&pa[10])), "=r"(*reinterpret_cast<uint32_t*>(&pa[11])), "=r"(*reinterpret_cast<uint32_t*>(&pa[12])), "=r"(*reinterpret_cast<uint32_t*>(&pa[13])), "=r"(*reinterpret_cast<uint32_t*>(&pa[14])), "=r"(*reinterpret_cast<uint32_t*>(&pa[15]))
                    : "r"(t_sl + 192));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&pb[0])), "=r"(*reinterpret_cast<uint32_t*>(&pb[1])), "=r"(*reinterpret_cast<uint32_t*>(&pb[2])), "=r"(*reinterpret_cast<uint32_t*>(&pb[3])), "=r"(*reinterpret_cast<uint32_t*>(&pb[4])), "=r"(*reinterpret_cast<uint32_t*>(&pb[5])), "=r"(*reinterpret_cast<uint32_t*>(&pb[6])), "=r"(*reinterpret_cast<uint32_t*>(&pb[7])), "=r"(*reinterpret_cast<uint32_t*>(&pb[8])), "=r"(*reinterpret_cast<uint32_t*>(&pb[9])), "=r"(*reinterpret_cast<uint32_t*>(&pb[10])), "=r"(*reinterpret_cast<uint32_t*>(&pb[11])), "=r"(*reinterpret_cast<uint32_t*>(&pb[12])), "=r"(*reinterpret_cast<uint32_t*>(&pb[13])), "=r"(*reinterpret_cast<uint32_t*>(&pb[14])), "=r"(*reinterpret_cast<uint32_t*>(&pb[15]))
                    : "r"(t_sl + 224));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_free_addr + (pf_c) * 8);
                pf_c += 1;
                if (pf_c == 2) { pf_c = 0; _phase_p_full ^= 1; }
                int col_0 = cq_2;
                float g_col_0 = s_gt[g_off_1 + col_0];
                float val_a_0 = 0.0f;
                float val_b_0 = 0.0f;
                if (col_0 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_0 = approx_exp2(g_a - g_col_0);
                        val_a_0 = pa[0] * _exp2_0 * scale;
                    }
                }
                if (col_0 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_1 = approx_exp2(g_b - g_col_0);
                        val_b_0 = pa[2] * _exp2_1 * scale;
                    }
                }
                pa[0] = val_a_0;
                pa[2] = val_b_0;
                int col_1 = cq_2 + 1;
                float g_col_1 = s_gt[g_off_1 + col_1];
                float val_a_1 = 0.0f;
                float val_b_1 = 0.0f;
                if (col_1 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_2 = approx_exp2(g_a - g_col_1);
                        val_a_1 = pa[1] * _exp2_2 * scale;
                    }
                }
                if (col_1 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_3 = approx_exp2(g_b - g_col_1);
                        val_b_1 = pa[3] * _exp2_3 * scale;
                    }
                }
                pa[1] = val_a_1;
                pa[3] = val_b_1;
                int col_2 = 8 + cq_2;
                float g_col_2 = s_gt[g_off_1 + col_2];
                float val_a_2 = 0.0f;
                float val_b_2 = 0.0f;
                if (col_2 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_4 = approx_exp2(g_a - g_col_2);
                        val_a_2 = pa[4] * _exp2_4 * scale;
                    }
                }
                if (col_2 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_5 = approx_exp2(g_b - g_col_2);
                        val_b_2 = pa[6] * _exp2_5 * scale;
                    }
                }
                pa[4] = val_a_2;
                pa[6] = val_b_2;
                int col_3 = 8 + cq_2 + 1;
                float g_col_3 = s_gt[g_off_1 + col_3];
                float val_a_3 = 0.0f;
                float val_b_3 = 0.0f;
                if (col_3 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_6 = approx_exp2(g_a - g_col_3);
                        val_a_3 = pa[5] * _exp2_6 * scale;
                    }
                }
                if (col_3 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_7 = approx_exp2(g_b - g_col_3);
                        val_b_3 = pa[7] * _exp2_7 * scale;
                    }
                }
                pa[5] = val_a_3;
                pa[7] = val_b_3;
                int col_4 = 16 + cq_2;
                float g_col_4 = s_gt[g_off_1 + col_4];
                float val_a_4 = 0.0f;
                float val_b_4 = 0.0f;
                if (col_4 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_8 = approx_exp2(g_a - g_col_4);
                        val_a_4 = pa[8] * _exp2_8 * scale;
                    }
                }
                if (col_4 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_9 = approx_exp2(g_b - g_col_4);
                        val_b_4 = pa[10] * _exp2_9 * scale;
                    }
                }
                pa[8] = val_a_4;
                pa[10] = val_b_4;
                int col_5 = 16 + cq_2 + 1;
                float g_col_5 = s_gt[g_off_1 + col_5];
                float val_a_5 = 0.0f;
                float val_b_5 = 0.0f;
                if (col_5 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_10 = approx_exp2(g_a - g_col_5);
                        val_a_5 = pa[9] * _exp2_10 * scale;
                    }
                }
                if (col_5 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_11 = approx_exp2(g_b - g_col_5);
                        val_b_5 = pa[11] * _exp2_11 * scale;
                    }
                }
                pa[9] = val_a_5;
                pa[11] = val_b_5;
                int col_6 = 24 + cq_2;
                float g_col_6 = s_gt[g_off_1 + col_6];
                float val_a_6 = 0.0f;
                float val_b_6 = 0.0f;
                if (col_6 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_12 = approx_exp2(g_a - g_col_6);
                        val_a_6 = pa[12] * _exp2_12 * scale;
                    }
                }
                if (col_6 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_13 = approx_exp2(g_b - g_col_6);
                        val_b_6 = pa[14] * _exp2_13 * scale;
                    }
                }
                pa[12] = val_a_6;
                pa[14] = val_b_6;
                int col_7 = 24 + cq_2 + 1;
                float g_col_7 = s_gt[g_off_1 + col_7];
                float val_a_7 = 0.0f;
                float val_b_7 = 0.0f;
                if (col_7 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_14 = approx_exp2(g_a - g_col_7);
                        val_a_7 = pa[13] * _exp2_14 * scale;
                    }
                }
                if (col_7 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_15 = approx_exp2(g_b - g_col_7);
                        val_b_7 = pa[15] * _exp2_15 * scale;
                    }
                }
                pa[13] = val_a_7;
                pa[15] = val_b_7;
                int col_8 = 32 + cq_2;
                float g_col_8 = s_gt[g_off_1 + col_8];
                float val_a_8 = 0.0f;
                float val_b_8 = 0.0f;
                if (col_8 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_16 = approx_exp2(g_a - g_col_8);
                        val_a_8 = pb[0] * _exp2_16 * scale;
                    }
                }
                if (col_8 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_17 = approx_exp2(g_b - g_col_8);
                        val_b_8 = pb[2] * _exp2_17 * scale;
                    }
                }
                pb[0] = val_a_8;
                pb[2] = val_b_8;
                int col_9 = 32 + cq_2 + 1;
                float g_col_9 = s_gt[g_off_1 + col_9];
                float val_a_9 = 0.0f;
                float val_b_9 = 0.0f;
                if (col_9 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_18 = approx_exp2(g_a - g_col_9);
                        val_a_9 = pb[1] * _exp2_18 * scale;
                    }
                }
                if (col_9 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_19 = approx_exp2(g_b - g_col_9);
                        val_b_9 = pb[3] * _exp2_19 * scale;
                    }
                }
                pb[1] = val_a_9;
                pb[3] = val_b_9;
                int col_10 = 40 + cq_2;
                float g_col_10 = s_gt[g_off_1 + col_10];
                float val_a_10 = 0.0f;
                float val_b_10 = 0.0f;
                if (col_10 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_20 = approx_exp2(g_a - g_col_10);
                        val_a_10 = pb[4] * _exp2_20 * scale;
                    }
                }
                if (col_10 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_21 = approx_exp2(g_b - g_col_10);
                        val_b_10 = pb[6] * _exp2_21 * scale;
                    }
                }
                pb[4] = val_a_10;
                pb[6] = val_b_10;
                int col_11 = 40 + cq_2 + 1;
                float g_col_11 = s_gt[g_off_1 + col_11];
                float val_a_11 = 0.0f;
                float val_b_11 = 0.0f;
                if (col_11 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_22 = approx_exp2(g_a - g_col_11);
                        val_a_11 = pb[5] * _exp2_22 * scale;
                    }
                }
                if (col_11 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_23 = approx_exp2(g_b - g_col_11);
                        val_b_11 = pb[7] * _exp2_23 * scale;
                    }
                }
                pb[5] = val_a_11;
                pb[7] = val_b_11;
                int col_12 = 48 + cq_2;
                float g_col_12 = s_gt[g_off_1 + col_12];
                float val_a_12 = 0.0f;
                float val_b_12 = 0.0f;
                if (col_12 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_24 = approx_exp2(g_a - g_col_12);
                        val_a_12 = pb[8] * _exp2_24 * scale;
                    }
                }
                if (col_12 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_25 = approx_exp2(g_b - g_col_12);
                        val_b_12 = pb[10] * _exp2_25 * scale;
                    }
                }
                pb[8] = val_a_12;
                pb[10] = val_b_12;
                int col_13 = 48 + cq_2 + 1;
                float g_col_13 = s_gt[g_off_1 + col_13];
                float val_a_13 = 0.0f;
                float val_b_13 = 0.0f;
                if (col_13 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_26 = approx_exp2(g_a - g_col_13);
                        val_a_13 = pb[9] * _exp2_26 * scale;
                    }
                }
                if (col_13 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_27 = approx_exp2(g_b - g_col_13);
                        val_b_13 = pb[11] * _exp2_27 * scale;
                    }
                }
                pb[9] = val_a_13;
                pb[11] = val_b_13;
                int col_14 = 56 + cq_2;
                float g_col_14 = s_gt[g_off_1 + col_14];
                float val_a_14 = 0.0f;
                float val_b_14 = 0.0f;
                if (col_14 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_28 = approx_exp2(g_a - g_col_14);
                        val_a_14 = pb[12] * _exp2_28 * scale;
                    }
                }
                if (col_14 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_29 = approx_exp2(g_b - g_col_14);
                        val_b_14 = pb[14] * _exp2_29 * scale;
                    }
                }
                pb[12] = val_a_14;
                pb[14] = val_b_14;
                int col_15 = 56 + cq_2 + 1;
                float g_col_15 = s_gt[g_off_1 + col_15];
                float val_a_15 = 0.0f;
                float val_b_15 = 0.0f;
                if (col_15 <= row_a) {
                    if (row_a < n_1) {
                        float _exp2_30 = approx_exp2(g_a - g_col_15);
                        val_a_15 = pb[13] * _exp2_30 * scale;
                    }
                }
                if (col_15 <= row_b) {
                    if (row_b < n_1) {
                        float _exp2_31 = approx_exp2(g_b - g_col_15);
                        val_b_15 = pb[15] * _exp2_31 * scale;
                    }
                }
                pb[13] = val_a_15;
                pb[15] = val_b_15;
                unsigned int pk_16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pa + 0)[_lp*2 + 0], (pa + 0)[_lp*2+1 + 0]));
                    pk_16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_32 = static_cast<uint32_t>(s_pg_addr + (unsigned int)(st_col_2 / 64 * 8192 + (st_row_2 * 128 + st_col_2 % 64 * 2 ^ (st_row_2 * 128 + st_col_2 % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_32), "r"(*reinterpret_cast<const uint32_t*>(&pk_16[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_16[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_16[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_16[3]))
                    : "memory");
                unsigned int pk_17[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pa + 8)[_lp*2 + 0], (pa + 8)[_lp*2+1 + 0]));
                    pk_17[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_33 = static_cast<uint32_t>(s_pg_addr + (unsigned int)((16 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (16 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (16 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_33), "r"(*reinterpret_cast<const uint32_t*>(&pk_17[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_17[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_17[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_17[3]))
                    : "memory");
                unsigned int pk_18[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pb + 0)[_lp*2 + 0], (pb + 0)[_lp*2+1 + 0]));
                    pk_18[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_34 = static_cast<uint32_t>(s_pg_addr + (unsigned int)((32 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (32 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (32 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_34), "r"(*reinterpret_cast<const uint32_t*>(&pk_18[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_18[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_18[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_18[3]))
                    : "memory");
                unsigned int pk_19[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pb + 8)[_lp*2 + 0], (pb + 8)[_lp*2+1 + 0]));
                    pk_19[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_35 = static_cast<uint32_t>(s_pg_addr + (unsigned int)((48 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (48 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (48 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_35), "r"(*reinterpret_cast<const uint32_t*>(&pk_19[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_19[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_19[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_19[3]))
                    : "memory");
                mbarrier_wait(o1_full_addr + (o1_c) * 8, _phase_o1_full);
                o1_c += 1;
                if (o1_c == 2) { o1_c = 0; _phase_o1_full ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&oa[0])), "=r"(*reinterpret_cast<uint32_t*>(&oa[1])), "=r"(*reinterpret_cast<uint32_t*>(&oa[2])), "=r"(*reinterpret_cast<uint32_t*>(&oa[3])), "=r"(*reinterpret_cast<uint32_t*>(&oa[4])), "=r"(*reinterpret_cast<uint32_t*>(&oa[5])), "=r"(*reinterpret_cast<uint32_t*>(&oa[6])), "=r"(*reinterpret_cast<uint32_t*>(&oa[7])), "=r"(*reinterpret_cast<uint32_t*>(&oa[8])), "=r"(*reinterpret_cast<uint32_t*>(&oa[9])), "=r"(*reinterpret_cast<uint32_t*>(&oa[10])), "=r"(*reinterpret_cast<uint32_t*>(&oa[11])), "=r"(*reinterpret_cast<uint32_t*>(&oa[12])), "=r"(*reinterpret_cast<uint32_t*>(&oa[13])), "=r"(*reinterpret_cast<uint32_t*>(&oa[14])), "=r"(*reinterpret_cast<uint32_t*>(&oa[15]))
                    : "r"(t_sl + 256));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ob[0])), "=r"(*reinterpret_cast<uint32_t*>(&ob[1])), "=r"(*reinterpret_cast<uint32_t*>(&ob[2])), "=r"(*reinterpret_cast<uint32_t*>(&ob[3])), "=r"(*reinterpret_cast<uint32_t*>(&ob[4])), "=r"(*reinterpret_cast<uint32_t*>(&ob[5])), "=r"(*reinterpret_cast<uint32_t*>(&ob[6])), "=r"(*reinterpret_cast<uint32_t*>(&ob[7])), "=r"(*reinterpret_cast<uint32_t*>(&ob[8])), "=r"(*reinterpret_cast<uint32_t*>(&ob[9])), "=r"(*reinterpret_cast<uint32_t*>(&ob[10])), "=r"(*reinterpret_cast<uint32_t*>(&ob[11])), "=r"(*reinterpret_cast<uint32_t*>(&ob[12])), "=r"(*reinterpret_cast<uint32_t*>(&ob[13])), "=r"(*reinterpret_cast<uint32_t*>(&ob[14])), "=r"(*reinterpret_cast<uint32_t*>(&ob[15]))
                    : "r"(t_sl + 288));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                oa[0] = oa[0] * gain_a;
                oa[1] = oa[1] * gain_a;
                oa[2] = oa[2] * gain_b;
                oa[3] = oa[3] * gain_b;
                oa[4] = oa[4] * gain_a;
                oa[5] = oa[5] * gain_a;
                oa[6] = oa[6] * gain_b;
                oa[7] = oa[7] * gain_b;
                oa[8] = oa[8] * gain_a;
                oa[9] = oa[9] * gain_a;
                oa[10] = oa[10] * gain_b;
                oa[11] = oa[11] * gain_b;
                oa[12] = oa[12] * gain_a;
                oa[13] = oa[13] * gain_a;
                oa[14] = oa[14] * gain_b;
                oa[15] = oa[15] * gain_b;
                ob[0] = ob[0] * gain_a;
                ob[1] = ob[1] * gain_a;
                ob[2] = ob[2] * gain_b;
                ob[3] = ob[3] * gain_b;
                ob[4] = ob[4] * gain_a;
                ob[5] = ob[5] * gain_a;
                ob[6] = ob[6] * gain_b;
                ob[7] = ob[7] * gain_b;
                ob[8] = ob[8] * gain_a;
                ob[9] = ob[9] * gain_a;
                ob[10] = ob[10] * gain_b;
                ob[11] = ob[11] * gain_b;
                ob[12] = ob[12] * gain_a;
                ob[13] = ob[13] * gain_a;
                ob[14] = ob[14] * gain_b;
                ob[15] = ob[15] * gain_b;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_sl + 256), "r"(*reinterpret_cast<const uint32_t*>(&oa[0])), "r"(*reinterpret_cast<const uint32_t*>(&oa[1])), "r"(*reinterpret_cast<const uint32_t*>(&oa[2])), "r"(*reinterpret_cast<const uint32_t*>(&oa[3])), "r"(*reinterpret_cast<const uint32_t*>(&oa[4])), "r"(*reinterpret_cast<const uint32_t*>(&oa[5])), "r"(*reinterpret_cast<const uint32_t*>(&oa[6])), "r"(*reinterpret_cast<const uint32_t*>(&oa[7])), "r"(*reinterpret_cast<const uint32_t*>(&oa[8])), "r"(*reinterpret_cast<const uint32_t*>(&oa[9])), "r"(*reinterpret_cast<const uint32_t*>(&oa[10])), "r"(*reinterpret_cast<const uint32_t*>(&oa[11])), "r"(*reinterpret_cast<const uint32_t*>(&oa[12])), "r"(*reinterpret_cast<const uint32_t*>(&oa[13])), "r"(*reinterpret_cast<const uint32_t*>(&oa[14])), "r"(*reinterpret_cast<const uint32_t*>(&oa[15])));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_sl + 288), "r"(*reinterpret_cast<const uint32_t*>(&ob[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob[3])), "r"(*reinterpret_cast<const uint32_t*>(&ob[4])), "r"(*reinterpret_cast<const uint32_t*>(&ob[5])), "r"(*reinterpret_cast<const uint32_t*>(&ob[6])), "r"(*reinterpret_cast<const uint32_t*>(&ob[7])), "r"(*reinterpret_cast<const uint32_t*>(&ob[8])), "r"(*reinterpret_cast<const uint32_t*>(&ob[9])), "r"(*reinterpret_cast<const uint32_t*>(&ob[10])), "r"(*reinterpret_cast<const uint32_t*>(&ob[11])), "r"(*reinterpret_cast<const uint32_t*>(&ob[12])), "r"(*reinterpret_cast<const uint32_t*>(&ob[13])), "r"(*reinterpret_cast<const uint32_t*>(&ob[14])), "r"(*reinterpret_cast<const uint32_t*>(&ob[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&oa[0])), "=r"(*reinterpret_cast<uint32_t*>(&oa[1])), "=r"(*reinterpret_cast<uint32_t*>(&oa[2])), "=r"(*reinterpret_cast<uint32_t*>(&oa[3])), "=r"(*reinterpret_cast<uint32_t*>(&oa[4])), "=r"(*reinterpret_cast<uint32_t*>(&oa[5])), "=r"(*reinterpret_cast<uint32_t*>(&oa[6])), "=r"(*reinterpret_cast<uint32_t*>(&oa[7])), "=r"(*reinterpret_cast<uint32_t*>(&oa[8])), "=r"(*reinterpret_cast<uint32_t*>(&oa[9])), "=r"(*reinterpret_cast<uint32_t*>(&oa[10])), "=r"(*reinterpret_cast<uint32_t*>(&oa[11])), "=r"(*reinterpret_cast<uint32_t*>(&oa[12])), "=r"(*reinterpret_cast<uint32_t*>(&oa[13])), "=r"(*reinterpret_cast<uint32_t*>(&oa[14])), "=r"(*reinterpret_cast<uint32_t*>(&oa[15]))
                    : "r"(t_sl + 320));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ob[0])), "=r"(*reinterpret_cast<uint32_t*>(&ob[1])), "=r"(*reinterpret_cast<uint32_t*>(&ob[2])), "=r"(*reinterpret_cast<uint32_t*>(&ob[3])), "=r"(*reinterpret_cast<uint32_t*>(&ob[4])), "=r"(*reinterpret_cast<uint32_t*>(&ob[5])), "=r"(*reinterpret_cast<uint32_t*>(&ob[6])), "=r"(*reinterpret_cast<uint32_t*>(&ob[7])), "=r"(*reinterpret_cast<uint32_t*>(&ob[8])), "=r"(*reinterpret_cast<uint32_t*>(&ob[9])), "=r"(*reinterpret_cast<uint32_t*>(&ob[10])), "=r"(*reinterpret_cast<uint32_t*>(&ob[11])), "=r"(*reinterpret_cast<uint32_t*>(&ob[12])), "=r"(*reinterpret_cast<uint32_t*>(&ob[13])), "=r"(*reinterpret_cast<uint32_t*>(&ob[14])), "=r"(*reinterpret_cast<uint32_t*>(&ob[15]))
                    : "r"(t_sl + 352));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                oa[0] = oa[0] * gain_a;
                oa[1] = oa[1] * gain_a;
                oa[2] = oa[2] * gain_b;
                oa[3] = oa[3] * gain_b;
                oa[4] = oa[4] * gain_a;
                oa[5] = oa[5] * gain_a;
                oa[6] = oa[6] * gain_b;
                oa[7] = oa[7] * gain_b;
                oa[8] = oa[8] * gain_a;
                oa[9] = oa[9] * gain_a;
                oa[10] = oa[10] * gain_b;
                oa[11] = oa[11] * gain_b;
                oa[12] = oa[12] * gain_a;
                oa[13] = oa[13] * gain_a;
                oa[14] = oa[14] * gain_b;
                oa[15] = oa[15] * gain_b;
                ob[0] = ob[0] * gain_a;
                ob[1] = ob[1] * gain_a;
                ob[2] = ob[2] * gain_b;
                ob[3] = ob[3] * gain_b;
                ob[4] = ob[4] * gain_a;
                ob[5] = ob[5] * gain_a;
                ob[6] = ob[6] * gain_b;
                ob[7] = ob[7] * gain_b;
                ob[8] = ob[8] * gain_a;
                ob[9] = ob[9] * gain_a;
                ob[10] = ob[10] * gain_b;
                ob[11] = ob[11] * gain_b;
                ob[12] = ob[12] * gain_a;
                ob[13] = ob[13] * gain_a;
                ob[14] = ob[14] * gain_b;
                ob[15] = ob[15] * gain_b;
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_sl + 320), "r"(*reinterpret_cast<const uint32_t*>(&oa[0])), "r"(*reinterpret_cast<const uint32_t*>(&oa[1])), "r"(*reinterpret_cast<const uint32_t*>(&oa[2])), "r"(*reinterpret_cast<const uint32_t*>(&oa[3])), "r"(*reinterpret_cast<const uint32_t*>(&oa[4])), "r"(*reinterpret_cast<const uint32_t*>(&oa[5])), "r"(*reinterpret_cast<const uint32_t*>(&oa[6])), "r"(*reinterpret_cast<const uint32_t*>(&oa[7])), "r"(*reinterpret_cast<const uint32_t*>(&oa[8])), "r"(*reinterpret_cast<const uint32_t*>(&oa[9])), "r"(*reinterpret_cast<const uint32_t*>(&oa[10])), "r"(*reinterpret_cast<const uint32_t*>(&oa[11])), "r"(*reinterpret_cast<const uint32_t*>(&oa[12])), "r"(*reinterpret_cast<const uint32_t*>(&oa[13])), "r"(*reinterpret_cast<const uint32_t*>(&oa[14])), "r"(*reinterpret_cast<const uint32_t*>(&oa[15])));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x256b.x4.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(t_sl + 352), "r"(*reinterpret_cast<const uint32_t*>(&ob[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob[3])), "r"(*reinterpret_cast<const uint32_t*>(&ob[4])), "r"(*reinterpret_cast<const uint32_t*>(&ob[5])), "r"(*reinterpret_cast<const uint32_t*>(&ob[6])), "r"(*reinterpret_cast<const uint32_t*>(&ob[7])), "r"(*reinterpret_cast<const uint32_t*>(&ob[8])), "r"(*reinterpret_cast<const uint32_t*>(&ob[9])), "r"(*reinterpret_cast<const uint32_t*>(&ob[10])), "r"(*reinterpret_cast<const uint32_t*>(&ob[11])), "r"(*reinterpret_cast<const uint32_t*>(&ob[12])), "r"(*reinterpret_cast<const uint32_t*>(&ob[13])), "r"(*reinterpret_cast<const uint32_t*>(&ob[14])), "r"(*reinterpret_cast<const uint32_t*>(&ob[15])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(gates_free_addr + (go_c) * 8);
                go_c += 1;
                if (go_c == 2) { go_c = 0; _phase_gates_ready_2 ^= 1; }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(pv_ready_addr);
                mbarrier_wait(o_full_addr + (of_c) * 8, _phase_o_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                long long out_a = ((long long)(tok0_1 + row_a) * (long long)num_v_heads + (long long)hv_2) * 128 + (long long)cq_2;
                long long out_b = ((long long)(tok0_1 + row_b) * (long long)num_v_heads + (long long)hv_2) * 128 + (long long)cq_2;
                if (n_1 == 64) {
                    if (tid == 256) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&oa[0])), "=r"(*reinterpret_cast<uint32_t*>(&oa[1])), "=r"(*reinterpret_cast<uint32_t*>(&oa[2])), "=r"(*reinterpret_cast<uint32_t*>(&oa[3])), "=r"(*reinterpret_cast<uint32_t*>(&oa[4])), "=r"(*reinterpret_cast<uint32_t*>(&oa[5])), "=r"(*reinterpret_cast<uint32_t*>(&oa[6])), "=r"(*reinterpret_cast<uint32_t*>(&oa[7])), "=r"(*reinterpret_cast<uint32_t*>(&oa[8])), "=r"(*reinterpret_cast<uint32_t*>(&oa[9])), "=r"(*reinterpret_cast<uint32_t*>(&oa[10])), "=r"(*reinterpret_cast<uint32_t*>(&oa[11])), "=r"(*reinterpret_cast<uint32_t*>(&oa[12])), "=r"(*reinterpret_cast<uint32_t*>(&oa[13])), "=r"(*reinterpret_cast<uint32_t*>(&oa[14])), "=r"(*reinterpret_cast<uint32_t*>(&oa[15]))
                    : "r"(t_sl + 256));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ob[0])), "=r"(*reinterpret_cast<uint32_t*>(&ob[1])), "=r"(*reinterpret_cast<uint32_t*>(&ob[2])), "=r"(*reinterpret_cast<uint32_t*>(&ob[3])), "=r"(*reinterpret_cast<uint32_t*>(&ob[4])), "=r"(*reinterpret_cast<uint32_t*>(&ob[5])), "=r"(*reinterpret_cast<uint32_t*>(&ob[6])), "=r"(*reinterpret_cast<uint32_t*>(&ob[7])), "=r"(*reinterpret_cast<uint32_t*>(&ob[8])), "=r"(*reinterpret_cast<uint32_t*>(&ob[9])), "=r"(*reinterpret_cast<uint32_t*>(&ob[10])), "=r"(*reinterpret_cast<uint32_t*>(&ob[11])), "=r"(*reinterpret_cast<uint32_t*>(&ob[12])), "=r"(*reinterpret_cast<uint32_t*>(&ob[13])), "=r"(*reinterpret_cast<uint32_t*>(&ob[14])), "=r"(*reinterpret_cast<uint32_t*>(&ob[15]))
                    : "r"(t_sl + 288));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (n_1 == 64) {
                    unsigned int pk_20[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 0)[_lp*2 + 0], (oa + 0)[_lp*2+1 + 0]));
                        pk_20[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_36 = static_cast<uint32_t>(s_ost_addr + (unsigned int)(st_col_2 / 64 * 8192 + (st_row_2 * 128 + st_col_2 % 64 * 2 ^ (st_row_2 * 128 + st_col_2 % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_36), "r"(*reinterpret_cast<const uint32_t*>(&pk_20[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_20[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_20[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_20[3]))
                        : "memory");
                    unsigned int pk_21[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 8)[_lp*2 + 0], (oa + 8)[_lp*2+1 + 0]));
                        pk_21[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_37 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((16 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (16 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (16 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_37), "r"(*reinterpret_cast<const uint32_t*>(&pk_21[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_21[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_21[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_21[3]))
                        : "memory");
                    unsigned int pk_22[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 0)[_lp*2 + 0], (ob + 0)[_lp*2+1 + 0]));
                        pk_22[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_38 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((32 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (32 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (32 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_38), "r"(*reinterpret_cast<const uint32_t*>(&pk_22[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_22[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_22[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_22[3]))
                        : "memory");
                    unsigned int pk_23[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 8)[_lp*2 + 0], (ob + 8)[_lp*2+1 + 0]));
                        pk_23[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_39 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((48 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (48 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (48 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_39), "r"(*reinterpret_cast<const uint32_t*>(&pk_23[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_23[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_23[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_23[3]))
                        : "memory");
                } else {
                    if (row_a < n_1) {
                        unsigned int wa_2[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 0)[_lp*2 + 0], (oa + 0)[_lp*2+1 + 0]));
                            wa_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 0))[0] = reinterpret_cast<unsigned*>(wa_2)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_2[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 2)[_lp*2 + 0], (oa + 2)[_lp*2+1 + 0]));
                            wb_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 0))[0] = reinterpret_cast<unsigned*>(wb_2)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_3[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 4)[_lp*2 + 0], (oa + 4)[_lp*2+1 + 0]));
                            wa_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 8))[0] = reinterpret_cast<unsigned*>(wa_3)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_3[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 6)[_lp*2 + 0], (oa + 6)[_lp*2+1 + 0]));
                            wb_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 8))[0] = reinterpret_cast<unsigned*>(wb_3)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_4[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 8)[_lp*2 + 0], (oa + 8)[_lp*2+1 + 0]));
                            wa_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 16))[0] = reinterpret_cast<unsigned*>(wa_4)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_4[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 10)[_lp*2 + 0], (oa + 10)[_lp*2+1 + 0]));
                            wb_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 16))[0] = reinterpret_cast<unsigned*>(wb_4)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_5[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 12)[_lp*2 + 0], (oa + 12)[_lp*2+1 + 0]));
                            wa_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 24))[0] = reinterpret_cast<unsigned*>(wa_5)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_5[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 14)[_lp*2 + 0], (oa + 14)[_lp*2+1 + 0]));
                            wb_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 24))[0] = reinterpret_cast<unsigned*>(wb_5)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_6[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 0)[_lp*2 + 0], (ob + 0)[_lp*2+1 + 0]));
                            wa_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 32))[0] = reinterpret_cast<unsigned*>(wa_6)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_6[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 2)[_lp*2 + 0], (ob + 2)[_lp*2+1 + 0]));
                            wb_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 32))[0] = reinterpret_cast<unsigned*>(wb_6)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_7[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 4)[_lp*2 + 0], (ob + 4)[_lp*2+1 + 0]));
                            wa_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 40))[0] = reinterpret_cast<unsigned*>(wa_7)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_7[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 6)[_lp*2 + 0], (ob + 6)[_lp*2+1 + 0]));
                            wb_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 40))[0] = reinterpret_cast<unsigned*>(wb_7)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_8[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 8)[_lp*2 + 0], (ob + 8)[_lp*2+1 + 0]));
                            wa_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 48))[0] = reinterpret_cast<unsigned*>(wa_8)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_8[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 10)[_lp*2 + 0], (ob + 10)[_lp*2+1 + 0]));
                            wb_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 48))[0] = reinterpret_cast<unsigned*>(wb_8)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_9[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 12)[_lp*2 + 0], (ob + 12)[_lp*2+1 + 0]));
                            wa_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 56))[0] = reinterpret_cast<unsigned*>(wa_9)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_9[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 14)[_lp*2 + 0], (ob + 14)[_lp*2+1 + 0]));
                            wb_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 56))[0] = reinterpret_cast<unsigned*>(wb_9)[0];
                    }
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&oa[0])), "=r"(*reinterpret_cast<uint32_t*>(&oa[1])), "=r"(*reinterpret_cast<uint32_t*>(&oa[2])), "=r"(*reinterpret_cast<uint32_t*>(&oa[3])), "=r"(*reinterpret_cast<uint32_t*>(&oa[4])), "=r"(*reinterpret_cast<uint32_t*>(&oa[5])), "=r"(*reinterpret_cast<uint32_t*>(&oa[6])), "=r"(*reinterpret_cast<uint32_t*>(&oa[7])), "=r"(*reinterpret_cast<uint32_t*>(&oa[8])), "=r"(*reinterpret_cast<uint32_t*>(&oa[9])), "=r"(*reinterpret_cast<uint32_t*>(&oa[10])), "=r"(*reinterpret_cast<uint32_t*>(&oa[11])), "=r"(*reinterpret_cast<uint32_t*>(&oa[12])), "=r"(*reinterpret_cast<uint32_t*>(&oa[13])), "=r"(*reinterpret_cast<uint32_t*>(&oa[14])), "=r"(*reinterpret_cast<uint32_t*>(&oa[15]))
                    : "r"(t_sl + 320));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&ob[0])), "=r"(*reinterpret_cast<uint32_t*>(&ob[1])), "=r"(*reinterpret_cast<uint32_t*>(&ob[2])), "=r"(*reinterpret_cast<uint32_t*>(&ob[3])), "=r"(*reinterpret_cast<uint32_t*>(&ob[4])), "=r"(*reinterpret_cast<uint32_t*>(&ob[5])), "=r"(*reinterpret_cast<uint32_t*>(&ob[6])), "=r"(*reinterpret_cast<uint32_t*>(&ob[7])), "=r"(*reinterpret_cast<uint32_t*>(&ob[8])), "=r"(*reinterpret_cast<uint32_t*>(&ob[9])), "=r"(*reinterpret_cast<uint32_t*>(&ob[10])), "=r"(*reinterpret_cast<uint32_t*>(&ob[11])), "=r"(*reinterpret_cast<uint32_t*>(&ob[12])), "=r"(*reinterpret_cast<uint32_t*>(&ob[13])), "=r"(*reinterpret_cast<uint32_t*>(&ob[14])), "=r"(*reinterpret_cast<uint32_t*>(&ob[15]))
                    : "r"(t_sl + 352));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(o_free_addr + (of_c) * 8);
                if (n_1 == 64) {
                    unsigned int pk_24[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 0)[_lp*2 + 0], (oa + 0)[_lp*2+1 + 0]));
                        pk_24[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_40 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((64 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (64 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (64 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_40), "r"(*reinterpret_cast<const uint32_t*>(&pk_24[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_24[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_24[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_24[3]))
                        : "memory");
                    unsigned int pk_25[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 8)[_lp*2 + 0], (oa + 8)[_lp*2+1 + 0]));
                        pk_25[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_41 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((80 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (80 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (80 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_41), "r"(*reinterpret_cast<const uint32_t*>(&pk_25[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_25[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_25[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_25[3]))
                        : "memory");
                    unsigned int pk_26[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 0)[_lp*2 + 0], (ob + 0)[_lp*2+1 + 0]));
                        pk_26[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_42 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((96 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (96 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (96 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_42), "r"(*reinterpret_cast<const uint32_t*>(&pk_26[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_26[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_26[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_26[3]))
                        : "memory");
                    unsigned int pk_27[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 8)[_lp*2 + 0], (ob + 8)[_lp*2+1 + 0]));
                        pk_27[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_43 = static_cast<uint32_t>(s_ost_addr + (unsigned int)((112 + st_col_2) / 64 * 8192 + (st_row_2 * 128 + (112 + st_col_2) % 64 * 2 ^ (st_row_2 * 128 + (112 + st_col_2) % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_43), "r"(*reinterpret_cast<const uint32_t*>(&pk_27[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk_27[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk_27[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk_27[3]))
                        : "memory");
                } else {
                    if (row_a < n_1) {
                        unsigned int wa_10[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 0)[_lp*2 + 0], (oa + 0)[_lp*2+1 + 0]));
                            wa_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 64))[0] = reinterpret_cast<unsigned*>(wa_10)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_10[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 2)[_lp*2 + 0], (oa + 2)[_lp*2+1 + 0]));
                            wb_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 64))[0] = reinterpret_cast<unsigned*>(wb_10)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_11[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 4)[_lp*2 + 0], (oa + 4)[_lp*2+1 + 0]));
                            wa_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 72))[0] = reinterpret_cast<unsigned*>(wa_11)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_11[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 6)[_lp*2 + 0], (oa + 6)[_lp*2+1 + 0]));
                            wb_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 72))[0] = reinterpret_cast<unsigned*>(wb_11)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_12[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 8)[_lp*2 + 0], (oa + 8)[_lp*2+1 + 0]));
                            wa_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 80))[0] = reinterpret_cast<unsigned*>(wa_12)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_12[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 10)[_lp*2 + 0], (oa + 10)[_lp*2+1 + 0]));
                            wb_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 80))[0] = reinterpret_cast<unsigned*>(wb_12)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_13[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 12)[_lp*2 + 0], (oa + 12)[_lp*2+1 + 0]));
                            wa_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 88))[0] = reinterpret_cast<unsigned*>(wa_13)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_13[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((oa + 14)[_lp*2 + 0], (oa + 14)[_lp*2+1 + 0]));
                            wb_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 88))[0] = reinterpret_cast<unsigned*>(wb_13)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_14[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 0)[_lp*2 + 0], (ob + 0)[_lp*2+1 + 0]));
                            wa_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 96))[0] = reinterpret_cast<unsigned*>(wa_14)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_14[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 2)[_lp*2 + 0], (ob + 2)[_lp*2+1 + 0]));
                            wb_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 96))[0] = reinterpret_cast<unsigned*>(wb_14)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_15[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 4)[_lp*2 + 0], (ob + 4)[_lp*2+1 + 0]));
                            wa_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 104))[0] = reinterpret_cast<unsigned*>(wa_15)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_15[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 6)[_lp*2 + 0], (ob + 6)[_lp*2+1 + 0]));
                            wb_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 104))[0] = reinterpret_cast<unsigned*>(wb_15)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_16[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 8)[_lp*2 + 0], (ob + 8)[_lp*2+1 + 0]));
                            wa_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 112))[0] = reinterpret_cast<unsigned*>(wa_16)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_16[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 10)[_lp*2 + 0], (ob + 10)[_lp*2+1 + 0]));
                            wb_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 112))[0] = reinterpret_cast<unsigned*>(wb_16)[0];
                    }
                    if (row_a < n_1) {
                        unsigned int wa_17[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 12)[_lp*2 + 0], (ob + 12)[_lp*2+1 + 0]));
                            wa_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_a + 120))[0] = reinterpret_cast<unsigned*>(wa_17)[0];
                    }
                    if (row_b < n_1) {
                        unsigned int wb_17[1];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((ob + 14)[_lp*2 + 0], (ob + 14)[_lp*2+1 + 0]));
                            wb_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<unsigned*>(o_out + (out_b + 120))[0] = reinterpret_cast<unsigned*>(wb_17)[0];
                    }
                }
                of_c += 1;
                if (of_c == 2) { of_c = 0; _phase_o_full ^= 1; }
                if (n_1 == 64) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    if (tid == 256) {
                        tma_store_3d((&o_tma), 0, hv_2, tok0_1, s_ost_addr + 0);
                        tma_store_3d((&o_tma), 64, hv_2, tok0_1, s_ost_addr + 8192);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
            }
            if (tid == 256) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            int blk_34 = blockIdx.x;
            int seg_3 = walk[blk_34 / num_v_heads];
            int hv_3 = blk_34 % num_v_heads;
            int c_begin_3 = seg_chunk_start[seg_3];
            int nc_3 = seg_chunk_start[seg_3 + 1] - c_begin_3;
            int seq_3 = seg_seq[seg_3];
            unsigned int tf_c_1 = 0;
            unsigned int pf_c_1 = 0;
            unsigned int o1_c_1 = 0;
            unsigned int po_c = 0;
            unsigned int _phase_tiles_full_1 = 0;
            unsigned int _phase_p_free = 1;
            unsigned int _phase_pv_ready_0 = 0;
            unsigned int _phase_hs_k0_0 = 0;
            unsigned int _phase_hs_k1_0 = 0;
            unsigned int _phase_o_free = 1;
            unsigned int _phase_vprime_ready_0 = 0;
            unsigned int _phase_s_scaled_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_3 = 0; c_3 < nc_3; c_3++) {
                    mbarrier_wait(tiles_full_addr + (tf_c_1) * 8, _phase_tiles_full_1);
                    mbarrier_wait(p_free_addr + (pf_c_1) * 8, _phase_p_free);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_0 = (((s_qn_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_1 = (((s_kn_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_0 = (((s_qn_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_1 = (((s_kn_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"((tmem_all + (192 + c_3 % 2 * 1048576))), "r"(0) : "memory");
                    tcgen05_commit(p_full_addr + (pf_c_1) * 8);
                    pf_c_1 += 1;
                    if (pf_c_1 == 2) { pf_c_1 = 0; _phase_p_free ^= 1; }
                    if (c_3 > 0) {
                        mbarrier_wait(pv_ready_addr, _phase_pv_ready_0);
                        _phase_pv_ready_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_2 = ((s_pg_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_3 = (((s_vnT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _smem_desc_lo_2 = ((s_pg_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_3 = (((s_vnT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"((tmem_all + (256 + (c_3 - 1) % 2 * 1048576))), "r"(1) : "memory");
                        tcgen05_commit(o_full_addr + (po_c) * 8);
                        po_c += 1;
                        if (po_c == 2) { po_c = 0; }
                        tcgen05_commit(vn_free_addr);
                    }
                    mbarrier_wait(hs_k0_addr, _phase_hs_k0_0);
                    _phase_hs_k0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_4 = ((s_hs0_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_5 = (((s_w_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_4 = ((s_hs0_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_5 = (((s_w_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"((tmem_all + (128))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_6 = ((s_hs1_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_7 = (((s_w_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_6 = ((s_hs1_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_7 = (((s_w_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"((tmem_all + (1048704))), "r"(0) : "memory");
                    mbarrier_wait(hs_k1_addr, _phase_hs_k1_0);
                    _phase_hs_k1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_8 = ((s_hs0_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_9 = (((s_w_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_8 = ((s_hs0_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_9 = (((s_w_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "mov.b32 id, 68158608;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"((tmem_all + (128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_10 = ((s_hs1_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_11 = (((s_w_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_10 = ((s_hs1_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_11 = (((s_w_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
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
                    "mov.b32 id, 68158608;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"((tmem_all + (1048704))), "r"(1) : "memory");
                    tcgen05_commit(ws_full_addr);
                    tcgen05_commit(w_free_addr + (tf_c_1) * 8);
                    mbarrier_wait(o_free_addr + (o1_c_1) * 8, _phase_o_free);
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_12 = (((s_qn_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_13 = ((s_hs0_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_12 = (((s_qn_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_13 = ((s_hs0_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"((tmem_all + (256 + c_3 % 2 * 1048576))), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_14 = (((s_qn_addr) >> 4) & 0x7FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_15 = ((s_hs1_addr) >> 4) & 0x7FFF;
#else
                    int _smem_desc_lo_14 = (((s_qn_addr) >> 4) & 0x3FFF) + (tf_c_1) * 4096;
                    int _smem_desc_lo_15 = ((s_hs1_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"((tmem_all + (320 + c_3 % 2 * 1048576))), "r"(0) : "memory");
                    tcgen05_commit(o1_full_addr + (o1_c_1) * 8);
                    o1_c_1 += 1;
                    if (o1_c_1 == 2) { o1_c_1 = 0; _phase_o_free ^= 1; }
                    mbarrier_wait(vprime_ready_addr, _phase_vprime_ready_0);
                    _phase_vprime_ready_0 ^= 1;
                    mbarrier_wait(s_scaled_addr, _phase_s_scaled_0);
                    _phase_s_scaled_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_16 = ((s_vp0_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_17 = ((((s_knT_addr) >> 4) & 0x7FFF) | 0x2000000) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_16 = ((s_vp0_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_17 = ((((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000) + (tf_c_1) * 4096;
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
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"(tmem_all), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_18 = ((s_vp1_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_19 = ((((s_knT_addr) >> 4) & 0x7FFF) | 0x2000000) + (tf_c_1) * 4096;
#else
                    int _smem_desc_lo_18 = ((s_vp1_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_19 = ((((s_knT_addr) >> 4) & 0x3FFF) | 0x2000000) + (tf_c_1) * 4096;
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
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"((tmem_all + (1048576))), "r"(1) : "memory");
                    tcgen05_commit(s_full_addr);
                    tcgen05_commit(qk_free_addr + (tf_c_1) * 8);
                    tf_c_1 += 1;
                    if (tf_c_1 == 2) { tf_c_1 = 0; _phase_tiles_full_1 ^= 1; }
                }
                if (nc_3 > 0) {
                    mbarrier_wait(pv_ready_addr, _phase_pv_ready_0);
                    _phase_pv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _smem_desc_lo_20 = ((s_pg_addr) >> 4) & 0x7FFF;
                    int _smem_desc_lo_21 = (((s_vnT_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                    int _smem_desc_lo_20 = ((s_pg_addr) >> 4) & 0x3FFF;
                    int _smem_desc_lo_21 = (((s_vnT_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    :: "r"(_smem_desc_lo_20), "r"(_smem_desc_lo_21), "r"((tmem_all + (256 + (nc_3 - 1) % 2 * 1048576))), "r"(1) : "memory");
                    tcgen05_commit(o_full_addr + (po_c) * 8);
                    po_c += 1;
                    if (po_c == 2) { po_c = 0; }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            int blk_35 = blockIdx.x;
            int seg_4 = walk[blk_35 / num_v_heads];
            int hv_4 = blk_35 % num_v_heads;
            int c_begin_4 = seg_chunk_start[seg_4];
            int nc_4 = seg_chunk_start[seg_4 + 1] - c_begin_4;
            int seq_4 = seg_seq[seg_4];
            int hq = hv_4 / (num_v_heads / num_heads);
            unsigned int ld_c = 0;
            unsigned int _phase_w_free = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_4 = 0; c_4 < nc_4; c_4++) {
                    int tok0_2 = chunk_start[c_begin_4 + c_4];
                    unsigned int so = ld_c * 65536;
                    mbarrier_wait(w_free_addr + (ld_c) * 8, _phase_w_free);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_c) * 8, 65536);
                    tma_3d_gmem2smem(s_w_addr + so + 0, (&w_tma), 0, hv_4, tok0_2, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_w_addr + so + 8192, (&w_tma), 64, hv_4, tok0_2, tiles_full_addr + (ld_c) * 8);
                    mbarrier_wait(u_free_addr + (ld_c) * 8, _phase_w_free);
                    tma_3d_gmem2smem(s_u_addr + so + 0, (&u_tma), 0, hv_4, tok0_2, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_u_addr + so + 8192, (&u_tma), 64, hv_4, tok0_2, tiles_full_addr + (ld_c) * 8);
                    mbarrier_wait(qk_free_addr + (ld_c) * 8, _phase_w_free);
                    tma_3d_gmem2smem(s_kn_addr + so + 0, (&kn_tma), 0, hq, tok0_2, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_qn_addr + so + 0, (&qn_tma), 0, hq, tok0_2, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_kn_addr + so + 8192, (&kn_tma), 64, hq, tok0_2, tiles_full_addr + (ld_c) * 8);
                    tma_3d_gmem2smem(s_qn_addr + so + 8192, (&qn_tma), 64, hq, tok0_2, tiles_full_addr + (ld_c) * 8);
                    ld_c += 1;
                    if (ld_c == 2) { ld_c = 0; _phase_w_free ^= 1; }
                }
            }
        }
    }
    // ---- Role: gate ----
    if (warp == 14) {
        { // gate_main
            int blk_36 = blockIdx.x;
            int seg_5 = walk[blk_36 / num_v_heads];
            int hv_5 = blk_36 % num_v_heads;
            int c_begin_5 = seg_chunk_start[seg_5];
            int nc_5 = seg_chunk_start[seg_5 + 1] - c_begin_5;
            int seq_5 = seg_seq[seg_5];
            int lane_4 = lane;
            unsigned int gt_c = 0;
            unsigned int _phase_gates_free = 1;
            #pragma unroll 1
            for (int c_5 = 0; c_5 < nc_5; c_5++) {
                mbarrier_wait(gates_free_addr + (gt_c) * 8, _phase_gates_free);
                int chunk_2 = c_begin_5 + c_5;
                int tok0_3 = chunk_start[chunk_2];
                int n_2 = chunk_len[chunk_2];
                int g_off_2 = c_5 % 2 * 256;
                int _max_0_1 = ((n_2) > (1) ? (n_2) : (1));
                float g_last = g_cs[(long long)(tok0_3 + _max_0_1 - 1) * (long long)num_v_heads + (long long)hv_5];
                float gv_0 = 0.0f;
                float dv_0 = 0.0f;
                if (lane_4 < n_2) {
                    gv_0 = g_cs[(long long)(tok0_3 + lane_4) * (long long)num_v_heads + (long long)hv_5];
                    float _exp2_1_1 = approx_exp2(g_last - gv_0);
                    dv_0 = _exp2_1_1;
                }
                s_gt[g_off_2 + lane_4] = gv_0;
                s_gt[g_off_2 + 64 + lane_4] = dv_0;
                float _exp2_2_1 = approx_exp2(gv_0);
                s_gt[g_off_2 + 128 + lane_4] = _exp2_2_1 * scale;
                float gv_1 = 0.0f;
                float dv_1 = 0.0f;
                if (lane_4 + 32 < n_2) {
                    gv_1 = g_cs[(long long)(tok0_3 + lane_4 + 32) * (long long)num_v_heads + (long long)hv_5];
                    float _exp2_3_1 = approx_exp2(g_last - gv_1);
                    dv_1 = _exp2_3_1;
                }
                s_gt[g_off_2 + lane_4 + 32] = gv_1;
                s_gt[g_off_2 + 64 + lane_4 + 32] = dv_1;
                float _exp2_4_1 = approx_exp2(gv_1);
                s_gt[g_off_2 + 128 + lane_4 + 32] = _exp2_4_1 * scale;
                if (lane_4 == 0) {
                    float _exp2_5_1 = approx_exp2(g_last);
                    s_gt[g_off_2 + 192] = _exp2_5_1;
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
            int blk_37 = blockIdx.x;
            int seg_6 = walk[blk_37 / num_v_heads];
            int hv_6 = blk_37 % num_v_heads;
            int c_begin_6 = seg_chunk_start[seg_6];
            int nc_6 = seg_chunk_start[seg_6 + 1] - c_begin_6;
            int seq_6 = seg_seq[seg_6];
            unsigned int _phase_hs_k1_0_1 = 0;
            unsigned int _phase_vprime_ready_0_1 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int c_6 = 0; c_6 < nc_6; c_6++) {
                    int chunk_3 = c_begin_6 + c_6;
                    int n_3 = chunk_len[chunk_3];
                    int tok0_4 = chunk_start[chunk_3];
                    int hrow = chunk_3 * num_v_heads + hv_6;
                    mbarrier_wait(hs_k1_addr, _phase_hs_k1_0_1);
                    _phase_hs_k1_0_1 ^= 1;
                    tma_store_3d((&h_tma), 0, 0, hrow, s_hs0_addr + 0);
                    tma_store_3d((&h_tma), 0, 64, hrow, s_hs1_addr + 0);
                    tma_store_3d((&h_tma), 64, 0, hrow, s_hs0_addr + 8192);
                    tma_store_3d((&h_tma), 64, 64, hrow, s_hs1_addr + 8192);
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    mbarrier_arrive(hs_free_addr);
                    mbarrier_wait(vprime_ready_addr, _phase_vprime_ready_0_1);
                    _phase_vprime_ready_0_1 ^= 1;
                    if (n_3 == 64) {
                        tma_store_3d((&v_new_tma), 0, hv_6, tok0_4, s_vn_addr + 0);
                        tma_store_3d((&v_new_tma), 64, hv_6, tok0_4, s_vn_addr + 8192);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    mbarrier_arrive(vn_free_addr);
                }
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
