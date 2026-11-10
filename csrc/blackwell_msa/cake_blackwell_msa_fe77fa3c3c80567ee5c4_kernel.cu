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
#define TMEM_NCOLS 512
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_OFFSET 256
#define NUM_KV_RING_STAGES 9
#define NUM_S_RING_STAGES 2
#define NUM_P_RING_STAGES 4
#define NUM_ITEM_RING_STAGES 2
#define SMEM_SMEM_EXCH_OFF 1024
#define SMEM_SMEM_EXCH_STAGE_BYTES 4096
#define SMEM_SMEM_EXCH_STRIDE 4096
#define SMEM_SMEM_SUM_OFF 5120
#define SMEM_SMEM_SUM_STAGE_BYTES 1024
#define SMEM_SMEM_SUM_STRIDE 1024
#define SMEM_SMEM_BLOCK_OFF 6144
#define SMEM_SMEM_BLOCK_STAGE_BYTES 400
#define SMEM_SMEM_BLOCK_STRIDE 400
#define SMEM_SMEM_PAGE_OFF 6544
#define SMEM_SMEM_PAGE_STAGE_BYTES 400
#define SMEM_SMEM_PAGE_STRIDE 400
#define SMEM_SMEM_TOKMASK_OFF 6944
#define SMEM_SMEM_TOKMASK_STAGE_BYTES 400
#define SMEM_SMEM_TOKMASK_STRIDE 400
#define SMEM_SMEM_NPAIRS_OFF 7344
#define SMEM_SMEM_NPAIRS_STAGE_BYTES 8
#define SMEM_SMEM_NPAIRS_STRIDE 8
#define SMEM_SMEM_KVLEN_OFF 7352
#define SMEM_SMEM_KVLEN_STAGE_BYTES 8
#define SMEM_SMEM_KVLEN_STRIDE 8
#define SMEM_SMEM_ISSUE_OFF 7360
#define SMEM_SMEM_ISSUE_STAGE_BYTES 16
#define SMEM_SMEM_ISSUE_STRIDE 16
#define SMEM_SMEM_QT_OFF 8192
#define SMEM_SMEM_QT_STAGE_BYTES 8192
#define SMEM_SMEM_QT_STRIDE 8192
#define SMEM_SMEM_P_OFF 16384
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_KV_OFF 81920
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 81920
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_TOTAL 229376
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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_blackwell_msa_fe77fa3c3c80567ee5c4(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, float* __restrict__ msa_lse, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages)
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
    #define q_empty_addr (mbar_base + 8)
    #define meta_full_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 104)
    #define s_full_addr (mbar_base + 176)
    #define s_empty_addr (mbar_base + 192)
    #define p_full_addr (mbar_base + 208)
    #define pv_done_addr (mbar_base + 240)
    #define item_tick_addr (mbar_base + 304)
    #define decode_done_addr (mbar_base + 320)
    #define meta_empty_addr (mbar_base + 328)
    #define tmem_dead_addr (mbar_base + 344)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_exch_addr = smem + 1024;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 5120);
    const int smem_sum_addr = smem + 5120;
    int* smem_block = reinterpret_cast<int*>(smem_raw + 6144);
    const int smem_block_addr = smem + 6144;
    int* smem_page = reinterpret_cast<int*>(smem_raw + 6544);
    const int smem_page_addr = smem + 6544;
    int* smem_tokmask = reinterpret_cast<int*>(smem_raw + 6944);
    const int smem_tokmask_addr = smem + 6944;
    int* smem_npairs = reinterpret_cast<int*>(smem_raw + 7344);
    const int smem_npairs_addr = smem + 7344;
    int* smem_kvlen = reinterpret_cast<int*>(smem_raw + 7352);
    const int smem_kvlen_addr = smem + 7352;
    unsigned int* smem_issue = reinterpret_cast<unsigned int*>(smem_raw + 7360);
    const int smem_issue_addr = smem + 7360;
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 8192);
    const int smem_qt_addr = smem + 8192;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 16384);
    const int smem_p_addr = smem + 16384;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 81920);
    const int smem_kv_addr = smem + 81920;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 81920);
    const int smem_v_addr = smem + 81920;
    #pragma unroll
    for (int word = 0; word < 3; word++) {
        asm volatile("st.relaxed.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + (unsigned int)(4 * word))), "r"(static_cast<uint32_t>(0)) : "memory");
    }

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 44 barriers)
    // Mbarriers at smem_raw[0..352)

    if (warp == 0) {
        // q_full: 1 barriers, init_count=1
        // q_empty: 1 barriers, init_count=1
        // --- pipeline 'item_ring' ---
        // meta_full: 2 barriers, init_count=1
        // --- pipeline 'kv_ring' ---
        // kv_full: 9 barriers, init_count=1
        // kv_empty: 9 barriers, init_count=1
        // --- pipeline 's_ring' ---
        // s_full: 2 barriers, init_count=1
        // s_empty: 2 barriers, init_count=8
        // --- pipeline 'p_ring' ---
        // p_full: 4 barriers, init_count=8
        // pv_done: 8 barriers, init_count=1
        // item_tick: 2 barriers, init_count=1
        // decode_done: 1 barriers, init_count=8
        // --- pipeline 'item_ring' ---
        // meta_empty: 2 barriers, init_count=10
        // tmem_dead: 1 barriers, init_count=8
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(30), "r"((uint32_t)(8)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(24), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 8;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(11), "r"((uint32_t)(10)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(9), "r"((uint32_t)(8)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(1)));
        if (lane < 12) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    if (warp == 0) {
        int _tmem_hold = smem + 352;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o = taddr + 256;
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // softmax_0_main
            const int lane_base = warp % 4 * 32 << 16;
            const int warp_in_role = warp % 4;
            const int key_row = warp_in_role * 32 + lane;
            const int own_half = warp_in_role / 2;
            const int grp_shift = lane / 8 * 8;
            const unsigned int grp_mask = 255;
            float tau = 8.0f;
            int g_real_w = num_q_heads / num_kv_heads;
            int heads_per_req_w = num_kv_heads * (g_real_w / 8);
            unsigned int total_items = total_q / 8 * heads_per_req_w;
            int item_stage = 0;
            int s_stage = 0;
            int p_stage = 0;
            int item_ord = 0;
            int gpair = 0;
            unsigned int _phase_meta_full = 0;
            unsigned int _phase_s_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx = blockIdx.x; work_idx < total_items; work_idx += gridDim.x) {
                int item = work_idx;
                int g_real = num_q_heads / num_kv_heads;
                int halves = g_real / 8;
                int heads_per_req = num_kv_heads * halves;
                int request = item / heads_per_req;
                int sub = item % heads_per_req;
                int kv_head = sub / halves;
                int head0 = kv_head * g_real + sub % halves * 8;
                mbarrier_wait(meta_full_addr + (item_stage) * 8, _phase_meta_full);
                int npairs = smem_npairs[item_stage];
                int kvlen = smem_kvlen[item_stage];
                int meta_base = item_stage * 50;
                item_stage += 1;
                if (item_stage == 2) { item_stage = 0; _phase_meta_full ^= 1; }
                float o_hold[4];
                float ms_hold[4];
                o_hold[0] = -CUDART_INF_F;
                ms_hold[0] = 0.0f;
                o_hold[1] = -CUDART_INF_F;
                ms_hold[1] = 0.0f;
                o_hold[2] = -CUDART_INF_F;
                ms_hold[2] = 0.0f;
                o_hold[3] = -CUDART_INF_F;
                ms_hold[3] = 0.0f;
                float lane_sum[64];
                lane_sum[0] = 0.0f;
                lane_sum[1] = 0.0f;
                lane_sum[2] = 0.0f;
                lane_sum[3] = 0.0f;
                lane_sum[4] = 0.0f;
                lane_sum[5] = 0.0f;
                lane_sum[6] = 0.0f;
                lane_sum[7] = 0.0f;
                lane_sum[8] = 0.0f;
                lane_sum[9] = 0.0f;
                lane_sum[10] = 0.0f;
                lane_sum[11] = 0.0f;
                lane_sum[12] = 0.0f;
                lane_sum[13] = 0.0f;
                lane_sum[14] = 0.0f;
                lane_sum[15] = 0.0f;
                lane_sum[16] = 0.0f;
                lane_sum[17] = 0.0f;
                lane_sum[18] = 0.0f;
                lane_sum[19] = 0.0f;
                lane_sum[20] = 0.0f;
                lane_sum[21] = 0.0f;
                lane_sum[22] = 0.0f;
                lane_sum[23] = 0.0f;
                lane_sum[24] = 0.0f;
                lane_sum[25] = 0.0f;
                lane_sum[26] = 0.0f;
                lane_sum[27] = 0.0f;
                lane_sum[28] = 0.0f;
                lane_sum[29] = 0.0f;
                lane_sum[30] = 0.0f;
                lane_sum[31] = 0.0f;
                lane_sum[32] = 0.0f;
                lane_sum[33] = 0.0f;
                lane_sum[34] = 0.0f;
                lane_sum[35] = 0.0f;
                lane_sum[36] = 0.0f;
                lane_sum[37] = 0.0f;
                lane_sum[38] = 0.0f;
                lane_sum[39] = 0.0f;
                lane_sum[40] = 0.0f;
                lane_sum[41] = 0.0f;
                lane_sum[42] = 0.0f;
                lane_sum[43] = 0.0f;
                lane_sum[44] = 0.0f;
                lane_sum[45] = 0.0f;
                lane_sum[46] = 0.0f;
                lane_sum[47] = 0.0f;
                lane_sum[48] = 0.0f;
                lane_sum[49] = 0.0f;
                lane_sum[50] = 0.0f;
                lane_sum[51] = 0.0f;
                lane_sum[52] = 0.0f;
                lane_sum[53] = 0.0f;
                lane_sum[54] = 0.0f;
                lane_sum[55] = 0.0f;
                lane_sum[56] = 0.0f;
                lane_sum[57] = 0.0f;
                lane_sum[58] = 0.0f;
                lane_sum[59] = 0.0f;
                lane_sum[60] = 0.0f;
                lane_sum[61] = 0.0f;
                lane_sum[62] = 0.0f;
                lane_sum[63] = 0.0f;
                #pragma unroll 1
                for (int pair = 0; pair < 25; pair++) {
                    if (pair >= npairs) {
                        break;
                    }
                    mbarrier_wait(s_full_addr + (s_stage) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)(s_stage * 128) + 0 + (unsigned int)lane_base;
                    float sv0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv0[0]), "=f"(sv0[1]), "=f"(sv0[2]), "=f"(sv0[3]), "=f"(sv0[4]), "=f"(sv0[5]), "=f"(sv0[6]), "=f"(sv0[7]), "=f"(sv0[8]), "=f"(sv0[9]), "=f"(sv0[10]), "=f"(sv0[11]), "=f"(sv0[12]), "=f"(sv0[13]), "=f"(sv0[14]), "=f"(sv0[15]), "=f"(sv0[16]), "=f"(sv0[17]), "=f"(sv0[18]), "=f"(sv0[19]), "=f"(sv0[20]), "=f"(sv0[21]), "=f"(sv0[22]), "=f"(sv0[23]), "=f"(sv0[24]), "=f"(sv0[25]), "=f"(sv0[26]), "=f"(sv0[27]), "=f"(sv0[28]), "=f"(sv0[29]), "=f"(sv0[30]), "=f"(sv0[31])
                        : "r"(s_addr));
                    float sv1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv1[0]), "=f"(sv1[1]), "=f"(sv1[2]), "=f"(sv1[3]), "=f"(sv1[4]), "=f"(sv1[5]), "=f"(sv1[6]), "=f"(sv1[7]), "=f"(sv1[8]), "=f"(sv1[9]), "=f"(sv1[10]), "=f"(sv1[11]), "=f"(sv1[12]), "=f"(sv1[13]), "=f"(sv1[14]), "=f"(sv1[15]), "=f"(sv1[16]), "=f"(sv1[17]), "=f"(sv1[18]), "=f"(sv1[19]), "=f"(sv1[20]), "=f"(sv1[21]), "=f"(sv1[22]), "=f"(sv1[23]), "=f"(sv1[24]), "=f"(sv1[25]), "=f"(sv1[26]), "=f"(sv1[27]), "=f"(sv1[28]), "=f"(sv1[29]), "=f"(sv1[30]), "=f"(sv1[31])
                        : "r"(s_addr + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage) * 8);
                    }
                    int blk_t = smem_block[meta_base + pair * 2 + 0];
                    unsigned int tm_t = (unsigned int)smem_tokmask[meta_base + pair * 2 + 0];
                    int vbase = kvlen - 7 - blk_t * 128;
                    if (key_row >= vbase + 0) {
                        sv0[0] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[0] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[1] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[1] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[2] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[2] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[3] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[3] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[4] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[4] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[5] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[5] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[6] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[6] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 0) {
                        sv0[7] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 0 & 1) == 0) {
                        sv0[7] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[8] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[8] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[9] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[9] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[10] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[10] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[11] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[11] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[12] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[12] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[13] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[13] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[14] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[14] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 1) {
                        sv0[15] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 1 & 1) == 0) {
                        sv0[15] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[16] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[16] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[17] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[17] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[18] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[18] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[19] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[19] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[20] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[20] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[21] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[21] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[22] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[22] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 2) {
                        sv0[23] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 2 & 1) == 0) {
                        sv0[23] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[24] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[24] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[25] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[25] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[26] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[26] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[27] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[27] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[28] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[28] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[29] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[29] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[30] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[30] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase + 3) {
                        sv0[31] = -CUDART_INF_F;
                    }
                    if ((tm_t >> 3 & 1) == 0) {
                        sv0[31] = -CUDART_INF_F;
                    }
                    float hold = -CUDART_INF_F;
                    float mx_0;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_0) : "f"(sv0[0]));
                    if (lane == 0) {
                        hold = mx_0;
                    }
                    float mx_1;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_1) : "f"(sv0[1]));
                    if (lane == 1) {
                        hold = mx_1;
                    }
                    float mx_2;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_2) : "f"(sv0[2]));
                    if (lane == 2) {
                        hold = mx_2;
                    }
                    float mx_3;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_3) : "f"(sv0[3]));
                    if (lane == 3) {
                        hold = mx_3;
                    }
                    float mx_4;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_4) : "f"(sv0[4]));
                    if (lane == 4) {
                        hold = mx_4;
                    }
                    float mx_5;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_5) : "f"(sv0[5]));
                    if (lane == 5) {
                        hold = mx_5;
                    }
                    float mx_6;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_6) : "f"(sv0[6]));
                    if (lane == 6) {
                        hold = mx_6;
                    }
                    float mx_7;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_7) : "f"(sv0[7]));
                    if (lane == 7) {
                        hold = mx_7;
                    }
                    float mx_8;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_8) : "f"(sv0[8]));
                    if (lane == 8) {
                        hold = mx_8;
                    }
                    float mx_9;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_9) : "f"(sv0[9]));
                    if (lane == 9) {
                        hold = mx_9;
                    }
                    float mx_10;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_10) : "f"(sv0[10]));
                    if (lane == 10) {
                        hold = mx_10;
                    }
                    float mx_11;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_11) : "f"(sv0[11]));
                    if (lane == 11) {
                        hold = mx_11;
                    }
                    float mx_12;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_12) : "f"(sv0[12]));
                    if (lane == 12) {
                        hold = mx_12;
                    }
                    float mx_13;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_13) : "f"(sv0[13]));
                    if (lane == 13) {
                        hold = mx_13;
                    }
                    float mx_14;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_14) : "f"(sv0[14]));
                    if (lane == 14) {
                        hold = mx_14;
                    }
                    float mx_15;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_15) : "f"(sv0[15]));
                    if (lane == 15) {
                        hold = mx_15;
                    }
                    float mx_16;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_16) : "f"(sv0[16]));
                    if (lane == 16) {
                        hold = mx_16;
                    }
                    float mx_17;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_17) : "f"(sv0[17]));
                    if (lane == 17) {
                        hold = mx_17;
                    }
                    float mx_18;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_18) : "f"(sv0[18]));
                    if (lane == 18) {
                        hold = mx_18;
                    }
                    float mx_19;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_19) : "f"(sv0[19]));
                    if (lane == 19) {
                        hold = mx_19;
                    }
                    float mx_20;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_20) : "f"(sv0[20]));
                    if (lane == 20) {
                        hold = mx_20;
                    }
                    float mx_21;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_21) : "f"(sv0[21]));
                    if (lane == 21) {
                        hold = mx_21;
                    }
                    float mx_22;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_22) : "f"(sv0[22]));
                    if (lane == 22) {
                        hold = mx_22;
                    }
                    float mx_23;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_23) : "f"(sv0[23]));
                    if (lane == 23) {
                        hold = mx_23;
                    }
                    float mx_24;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_24) : "f"(sv0[24]));
                    if (lane == 24) {
                        hold = mx_24;
                    }
                    float mx_25;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_25) : "f"(sv0[25]));
                    if (lane == 25) {
                        hold = mx_25;
                    }
                    float mx_26;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_26) : "f"(sv0[26]));
                    if (lane == 26) {
                        hold = mx_26;
                    }
                    float mx_27;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_27) : "f"(sv0[27]));
                    if (lane == 27) {
                        hold = mx_27;
                    }
                    float mx_28;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_28) : "f"(sv0[28]));
                    if (lane == 28) {
                        hold = mx_28;
                    }
                    float mx_29;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_29) : "f"(sv0[29]));
                    if (lane == 29) {
                        hold = mx_29;
                    }
                    float mx_30;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_30) : "f"(sv0[30]));
                    if (lane == 30) {
                        hold = mx_30;
                    }
                    float mx_31;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_31) : "f"(sv0[31]));
                    if (lane == 31) {
                        hold = mx_31;
                    }
                    int exch_slot = ((0 + pair % 2) * 2 + 0) * 4 * 32 + warp_in_role * 32;
                    smem_exch[exch_slot + lane] = hold;
                    int blk_t_0 = smem_block[meta_base + pair * 2 + 1];
                    unsigned int tm_t_0 = (unsigned int)smem_tokmask[meta_base + pair * 2 + 1];
                    int vbase_0 = kvlen - 7 - blk_t_0 * 128;
                    if (key_row >= vbase_0 + 0) {
                        sv1[0] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[0] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[1] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[1] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[2] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[2] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[3] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[3] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[4] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[4] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[5] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[5] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[6] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[6] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 0) {
                        sv1[7] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 0 & 1) == 0) {
                        sv1[7] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[8] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[8] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[9] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[9] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[10] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[10] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[11] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[11] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[12] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[12] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[13] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[13] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[14] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[14] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 1) {
                        sv1[15] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 1 & 1) == 0) {
                        sv1[15] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[16] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[16] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[17] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[17] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[18] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[18] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[19] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[19] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[20] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[20] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[21] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[21] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[22] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[22] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 2) {
                        sv1[23] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 2 & 1) == 0) {
                        sv1[23] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[24] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[24] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[25] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[25] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[26] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[26] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[27] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[27] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[28] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[28] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[29] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[29] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[30] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[30] = -CUDART_INF_F;
                    }
                    if (key_row >= vbase_0 + 3) {
                        sv1[31] = -CUDART_INF_F;
                    }
                    if ((tm_t_0 >> 3 & 1) == 0) {
                        sv1[31] = -CUDART_INF_F;
                    }
                    float hold_0 = -CUDART_INF_F;
                    float mx_32;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_32) : "f"(sv1[0]));
                    if (lane == 0) {
                        hold_0 = mx_32;
                    }
                    float mx_33;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_33) : "f"(sv1[1]));
                    if (lane == 1) {
                        hold_0 = mx_33;
                    }
                    float mx_34;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_34) : "f"(sv1[2]));
                    if (lane == 2) {
                        hold_0 = mx_34;
                    }
                    float mx_35;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_35) : "f"(sv1[3]));
                    if (lane == 3) {
                        hold_0 = mx_35;
                    }
                    float mx_36;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_36) : "f"(sv1[4]));
                    if (lane == 4) {
                        hold_0 = mx_36;
                    }
                    float mx_37;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_37) : "f"(sv1[5]));
                    if (lane == 5) {
                        hold_0 = mx_37;
                    }
                    float mx_38;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_38) : "f"(sv1[6]));
                    if (lane == 6) {
                        hold_0 = mx_38;
                    }
                    float mx_39;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_39) : "f"(sv1[7]));
                    if (lane == 7) {
                        hold_0 = mx_39;
                    }
                    float mx_40;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_40) : "f"(sv1[8]));
                    if (lane == 8) {
                        hold_0 = mx_40;
                    }
                    float mx_41;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_41) : "f"(sv1[9]));
                    if (lane == 9) {
                        hold_0 = mx_41;
                    }
                    float mx_42;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_42) : "f"(sv1[10]));
                    if (lane == 10) {
                        hold_0 = mx_42;
                    }
                    float mx_43;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_43) : "f"(sv1[11]));
                    if (lane == 11) {
                        hold_0 = mx_43;
                    }
                    float mx_44;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_44) : "f"(sv1[12]));
                    if (lane == 12) {
                        hold_0 = mx_44;
                    }
                    float mx_45;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_45) : "f"(sv1[13]));
                    if (lane == 13) {
                        hold_0 = mx_45;
                    }
                    float mx_46;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_46) : "f"(sv1[14]));
                    if (lane == 14) {
                        hold_0 = mx_46;
                    }
                    float mx_47;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_47) : "f"(sv1[15]));
                    if (lane == 15) {
                        hold_0 = mx_47;
                    }
                    float mx_48;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_48) : "f"(sv1[16]));
                    if (lane == 16) {
                        hold_0 = mx_48;
                    }
                    float mx_49;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_49) : "f"(sv1[17]));
                    if (lane == 17) {
                        hold_0 = mx_49;
                    }
                    float mx_50;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_50) : "f"(sv1[18]));
                    if (lane == 18) {
                        hold_0 = mx_50;
                    }
                    float mx_51;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_51) : "f"(sv1[19]));
                    if (lane == 19) {
                        hold_0 = mx_51;
                    }
                    float mx_52;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_52) : "f"(sv1[20]));
                    if (lane == 20) {
                        hold_0 = mx_52;
                    }
                    float mx_53;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_53) : "f"(sv1[21]));
                    if (lane == 21) {
                        hold_0 = mx_53;
                    }
                    float mx_54;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_54) : "f"(sv1[22]));
                    if (lane == 22) {
                        hold_0 = mx_54;
                    }
                    float mx_55;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_55) : "f"(sv1[23]));
                    if (lane == 23) {
                        hold_0 = mx_55;
                    }
                    float mx_56;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_56) : "f"(sv1[24]));
                    if (lane == 24) {
                        hold_0 = mx_56;
                    }
                    float mx_57;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_57) : "f"(sv1[25]));
                    if (lane == 25) {
                        hold_0 = mx_57;
                    }
                    float mx_58;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_58) : "f"(sv1[26]));
                    if (lane == 26) {
                        hold_0 = mx_58;
                    }
                    float mx_59;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_59) : "f"(sv1[27]));
                    if (lane == 27) {
                        hold_0 = mx_59;
                    }
                    float mx_60;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_60) : "f"(sv1[28]));
                    if (lane == 28) {
                        hold_0 = mx_60;
                    }
                    float mx_61;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_61) : "f"(sv1[29]));
                    if (lane == 29) {
                        hold_0 = mx_61;
                    }
                    float mx_62;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_62) : "f"(sv1[30]));
                    if (lane == 30) {
                        hold_0 = mx_62;
                    }
                    float mx_63;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_63) : "f"(sv1[31]));
                    if (lane == 31) {
                        hold_0 = mx_63;
                    }
                    int exch_slot_0 = ((0 + pair % 2) * 2 + 1) * 4 * 32 + warp_in_role * 32;
                    smem_exch[exch_slot_0 + lane] = hold_0;
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int grow_any = 0;
                    float alpha[4];
                    unsigned int grown_mask[4];
                    alpha[0] = 1.0f;
                    grown_mask[0] = 0;
                    alpha[1] = 1.0f;
                    grown_mask[1] = 0;
                    alpha[2] = 1.0f;
                    grown_mask[2] = 0;
                    alpha[3] = 1.0f;
                    grown_mask[3] = 0;
                    int base_0 = ((0 + pair % 2) * 2 + 0) * 4 * 32 + 0;
                    float m_s_0;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_0) : "f"(smem_exch[base_0 + lane]), "f"(smem_exch[base_0 + 32 + lane]));
                    float cand_0;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_0) : "f"(o_hold[0]), "f"(m_s_0));
                    int moved_0 = 0;
                    if ((cand_0 - o_hold[0]) * softmax_scale_log2 > tau) {
                        moved_0 = 1;
                    }
                    unsigned int bal_0 = __ballot_sync(0xFFFFFFFF, moved_0 != 0);
                    if ((bal_0 >> (unsigned int)grp_shift & grp_mask) != 0) {
                        if (o_hold[0] > -CUDART_INF_F) {
                            float _exp2_0 = approx_exp2(softmax_scale_log2 * (o_hold[0] - cand_0));
                            alpha[0] = _exp2_0;
                        }
                        o_hold[0] = cand_0;
                        float safe_0 = ((cand_0 == -CUDART_INF_F) ? 0.0f : cand_0);
                        ms_hold[0] = safe_0 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_1 = __ballot_sync(0xFFFFFFFF, (bal_0 >> (unsigned int)grp_shift & grp_mask) != 0);
                    grown_mask[0] = _warp_vote_1;
                    if (grown_mask[0] != 0) {
                        grow_any = 1;
                    }
                    int base_1 = ((0 + pair % 2) * 2 + 0) * 4 * 32 + 64;
                    float m_s_1;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_1) : "f"(smem_exch[base_1 + lane]), "f"(smem_exch[base_1 + 32 + lane]));
                    float cand_1;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_1) : "f"(o_hold[1]), "f"(m_s_1));
                    int moved_1 = 0;
                    if ((cand_1 - o_hold[1]) * softmax_scale_log2 > tau) {
                        moved_1 = 1;
                    }
                    unsigned int bal_1 = __ballot_sync(0xFFFFFFFF, moved_1 != 0);
                    if ((bal_1 >> (unsigned int)grp_shift & grp_mask) != 0) {
                        if (o_hold[1] > -CUDART_INF_F) {
                            float _exp2_2 = approx_exp2(softmax_scale_log2 * (o_hold[1] - cand_1));
                            alpha[1] = _exp2_2;
                        }
                        o_hold[1] = cand_1;
                        float safe_1 = ((cand_1 == -CUDART_INF_F) ? 0.0f : cand_1);
                        ms_hold[1] = safe_1 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_3 = __ballot_sync(0xFFFFFFFF, (bal_1 >> (unsigned int)grp_shift & grp_mask) != 0);
                    grown_mask[1] = _warp_vote_3;
                    if (grown_mask[1] != 0) {
                        grow_any = 1;
                    }
                    int base_2 = ((0 + pair % 2) * 2 + 1) * 4 * 32 + 0;
                    float m_s_2;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_2) : "f"(smem_exch[base_2 + lane]), "f"(smem_exch[base_2 + 32 + lane]));
                    float cand_2;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_2) : "f"(o_hold[2]), "f"(m_s_2));
                    int moved_2 = 0;
                    if ((cand_2 - o_hold[2]) * softmax_scale_log2 > tau) {
                        moved_2 = 1;
                    }
                    unsigned int bal_2 = __ballot_sync(0xFFFFFFFF, moved_2 != 0);
                    if ((bal_2 >> (unsigned int)grp_shift & grp_mask) != 0) {
                        if (o_hold[2] > -CUDART_INF_F) {
                            float _exp2_4 = approx_exp2(softmax_scale_log2 * (o_hold[2] - cand_2));
                            alpha[2] = _exp2_4;
                        }
                        o_hold[2] = cand_2;
                        float safe_2 = ((cand_2 == -CUDART_INF_F) ? 0.0f : cand_2);
                        ms_hold[2] = safe_2 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_5 = __ballot_sync(0xFFFFFFFF, (bal_2 >> (unsigned int)grp_shift & grp_mask) != 0);
                    grown_mask[2] = _warp_vote_5;
                    if (grown_mask[2] != 0) {
                        grow_any = 1;
                    }
                    int base_3 = ((0 + pair % 2) * 2 + 1) * 4 * 32 + 64;
                    float m_s_3;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_3) : "f"(smem_exch[base_3 + lane]), "f"(smem_exch[base_3 + 32 + lane]));
                    float cand_3;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_3) : "f"(o_hold[3]), "f"(m_s_3));
                    int moved_3 = 0;
                    if ((cand_3 - o_hold[3]) * softmax_scale_log2 > tau) {
                        moved_3 = 1;
                    }
                    unsigned int bal_3 = __ballot_sync(0xFFFFFFFF, moved_3 != 0);
                    if ((bal_3 >> (unsigned int)grp_shift & grp_mask) != 0) {
                        if (o_hold[3] > -CUDART_INF_F) {
                            float _exp2_6 = approx_exp2(softmax_scale_log2 * (o_hold[3] - cand_3));
                            alpha[3] = _exp2_6;
                        }
                        o_hold[3] = cand_3;
                        float safe_3 = ((cand_3 == -CUDART_INF_F) ? 0.0f : cand_3);
                        ms_hold[3] = safe_3 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_7 = __ballot_sync(0xFFFFFFFF, (bal_3 >> (unsigned int)grp_shift & grp_mask) != 0);
                    grown_mask[3] = _warp_vote_7;
                    if (grown_mask[3] != 0) {
                        grow_any = 1;
                    }
                    if (grow_any != 0) {
                        float a_sel_0 = ((own_half == 0) ? alpha[0] : alpha[1]);
                        float _shfl_sync_8 = __shfl_sync(0xFFFFFFFF, a_sel_0, 0);
                        lane_sum[0] = lane_sum[0] * _shfl_sync_8;
                        float _shfl_sync_9 = __shfl_sync(0xFFFFFFFF, a_sel_0, 1);
                        lane_sum[1] = lane_sum[1] * _shfl_sync_9;
                        float _shfl_sync_10 = __shfl_sync(0xFFFFFFFF, a_sel_0, 2);
                        lane_sum[2] = lane_sum[2] * _shfl_sync_10;
                        float _shfl_sync_11 = __shfl_sync(0xFFFFFFFF, a_sel_0, 3);
                        lane_sum[3] = lane_sum[3] * _shfl_sync_11;
                        float _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, a_sel_0, 4);
                        lane_sum[4] = lane_sum[4] * _shfl_sync_12;
                        float _shfl_sync_13 = __shfl_sync(0xFFFFFFFF, a_sel_0, 5);
                        lane_sum[5] = lane_sum[5] * _shfl_sync_13;
                        float _shfl_sync_14 = __shfl_sync(0xFFFFFFFF, a_sel_0, 6);
                        lane_sum[6] = lane_sum[6] * _shfl_sync_14;
                        float _shfl_sync_15 = __shfl_sync(0xFFFFFFFF, a_sel_0, 7);
                        lane_sum[7] = lane_sum[7] * _shfl_sync_15;
                        float _shfl_sync_16 = __shfl_sync(0xFFFFFFFF, a_sel_0, 8);
                        lane_sum[8] = lane_sum[8] * _shfl_sync_16;
                        float _shfl_sync_17 = __shfl_sync(0xFFFFFFFF, a_sel_0, 9);
                        lane_sum[9] = lane_sum[9] * _shfl_sync_17;
                        float _shfl_sync_18 = __shfl_sync(0xFFFFFFFF, a_sel_0, 10);
                        lane_sum[10] = lane_sum[10] * _shfl_sync_18;
                        float _shfl_sync_19 = __shfl_sync(0xFFFFFFFF, a_sel_0, 11);
                        lane_sum[11] = lane_sum[11] * _shfl_sync_19;
                        float _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, a_sel_0, 12);
                        lane_sum[12] = lane_sum[12] * _shfl_sync_20;
                        float _shfl_sync_21 = __shfl_sync(0xFFFFFFFF, a_sel_0, 13);
                        lane_sum[13] = lane_sum[13] * _shfl_sync_21;
                        float _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, a_sel_0, 14);
                        lane_sum[14] = lane_sum[14] * _shfl_sync_22;
                        float _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, a_sel_0, 15);
                        lane_sum[15] = lane_sum[15] * _shfl_sync_23;
                        float _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, a_sel_0, 16);
                        lane_sum[16] = lane_sum[16] * _shfl_sync_24;
                        float _shfl_sync_25 = __shfl_sync(0xFFFFFFFF, a_sel_0, 17);
                        lane_sum[17] = lane_sum[17] * _shfl_sync_25;
                        float _shfl_sync_26 = __shfl_sync(0xFFFFFFFF, a_sel_0, 18);
                        lane_sum[18] = lane_sum[18] * _shfl_sync_26;
                        float _shfl_sync_27 = __shfl_sync(0xFFFFFFFF, a_sel_0, 19);
                        lane_sum[19] = lane_sum[19] * _shfl_sync_27;
                        float _shfl_sync_28 = __shfl_sync(0xFFFFFFFF, a_sel_0, 20);
                        lane_sum[20] = lane_sum[20] * _shfl_sync_28;
                        float _shfl_sync_29 = __shfl_sync(0xFFFFFFFF, a_sel_0, 21);
                        lane_sum[21] = lane_sum[21] * _shfl_sync_29;
                        float _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, a_sel_0, 22);
                        lane_sum[22] = lane_sum[22] * _shfl_sync_30;
                        float _shfl_sync_31 = __shfl_sync(0xFFFFFFFF, a_sel_0, 23);
                        lane_sum[23] = lane_sum[23] * _shfl_sync_31;
                        float _shfl_sync_32 = __shfl_sync(0xFFFFFFFF, a_sel_0, 24);
                        lane_sum[24] = lane_sum[24] * _shfl_sync_32;
                        float _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, a_sel_0, 25);
                        lane_sum[25] = lane_sum[25] * _shfl_sync_33;
                        float _shfl_sync_34 = __shfl_sync(0xFFFFFFFF, a_sel_0, 26);
                        lane_sum[26] = lane_sum[26] * _shfl_sync_34;
                        float _shfl_sync_35 = __shfl_sync(0xFFFFFFFF, a_sel_0, 27);
                        lane_sum[27] = lane_sum[27] * _shfl_sync_35;
                        float _shfl_sync_36 = __shfl_sync(0xFFFFFFFF, a_sel_0, 28);
                        lane_sum[28] = lane_sum[28] * _shfl_sync_36;
                        float _shfl_sync_37 = __shfl_sync(0xFFFFFFFF, a_sel_0, 29);
                        lane_sum[29] = lane_sum[29] * _shfl_sync_37;
                        float _shfl_sync_38 = __shfl_sync(0xFFFFFFFF, a_sel_0, 30);
                        lane_sum[30] = lane_sum[30] * _shfl_sync_38;
                        float _shfl_sync_39 = __shfl_sync(0xFFFFFFFF, a_sel_0, 31);
                        lane_sum[31] = lane_sum[31] * _shfl_sync_39;
                        float a_sel_1 = ((own_half == 0) ? alpha[2] : alpha[3]);
                        float _shfl_sync_40 = __shfl_sync(0xFFFFFFFF, a_sel_1, 0);
                        lane_sum[32] = lane_sum[32] * _shfl_sync_40;
                        float _shfl_sync_41 = __shfl_sync(0xFFFFFFFF, a_sel_1, 1);
                        lane_sum[33] = lane_sum[33] * _shfl_sync_41;
                        float _shfl_sync_42 = __shfl_sync(0xFFFFFFFF, a_sel_1, 2);
                        lane_sum[34] = lane_sum[34] * _shfl_sync_42;
                        float _shfl_sync_43 = __shfl_sync(0xFFFFFFFF, a_sel_1, 3);
                        lane_sum[35] = lane_sum[35] * _shfl_sync_43;
                        float _shfl_sync_44 = __shfl_sync(0xFFFFFFFF, a_sel_1, 4);
                        lane_sum[36] = lane_sum[36] * _shfl_sync_44;
                        float _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, a_sel_1, 5);
                        lane_sum[37] = lane_sum[37] * _shfl_sync_45;
                        float _shfl_sync_46 = __shfl_sync(0xFFFFFFFF, a_sel_1, 6);
                        lane_sum[38] = lane_sum[38] * _shfl_sync_46;
                        float _shfl_sync_47 = __shfl_sync(0xFFFFFFFF, a_sel_1, 7);
                        lane_sum[39] = lane_sum[39] * _shfl_sync_47;
                        float _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, a_sel_1, 8);
                        lane_sum[40] = lane_sum[40] * _shfl_sync_48;
                        float _shfl_sync_49 = __shfl_sync(0xFFFFFFFF, a_sel_1, 9);
                        lane_sum[41] = lane_sum[41] * _shfl_sync_49;
                        float _shfl_sync_50 = __shfl_sync(0xFFFFFFFF, a_sel_1, 10);
                        lane_sum[42] = lane_sum[42] * _shfl_sync_50;
                        float _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, a_sel_1, 11);
                        lane_sum[43] = lane_sum[43] * _shfl_sync_51;
                        float _shfl_sync_52 = __shfl_sync(0xFFFFFFFF, a_sel_1, 12);
                        lane_sum[44] = lane_sum[44] * _shfl_sync_52;
                        float _shfl_sync_53 = __shfl_sync(0xFFFFFFFF, a_sel_1, 13);
                        lane_sum[45] = lane_sum[45] * _shfl_sync_53;
                        float _shfl_sync_54 = __shfl_sync(0xFFFFFFFF, a_sel_1, 14);
                        lane_sum[46] = lane_sum[46] * _shfl_sync_54;
                        float _shfl_sync_55 = __shfl_sync(0xFFFFFFFF, a_sel_1, 15);
                        lane_sum[47] = lane_sum[47] * _shfl_sync_55;
                        float _shfl_sync_56 = __shfl_sync(0xFFFFFFFF, a_sel_1, 16);
                        lane_sum[48] = lane_sum[48] * _shfl_sync_56;
                        float _shfl_sync_57 = __shfl_sync(0xFFFFFFFF, a_sel_1, 17);
                        lane_sum[49] = lane_sum[49] * _shfl_sync_57;
                        float _shfl_sync_58 = __shfl_sync(0xFFFFFFFF, a_sel_1, 18);
                        lane_sum[50] = lane_sum[50] * _shfl_sync_58;
                        float _shfl_sync_59 = __shfl_sync(0xFFFFFFFF, a_sel_1, 19);
                        lane_sum[51] = lane_sum[51] * _shfl_sync_59;
                        float _shfl_sync_60 = __shfl_sync(0xFFFFFFFF, a_sel_1, 20);
                        lane_sum[52] = lane_sum[52] * _shfl_sync_60;
                        float _shfl_sync_61 = __shfl_sync(0xFFFFFFFF, a_sel_1, 21);
                        lane_sum[53] = lane_sum[53] * _shfl_sync_61;
                        float _shfl_sync_62 = __shfl_sync(0xFFFFFFFF, a_sel_1, 22);
                        lane_sum[54] = lane_sum[54] * _shfl_sync_62;
                        float _shfl_sync_63 = __shfl_sync(0xFFFFFFFF, a_sel_1, 23);
                        lane_sum[55] = lane_sum[55] * _shfl_sync_63;
                        float _shfl_sync_64 = __shfl_sync(0xFFFFFFFF, a_sel_1, 24);
                        lane_sum[56] = lane_sum[56] * _shfl_sync_64;
                        float _shfl_sync_65 = __shfl_sync(0xFFFFFFFF, a_sel_1, 25);
                        lane_sum[57] = lane_sum[57] * _shfl_sync_65;
                        float _shfl_sync_66 = __shfl_sync(0xFFFFFFFF, a_sel_1, 26);
                        lane_sum[58] = lane_sum[58] * _shfl_sync_66;
                        float _shfl_sync_67 = __shfl_sync(0xFFFFFFFF, a_sel_1, 27);
                        lane_sum[59] = lane_sum[59] * _shfl_sync_67;
                        float _shfl_sync_68 = __shfl_sync(0xFFFFFFFF, a_sel_1, 28);
                        lane_sum[60] = lane_sum[60] * _shfl_sync_68;
                        float _shfl_sync_69 = __shfl_sync(0xFFFFFFFF, a_sel_1, 29);
                        lane_sum[61] = lane_sum[61] * _shfl_sync_69;
                        float _shfl_sync_70 = __shfl_sync(0xFFFFFFFF, a_sel_1, 30);
                        lane_sum[62] = lane_sum[62] * _shfl_sync_70;
                        float _shfl_sync_71 = __shfl_sync(0xFFFFFFFF, a_sel_1, 31);
                        lane_sum[63] = lane_sum[63] * _shfl_sync_71;
                        if (pair > 0) {
                            int _stage_pv_done = (int)((unsigned int)(gpair - 1) % 8);
                            int _parity_pv_done = (int)((unsigned int)(gpair - 1) / 8 % 2);
                            mbarrier_wait(pv_done_addr + (_stage_pv_done) * 8, _parity_pv_done);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (grown_mask[0] != 0) {
                                int o_addr_0 = taddr + 256 + (unsigned int)lane_base;
                                float o_acc_0[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_0[0]), "=f"(o_acc_0[1]), "=f"(o_acc_0[2]), "=f"(o_acc_0[3]), "=f"(o_acc_0[4]), "=f"(o_acc_0[5]), "=f"(o_acc_0[6]), "=f"(o_acc_0[7]), "=f"(o_acc_0[8]), "=f"(o_acc_0[9]), "=f"(o_acc_0[10]), "=f"(o_acc_0[11]), "=f"(o_acc_0[12]), "=f"(o_acc_0[13]), "=f"(o_acc_0[14]), "=f"(o_acc_0[15]), "=f"(o_acc_0[16]), "=f"(o_acc_0[17]), "=f"(o_acc_0[18]), "=f"(o_acc_0[19]), "=f"(o_acc_0[20]), "=f"(o_acc_0[21]), "=f"(o_acc_0[22]), "=f"(o_acc_0[23]), "=f"(o_acc_0[24]), "=f"(o_acc_0[25]), "=f"(o_acc_0[26]), "=f"(o_acc_0[27]), "=f"(o_acc_0[28]), "=f"(o_acc_0[29]), "=f"(o_acc_0[30]), "=f"(o_acc_0[31])
                                    : "r"(o_addr_0));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_72 = __shfl_sync(0xFFFFFFFF, alpha[0], 0);
                                o_acc_0[0] = o_acc_0[0] * _shfl_sync_72;
                                float _shfl_sync_73 = __shfl_sync(0xFFFFFFFF, alpha[0], 1);
                                o_acc_0[1] = o_acc_0[1] * _shfl_sync_73;
                                float _shfl_sync_74 = __shfl_sync(0xFFFFFFFF, alpha[0], 2);
                                o_acc_0[2] = o_acc_0[2] * _shfl_sync_74;
                                float _shfl_sync_75 = __shfl_sync(0xFFFFFFFF, alpha[0], 3);
                                o_acc_0[3] = o_acc_0[3] * _shfl_sync_75;
                                float _shfl_sync_76 = __shfl_sync(0xFFFFFFFF, alpha[0], 4);
                                o_acc_0[4] = o_acc_0[4] * _shfl_sync_76;
                                float _shfl_sync_77 = __shfl_sync(0xFFFFFFFF, alpha[0], 5);
                                o_acc_0[5] = o_acc_0[5] * _shfl_sync_77;
                                float _shfl_sync_78 = __shfl_sync(0xFFFFFFFF, alpha[0], 6);
                                o_acc_0[6] = o_acc_0[6] * _shfl_sync_78;
                                float _shfl_sync_79 = __shfl_sync(0xFFFFFFFF, alpha[0], 7);
                                o_acc_0[7] = o_acc_0[7] * _shfl_sync_79;
                                float _shfl_sync_80 = __shfl_sync(0xFFFFFFFF, alpha[0], 8);
                                o_acc_0[8] = o_acc_0[8] * _shfl_sync_80;
                                float _shfl_sync_81 = __shfl_sync(0xFFFFFFFF, alpha[0], 9);
                                o_acc_0[9] = o_acc_0[9] * _shfl_sync_81;
                                float _shfl_sync_82 = __shfl_sync(0xFFFFFFFF, alpha[0], 10);
                                o_acc_0[10] = o_acc_0[10] * _shfl_sync_82;
                                float _shfl_sync_83 = __shfl_sync(0xFFFFFFFF, alpha[0], 11);
                                o_acc_0[11] = o_acc_0[11] * _shfl_sync_83;
                                float _shfl_sync_84 = __shfl_sync(0xFFFFFFFF, alpha[0], 12);
                                o_acc_0[12] = o_acc_0[12] * _shfl_sync_84;
                                float _shfl_sync_85 = __shfl_sync(0xFFFFFFFF, alpha[0], 13);
                                o_acc_0[13] = o_acc_0[13] * _shfl_sync_85;
                                float _shfl_sync_86 = __shfl_sync(0xFFFFFFFF, alpha[0], 14);
                                o_acc_0[14] = o_acc_0[14] * _shfl_sync_86;
                                float _shfl_sync_87 = __shfl_sync(0xFFFFFFFF, alpha[0], 15);
                                o_acc_0[15] = o_acc_0[15] * _shfl_sync_87;
                                float _shfl_sync_88 = __shfl_sync(0xFFFFFFFF, alpha[0], 16);
                                o_acc_0[16] = o_acc_0[16] * _shfl_sync_88;
                                float _shfl_sync_89 = __shfl_sync(0xFFFFFFFF, alpha[0], 17);
                                o_acc_0[17] = o_acc_0[17] * _shfl_sync_89;
                                float _shfl_sync_90 = __shfl_sync(0xFFFFFFFF, alpha[0], 18);
                                o_acc_0[18] = o_acc_0[18] * _shfl_sync_90;
                                float _shfl_sync_91 = __shfl_sync(0xFFFFFFFF, alpha[0], 19);
                                o_acc_0[19] = o_acc_0[19] * _shfl_sync_91;
                                float _shfl_sync_92 = __shfl_sync(0xFFFFFFFF, alpha[0], 20);
                                o_acc_0[20] = o_acc_0[20] * _shfl_sync_92;
                                float _shfl_sync_93 = __shfl_sync(0xFFFFFFFF, alpha[0], 21);
                                o_acc_0[21] = o_acc_0[21] * _shfl_sync_93;
                                float _shfl_sync_94 = __shfl_sync(0xFFFFFFFF, alpha[0], 22);
                                o_acc_0[22] = o_acc_0[22] * _shfl_sync_94;
                                float _shfl_sync_95 = __shfl_sync(0xFFFFFFFF, alpha[0], 23);
                                o_acc_0[23] = o_acc_0[23] * _shfl_sync_95;
                                float _shfl_sync_96 = __shfl_sync(0xFFFFFFFF, alpha[0], 24);
                                o_acc_0[24] = o_acc_0[24] * _shfl_sync_96;
                                float _shfl_sync_97 = __shfl_sync(0xFFFFFFFF, alpha[0], 25);
                                o_acc_0[25] = o_acc_0[25] * _shfl_sync_97;
                                float _shfl_sync_98 = __shfl_sync(0xFFFFFFFF, alpha[0], 26);
                                o_acc_0[26] = o_acc_0[26] * _shfl_sync_98;
                                float _shfl_sync_99 = __shfl_sync(0xFFFFFFFF, alpha[0], 27);
                                o_acc_0[27] = o_acc_0[27] * _shfl_sync_99;
                                float _shfl_sync_100 = __shfl_sync(0xFFFFFFFF, alpha[0], 28);
                                o_acc_0[28] = o_acc_0[28] * _shfl_sync_100;
                                float _shfl_sync_101 = __shfl_sync(0xFFFFFFFF, alpha[0], 29);
                                o_acc_0[29] = o_acc_0[29] * _shfl_sync_101;
                                float _shfl_sync_102 = __shfl_sync(0xFFFFFFFF, alpha[0], 30);
                                o_acc_0[30] = o_acc_0[30] * _shfl_sync_102;
                                float _shfl_sync_103 = __shfl_sync(0xFFFFFFFF, alpha[0], 31);
                                o_acc_0[31] = o_acc_0[31] * _shfl_sync_103;
                                tmem_st_x32_f32(o_addr_0, o_acc_0);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if (grown_mask[1] != 0) {
                                int o_addr_1 = taddr + 320 + (unsigned int)lane_base;
                                float o_acc_1[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_1[0]), "=f"(o_acc_1[1]), "=f"(o_acc_1[2]), "=f"(o_acc_1[3]), "=f"(o_acc_1[4]), "=f"(o_acc_1[5]), "=f"(o_acc_1[6]), "=f"(o_acc_1[7]), "=f"(o_acc_1[8]), "=f"(o_acc_1[9]), "=f"(o_acc_1[10]), "=f"(o_acc_1[11]), "=f"(o_acc_1[12]), "=f"(o_acc_1[13]), "=f"(o_acc_1[14]), "=f"(o_acc_1[15]), "=f"(o_acc_1[16]), "=f"(o_acc_1[17]), "=f"(o_acc_1[18]), "=f"(o_acc_1[19]), "=f"(o_acc_1[20]), "=f"(o_acc_1[21]), "=f"(o_acc_1[22]), "=f"(o_acc_1[23]), "=f"(o_acc_1[24]), "=f"(o_acc_1[25]), "=f"(o_acc_1[26]), "=f"(o_acc_1[27]), "=f"(o_acc_1[28]), "=f"(o_acc_1[29]), "=f"(o_acc_1[30]), "=f"(o_acc_1[31])
                                    : "r"(o_addr_1));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_104 = __shfl_sync(0xFFFFFFFF, alpha[1], 0);
                                o_acc_1[0] = o_acc_1[0] * _shfl_sync_104;
                                float _shfl_sync_105 = __shfl_sync(0xFFFFFFFF, alpha[1], 1);
                                o_acc_1[1] = o_acc_1[1] * _shfl_sync_105;
                                float _shfl_sync_106 = __shfl_sync(0xFFFFFFFF, alpha[1], 2);
                                o_acc_1[2] = o_acc_1[2] * _shfl_sync_106;
                                float _shfl_sync_107 = __shfl_sync(0xFFFFFFFF, alpha[1], 3);
                                o_acc_1[3] = o_acc_1[3] * _shfl_sync_107;
                                float _shfl_sync_108 = __shfl_sync(0xFFFFFFFF, alpha[1], 4);
                                o_acc_1[4] = o_acc_1[4] * _shfl_sync_108;
                                float _shfl_sync_109 = __shfl_sync(0xFFFFFFFF, alpha[1], 5);
                                o_acc_1[5] = o_acc_1[5] * _shfl_sync_109;
                                float _shfl_sync_110 = __shfl_sync(0xFFFFFFFF, alpha[1], 6);
                                o_acc_1[6] = o_acc_1[6] * _shfl_sync_110;
                                float _shfl_sync_111 = __shfl_sync(0xFFFFFFFF, alpha[1], 7);
                                o_acc_1[7] = o_acc_1[7] * _shfl_sync_111;
                                float _shfl_sync_112 = __shfl_sync(0xFFFFFFFF, alpha[1], 8);
                                o_acc_1[8] = o_acc_1[8] * _shfl_sync_112;
                                float _shfl_sync_113 = __shfl_sync(0xFFFFFFFF, alpha[1], 9);
                                o_acc_1[9] = o_acc_1[9] * _shfl_sync_113;
                                float _shfl_sync_114 = __shfl_sync(0xFFFFFFFF, alpha[1], 10);
                                o_acc_1[10] = o_acc_1[10] * _shfl_sync_114;
                                float _shfl_sync_115 = __shfl_sync(0xFFFFFFFF, alpha[1], 11);
                                o_acc_1[11] = o_acc_1[11] * _shfl_sync_115;
                                float _shfl_sync_116 = __shfl_sync(0xFFFFFFFF, alpha[1], 12);
                                o_acc_1[12] = o_acc_1[12] * _shfl_sync_116;
                                float _shfl_sync_117 = __shfl_sync(0xFFFFFFFF, alpha[1], 13);
                                o_acc_1[13] = o_acc_1[13] * _shfl_sync_117;
                                float _shfl_sync_118 = __shfl_sync(0xFFFFFFFF, alpha[1], 14);
                                o_acc_1[14] = o_acc_1[14] * _shfl_sync_118;
                                float _shfl_sync_119 = __shfl_sync(0xFFFFFFFF, alpha[1], 15);
                                o_acc_1[15] = o_acc_1[15] * _shfl_sync_119;
                                float _shfl_sync_120 = __shfl_sync(0xFFFFFFFF, alpha[1], 16);
                                o_acc_1[16] = o_acc_1[16] * _shfl_sync_120;
                                float _shfl_sync_121 = __shfl_sync(0xFFFFFFFF, alpha[1], 17);
                                o_acc_1[17] = o_acc_1[17] * _shfl_sync_121;
                                float _shfl_sync_122 = __shfl_sync(0xFFFFFFFF, alpha[1], 18);
                                o_acc_1[18] = o_acc_1[18] * _shfl_sync_122;
                                float _shfl_sync_123 = __shfl_sync(0xFFFFFFFF, alpha[1], 19);
                                o_acc_1[19] = o_acc_1[19] * _shfl_sync_123;
                                float _shfl_sync_124 = __shfl_sync(0xFFFFFFFF, alpha[1], 20);
                                o_acc_1[20] = o_acc_1[20] * _shfl_sync_124;
                                float _shfl_sync_125 = __shfl_sync(0xFFFFFFFF, alpha[1], 21);
                                o_acc_1[21] = o_acc_1[21] * _shfl_sync_125;
                                float _shfl_sync_126 = __shfl_sync(0xFFFFFFFF, alpha[1], 22);
                                o_acc_1[22] = o_acc_1[22] * _shfl_sync_126;
                                float _shfl_sync_127 = __shfl_sync(0xFFFFFFFF, alpha[1], 23);
                                o_acc_1[23] = o_acc_1[23] * _shfl_sync_127;
                                float _shfl_sync_128 = __shfl_sync(0xFFFFFFFF, alpha[1], 24);
                                o_acc_1[24] = o_acc_1[24] * _shfl_sync_128;
                                float _shfl_sync_129 = __shfl_sync(0xFFFFFFFF, alpha[1], 25);
                                o_acc_1[25] = o_acc_1[25] * _shfl_sync_129;
                                float _shfl_sync_130 = __shfl_sync(0xFFFFFFFF, alpha[1], 26);
                                o_acc_1[26] = o_acc_1[26] * _shfl_sync_130;
                                float _shfl_sync_131 = __shfl_sync(0xFFFFFFFF, alpha[1], 27);
                                o_acc_1[27] = o_acc_1[27] * _shfl_sync_131;
                                float _shfl_sync_132 = __shfl_sync(0xFFFFFFFF, alpha[1], 28);
                                o_acc_1[28] = o_acc_1[28] * _shfl_sync_132;
                                float _shfl_sync_133 = __shfl_sync(0xFFFFFFFF, alpha[1], 29);
                                o_acc_1[29] = o_acc_1[29] * _shfl_sync_133;
                                float _shfl_sync_134 = __shfl_sync(0xFFFFFFFF, alpha[1], 30);
                                o_acc_1[30] = o_acc_1[30] * _shfl_sync_134;
                                float _shfl_sync_135 = __shfl_sync(0xFFFFFFFF, alpha[1], 31);
                                o_acc_1[31] = o_acc_1[31] * _shfl_sync_135;
                                tmem_st_x32_f32(o_addr_1, o_acc_1);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if (grown_mask[2] != 0) {
                                int o_addr_2 = taddr + 384 + (unsigned int)lane_base;
                                float o_acc_2[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_2[0]), "=f"(o_acc_2[1]), "=f"(o_acc_2[2]), "=f"(o_acc_2[3]), "=f"(o_acc_2[4]), "=f"(o_acc_2[5]), "=f"(o_acc_2[6]), "=f"(o_acc_2[7]), "=f"(o_acc_2[8]), "=f"(o_acc_2[9]), "=f"(o_acc_2[10]), "=f"(o_acc_2[11]), "=f"(o_acc_2[12]), "=f"(o_acc_2[13]), "=f"(o_acc_2[14]), "=f"(o_acc_2[15]), "=f"(o_acc_2[16]), "=f"(o_acc_2[17]), "=f"(o_acc_2[18]), "=f"(o_acc_2[19]), "=f"(o_acc_2[20]), "=f"(o_acc_2[21]), "=f"(o_acc_2[22]), "=f"(o_acc_2[23]), "=f"(o_acc_2[24]), "=f"(o_acc_2[25]), "=f"(o_acc_2[26]), "=f"(o_acc_2[27]), "=f"(o_acc_2[28]), "=f"(o_acc_2[29]), "=f"(o_acc_2[30]), "=f"(o_acc_2[31])
                                    : "r"(o_addr_2));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_136 = __shfl_sync(0xFFFFFFFF, alpha[2], 0);
                                o_acc_2[0] = o_acc_2[0] * _shfl_sync_136;
                                float _shfl_sync_137 = __shfl_sync(0xFFFFFFFF, alpha[2], 1);
                                o_acc_2[1] = o_acc_2[1] * _shfl_sync_137;
                                float _shfl_sync_138 = __shfl_sync(0xFFFFFFFF, alpha[2], 2);
                                o_acc_2[2] = o_acc_2[2] * _shfl_sync_138;
                                float _shfl_sync_139 = __shfl_sync(0xFFFFFFFF, alpha[2], 3);
                                o_acc_2[3] = o_acc_2[3] * _shfl_sync_139;
                                float _shfl_sync_140 = __shfl_sync(0xFFFFFFFF, alpha[2], 4);
                                o_acc_2[4] = o_acc_2[4] * _shfl_sync_140;
                                float _shfl_sync_141 = __shfl_sync(0xFFFFFFFF, alpha[2], 5);
                                o_acc_2[5] = o_acc_2[5] * _shfl_sync_141;
                                float _shfl_sync_142 = __shfl_sync(0xFFFFFFFF, alpha[2], 6);
                                o_acc_2[6] = o_acc_2[6] * _shfl_sync_142;
                                float _shfl_sync_143 = __shfl_sync(0xFFFFFFFF, alpha[2], 7);
                                o_acc_2[7] = o_acc_2[7] * _shfl_sync_143;
                                float _shfl_sync_144 = __shfl_sync(0xFFFFFFFF, alpha[2], 8);
                                o_acc_2[8] = o_acc_2[8] * _shfl_sync_144;
                                float _shfl_sync_145 = __shfl_sync(0xFFFFFFFF, alpha[2], 9);
                                o_acc_2[9] = o_acc_2[9] * _shfl_sync_145;
                                float _shfl_sync_146 = __shfl_sync(0xFFFFFFFF, alpha[2], 10);
                                o_acc_2[10] = o_acc_2[10] * _shfl_sync_146;
                                float _shfl_sync_147 = __shfl_sync(0xFFFFFFFF, alpha[2], 11);
                                o_acc_2[11] = o_acc_2[11] * _shfl_sync_147;
                                float _shfl_sync_148 = __shfl_sync(0xFFFFFFFF, alpha[2], 12);
                                o_acc_2[12] = o_acc_2[12] * _shfl_sync_148;
                                float _shfl_sync_149 = __shfl_sync(0xFFFFFFFF, alpha[2], 13);
                                o_acc_2[13] = o_acc_2[13] * _shfl_sync_149;
                                float _shfl_sync_150 = __shfl_sync(0xFFFFFFFF, alpha[2], 14);
                                o_acc_2[14] = o_acc_2[14] * _shfl_sync_150;
                                float _shfl_sync_151 = __shfl_sync(0xFFFFFFFF, alpha[2], 15);
                                o_acc_2[15] = o_acc_2[15] * _shfl_sync_151;
                                float _shfl_sync_152 = __shfl_sync(0xFFFFFFFF, alpha[2], 16);
                                o_acc_2[16] = o_acc_2[16] * _shfl_sync_152;
                                float _shfl_sync_153 = __shfl_sync(0xFFFFFFFF, alpha[2], 17);
                                o_acc_2[17] = o_acc_2[17] * _shfl_sync_153;
                                float _shfl_sync_154 = __shfl_sync(0xFFFFFFFF, alpha[2], 18);
                                o_acc_2[18] = o_acc_2[18] * _shfl_sync_154;
                                float _shfl_sync_155 = __shfl_sync(0xFFFFFFFF, alpha[2], 19);
                                o_acc_2[19] = o_acc_2[19] * _shfl_sync_155;
                                float _shfl_sync_156 = __shfl_sync(0xFFFFFFFF, alpha[2], 20);
                                o_acc_2[20] = o_acc_2[20] * _shfl_sync_156;
                                float _shfl_sync_157 = __shfl_sync(0xFFFFFFFF, alpha[2], 21);
                                o_acc_2[21] = o_acc_2[21] * _shfl_sync_157;
                                float _shfl_sync_158 = __shfl_sync(0xFFFFFFFF, alpha[2], 22);
                                o_acc_2[22] = o_acc_2[22] * _shfl_sync_158;
                                float _shfl_sync_159 = __shfl_sync(0xFFFFFFFF, alpha[2], 23);
                                o_acc_2[23] = o_acc_2[23] * _shfl_sync_159;
                                float _shfl_sync_160 = __shfl_sync(0xFFFFFFFF, alpha[2], 24);
                                o_acc_2[24] = o_acc_2[24] * _shfl_sync_160;
                                float _shfl_sync_161 = __shfl_sync(0xFFFFFFFF, alpha[2], 25);
                                o_acc_2[25] = o_acc_2[25] * _shfl_sync_161;
                                float _shfl_sync_162 = __shfl_sync(0xFFFFFFFF, alpha[2], 26);
                                o_acc_2[26] = o_acc_2[26] * _shfl_sync_162;
                                float _shfl_sync_163 = __shfl_sync(0xFFFFFFFF, alpha[2], 27);
                                o_acc_2[27] = o_acc_2[27] * _shfl_sync_163;
                                float _shfl_sync_164 = __shfl_sync(0xFFFFFFFF, alpha[2], 28);
                                o_acc_2[28] = o_acc_2[28] * _shfl_sync_164;
                                float _shfl_sync_165 = __shfl_sync(0xFFFFFFFF, alpha[2], 29);
                                o_acc_2[29] = o_acc_2[29] * _shfl_sync_165;
                                float _shfl_sync_166 = __shfl_sync(0xFFFFFFFF, alpha[2], 30);
                                o_acc_2[30] = o_acc_2[30] * _shfl_sync_166;
                                float _shfl_sync_167 = __shfl_sync(0xFFFFFFFF, alpha[2], 31);
                                o_acc_2[31] = o_acc_2[31] * _shfl_sync_167;
                                tmem_st_x32_f32(o_addr_2, o_acc_2);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if (grown_mask[3] != 0) {
                                int o_addr_3 = taddr + 448 + (unsigned int)lane_base;
                                float o_acc_3[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_3[0]), "=f"(o_acc_3[1]), "=f"(o_acc_3[2]), "=f"(o_acc_3[3]), "=f"(o_acc_3[4]), "=f"(o_acc_3[5]), "=f"(o_acc_3[6]), "=f"(o_acc_3[7]), "=f"(o_acc_3[8]), "=f"(o_acc_3[9]), "=f"(o_acc_3[10]), "=f"(o_acc_3[11]), "=f"(o_acc_3[12]), "=f"(o_acc_3[13]), "=f"(o_acc_3[14]), "=f"(o_acc_3[15]), "=f"(o_acc_3[16]), "=f"(o_acc_3[17]), "=f"(o_acc_3[18]), "=f"(o_acc_3[19]), "=f"(o_acc_3[20]), "=f"(o_acc_3[21]), "=f"(o_acc_3[22]), "=f"(o_acc_3[23]), "=f"(o_acc_3[24]), "=f"(o_acc_3[25]), "=f"(o_acc_3[26]), "=f"(o_acc_3[27]), "=f"(o_acc_3[28]), "=f"(o_acc_3[29]), "=f"(o_acc_3[30]), "=f"(o_acc_3[31])
                                    : "r"(o_addr_3));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_168 = __shfl_sync(0xFFFFFFFF, alpha[3], 0);
                                o_acc_3[0] = o_acc_3[0] * _shfl_sync_168;
                                float _shfl_sync_169 = __shfl_sync(0xFFFFFFFF, alpha[3], 1);
                                o_acc_3[1] = o_acc_3[1] * _shfl_sync_169;
                                float _shfl_sync_170 = __shfl_sync(0xFFFFFFFF, alpha[3], 2);
                                o_acc_3[2] = o_acc_3[2] * _shfl_sync_170;
                                float _shfl_sync_171 = __shfl_sync(0xFFFFFFFF, alpha[3], 3);
                                o_acc_3[3] = o_acc_3[3] * _shfl_sync_171;
                                float _shfl_sync_172 = __shfl_sync(0xFFFFFFFF, alpha[3], 4);
                                o_acc_3[4] = o_acc_3[4] * _shfl_sync_172;
                                float _shfl_sync_173 = __shfl_sync(0xFFFFFFFF, alpha[3], 5);
                                o_acc_3[5] = o_acc_3[5] * _shfl_sync_173;
                                float _shfl_sync_174 = __shfl_sync(0xFFFFFFFF, alpha[3], 6);
                                o_acc_3[6] = o_acc_3[6] * _shfl_sync_174;
                                float _shfl_sync_175 = __shfl_sync(0xFFFFFFFF, alpha[3], 7);
                                o_acc_3[7] = o_acc_3[7] * _shfl_sync_175;
                                float _shfl_sync_176 = __shfl_sync(0xFFFFFFFF, alpha[3], 8);
                                o_acc_3[8] = o_acc_3[8] * _shfl_sync_176;
                                float _shfl_sync_177 = __shfl_sync(0xFFFFFFFF, alpha[3], 9);
                                o_acc_3[9] = o_acc_3[9] * _shfl_sync_177;
                                float _shfl_sync_178 = __shfl_sync(0xFFFFFFFF, alpha[3], 10);
                                o_acc_3[10] = o_acc_3[10] * _shfl_sync_178;
                                float _shfl_sync_179 = __shfl_sync(0xFFFFFFFF, alpha[3], 11);
                                o_acc_3[11] = o_acc_3[11] * _shfl_sync_179;
                                float _shfl_sync_180 = __shfl_sync(0xFFFFFFFF, alpha[3], 12);
                                o_acc_3[12] = o_acc_3[12] * _shfl_sync_180;
                                float _shfl_sync_181 = __shfl_sync(0xFFFFFFFF, alpha[3], 13);
                                o_acc_3[13] = o_acc_3[13] * _shfl_sync_181;
                                float _shfl_sync_182 = __shfl_sync(0xFFFFFFFF, alpha[3], 14);
                                o_acc_3[14] = o_acc_3[14] * _shfl_sync_182;
                                float _shfl_sync_183 = __shfl_sync(0xFFFFFFFF, alpha[3], 15);
                                o_acc_3[15] = o_acc_3[15] * _shfl_sync_183;
                                float _shfl_sync_184 = __shfl_sync(0xFFFFFFFF, alpha[3], 16);
                                o_acc_3[16] = o_acc_3[16] * _shfl_sync_184;
                                float _shfl_sync_185 = __shfl_sync(0xFFFFFFFF, alpha[3], 17);
                                o_acc_3[17] = o_acc_3[17] * _shfl_sync_185;
                                float _shfl_sync_186 = __shfl_sync(0xFFFFFFFF, alpha[3], 18);
                                o_acc_3[18] = o_acc_3[18] * _shfl_sync_186;
                                float _shfl_sync_187 = __shfl_sync(0xFFFFFFFF, alpha[3], 19);
                                o_acc_3[19] = o_acc_3[19] * _shfl_sync_187;
                                float _shfl_sync_188 = __shfl_sync(0xFFFFFFFF, alpha[3], 20);
                                o_acc_3[20] = o_acc_3[20] * _shfl_sync_188;
                                float _shfl_sync_189 = __shfl_sync(0xFFFFFFFF, alpha[3], 21);
                                o_acc_3[21] = o_acc_3[21] * _shfl_sync_189;
                                float _shfl_sync_190 = __shfl_sync(0xFFFFFFFF, alpha[3], 22);
                                o_acc_3[22] = o_acc_3[22] * _shfl_sync_190;
                                float _shfl_sync_191 = __shfl_sync(0xFFFFFFFF, alpha[3], 23);
                                o_acc_3[23] = o_acc_3[23] * _shfl_sync_191;
                                float _shfl_sync_192 = __shfl_sync(0xFFFFFFFF, alpha[3], 24);
                                o_acc_3[24] = o_acc_3[24] * _shfl_sync_192;
                                float _shfl_sync_193 = __shfl_sync(0xFFFFFFFF, alpha[3], 25);
                                o_acc_3[25] = o_acc_3[25] * _shfl_sync_193;
                                float _shfl_sync_194 = __shfl_sync(0xFFFFFFFF, alpha[3], 26);
                                o_acc_3[26] = o_acc_3[26] * _shfl_sync_194;
                                float _shfl_sync_195 = __shfl_sync(0xFFFFFFFF, alpha[3], 27);
                                o_acc_3[27] = o_acc_3[27] * _shfl_sync_195;
                                float _shfl_sync_196 = __shfl_sync(0xFFFFFFFF, alpha[3], 28);
                                o_acc_3[28] = o_acc_3[28] * _shfl_sync_196;
                                float _shfl_sync_197 = __shfl_sync(0xFFFFFFFF, alpha[3], 29);
                                o_acc_3[29] = o_acc_3[29] * _shfl_sync_197;
                                float _shfl_sync_198 = __shfl_sync(0xFFFFFFFF, alpha[3], 30);
                                o_acc_3[30] = o_acc_3[30] * _shfl_sync_198;
                                float _shfl_sync_199 = __shfl_sync(0xFFFFFFFF, alpha[3], 31);
                                o_acc_3[31] = o_acc_3[31] * _shfl_sync_199;
                                tmem_st_x32_f32(o_addr_3, o_acc_3);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                    }
                    if (gpair >= 4) {
                        int _stage_pv_done_1 = (int)((unsigned int)(gpair - 4) % 8);
                        int _parity_pv_done_1 = (int)((unsigned int)(gpair - 4) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_1) * 8, _parity_pv_done_1);
                    }
                    float ms_sel = ((own_half == 0) ? ms_hold[0] : ms_hold[1]);
                    float _shfl_sync_201 = __shfl_sync(0xFFFFFFFF, ms_sel, 0);
                    float _exp2_200 = approx_exp2(__fmaf_rn(sv0[0], softmax_scale_log2, -_shfl_sync_201));
                    sv0[0] = _exp2_200;
                    lane_sum[0] = lane_sum[0] + sv0[0];
                    float _shfl_sync_203 = __shfl_sync(0xFFFFFFFF, ms_sel, 1);
                    float _exp2_202 = approx_exp2(__fmaf_rn(sv0[1], softmax_scale_log2, -_shfl_sync_203));
                    sv0[1] = _exp2_202;
                    lane_sum[1] = lane_sum[1] + sv0[1];
                    float _shfl_sync_205 = __shfl_sync(0xFFFFFFFF, ms_sel, 2);
                    float _exp2_204 = approx_exp2(__fmaf_rn(sv0[2], softmax_scale_log2, -_shfl_sync_205));
                    sv0[2] = _exp2_204;
                    lane_sum[2] = lane_sum[2] + sv0[2];
                    float _shfl_sync_207 = __shfl_sync(0xFFFFFFFF, ms_sel, 3);
                    float _exp2_206 = approx_exp2(__fmaf_rn(sv0[3], softmax_scale_log2, -_shfl_sync_207));
                    sv0[3] = _exp2_206;
                    lane_sum[3] = lane_sum[3] + sv0[3];
                    float _shfl_sync_209 = __shfl_sync(0xFFFFFFFF, ms_sel, 4);
                    float _exp2_208 = approx_exp2(__fmaf_rn(sv0[4], softmax_scale_log2, -_shfl_sync_209));
                    sv0[4] = _exp2_208;
                    lane_sum[4] = lane_sum[4] + sv0[4];
                    float _shfl_sync_211 = __shfl_sync(0xFFFFFFFF, ms_sel, 5);
                    float _exp2_210 = approx_exp2(__fmaf_rn(sv0[5], softmax_scale_log2, -_shfl_sync_211));
                    sv0[5] = _exp2_210;
                    lane_sum[5] = lane_sum[5] + sv0[5];
                    float _shfl_sync_213 = __shfl_sync(0xFFFFFFFF, ms_sel, 6);
                    float _exp2_212 = approx_exp2(__fmaf_rn(sv0[6], softmax_scale_log2, -_shfl_sync_213));
                    sv0[6] = _exp2_212;
                    lane_sum[6] = lane_sum[6] + sv0[6];
                    float _shfl_sync_215 = __shfl_sync(0xFFFFFFFF, ms_sel, 7);
                    float _exp2_214 = approx_exp2(__fmaf_rn(sv0[7], softmax_scale_log2, -_shfl_sync_215));
                    sv0[7] = _exp2_214;
                    lane_sum[7] = lane_sum[7] + sv0[7];
                    float _shfl_sync_217 = __shfl_sync(0xFFFFFFFF, ms_sel, 8);
                    float _exp2_216 = approx_exp2(__fmaf_rn(sv0[8], softmax_scale_log2, -_shfl_sync_217));
                    sv0[8] = _exp2_216;
                    lane_sum[8] = lane_sum[8] + sv0[8];
                    float _shfl_sync_219 = __shfl_sync(0xFFFFFFFF, ms_sel, 9);
                    float _exp2_218 = approx_exp2(__fmaf_rn(sv0[9], softmax_scale_log2, -_shfl_sync_219));
                    sv0[9] = _exp2_218;
                    lane_sum[9] = lane_sum[9] + sv0[9];
                    float _shfl_sync_221 = __shfl_sync(0xFFFFFFFF, ms_sel, 10);
                    float _exp2_220 = approx_exp2(__fmaf_rn(sv0[10], softmax_scale_log2, -_shfl_sync_221));
                    sv0[10] = _exp2_220;
                    lane_sum[10] = lane_sum[10] + sv0[10];
                    float _shfl_sync_223 = __shfl_sync(0xFFFFFFFF, ms_sel, 11);
                    float _exp2_222 = approx_exp2(__fmaf_rn(sv0[11], softmax_scale_log2, -_shfl_sync_223));
                    sv0[11] = _exp2_222;
                    lane_sum[11] = lane_sum[11] + sv0[11];
                    float _shfl_sync_225 = __shfl_sync(0xFFFFFFFF, ms_sel, 12);
                    float _exp2_224 = approx_exp2(__fmaf_rn(sv0[12], softmax_scale_log2, -_shfl_sync_225));
                    sv0[12] = _exp2_224;
                    lane_sum[12] = lane_sum[12] + sv0[12];
                    float _shfl_sync_227 = __shfl_sync(0xFFFFFFFF, ms_sel, 13);
                    float _exp2_226 = approx_exp2(__fmaf_rn(sv0[13], softmax_scale_log2, -_shfl_sync_227));
                    sv0[13] = _exp2_226;
                    lane_sum[13] = lane_sum[13] + sv0[13];
                    float _shfl_sync_229 = __shfl_sync(0xFFFFFFFF, ms_sel, 14);
                    float _exp2_228 = approx_exp2(__fmaf_rn(sv0[14], softmax_scale_log2, -_shfl_sync_229));
                    sv0[14] = _exp2_228;
                    lane_sum[14] = lane_sum[14] + sv0[14];
                    float _shfl_sync_231 = __shfl_sync(0xFFFFFFFF, ms_sel, 15);
                    float _exp2_230 = approx_exp2(__fmaf_rn(sv0[15], softmax_scale_log2, -_shfl_sync_231));
                    sv0[15] = _exp2_230;
                    lane_sum[15] = lane_sum[15] + sv0[15];
                    float _shfl_sync_233 = __shfl_sync(0xFFFFFFFF, ms_sel, 16);
                    float _exp2_232 = approx_exp2(__fmaf_rn(sv0[16], softmax_scale_log2, -_shfl_sync_233));
                    sv0[16] = _exp2_232;
                    lane_sum[16] = lane_sum[16] + sv0[16];
                    float _shfl_sync_235 = __shfl_sync(0xFFFFFFFF, ms_sel, 17);
                    float _exp2_234 = approx_exp2(__fmaf_rn(sv0[17], softmax_scale_log2, -_shfl_sync_235));
                    sv0[17] = _exp2_234;
                    lane_sum[17] = lane_sum[17] + sv0[17];
                    float _shfl_sync_237 = __shfl_sync(0xFFFFFFFF, ms_sel, 18);
                    float _exp2_236 = approx_exp2(__fmaf_rn(sv0[18], softmax_scale_log2, -_shfl_sync_237));
                    sv0[18] = _exp2_236;
                    lane_sum[18] = lane_sum[18] + sv0[18];
                    float _shfl_sync_239 = __shfl_sync(0xFFFFFFFF, ms_sel, 19);
                    float _exp2_238 = approx_exp2(__fmaf_rn(sv0[19], softmax_scale_log2, -_shfl_sync_239));
                    sv0[19] = _exp2_238;
                    lane_sum[19] = lane_sum[19] + sv0[19];
                    float _shfl_sync_241 = __shfl_sync(0xFFFFFFFF, ms_sel, 20);
                    float _exp2_240 = approx_exp2(__fmaf_rn(sv0[20], softmax_scale_log2, -_shfl_sync_241));
                    sv0[20] = _exp2_240;
                    lane_sum[20] = lane_sum[20] + sv0[20];
                    float _shfl_sync_243 = __shfl_sync(0xFFFFFFFF, ms_sel, 21);
                    float _exp2_242 = approx_exp2(__fmaf_rn(sv0[21], softmax_scale_log2, -_shfl_sync_243));
                    sv0[21] = _exp2_242;
                    lane_sum[21] = lane_sum[21] + sv0[21];
                    float _shfl_sync_245 = __shfl_sync(0xFFFFFFFF, ms_sel, 22);
                    float _exp2_244 = approx_exp2(__fmaf_rn(sv0[22], softmax_scale_log2, -_shfl_sync_245));
                    sv0[22] = _exp2_244;
                    lane_sum[22] = lane_sum[22] + sv0[22];
                    float _shfl_sync_247 = __shfl_sync(0xFFFFFFFF, ms_sel, 23);
                    float _exp2_246 = approx_exp2(__fmaf_rn(sv0[23], softmax_scale_log2, -_shfl_sync_247));
                    sv0[23] = _exp2_246;
                    lane_sum[23] = lane_sum[23] + sv0[23];
                    float _shfl_sync_249 = __shfl_sync(0xFFFFFFFF, ms_sel, 24);
                    float _exp2_248 = approx_exp2(__fmaf_rn(sv0[24], softmax_scale_log2, -_shfl_sync_249));
                    sv0[24] = _exp2_248;
                    lane_sum[24] = lane_sum[24] + sv0[24];
                    float _shfl_sync_251 = __shfl_sync(0xFFFFFFFF, ms_sel, 25);
                    float _exp2_250 = approx_exp2(__fmaf_rn(sv0[25], softmax_scale_log2, -_shfl_sync_251));
                    sv0[25] = _exp2_250;
                    lane_sum[25] = lane_sum[25] + sv0[25];
                    float _shfl_sync_253 = __shfl_sync(0xFFFFFFFF, ms_sel, 26);
                    float _exp2_252 = approx_exp2(__fmaf_rn(sv0[26], softmax_scale_log2, -_shfl_sync_253));
                    sv0[26] = _exp2_252;
                    lane_sum[26] = lane_sum[26] + sv0[26];
                    float _shfl_sync_255 = __shfl_sync(0xFFFFFFFF, ms_sel, 27);
                    float _exp2_254 = approx_exp2(__fmaf_rn(sv0[27], softmax_scale_log2, -_shfl_sync_255));
                    sv0[27] = _exp2_254;
                    lane_sum[27] = lane_sum[27] + sv0[27];
                    float _shfl_sync_257 = __shfl_sync(0xFFFFFFFF, ms_sel, 28);
                    float _exp2_256 = approx_exp2(__fmaf_rn(sv0[28], softmax_scale_log2, -_shfl_sync_257));
                    sv0[28] = _exp2_256;
                    lane_sum[28] = lane_sum[28] + sv0[28];
                    float _shfl_sync_259 = __shfl_sync(0xFFFFFFFF, ms_sel, 29);
                    float _exp2_258 = approx_exp2(__fmaf_rn(sv0[29], softmax_scale_log2, -_shfl_sync_259));
                    sv0[29] = _exp2_258;
                    lane_sum[29] = lane_sum[29] + sv0[29];
                    float _shfl_sync_261 = __shfl_sync(0xFFFFFFFF, ms_sel, 30);
                    float _exp2_260 = approx_exp2(__fmaf_rn(sv0[30], softmax_scale_log2, -_shfl_sync_261));
                    sv0[30] = _exp2_260;
                    lane_sum[30] = lane_sum[30] + sv0[30];
                    float _shfl_sync_263 = __shfl_sync(0xFFFFFFFF, ms_sel, 31);
                    float _exp2_262 = approx_exp2(__fmaf_rn(sv0[31], softmax_scale_log2, -_shfl_sync_263));
                    sv0[31] = _exp2_262;
                    lane_sum[31] = lane_sum[31] + sv0[31];
                    unsigned int words[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[0]), "f"(sv0[1]),
                                               "f"(sv0[2]), "f"(sv0[3]));
                        words[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[4]), "f"(sv0[5]),
                                               "f"(sv0[6]), "f"(sv0[7]));
                        words[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[8]), "f"(sv0[9]),
                                               "f"(sv0[10]), "f"(sv0[11]));
                        words[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[12]), "f"(sv0[13]),
                                               "f"(sv0[14]), "f"(sv0[15]));
                        words[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[16]), "f"(sv0[17]),
                                               "f"(sv0[18]), "f"(sv0[19]));
                        words[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[20]), "f"(sv0[21]),
                                               "f"(sv0[22]), "f"(sv0[23]));
                        words[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[24]), "f"(sv0[25]),
                                               "f"(sv0[26]), "f"(sv0[27]));
                        words[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0[28]), "f"(sv0[29]),
                                               "f"(sv0[30]), "f"(sv0[31]));
                        words[7] = _packed;
                    }
                    int p_base = smem_p_addr + (unsigned int)((p_stage * 2 + 0) * 8192);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (key_row * 64 + 0 ^ (key_row * 64 + 0 >> 7 & 3) << 4))), "r"(words[0]), "r"(words[1]), "r"(words[2]), "r"(words[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (key_row * 64 + 16 ^ (key_row * 64 + 16 >> 7 & 3) << 4))), "r"(words[4]), "r"(words[5]), "r"(words[6]), "r"(words[7]) : "memory");
                    float ms_sel_0 = ((own_half == 0) ? ms_hold[2] : ms_hold[3]);
                    float _shfl_sync_265 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 0);
                    float _exp2_264 = approx_exp2(__fmaf_rn(sv1[0], softmax_scale_log2, -_shfl_sync_265));
                    sv1[0] = _exp2_264;
                    lane_sum[32] = lane_sum[32] + sv1[0];
                    float _shfl_sync_267 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 1);
                    float _exp2_266 = approx_exp2(__fmaf_rn(sv1[1], softmax_scale_log2, -_shfl_sync_267));
                    sv1[1] = _exp2_266;
                    lane_sum[33] = lane_sum[33] + sv1[1];
                    float _shfl_sync_269 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 2);
                    float _exp2_268 = approx_exp2(__fmaf_rn(sv1[2], softmax_scale_log2, -_shfl_sync_269));
                    sv1[2] = _exp2_268;
                    lane_sum[34] = lane_sum[34] + sv1[2];
                    float _shfl_sync_271 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 3);
                    float _exp2_270 = approx_exp2(__fmaf_rn(sv1[3], softmax_scale_log2, -_shfl_sync_271));
                    sv1[3] = _exp2_270;
                    lane_sum[35] = lane_sum[35] + sv1[3];
                    float _shfl_sync_273 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 4);
                    float _exp2_272 = approx_exp2(__fmaf_rn(sv1[4], softmax_scale_log2, -_shfl_sync_273));
                    sv1[4] = _exp2_272;
                    lane_sum[36] = lane_sum[36] + sv1[4];
                    float _shfl_sync_275 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 5);
                    float _exp2_274 = approx_exp2(__fmaf_rn(sv1[5], softmax_scale_log2, -_shfl_sync_275));
                    sv1[5] = _exp2_274;
                    lane_sum[37] = lane_sum[37] + sv1[5];
                    float _shfl_sync_277 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 6);
                    float _exp2_276 = approx_exp2(__fmaf_rn(sv1[6], softmax_scale_log2, -_shfl_sync_277));
                    sv1[6] = _exp2_276;
                    lane_sum[38] = lane_sum[38] + sv1[6];
                    float _shfl_sync_279 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 7);
                    float _exp2_278 = approx_exp2(__fmaf_rn(sv1[7], softmax_scale_log2, -_shfl_sync_279));
                    sv1[7] = _exp2_278;
                    lane_sum[39] = lane_sum[39] + sv1[7];
                    float _shfl_sync_281 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 8);
                    float _exp2_280 = approx_exp2(__fmaf_rn(sv1[8], softmax_scale_log2, -_shfl_sync_281));
                    sv1[8] = _exp2_280;
                    lane_sum[40] = lane_sum[40] + sv1[8];
                    float _shfl_sync_283 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 9);
                    float _exp2_282 = approx_exp2(__fmaf_rn(sv1[9], softmax_scale_log2, -_shfl_sync_283));
                    sv1[9] = _exp2_282;
                    lane_sum[41] = lane_sum[41] + sv1[9];
                    float _shfl_sync_285 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 10);
                    float _exp2_284 = approx_exp2(__fmaf_rn(sv1[10], softmax_scale_log2, -_shfl_sync_285));
                    sv1[10] = _exp2_284;
                    lane_sum[42] = lane_sum[42] + sv1[10];
                    float _shfl_sync_287 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 11);
                    float _exp2_286 = approx_exp2(__fmaf_rn(sv1[11], softmax_scale_log2, -_shfl_sync_287));
                    sv1[11] = _exp2_286;
                    lane_sum[43] = lane_sum[43] + sv1[11];
                    float _shfl_sync_289 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 12);
                    float _exp2_288 = approx_exp2(__fmaf_rn(sv1[12], softmax_scale_log2, -_shfl_sync_289));
                    sv1[12] = _exp2_288;
                    lane_sum[44] = lane_sum[44] + sv1[12];
                    float _shfl_sync_291 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 13);
                    float _exp2_290 = approx_exp2(__fmaf_rn(sv1[13], softmax_scale_log2, -_shfl_sync_291));
                    sv1[13] = _exp2_290;
                    lane_sum[45] = lane_sum[45] + sv1[13];
                    float _shfl_sync_293 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 14);
                    float _exp2_292 = approx_exp2(__fmaf_rn(sv1[14], softmax_scale_log2, -_shfl_sync_293));
                    sv1[14] = _exp2_292;
                    lane_sum[46] = lane_sum[46] + sv1[14];
                    float _shfl_sync_295 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 15);
                    float _exp2_294 = approx_exp2(__fmaf_rn(sv1[15], softmax_scale_log2, -_shfl_sync_295));
                    sv1[15] = _exp2_294;
                    lane_sum[47] = lane_sum[47] + sv1[15];
                    float _shfl_sync_297 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 16);
                    float _exp2_296 = approx_exp2(__fmaf_rn(sv1[16], softmax_scale_log2, -_shfl_sync_297));
                    sv1[16] = _exp2_296;
                    lane_sum[48] = lane_sum[48] + sv1[16];
                    float _shfl_sync_299 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 17);
                    float _exp2_298 = approx_exp2(__fmaf_rn(sv1[17], softmax_scale_log2, -_shfl_sync_299));
                    sv1[17] = _exp2_298;
                    lane_sum[49] = lane_sum[49] + sv1[17];
                    float _shfl_sync_301 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 18);
                    float _exp2_300 = approx_exp2(__fmaf_rn(sv1[18], softmax_scale_log2, -_shfl_sync_301));
                    sv1[18] = _exp2_300;
                    lane_sum[50] = lane_sum[50] + sv1[18];
                    float _shfl_sync_303 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 19);
                    float _exp2_302 = approx_exp2(__fmaf_rn(sv1[19], softmax_scale_log2, -_shfl_sync_303));
                    sv1[19] = _exp2_302;
                    lane_sum[51] = lane_sum[51] + sv1[19];
                    float _shfl_sync_305 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 20);
                    float _exp2_304 = approx_exp2(__fmaf_rn(sv1[20], softmax_scale_log2, -_shfl_sync_305));
                    sv1[20] = _exp2_304;
                    lane_sum[52] = lane_sum[52] + sv1[20];
                    float _shfl_sync_307 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 21);
                    float _exp2_306 = approx_exp2(__fmaf_rn(sv1[21], softmax_scale_log2, -_shfl_sync_307));
                    sv1[21] = _exp2_306;
                    lane_sum[53] = lane_sum[53] + sv1[21];
                    float _shfl_sync_309 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 22);
                    float _exp2_308 = approx_exp2(__fmaf_rn(sv1[22], softmax_scale_log2, -_shfl_sync_309));
                    sv1[22] = _exp2_308;
                    lane_sum[54] = lane_sum[54] + sv1[22];
                    float _shfl_sync_311 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 23);
                    float _exp2_310 = approx_exp2(__fmaf_rn(sv1[23], softmax_scale_log2, -_shfl_sync_311));
                    sv1[23] = _exp2_310;
                    lane_sum[55] = lane_sum[55] + sv1[23];
                    float _shfl_sync_313 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 24);
                    float _exp2_312 = approx_exp2(__fmaf_rn(sv1[24], softmax_scale_log2, -_shfl_sync_313));
                    sv1[24] = _exp2_312;
                    lane_sum[56] = lane_sum[56] + sv1[24];
                    float _shfl_sync_315 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 25);
                    float _exp2_314 = approx_exp2(__fmaf_rn(sv1[25], softmax_scale_log2, -_shfl_sync_315));
                    sv1[25] = _exp2_314;
                    lane_sum[57] = lane_sum[57] + sv1[25];
                    float _shfl_sync_317 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 26);
                    float _exp2_316 = approx_exp2(__fmaf_rn(sv1[26], softmax_scale_log2, -_shfl_sync_317));
                    sv1[26] = _exp2_316;
                    lane_sum[58] = lane_sum[58] + sv1[26];
                    float _shfl_sync_319 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 27);
                    float _exp2_318 = approx_exp2(__fmaf_rn(sv1[27], softmax_scale_log2, -_shfl_sync_319));
                    sv1[27] = _exp2_318;
                    lane_sum[59] = lane_sum[59] + sv1[27];
                    float _shfl_sync_321 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 28);
                    float _exp2_320 = approx_exp2(__fmaf_rn(sv1[28], softmax_scale_log2, -_shfl_sync_321));
                    sv1[28] = _exp2_320;
                    lane_sum[60] = lane_sum[60] + sv1[28];
                    float _shfl_sync_323 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 29);
                    float _exp2_322 = approx_exp2(__fmaf_rn(sv1[29], softmax_scale_log2, -_shfl_sync_323));
                    sv1[29] = _exp2_322;
                    lane_sum[61] = lane_sum[61] + sv1[29];
                    float _shfl_sync_325 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 30);
                    float _exp2_324 = approx_exp2(__fmaf_rn(sv1[30], softmax_scale_log2, -_shfl_sync_325));
                    sv1[30] = _exp2_324;
                    lane_sum[62] = lane_sum[62] + sv1[30];
                    float _shfl_sync_327 = __shfl_sync(0xFFFFFFFF, ms_sel_0, 31);
                    float _exp2_326 = approx_exp2(__fmaf_rn(sv1[31], softmax_scale_log2, -_shfl_sync_327));
                    sv1[31] = _exp2_326;
                    lane_sum[63] = lane_sum[63] + sv1[31];
                    unsigned int words_0[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[0]), "f"(sv1[1]),
                                               "f"(sv1[2]), "f"(sv1[3]));
                        words_0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[4]), "f"(sv1[5]),
                                               "f"(sv1[6]), "f"(sv1[7]));
                        words_0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[8]), "f"(sv1[9]),
                                               "f"(sv1[10]), "f"(sv1[11]));
                        words_0[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[12]), "f"(sv1[13]),
                                               "f"(sv1[14]), "f"(sv1[15]));
                        words_0[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[16]), "f"(sv1[17]),
                                               "f"(sv1[18]), "f"(sv1[19]));
                        words_0[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[20]), "f"(sv1[21]),
                                               "f"(sv1[22]), "f"(sv1[23]));
                        words_0[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[24]), "f"(sv1[25]),
                                               "f"(sv1[26]), "f"(sv1[27]));
                        words_0[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1[28]), "f"(sv1[29]),
                                               "f"(sv1[30]), "f"(sv1[31]));
                        words_0[7] = _packed;
                    }
                    int p_base_0 = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 8192);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_0 + (key_row * 64 + 0 ^ (key_row * 64 + 0 >> 7 & 3) << 4))), "r"(words_0[0]), "r"(words_0[1]), "r"(words_0[2]), "r"(words_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_0 + (key_row * 64 + 16 ^ (key_row * 64 + 16 >> 7 & 3) << 4))), "r"(words_0[4]), "r"(words_0[5]), "r"(words_0[6]), "r"(words_0[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage) * 8);
                    }
                    s_stage += 1;
                    if (s_stage == 2) { s_stage = 0; _phase_s_full ^= 1; }
                    p_stage += 1;
                    if (p_stage == 4) { p_stage = 0; }
                    gpair += 1;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)item_ord % 2)) * 8);
                }
                float m_fin = o_hold[0];
                m_fin = max_noftz(m_fin, o_hold[1]);
                m_fin = max_noftz(m_fin, o_hold[2]);
                m_fin = max_noftz(m_fin, o_hold[3]);
                float w_hold[4];
                float _exp2_328 = approx_exp2((o_hold[0] - m_fin) * softmax_scale_log2);
                w_hold[0] = ((o_hold[0] == -CUDART_INF_F) ? 0.0f : _exp2_328);
                float _exp2_329 = approx_exp2((o_hold[1] - m_fin) * softmax_scale_log2);
                w_hold[1] = ((o_hold[1] == -CUDART_INF_F) ? 0.0f : _exp2_329);
                float _exp2_330 = approx_exp2((o_hold[2] - m_fin) * softmax_scale_log2);
                w_hold[2] = ((o_hold[2] == -CUDART_INF_F) ? 0.0f : _exp2_330);
                float _exp2_331 = approx_exp2((o_hold[3] - m_fin) * softmax_scale_log2);
                w_hold[3] = ((o_hold[3] == -CUDART_INF_F) ? 0.0f : _exp2_331);
                float colsum[32];
                float acc_0 = 0.0f;
                float w_sel_0 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_332 = __shfl_sync(0xFFFFFFFF, w_sel_0, 0);
                acc_0 = __fmaf_rn(lane_sum[0], _shfl_sync_332, acc_0);
                float w_sel_1 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_333 = __shfl_sync(0xFFFFFFFF, w_sel_1, 0);
                acc_0 = __fmaf_rn(lane_sum[32], _shfl_sync_333, acc_0);
                colsum[0] = acc_0;
                float acc_1 = 0.0f;
                float w_sel_2 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_334 = __shfl_sync(0xFFFFFFFF, w_sel_2, 1);
                acc_1 = __fmaf_rn(lane_sum[1], _shfl_sync_334, acc_1);
                float w_sel_3 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_335 = __shfl_sync(0xFFFFFFFF, w_sel_3, 1);
                acc_1 = __fmaf_rn(lane_sum[33], _shfl_sync_335, acc_1);
                colsum[1] = acc_1;
                float acc_2 = 0.0f;
                float w_sel_4 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_336 = __shfl_sync(0xFFFFFFFF, w_sel_4, 2);
                acc_2 = __fmaf_rn(lane_sum[2], _shfl_sync_336, acc_2);
                float w_sel_5 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_337 = __shfl_sync(0xFFFFFFFF, w_sel_5, 2);
                acc_2 = __fmaf_rn(lane_sum[34], _shfl_sync_337, acc_2);
                colsum[2] = acc_2;
                float acc_3 = 0.0f;
                float w_sel_6 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_338 = __shfl_sync(0xFFFFFFFF, w_sel_6, 3);
                acc_3 = __fmaf_rn(lane_sum[3], _shfl_sync_338, acc_3);
                float w_sel_7 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_339 = __shfl_sync(0xFFFFFFFF, w_sel_7, 3);
                acc_3 = __fmaf_rn(lane_sum[35], _shfl_sync_339, acc_3);
                colsum[3] = acc_3;
                float acc_4 = 0.0f;
                float w_sel_8 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_340 = __shfl_sync(0xFFFFFFFF, w_sel_8, 4);
                acc_4 = __fmaf_rn(lane_sum[4], _shfl_sync_340, acc_4);
                float w_sel_9 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_341 = __shfl_sync(0xFFFFFFFF, w_sel_9, 4);
                acc_4 = __fmaf_rn(lane_sum[36], _shfl_sync_341, acc_4);
                colsum[4] = acc_4;
                float acc_5 = 0.0f;
                float w_sel_10 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_342 = __shfl_sync(0xFFFFFFFF, w_sel_10, 5);
                acc_5 = __fmaf_rn(lane_sum[5], _shfl_sync_342, acc_5);
                float w_sel_11 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_343 = __shfl_sync(0xFFFFFFFF, w_sel_11, 5);
                acc_5 = __fmaf_rn(lane_sum[37], _shfl_sync_343, acc_5);
                colsum[5] = acc_5;
                float acc_6 = 0.0f;
                float w_sel_12 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_344 = __shfl_sync(0xFFFFFFFF, w_sel_12, 6);
                acc_6 = __fmaf_rn(lane_sum[6], _shfl_sync_344, acc_6);
                float w_sel_13 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_345 = __shfl_sync(0xFFFFFFFF, w_sel_13, 6);
                acc_6 = __fmaf_rn(lane_sum[38], _shfl_sync_345, acc_6);
                colsum[6] = acc_6;
                float acc_7 = 0.0f;
                float w_sel_14 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_346 = __shfl_sync(0xFFFFFFFF, w_sel_14, 7);
                acc_7 = __fmaf_rn(lane_sum[7], _shfl_sync_346, acc_7);
                float w_sel_15 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_347 = __shfl_sync(0xFFFFFFFF, w_sel_15, 7);
                acc_7 = __fmaf_rn(lane_sum[39], _shfl_sync_347, acc_7);
                colsum[7] = acc_7;
                float acc_8 = 0.0f;
                float w_sel_16 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_348 = __shfl_sync(0xFFFFFFFF, w_sel_16, 8);
                acc_8 = __fmaf_rn(lane_sum[8], _shfl_sync_348, acc_8);
                float w_sel_17 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_349 = __shfl_sync(0xFFFFFFFF, w_sel_17, 8);
                acc_8 = __fmaf_rn(lane_sum[40], _shfl_sync_349, acc_8);
                colsum[8] = acc_8;
                float acc_9 = 0.0f;
                float w_sel_18 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_350 = __shfl_sync(0xFFFFFFFF, w_sel_18, 9);
                acc_9 = __fmaf_rn(lane_sum[9], _shfl_sync_350, acc_9);
                float w_sel_19 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_351 = __shfl_sync(0xFFFFFFFF, w_sel_19, 9);
                acc_9 = __fmaf_rn(lane_sum[41], _shfl_sync_351, acc_9);
                colsum[9] = acc_9;
                float acc_10 = 0.0f;
                float w_sel_20 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_352 = __shfl_sync(0xFFFFFFFF, w_sel_20, 10);
                acc_10 = __fmaf_rn(lane_sum[10], _shfl_sync_352, acc_10);
                float w_sel_21 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_353 = __shfl_sync(0xFFFFFFFF, w_sel_21, 10);
                acc_10 = __fmaf_rn(lane_sum[42], _shfl_sync_353, acc_10);
                colsum[10] = acc_10;
                float acc_11 = 0.0f;
                float w_sel_22 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_354 = __shfl_sync(0xFFFFFFFF, w_sel_22, 11);
                acc_11 = __fmaf_rn(lane_sum[11], _shfl_sync_354, acc_11);
                float w_sel_23 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_355 = __shfl_sync(0xFFFFFFFF, w_sel_23, 11);
                acc_11 = __fmaf_rn(lane_sum[43], _shfl_sync_355, acc_11);
                colsum[11] = acc_11;
                float acc_12 = 0.0f;
                float w_sel_24 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_356 = __shfl_sync(0xFFFFFFFF, w_sel_24, 12);
                acc_12 = __fmaf_rn(lane_sum[12], _shfl_sync_356, acc_12);
                float w_sel_25 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_357 = __shfl_sync(0xFFFFFFFF, w_sel_25, 12);
                acc_12 = __fmaf_rn(lane_sum[44], _shfl_sync_357, acc_12);
                colsum[12] = acc_12;
                float acc_13 = 0.0f;
                float w_sel_26 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_358 = __shfl_sync(0xFFFFFFFF, w_sel_26, 13);
                acc_13 = __fmaf_rn(lane_sum[13], _shfl_sync_358, acc_13);
                float w_sel_27 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_359 = __shfl_sync(0xFFFFFFFF, w_sel_27, 13);
                acc_13 = __fmaf_rn(lane_sum[45], _shfl_sync_359, acc_13);
                colsum[13] = acc_13;
                float acc_14 = 0.0f;
                float w_sel_28 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_360 = __shfl_sync(0xFFFFFFFF, w_sel_28, 14);
                acc_14 = __fmaf_rn(lane_sum[14], _shfl_sync_360, acc_14);
                float w_sel_29 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_361 = __shfl_sync(0xFFFFFFFF, w_sel_29, 14);
                acc_14 = __fmaf_rn(lane_sum[46], _shfl_sync_361, acc_14);
                colsum[14] = acc_14;
                float acc_15 = 0.0f;
                float w_sel_30 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_362 = __shfl_sync(0xFFFFFFFF, w_sel_30, 15);
                acc_15 = __fmaf_rn(lane_sum[15], _shfl_sync_362, acc_15);
                float w_sel_31 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_363 = __shfl_sync(0xFFFFFFFF, w_sel_31, 15);
                acc_15 = __fmaf_rn(lane_sum[47], _shfl_sync_363, acc_15);
                colsum[15] = acc_15;
                float acc_16 = 0.0f;
                float w_sel_32 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_364 = __shfl_sync(0xFFFFFFFF, w_sel_32, 16);
                acc_16 = __fmaf_rn(lane_sum[16], _shfl_sync_364, acc_16);
                float w_sel_33 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_365 = __shfl_sync(0xFFFFFFFF, w_sel_33, 16);
                acc_16 = __fmaf_rn(lane_sum[48], _shfl_sync_365, acc_16);
                colsum[16] = acc_16;
                float acc_17 = 0.0f;
                float w_sel_34 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_366 = __shfl_sync(0xFFFFFFFF, w_sel_34, 17);
                acc_17 = __fmaf_rn(lane_sum[17], _shfl_sync_366, acc_17);
                float w_sel_35 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_367 = __shfl_sync(0xFFFFFFFF, w_sel_35, 17);
                acc_17 = __fmaf_rn(lane_sum[49], _shfl_sync_367, acc_17);
                colsum[17] = acc_17;
                float acc_18 = 0.0f;
                float w_sel_36 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_368 = __shfl_sync(0xFFFFFFFF, w_sel_36, 18);
                acc_18 = __fmaf_rn(lane_sum[18], _shfl_sync_368, acc_18);
                float w_sel_37 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_369 = __shfl_sync(0xFFFFFFFF, w_sel_37, 18);
                acc_18 = __fmaf_rn(lane_sum[50], _shfl_sync_369, acc_18);
                colsum[18] = acc_18;
                float acc_19 = 0.0f;
                float w_sel_38 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_370 = __shfl_sync(0xFFFFFFFF, w_sel_38, 19);
                acc_19 = __fmaf_rn(lane_sum[19], _shfl_sync_370, acc_19);
                float w_sel_39 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_371 = __shfl_sync(0xFFFFFFFF, w_sel_39, 19);
                acc_19 = __fmaf_rn(lane_sum[51], _shfl_sync_371, acc_19);
                colsum[19] = acc_19;
                float acc_20 = 0.0f;
                float w_sel_40 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_372 = __shfl_sync(0xFFFFFFFF, w_sel_40, 20);
                acc_20 = __fmaf_rn(lane_sum[20], _shfl_sync_372, acc_20);
                float w_sel_41 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_373 = __shfl_sync(0xFFFFFFFF, w_sel_41, 20);
                acc_20 = __fmaf_rn(lane_sum[52], _shfl_sync_373, acc_20);
                colsum[20] = acc_20;
                float acc_21 = 0.0f;
                float w_sel_42 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_374 = __shfl_sync(0xFFFFFFFF, w_sel_42, 21);
                acc_21 = __fmaf_rn(lane_sum[21], _shfl_sync_374, acc_21);
                float w_sel_43 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_375 = __shfl_sync(0xFFFFFFFF, w_sel_43, 21);
                acc_21 = __fmaf_rn(lane_sum[53], _shfl_sync_375, acc_21);
                colsum[21] = acc_21;
                float acc_22 = 0.0f;
                float w_sel_44 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_376 = __shfl_sync(0xFFFFFFFF, w_sel_44, 22);
                acc_22 = __fmaf_rn(lane_sum[22], _shfl_sync_376, acc_22);
                float w_sel_45 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_377 = __shfl_sync(0xFFFFFFFF, w_sel_45, 22);
                acc_22 = __fmaf_rn(lane_sum[54], _shfl_sync_377, acc_22);
                colsum[22] = acc_22;
                float acc_23 = 0.0f;
                float w_sel_46 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_378 = __shfl_sync(0xFFFFFFFF, w_sel_46, 23);
                acc_23 = __fmaf_rn(lane_sum[23], _shfl_sync_378, acc_23);
                float w_sel_47 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_379 = __shfl_sync(0xFFFFFFFF, w_sel_47, 23);
                acc_23 = __fmaf_rn(lane_sum[55], _shfl_sync_379, acc_23);
                colsum[23] = acc_23;
                float acc_24 = 0.0f;
                float w_sel_48 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_380 = __shfl_sync(0xFFFFFFFF, w_sel_48, 24);
                acc_24 = __fmaf_rn(lane_sum[24], _shfl_sync_380, acc_24);
                float w_sel_49 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_381 = __shfl_sync(0xFFFFFFFF, w_sel_49, 24);
                acc_24 = __fmaf_rn(lane_sum[56], _shfl_sync_381, acc_24);
                colsum[24] = acc_24;
                float acc_25 = 0.0f;
                float w_sel_50 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_382 = __shfl_sync(0xFFFFFFFF, w_sel_50, 25);
                acc_25 = __fmaf_rn(lane_sum[25], _shfl_sync_382, acc_25);
                float w_sel_51 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_383 = __shfl_sync(0xFFFFFFFF, w_sel_51, 25);
                acc_25 = __fmaf_rn(lane_sum[57], _shfl_sync_383, acc_25);
                colsum[25] = acc_25;
                float acc_26 = 0.0f;
                float w_sel_52 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_384 = __shfl_sync(0xFFFFFFFF, w_sel_52, 26);
                acc_26 = __fmaf_rn(lane_sum[26], _shfl_sync_384, acc_26);
                float w_sel_53 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_385 = __shfl_sync(0xFFFFFFFF, w_sel_53, 26);
                acc_26 = __fmaf_rn(lane_sum[58], _shfl_sync_385, acc_26);
                colsum[26] = acc_26;
                float acc_27 = 0.0f;
                float w_sel_54 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_386 = __shfl_sync(0xFFFFFFFF, w_sel_54, 27);
                acc_27 = __fmaf_rn(lane_sum[27], _shfl_sync_386, acc_27);
                float w_sel_55 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_387 = __shfl_sync(0xFFFFFFFF, w_sel_55, 27);
                acc_27 = __fmaf_rn(lane_sum[59], _shfl_sync_387, acc_27);
                colsum[27] = acc_27;
                float acc_28 = 0.0f;
                float w_sel_56 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_388 = __shfl_sync(0xFFFFFFFF, w_sel_56, 28);
                acc_28 = __fmaf_rn(lane_sum[28], _shfl_sync_388, acc_28);
                float w_sel_57 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_389 = __shfl_sync(0xFFFFFFFF, w_sel_57, 28);
                acc_28 = __fmaf_rn(lane_sum[60], _shfl_sync_389, acc_28);
                colsum[28] = acc_28;
                float acc_29 = 0.0f;
                float w_sel_58 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_390 = __shfl_sync(0xFFFFFFFF, w_sel_58, 29);
                acc_29 = __fmaf_rn(lane_sum[29], _shfl_sync_390, acc_29);
                float w_sel_59 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_391 = __shfl_sync(0xFFFFFFFF, w_sel_59, 29);
                acc_29 = __fmaf_rn(lane_sum[61], _shfl_sync_391, acc_29);
                colsum[29] = acc_29;
                float acc_30 = 0.0f;
                float w_sel_60 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_392 = __shfl_sync(0xFFFFFFFF, w_sel_60, 30);
                acc_30 = __fmaf_rn(lane_sum[30], _shfl_sync_392, acc_30);
                float w_sel_61 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_393 = __shfl_sync(0xFFFFFFFF, w_sel_61, 30);
                acc_30 = __fmaf_rn(lane_sum[62], _shfl_sync_393, acc_30);
                colsum[30] = acc_30;
                float acc_31 = 0.0f;
                float w_sel_62 = ((own_half == 0) ? w_hold[0] : w_hold[1]);
                float _shfl_sync_394 = __shfl_sync(0xFFFFFFFF, w_sel_62, 31);
                acc_31 = __fmaf_rn(lane_sum[31], _shfl_sync_394, acc_31);
                float w_sel_63 = ((own_half == 0) ? w_hold[2] : w_hold[3]);
                float _shfl_sync_395 = __shfl_sync(0xFFFFFFFF, w_sel_63, 31);
                acc_31 = __fmaf_rn(lane_sum[63], _shfl_sync_395, acc_31);
                colsum[31] = acc_31;
                int b4 = lane / 16 % 2;
                int b3 = lane / 8 % 2;
                int b2 = lane / 4 % 2;
                int b1 = lane / 2 % 2;
                int b0 = lane % 2;
                float r16[16];
                float snd_0 = ((b4 == 0) ? colsum[16] : colsum[0]);
                float kp_0 = ((b4 == 0) ? colsum[0] : colsum[16]);
                float _shfl_xor_sync_396 = __shfl_xor_sync(0xFFFFFFFF, snd_0, 16);
                r16[0] = kp_0 + _shfl_xor_sync_396;
                float snd_1 = ((b4 == 0) ? colsum[17] : colsum[1]);
                float kp_1 = ((b4 == 0) ? colsum[1] : colsum[17]);
                float _shfl_xor_sync_397 = __shfl_xor_sync(0xFFFFFFFF, snd_1, 16);
                r16[1] = kp_1 + _shfl_xor_sync_397;
                float snd_2 = ((b4 == 0) ? colsum[18] : colsum[2]);
                float kp_2 = ((b4 == 0) ? colsum[2] : colsum[18]);
                float _shfl_xor_sync_398 = __shfl_xor_sync(0xFFFFFFFF, snd_2, 16);
                r16[2] = kp_2 + _shfl_xor_sync_398;
                float snd_3 = ((b4 == 0) ? colsum[19] : colsum[3]);
                float kp_3 = ((b4 == 0) ? colsum[3] : colsum[19]);
                float _shfl_xor_sync_399 = __shfl_xor_sync(0xFFFFFFFF, snd_3, 16);
                r16[3] = kp_3 + _shfl_xor_sync_399;
                float snd_4 = ((b4 == 0) ? colsum[20] : colsum[4]);
                float kp_4 = ((b4 == 0) ? colsum[4] : colsum[20]);
                float _shfl_xor_sync_400 = __shfl_xor_sync(0xFFFFFFFF, snd_4, 16);
                r16[4] = kp_4 + _shfl_xor_sync_400;
                float snd_5 = ((b4 == 0) ? colsum[21] : colsum[5]);
                float kp_5 = ((b4 == 0) ? colsum[5] : colsum[21]);
                float _shfl_xor_sync_401 = __shfl_xor_sync(0xFFFFFFFF, snd_5, 16);
                r16[5] = kp_5 + _shfl_xor_sync_401;
                float snd_6 = ((b4 == 0) ? colsum[22] : colsum[6]);
                float kp_6 = ((b4 == 0) ? colsum[6] : colsum[22]);
                float _shfl_xor_sync_402 = __shfl_xor_sync(0xFFFFFFFF, snd_6, 16);
                r16[6] = kp_6 + _shfl_xor_sync_402;
                float snd_7 = ((b4 == 0) ? colsum[23] : colsum[7]);
                float kp_7 = ((b4 == 0) ? colsum[7] : colsum[23]);
                float _shfl_xor_sync_403 = __shfl_xor_sync(0xFFFFFFFF, snd_7, 16);
                r16[7] = kp_7 + _shfl_xor_sync_403;
                float snd_8 = ((b4 == 0) ? colsum[24] : colsum[8]);
                float kp_8 = ((b4 == 0) ? colsum[8] : colsum[24]);
                float _shfl_xor_sync_404 = __shfl_xor_sync(0xFFFFFFFF, snd_8, 16);
                r16[8] = kp_8 + _shfl_xor_sync_404;
                float snd_9 = ((b4 == 0) ? colsum[25] : colsum[9]);
                float kp_9 = ((b4 == 0) ? colsum[9] : colsum[25]);
                float _shfl_xor_sync_405 = __shfl_xor_sync(0xFFFFFFFF, snd_9, 16);
                r16[9] = kp_9 + _shfl_xor_sync_405;
                float snd_10 = ((b4 == 0) ? colsum[26] : colsum[10]);
                float kp_10 = ((b4 == 0) ? colsum[10] : colsum[26]);
                float _shfl_xor_sync_406 = __shfl_xor_sync(0xFFFFFFFF, snd_10, 16);
                r16[10] = kp_10 + _shfl_xor_sync_406;
                float snd_11 = ((b4 == 0) ? colsum[27] : colsum[11]);
                float kp_11 = ((b4 == 0) ? colsum[11] : colsum[27]);
                float _shfl_xor_sync_407 = __shfl_xor_sync(0xFFFFFFFF, snd_11, 16);
                r16[11] = kp_11 + _shfl_xor_sync_407;
                float snd_12 = ((b4 == 0) ? colsum[28] : colsum[12]);
                float kp_12 = ((b4 == 0) ? colsum[12] : colsum[28]);
                float _shfl_xor_sync_408 = __shfl_xor_sync(0xFFFFFFFF, snd_12, 16);
                r16[12] = kp_12 + _shfl_xor_sync_408;
                float snd_13 = ((b4 == 0) ? colsum[29] : colsum[13]);
                float kp_13 = ((b4 == 0) ? colsum[13] : colsum[29]);
                float _shfl_xor_sync_409 = __shfl_xor_sync(0xFFFFFFFF, snd_13, 16);
                r16[13] = kp_13 + _shfl_xor_sync_409;
                float snd_14 = ((b4 == 0) ? colsum[30] : colsum[14]);
                float kp_14 = ((b4 == 0) ? colsum[14] : colsum[30]);
                float _shfl_xor_sync_410 = __shfl_xor_sync(0xFFFFFFFF, snd_14, 16);
                r16[14] = kp_14 + _shfl_xor_sync_410;
                float snd_15 = ((b4 == 0) ? colsum[31] : colsum[15]);
                float kp_15 = ((b4 == 0) ? colsum[15] : colsum[31]);
                float _shfl_xor_sync_411 = __shfl_xor_sync(0xFFFFFFFF, snd_15, 16);
                r16[15] = kp_15 + _shfl_xor_sync_411;
                float r8[8];
                float snd8_0 = ((b3 == 0) ? r16[8] : r16[0]);
                float kp8_0 = ((b3 == 0) ? r16[0] : r16[8]);
                float _shfl_xor_sync_412 = __shfl_xor_sync(0xFFFFFFFF, snd8_0, 8);
                r8[0] = kp8_0 + _shfl_xor_sync_412;
                float snd8_1 = ((b3 == 0) ? r16[9] : r16[1]);
                float kp8_1 = ((b3 == 0) ? r16[1] : r16[9]);
                float _shfl_xor_sync_413 = __shfl_xor_sync(0xFFFFFFFF, snd8_1, 8);
                r8[1] = kp8_1 + _shfl_xor_sync_413;
                float snd8_2 = ((b3 == 0) ? r16[10] : r16[2]);
                float kp8_2 = ((b3 == 0) ? r16[2] : r16[10]);
                float _shfl_xor_sync_414 = __shfl_xor_sync(0xFFFFFFFF, snd8_2, 8);
                r8[2] = kp8_2 + _shfl_xor_sync_414;
                float snd8_3 = ((b3 == 0) ? r16[11] : r16[3]);
                float kp8_3 = ((b3 == 0) ? r16[3] : r16[11]);
                float _shfl_xor_sync_415 = __shfl_xor_sync(0xFFFFFFFF, snd8_3, 8);
                r8[3] = kp8_3 + _shfl_xor_sync_415;
                float snd8_4 = ((b3 == 0) ? r16[12] : r16[4]);
                float kp8_4 = ((b3 == 0) ? r16[4] : r16[12]);
                float _shfl_xor_sync_416 = __shfl_xor_sync(0xFFFFFFFF, snd8_4, 8);
                r8[4] = kp8_4 + _shfl_xor_sync_416;
                float snd8_5 = ((b3 == 0) ? r16[13] : r16[5]);
                float kp8_5 = ((b3 == 0) ? r16[5] : r16[13]);
                float _shfl_xor_sync_417 = __shfl_xor_sync(0xFFFFFFFF, snd8_5, 8);
                r8[5] = kp8_5 + _shfl_xor_sync_417;
                float snd8_6 = ((b3 == 0) ? r16[14] : r16[6]);
                float kp8_6 = ((b3 == 0) ? r16[6] : r16[14]);
                float _shfl_xor_sync_418 = __shfl_xor_sync(0xFFFFFFFF, snd8_6, 8);
                r8[6] = kp8_6 + _shfl_xor_sync_418;
                float snd8_7 = ((b3 == 0) ? r16[15] : r16[7]);
                float kp8_7 = ((b3 == 0) ? r16[7] : r16[15]);
                float _shfl_xor_sync_419 = __shfl_xor_sync(0xFFFFFFFF, snd8_7, 8);
                r8[7] = kp8_7 + _shfl_xor_sync_419;
                float r4[4];
                float snd4_0 = ((b2 == 0) ? r8[4] : r8[0]);
                float kp4_0 = ((b2 == 0) ? r8[0] : r8[4]);
                float _shfl_xor_sync_420 = __shfl_xor_sync(0xFFFFFFFF, snd4_0, 4);
                r4[0] = kp4_0 + _shfl_xor_sync_420;
                float snd4_1 = ((b2 == 0) ? r8[5] : r8[1]);
                float kp4_1 = ((b2 == 0) ? r8[1] : r8[5]);
                float _shfl_xor_sync_421 = __shfl_xor_sync(0xFFFFFFFF, snd4_1, 4);
                r4[1] = kp4_1 + _shfl_xor_sync_421;
                float snd4_2 = ((b2 == 0) ? r8[6] : r8[2]);
                float kp4_2 = ((b2 == 0) ? r8[2] : r8[6]);
                float _shfl_xor_sync_422 = __shfl_xor_sync(0xFFFFFFFF, snd4_2, 4);
                r4[2] = kp4_2 + _shfl_xor_sync_422;
                float snd4_3 = ((b2 == 0) ? r8[7] : r8[3]);
                float kp4_3 = ((b2 == 0) ? r8[3] : r8[7]);
                float _shfl_xor_sync_423 = __shfl_xor_sync(0xFFFFFFFF, snd4_3, 4);
                r4[3] = kp4_3 + _shfl_xor_sync_423;
                float r2[2];
                float snd2_0 = ((b1 == 0) ? r4[2] : r4[0]);
                float kp2_0 = ((b1 == 0) ? r4[0] : r4[2]);
                float _shfl_xor_sync_424 = __shfl_xor_sync(0xFFFFFFFF, snd2_0, 2);
                r2[0] = kp2_0 + _shfl_xor_sync_424;
                float snd2_1 = ((b1 == 0) ? r4[3] : r4[1]);
                float kp2_1 = ((b1 == 0) ? r4[1] : r4[3]);
                float _shfl_xor_sync_425 = __shfl_xor_sync(0xFFFFFFFF, snd2_1, 2);
                r2[1] = kp2_1 + _shfl_xor_sync_425;
                float snd1 = ((b0 == 0) ? r2[1] : r2[0]);
                float kp1 = ((b0 == 0) ? r2[0] : r2[1]);
                float _shfl_xor_sync_426 = __shfl_xor_sync(0xFFFFFFFF, snd1, 1);
                float r1 = kp1 + _shfl_xor_sync_426;
                float xs = r1;
                if (npairs == 0) {
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
                smem_sum[(0 + warp_in_role) * 32 + lane] = xs;
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float head_sum = 0.0f;
                head_sum = head_sum + smem_sum[0 + lane];
                head_sum = head_sum + smem_sum[32 + lane];
                head_sum = head_sum + smem_sum[64 + lane];
                head_sum = head_sum + smem_sum[96 + lane];
                float _rcp_427 = approx_rcp(head_sum);
                float inv = ((head_sum > 0.0f) ? output_scale * _rcp_427 : 0.0f);
                int _stage_item_tick = (int)((unsigned int)item_ord % 2);
                int _parity_item_tick = (int)((unsigned int)item_ord / 2 % 2);
                mbarrier_wait(item_tick_addr + (_stage_item_tick) * 8, _parity_item_tick);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (npairs > 0) {
                    int _stage_pv_done_2 = (int)((unsigned int)(gpair - 1) % 8);
                    int _parity_pv_done_2 = (int)((unsigned int)(gpair - 1) / 8 % 2);
                    mbarrier_wait(pv_done_addr + (_stage_pv_done_2) * 8, _parity_pv_done_2);
                    if (npairs > 1) {
                        int _stage_pv_done_3 = (int)((unsigned int)(gpair - 1 - 1) % 8);
                        int _parity_pv_done_3 = (int)((unsigned int)(gpair - 1 - 1) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_3) * 8, _parity_pv_done_3);
                    }
                    if (npairs > 2) {
                        int _stage_pv_done_4 = (int)((unsigned int)(gpair - 1 - 2) % 8);
                        int _parity_pv_done_4 = (int)((unsigned int)(gpair - 1 - 2) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_4) * 8, _parity_pv_done_4);
                    }
                    if (npairs > 3) {
                        int _stage_pv_done_5 = (int)((unsigned int)(gpair - 1 - 3) % 8);
                        int _parity_pv_done_5 = (int)((unsigned int)(gpair - 1 - 3) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_5) * 8, _parity_pv_done_5);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_epi[32];
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
                if (npairs > 0) {
                    float o_read_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_0[0]), "=f"(o_read_0[1]), "=f"(o_read_0[2]), "=f"(o_read_0[3]), "=f"(o_read_0[4]), "=f"(o_read_0[5]), "=f"(o_read_0[6]), "=f"(o_read_0[7]), "=f"(o_read_0[8]), "=f"(o_read_0[9]), "=f"(o_read_0[10]), "=f"(o_read_0[11]), "=f"(o_read_0[12]), "=f"(o_read_0[13]), "=f"(o_read_0[14]), "=f"(o_read_0[15]), "=f"(o_read_0[16]), "=f"(o_read_0[17]), "=f"(o_read_0[18]), "=f"(o_read_0[19]), "=f"(o_read_0[20]), "=f"(o_read_0[21]), "=f"(o_read_0[22]), "=f"(o_read_0[23]), "=f"(o_read_0[24]), "=f"(o_read_0[25]), "=f"(o_read_0[26]), "=f"(o_read_0[27]), "=f"(o_read_0[28]), "=f"(o_read_0[29]), "=f"(o_read_0[30]), "=f"(o_read_0[31])
                        : "r"(taddr + 256 + (unsigned int)lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_428 = __shfl_sync(0xFFFFFFFF, w_hold[0], 0);
                    o_epi[0] = __fmaf_rn(o_read_0[0], _shfl_sync_428, o_epi[0]);
                    float _shfl_sync_429 = __shfl_sync(0xFFFFFFFF, w_hold[0], 1);
                    o_epi[1] = __fmaf_rn(o_read_0[1], _shfl_sync_429, o_epi[1]);
                    float _shfl_sync_430 = __shfl_sync(0xFFFFFFFF, w_hold[0], 2);
                    o_epi[2] = __fmaf_rn(o_read_0[2], _shfl_sync_430, o_epi[2]);
                    float _shfl_sync_431 = __shfl_sync(0xFFFFFFFF, w_hold[0], 3);
                    o_epi[3] = __fmaf_rn(o_read_0[3], _shfl_sync_431, o_epi[3]);
                    float _shfl_sync_432 = __shfl_sync(0xFFFFFFFF, w_hold[0], 4);
                    o_epi[4] = __fmaf_rn(o_read_0[4], _shfl_sync_432, o_epi[4]);
                    float _shfl_sync_433 = __shfl_sync(0xFFFFFFFF, w_hold[0], 5);
                    o_epi[5] = __fmaf_rn(o_read_0[5], _shfl_sync_433, o_epi[5]);
                    float _shfl_sync_434 = __shfl_sync(0xFFFFFFFF, w_hold[0], 6);
                    o_epi[6] = __fmaf_rn(o_read_0[6], _shfl_sync_434, o_epi[6]);
                    float _shfl_sync_435 = __shfl_sync(0xFFFFFFFF, w_hold[0], 7);
                    o_epi[7] = __fmaf_rn(o_read_0[7], _shfl_sync_435, o_epi[7]);
                    float _shfl_sync_436 = __shfl_sync(0xFFFFFFFF, w_hold[0], 8);
                    o_epi[8] = __fmaf_rn(o_read_0[8], _shfl_sync_436, o_epi[8]);
                    float _shfl_sync_437 = __shfl_sync(0xFFFFFFFF, w_hold[0], 9);
                    o_epi[9] = __fmaf_rn(o_read_0[9], _shfl_sync_437, o_epi[9]);
                    float _shfl_sync_438 = __shfl_sync(0xFFFFFFFF, w_hold[0], 10);
                    o_epi[10] = __fmaf_rn(o_read_0[10], _shfl_sync_438, o_epi[10]);
                    float _shfl_sync_439 = __shfl_sync(0xFFFFFFFF, w_hold[0], 11);
                    o_epi[11] = __fmaf_rn(o_read_0[11], _shfl_sync_439, o_epi[11]);
                    float _shfl_sync_440 = __shfl_sync(0xFFFFFFFF, w_hold[0], 12);
                    o_epi[12] = __fmaf_rn(o_read_0[12], _shfl_sync_440, o_epi[12]);
                    float _shfl_sync_441 = __shfl_sync(0xFFFFFFFF, w_hold[0], 13);
                    o_epi[13] = __fmaf_rn(o_read_0[13], _shfl_sync_441, o_epi[13]);
                    float _shfl_sync_442 = __shfl_sync(0xFFFFFFFF, w_hold[0], 14);
                    o_epi[14] = __fmaf_rn(o_read_0[14], _shfl_sync_442, o_epi[14]);
                    float _shfl_sync_443 = __shfl_sync(0xFFFFFFFF, w_hold[0], 15);
                    o_epi[15] = __fmaf_rn(o_read_0[15], _shfl_sync_443, o_epi[15]);
                    float _shfl_sync_444 = __shfl_sync(0xFFFFFFFF, w_hold[0], 16);
                    o_epi[16] = __fmaf_rn(o_read_0[16], _shfl_sync_444, o_epi[16]);
                    float _shfl_sync_445 = __shfl_sync(0xFFFFFFFF, w_hold[0], 17);
                    o_epi[17] = __fmaf_rn(o_read_0[17], _shfl_sync_445, o_epi[17]);
                    float _shfl_sync_446 = __shfl_sync(0xFFFFFFFF, w_hold[0], 18);
                    o_epi[18] = __fmaf_rn(o_read_0[18], _shfl_sync_446, o_epi[18]);
                    float _shfl_sync_447 = __shfl_sync(0xFFFFFFFF, w_hold[0], 19);
                    o_epi[19] = __fmaf_rn(o_read_0[19], _shfl_sync_447, o_epi[19]);
                    float _shfl_sync_448 = __shfl_sync(0xFFFFFFFF, w_hold[0], 20);
                    o_epi[20] = __fmaf_rn(o_read_0[20], _shfl_sync_448, o_epi[20]);
                    float _shfl_sync_449 = __shfl_sync(0xFFFFFFFF, w_hold[0], 21);
                    o_epi[21] = __fmaf_rn(o_read_0[21], _shfl_sync_449, o_epi[21]);
                    float _shfl_sync_450 = __shfl_sync(0xFFFFFFFF, w_hold[0], 22);
                    o_epi[22] = __fmaf_rn(o_read_0[22], _shfl_sync_450, o_epi[22]);
                    float _shfl_sync_451 = __shfl_sync(0xFFFFFFFF, w_hold[0], 23);
                    o_epi[23] = __fmaf_rn(o_read_0[23], _shfl_sync_451, o_epi[23]);
                    float _shfl_sync_452 = __shfl_sync(0xFFFFFFFF, w_hold[0], 24);
                    o_epi[24] = __fmaf_rn(o_read_0[24], _shfl_sync_452, o_epi[24]);
                    float _shfl_sync_453 = __shfl_sync(0xFFFFFFFF, w_hold[0], 25);
                    o_epi[25] = __fmaf_rn(o_read_0[25], _shfl_sync_453, o_epi[25]);
                    float _shfl_sync_454 = __shfl_sync(0xFFFFFFFF, w_hold[0], 26);
                    o_epi[26] = __fmaf_rn(o_read_0[26], _shfl_sync_454, o_epi[26]);
                    float _shfl_sync_455 = __shfl_sync(0xFFFFFFFF, w_hold[0], 27);
                    o_epi[27] = __fmaf_rn(o_read_0[27], _shfl_sync_455, o_epi[27]);
                    float _shfl_sync_456 = __shfl_sync(0xFFFFFFFF, w_hold[0], 28);
                    o_epi[28] = __fmaf_rn(o_read_0[28], _shfl_sync_456, o_epi[28]);
                    float _shfl_sync_457 = __shfl_sync(0xFFFFFFFF, w_hold[0], 29);
                    o_epi[29] = __fmaf_rn(o_read_0[29], _shfl_sync_457, o_epi[29]);
                    float _shfl_sync_458 = __shfl_sync(0xFFFFFFFF, w_hold[0], 30);
                    o_epi[30] = __fmaf_rn(o_read_0[30], _shfl_sync_458, o_epi[30]);
                    float _shfl_sync_459 = __shfl_sync(0xFFFFFFFF, w_hold[0], 31);
                    o_epi[31] = __fmaf_rn(o_read_0[31], _shfl_sync_459, o_epi[31]);
                    float o_read_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_1[0]), "=f"(o_read_1[1]), "=f"(o_read_1[2]), "=f"(o_read_1[3]), "=f"(o_read_1[4]), "=f"(o_read_1[5]), "=f"(o_read_1[6]), "=f"(o_read_1[7]), "=f"(o_read_1[8]), "=f"(o_read_1[9]), "=f"(o_read_1[10]), "=f"(o_read_1[11]), "=f"(o_read_1[12]), "=f"(o_read_1[13]), "=f"(o_read_1[14]), "=f"(o_read_1[15]), "=f"(o_read_1[16]), "=f"(o_read_1[17]), "=f"(o_read_1[18]), "=f"(o_read_1[19]), "=f"(o_read_1[20]), "=f"(o_read_1[21]), "=f"(o_read_1[22]), "=f"(o_read_1[23]), "=f"(o_read_1[24]), "=f"(o_read_1[25]), "=f"(o_read_1[26]), "=f"(o_read_1[27]), "=f"(o_read_1[28]), "=f"(o_read_1[29]), "=f"(o_read_1[30]), "=f"(o_read_1[31])
                        : "r"(taddr + 320 + (unsigned int)lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_460 = __shfl_sync(0xFFFFFFFF, w_hold[1], 0);
                    o_epi[0] = __fmaf_rn(o_read_1[0], _shfl_sync_460, o_epi[0]);
                    float _shfl_sync_461 = __shfl_sync(0xFFFFFFFF, w_hold[1], 1);
                    o_epi[1] = __fmaf_rn(o_read_1[1], _shfl_sync_461, o_epi[1]);
                    float _shfl_sync_462 = __shfl_sync(0xFFFFFFFF, w_hold[1], 2);
                    o_epi[2] = __fmaf_rn(o_read_1[2], _shfl_sync_462, o_epi[2]);
                    float _shfl_sync_463 = __shfl_sync(0xFFFFFFFF, w_hold[1], 3);
                    o_epi[3] = __fmaf_rn(o_read_1[3], _shfl_sync_463, o_epi[3]);
                    float _shfl_sync_464 = __shfl_sync(0xFFFFFFFF, w_hold[1], 4);
                    o_epi[4] = __fmaf_rn(o_read_1[4], _shfl_sync_464, o_epi[4]);
                    float _shfl_sync_465 = __shfl_sync(0xFFFFFFFF, w_hold[1], 5);
                    o_epi[5] = __fmaf_rn(o_read_1[5], _shfl_sync_465, o_epi[5]);
                    float _shfl_sync_466 = __shfl_sync(0xFFFFFFFF, w_hold[1], 6);
                    o_epi[6] = __fmaf_rn(o_read_1[6], _shfl_sync_466, o_epi[6]);
                    float _shfl_sync_467 = __shfl_sync(0xFFFFFFFF, w_hold[1], 7);
                    o_epi[7] = __fmaf_rn(o_read_1[7], _shfl_sync_467, o_epi[7]);
                    float _shfl_sync_468 = __shfl_sync(0xFFFFFFFF, w_hold[1], 8);
                    o_epi[8] = __fmaf_rn(o_read_1[8], _shfl_sync_468, o_epi[8]);
                    float _shfl_sync_469 = __shfl_sync(0xFFFFFFFF, w_hold[1], 9);
                    o_epi[9] = __fmaf_rn(o_read_1[9], _shfl_sync_469, o_epi[9]);
                    float _shfl_sync_470 = __shfl_sync(0xFFFFFFFF, w_hold[1], 10);
                    o_epi[10] = __fmaf_rn(o_read_1[10], _shfl_sync_470, o_epi[10]);
                    float _shfl_sync_471 = __shfl_sync(0xFFFFFFFF, w_hold[1], 11);
                    o_epi[11] = __fmaf_rn(o_read_1[11], _shfl_sync_471, o_epi[11]);
                    float _shfl_sync_472 = __shfl_sync(0xFFFFFFFF, w_hold[1], 12);
                    o_epi[12] = __fmaf_rn(o_read_1[12], _shfl_sync_472, o_epi[12]);
                    float _shfl_sync_473 = __shfl_sync(0xFFFFFFFF, w_hold[1], 13);
                    o_epi[13] = __fmaf_rn(o_read_1[13], _shfl_sync_473, o_epi[13]);
                    float _shfl_sync_474 = __shfl_sync(0xFFFFFFFF, w_hold[1], 14);
                    o_epi[14] = __fmaf_rn(o_read_1[14], _shfl_sync_474, o_epi[14]);
                    float _shfl_sync_475 = __shfl_sync(0xFFFFFFFF, w_hold[1], 15);
                    o_epi[15] = __fmaf_rn(o_read_1[15], _shfl_sync_475, o_epi[15]);
                    float _shfl_sync_476 = __shfl_sync(0xFFFFFFFF, w_hold[1], 16);
                    o_epi[16] = __fmaf_rn(o_read_1[16], _shfl_sync_476, o_epi[16]);
                    float _shfl_sync_477 = __shfl_sync(0xFFFFFFFF, w_hold[1], 17);
                    o_epi[17] = __fmaf_rn(o_read_1[17], _shfl_sync_477, o_epi[17]);
                    float _shfl_sync_478 = __shfl_sync(0xFFFFFFFF, w_hold[1], 18);
                    o_epi[18] = __fmaf_rn(o_read_1[18], _shfl_sync_478, o_epi[18]);
                    float _shfl_sync_479 = __shfl_sync(0xFFFFFFFF, w_hold[1], 19);
                    o_epi[19] = __fmaf_rn(o_read_1[19], _shfl_sync_479, o_epi[19]);
                    float _shfl_sync_480 = __shfl_sync(0xFFFFFFFF, w_hold[1], 20);
                    o_epi[20] = __fmaf_rn(o_read_1[20], _shfl_sync_480, o_epi[20]);
                    float _shfl_sync_481 = __shfl_sync(0xFFFFFFFF, w_hold[1], 21);
                    o_epi[21] = __fmaf_rn(o_read_1[21], _shfl_sync_481, o_epi[21]);
                    float _shfl_sync_482 = __shfl_sync(0xFFFFFFFF, w_hold[1], 22);
                    o_epi[22] = __fmaf_rn(o_read_1[22], _shfl_sync_482, o_epi[22]);
                    float _shfl_sync_483 = __shfl_sync(0xFFFFFFFF, w_hold[1], 23);
                    o_epi[23] = __fmaf_rn(o_read_1[23], _shfl_sync_483, o_epi[23]);
                    float _shfl_sync_484 = __shfl_sync(0xFFFFFFFF, w_hold[1], 24);
                    o_epi[24] = __fmaf_rn(o_read_1[24], _shfl_sync_484, o_epi[24]);
                    float _shfl_sync_485 = __shfl_sync(0xFFFFFFFF, w_hold[1], 25);
                    o_epi[25] = __fmaf_rn(o_read_1[25], _shfl_sync_485, o_epi[25]);
                    float _shfl_sync_486 = __shfl_sync(0xFFFFFFFF, w_hold[1], 26);
                    o_epi[26] = __fmaf_rn(o_read_1[26], _shfl_sync_486, o_epi[26]);
                    float _shfl_sync_487 = __shfl_sync(0xFFFFFFFF, w_hold[1], 27);
                    o_epi[27] = __fmaf_rn(o_read_1[27], _shfl_sync_487, o_epi[27]);
                    float _shfl_sync_488 = __shfl_sync(0xFFFFFFFF, w_hold[1], 28);
                    o_epi[28] = __fmaf_rn(o_read_1[28], _shfl_sync_488, o_epi[28]);
                    float _shfl_sync_489 = __shfl_sync(0xFFFFFFFF, w_hold[1], 29);
                    o_epi[29] = __fmaf_rn(o_read_1[29], _shfl_sync_489, o_epi[29]);
                    float _shfl_sync_490 = __shfl_sync(0xFFFFFFFF, w_hold[1], 30);
                    o_epi[30] = __fmaf_rn(o_read_1[30], _shfl_sync_490, o_epi[30]);
                    float _shfl_sync_491 = __shfl_sync(0xFFFFFFFF, w_hold[1], 31);
                    o_epi[31] = __fmaf_rn(o_read_1[31], _shfl_sync_491, o_epi[31]);
                    float o_read_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_2[0]), "=f"(o_read_2[1]), "=f"(o_read_2[2]), "=f"(o_read_2[3]), "=f"(o_read_2[4]), "=f"(o_read_2[5]), "=f"(o_read_2[6]), "=f"(o_read_2[7]), "=f"(o_read_2[8]), "=f"(o_read_2[9]), "=f"(o_read_2[10]), "=f"(o_read_2[11]), "=f"(o_read_2[12]), "=f"(o_read_2[13]), "=f"(o_read_2[14]), "=f"(o_read_2[15]), "=f"(o_read_2[16]), "=f"(o_read_2[17]), "=f"(o_read_2[18]), "=f"(o_read_2[19]), "=f"(o_read_2[20]), "=f"(o_read_2[21]), "=f"(o_read_2[22]), "=f"(o_read_2[23]), "=f"(o_read_2[24]), "=f"(o_read_2[25]), "=f"(o_read_2[26]), "=f"(o_read_2[27]), "=f"(o_read_2[28]), "=f"(o_read_2[29]), "=f"(o_read_2[30]), "=f"(o_read_2[31])
                        : "r"(taddr + 384 + (unsigned int)lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_492 = __shfl_sync(0xFFFFFFFF, w_hold[2], 0);
                    o_epi[0] = __fmaf_rn(o_read_2[0], _shfl_sync_492, o_epi[0]);
                    float _shfl_sync_493 = __shfl_sync(0xFFFFFFFF, w_hold[2], 1);
                    o_epi[1] = __fmaf_rn(o_read_2[1], _shfl_sync_493, o_epi[1]);
                    float _shfl_sync_494 = __shfl_sync(0xFFFFFFFF, w_hold[2], 2);
                    o_epi[2] = __fmaf_rn(o_read_2[2], _shfl_sync_494, o_epi[2]);
                    float _shfl_sync_495 = __shfl_sync(0xFFFFFFFF, w_hold[2], 3);
                    o_epi[3] = __fmaf_rn(o_read_2[3], _shfl_sync_495, o_epi[3]);
                    float _shfl_sync_496 = __shfl_sync(0xFFFFFFFF, w_hold[2], 4);
                    o_epi[4] = __fmaf_rn(o_read_2[4], _shfl_sync_496, o_epi[4]);
                    float _shfl_sync_497 = __shfl_sync(0xFFFFFFFF, w_hold[2], 5);
                    o_epi[5] = __fmaf_rn(o_read_2[5], _shfl_sync_497, o_epi[5]);
                    float _shfl_sync_498 = __shfl_sync(0xFFFFFFFF, w_hold[2], 6);
                    o_epi[6] = __fmaf_rn(o_read_2[6], _shfl_sync_498, o_epi[6]);
                    float _shfl_sync_499 = __shfl_sync(0xFFFFFFFF, w_hold[2], 7);
                    o_epi[7] = __fmaf_rn(o_read_2[7], _shfl_sync_499, o_epi[7]);
                    float _shfl_sync_500 = __shfl_sync(0xFFFFFFFF, w_hold[2], 8);
                    o_epi[8] = __fmaf_rn(o_read_2[8], _shfl_sync_500, o_epi[8]);
                    float _shfl_sync_501 = __shfl_sync(0xFFFFFFFF, w_hold[2], 9);
                    o_epi[9] = __fmaf_rn(o_read_2[9], _shfl_sync_501, o_epi[9]);
                    float _shfl_sync_502 = __shfl_sync(0xFFFFFFFF, w_hold[2], 10);
                    o_epi[10] = __fmaf_rn(o_read_2[10], _shfl_sync_502, o_epi[10]);
                    float _shfl_sync_503 = __shfl_sync(0xFFFFFFFF, w_hold[2], 11);
                    o_epi[11] = __fmaf_rn(o_read_2[11], _shfl_sync_503, o_epi[11]);
                    float _shfl_sync_504 = __shfl_sync(0xFFFFFFFF, w_hold[2], 12);
                    o_epi[12] = __fmaf_rn(o_read_2[12], _shfl_sync_504, o_epi[12]);
                    float _shfl_sync_505 = __shfl_sync(0xFFFFFFFF, w_hold[2], 13);
                    o_epi[13] = __fmaf_rn(o_read_2[13], _shfl_sync_505, o_epi[13]);
                    float _shfl_sync_506 = __shfl_sync(0xFFFFFFFF, w_hold[2], 14);
                    o_epi[14] = __fmaf_rn(o_read_2[14], _shfl_sync_506, o_epi[14]);
                    float _shfl_sync_507 = __shfl_sync(0xFFFFFFFF, w_hold[2], 15);
                    o_epi[15] = __fmaf_rn(o_read_2[15], _shfl_sync_507, o_epi[15]);
                    float _shfl_sync_508 = __shfl_sync(0xFFFFFFFF, w_hold[2], 16);
                    o_epi[16] = __fmaf_rn(o_read_2[16], _shfl_sync_508, o_epi[16]);
                    float _shfl_sync_509 = __shfl_sync(0xFFFFFFFF, w_hold[2], 17);
                    o_epi[17] = __fmaf_rn(o_read_2[17], _shfl_sync_509, o_epi[17]);
                    float _shfl_sync_510 = __shfl_sync(0xFFFFFFFF, w_hold[2], 18);
                    o_epi[18] = __fmaf_rn(o_read_2[18], _shfl_sync_510, o_epi[18]);
                    float _shfl_sync_511 = __shfl_sync(0xFFFFFFFF, w_hold[2], 19);
                    o_epi[19] = __fmaf_rn(o_read_2[19], _shfl_sync_511, o_epi[19]);
                    float _shfl_sync_512 = __shfl_sync(0xFFFFFFFF, w_hold[2], 20);
                    o_epi[20] = __fmaf_rn(o_read_2[20], _shfl_sync_512, o_epi[20]);
                    float _shfl_sync_513 = __shfl_sync(0xFFFFFFFF, w_hold[2], 21);
                    o_epi[21] = __fmaf_rn(o_read_2[21], _shfl_sync_513, o_epi[21]);
                    float _shfl_sync_514 = __shfl_sync(0xFFFFFFFF, w_hold[2], 22);
                    o_epi[22] = __fmaf_rn(o_read_2[22], _shfl_sync_514, o_epi[22]);
                    float _shfl_sync_515 = __shfl_sync(0xFFFFFFFF, w_hold[2], 23);
                    o_epi[23] = __fmaf_rn(o_read_2[23], _shfl_sync_515, o_epi[23]);
                    float _shfl_sync_516 = __shfl_sync(0xFFFFFFFF, w_hold[2], 24);
                    o_epi[24] = __fmaf_rn(o_read_2[24], _shfl_sync_516, o_epi[24]);
                    float _shfl_sync_517 = __shfl_sync(0xFFFFFFFF, w_hold[2], 25);
                    o_epi[25] = __fmaf_rn(o_read_2[25], _shfl_sync_517, o_epi[25]);
                    float _shfl_sync_518 = __shfl_sync(0xFFFFFFFF, w_hold[2], 26);
                    o_epi[26] = __fmaf_rn(o_read_2[26], _shfl_sync_518, o_epi[26]);
                    float _shfl_sync_519 = __shfl_sync(0xFFFFFFFF, w_hold[2], 27);
                    o_epi[27] = __fmaf_rn(o_read_2[27], _shfl_sync_519, o_epi[27]);
                    float _shfl_sync_520 = __shfl_sync(0xFFFFFFFF, w_hold[2], 28);
                    o_epi[28] = __fmaf_rn(o_read_2[28], _shfl_sync_520, o_epi[28]);
                    float _shfl_sync_521 = __shfl_sync(0xFFFFFFFF, w_hold[2], 29);
                    o_epi[29] = __fmaf_rn(o_read_2[29], _shfl_sync_521, o_epi[29]);
                    float _shfl_sync_522 = __shfl_sync(0xFFFFFFFF, w_hold[2], 30);
                    o_epi[30] = __fmaf_rn(o_read_2[30], _shfl_sync_522, o_epi[30]);
                    float _shfl_sync_523 = __shfl_sync(0xFFFFFFFF, w_hold[2], 31);
                    o_epi[31] = __fmaf_rn(o_read_2[31], _shfl_sync_523, o_epi[31]);
                    float o_read_3[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_3[0]), "=f"(o_read_3[1]), "=f"(o_read_3[2]), "=f"(o_read_3[3]), "=f"(o_read_3[4]), "=f"(o_read_3[5]), "=f"(o_read_3[6]), "=f"(o_read_3[7]), "=f"(o_read_3[8]), "=f"(o_read_3[9]), "=f"(o_read_3[10]), "=f"(o_read_3[11]), "=f"(o_read_3[12]), "=f"(o_read_3[13]), "=f"(o_read_3[14]), "=f"(o_read_3[15]), "=f"(o_read_3[16]), "=f"(o_read_3[17]), "=f"(o_read_3[18]), "=f"(o_read_3[19]), "=f"(o_read_3[20]), "=f"(o_read_3[21]), "=f"(o_read_3[22]), "=f"(o_read_3[23]), "=f"(o_read_3[24]), "=f"(o_read_3[25]), "=f"(o_read_3[26]), "=f"(o_read_3[27]), "=f"(o_read_3[28]), "=f"(o_read_3[29]), "=f"(o_read_3[30]), "=f"(o_read_3[31])
                        : "r"(taddr + 448 + (unsigned int)lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_524 = __shfl_sync(0xFFFFFFFF, w_hold[3], 0);
                    o_epi[0] = __fmaf_rn(o_read_3[0], _shfl_sync_524, o_epi[0]);
                    float _shfl_sync_525 = __shfl_sync(0xFFFFFFFF, w_hold[3], 1);
                    o_epi[1] = __fmaf_rn(o_read_3[1], _shfl_sync_525, o_epi[1]);
                    float _shfl_sync_526 = __shfl_sync(0xFFFFFFFF, w_hold[3], 2);
                    o_epi[2] = __fmaf_rn(o_read_3[2], _shfl_sync_526, o_epi[2]);
                    float _shfl_sync_527 = __shfl_sync(0xFFFFFFFF, w_hold[3], 3);
                    o_epi[3] = __fmaf_rn(o_read_3[3], _shfl_sync_527, o_epi[3]);
                    float _shfl_sync_528 = __shfl_sync(0xFFFFFFFF, w_hold[3], 4);
                    o_epi[4] = __fmaf_rn(o_read_3[4], _shfl_sync_528, o_epi[4]);
                    float _shfl_sync_529 = __shfl_sync(0xFFFFFFFF, w_hold[3], 5);
                    o_epi[5] = __fmaf_rn(o_read_3[5], _shfl_sync_529, o_epi[5]);
                    float _shfl_sync_530 = __shfl_sync(0xFFFFFFFF, w_hold[3], 6);
                    o_epi[6] = __fmaf_rn(o_read_3[6], _shfl_sync_530, o_epi[6]);
                    float _shfl_sync_531 = __shfl_sync(0xFFFFFFFF, w_hold[3], 7);
                    o_epi[7] = __fmaf_rn(o_read_3[7], _shfl_sync_531, o_epi[7]);
                    float _shfl_sync_532 = __shfl_sync(0xFFFFFFFF, w_hold[3], 8);
                    o_epi[8] = __fmaf_rn(o_read_3[8], _shfl_sync_532, o_epi[8]);
                    float _shfl_sync_533 = __shfl_sync(0xFFFFFFFF, w_hold[3], 9);
                    o_epi[9] = __fmaf_rn(o_read_3[9], _shfl_sync_533, o_epi[9]);
                    float _shfl_sync_534 = __shfl_sync(0xFFFFFFFF, w_hold[3], 10);
                    o_epi[10] = __fmaf_rn(o_read_3[10], _shfl_sync_534, o_epi[10]);
                    float _shfl_sync_535 = __shfl_sync(0xFFFFFFFF, w_hold[3], 11);
                    o_epi[11] = __fmaf_rn(o_read_3[11], _shfl_sync_535, o_epi[11]);
                    float _shfl_sync_536 = __shfl_sync(0xFFFFFFFF, w_hold[3], 12);
                    o_epi[12] = __fmaf_rn(o_read_3[12], _shfl_sync_536, o_epi[12]);
                    float _shfl_sync_537 = __shfl_sync(0xFFFFFFFF, w_hold[3], 13);
                    o_epi[13] = __fmaf_rn(o_read_3[13], _shfl_sync_537, o_epi[13]);
                    float _shfl_sync_538 = __shfl_sync(0xFFFFFFFF, w_hold[3], 14);
                    o_epi[14] = __fmaf_rn(o_read_3[14], _shfl_sync_538, o_epi[14]);
                    float _shfl_sync_539 = __shfl_sync(0xFFFFFFFF, w_hold[3], 15);
                    o_epi[15] = __fmaf_rn(o_read_3[15], _shfl_sync_539, o_epi[15]);
                    float _shfl_sync_540 = __shfl_sync(0xFFFFFFFF, w_hold[3], 16);
                    o_epi[16] = __fmaf_rn(o_read_3[16], _shfl_sync_540, o_epi[16]);
                    float _shfl_sync_541 = __shfl_sync(0xFFFFFFFF, w_hold[3], 17);
                    o_epi[17] = __fmaf_rn(o_read_3[17], _shfl_sync_541, o_epi[17]);
                    float _shfl_sync_542 = __shfl_sync(0xFFFFFFFF, w_hold[3], 18);
                    o_epi[18] = __fmaf_rn(o_read_3[18], _shfl_sync_542, o_epi[18]);
                    float _shfl_sync_543 = __shfl_sync(0xFFFFFFFF, w_hold[3], 19);
                    o_epi[19] = __fmaf_rn(o_read_3[19], _shfl_sync_543, o_epi[19]);
                    float _shfl_sync_544 = __shfl_sync(0xFFFFFFFF, w_hold[3], 20);
                    o_epi[20] = __fmaf_rn(o_read_3[20], _shfl_sync_544, o_epi[20]);
                    float _shfl_sync_545 = __shfl_sync(0xFFFFFFFF, w_hold[3], 21);
                    o_epi[21] = __fmaf_rn(o_read_3[21], _shfl_sync_545, o_epi[21]);
                    float _shfl_sync_546 = __shfl_sync(0xFFFFFFFF, w_hold[3], 22);
                    o_epi[22] = __fmaf_rn(o_read_3[22], _shfl_sync_546, o_epi[22]);
                    float _shfl_sync_547 = __shfl_sync(0xFFFFFFFF, w_hold[3], 23);
                    o_epi[23] = __fmaf_rn(o_read_3[23], _shfl_sync_547, o_epi[23]);
                    float _shfl_sync_548 = __shfl_sync(0xFFFFFFFF, w_hold[3], 24);
                    o_epi[24] = __fmaf_rn(o_read_3[24], _shfl_sync_548, o_epi[24]);
                    float _shfl_sync_549 = __shfl_sync(0xFFFFFFFF, w_hold[3], 25);
                    o_epi[25] = __fmaf_rn(o_read_3[25], _shfl_sync_549, o_epi[25]);
                    float _shfl_sync_550 = __shfl_sync(0xFFFFFFFF, w_hold[3], 26);
                    o_epi[26] = __fmaf_rn(o_read_3[26], _shfl_sync_550, o_epi[26]);
                    float _shfl_sync_551 = __shfl_sync(0xFFFFFFFF, w_hold[3], 27);
                    o_epi[27] = __fmaf_rn(o_read_3[27], _shfl_sync_551, o_epi[27]);
                    float _shfl_sync_552 = __shfl_sync(0xFFFFFFFF, w_hold[3], 28);
                    o_epi[28] = __fmaf_rn(o_read_3[28], _shfl_sync_552, o_epi[28]);
                    float _shfl_sync_553 = __shfl_sync(0xFFFFFFFF, w_hold[3], 29);
                    o_epi[29] = __fmaf_rn(o_read_3[29], _shfl_sync_553, o_epi[29]);
                    float _shfl_sync_554 = __shfl_sync(0xFFFFFFFF, w_hold[3], 30);
                    o_epi[30] = __fmaf_rn(o_read_3[30], _shfl_sync_554, o_epi[30]);
                    float _shfl_sync_555 = __shfl_sync(0xFFFFFFFF, w_hold[3], 31);
                    o_epi[31] = __fmaf_rn(o_read_3[31], _shfl_sync_555, o_epi[31]);
                }
                if (work_idx + (unsigned int)gridDim.x >= total_items) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                int q_row0 = request * 8 * num_q_heads + head0;
                if (warp_in_role == 0) {
                    int gc = 0 + lane;
                    int lse_row = q_row0 + gc / 8 * num_q_heads + gc % 8;
                    float lse = -CUDART_INF_F;
                    if (head_sum > 0.0f) {
                        float _log2_556;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_556) : "f"(head_sum));
                        lse = __fmaf_rn(m_fin * softmax_scale_log2, 0.6931472f, _log2_556 * 0.6931472f);
                    }
                    *(reinterpret_cast<float*>(msa_lse + lse_row) + (0)) = lse;
                }
                int row_c_0 = q_row0 + 0 * num_q_heads + 0;
                float _shfl_sync_557 = __shfl_sync(0xFFFFFFFF, inv, 0);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_0 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[0] * _shfl_sync_557);
                int row_c_1 = q_row0 + 0 * num_q_heads + 1;
                float _shfl_sync_558 = __shfl_sync(0xFFFFFFFF, inv, 1);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_1 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[1] * _shfl_sync_558);
                int row_c_2 = q_row0 + 0 * num_q_heads + 2;
                float _shfl_sync_559 = __shfl_sync(0xFFFFFFFF, inv, 2);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_2 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[2] * _shfl_sync_559);
                int row_c_3 = q_row0 + 0 * num_q_heads + 3;
                float _shfl_sync_560 = __shfl_sync(0xFFFFFFFF, inv, 3);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_3 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[3] * _shfl_sync_560);
                int row_c_4 = q_row0 + 0 * num_q_heads + 4;
                float _shfl_sync_561 = __shfl_sync(0xFFFFFFFF, inv, 4);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_4 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[4] * _shfl_sync_561);
                int row_c_5 = q_row0 + 0 * num_q_heads + 5;
                float _shfl_sync_562 = __shfl_sync(0xFFFFFFFF, inv, 5);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_5 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[5] * _shfl_sync_562);
                int row_c_6 = q_row0 + 0 * num_q_heads + 6;
                float _shfl_sync_563 = __shfl_sync(0xFFFFFFFF, inv, 6);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_6 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[6] * _shfl_sync_563);
                int row_c_7 = q_row0 + 0 * num_q_heads + 7;
                float _shfl_sync_564 = __shfl_sync(0xFFFFFFFF, inv, 7);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_7 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[7] * _shfl_sync_564);
                int row_c_8 = q_row0 + 1 * num_q_heads + 0;
                float _shfl_sync_565 = __shfl_sync(0xFFFFFFFF, inv, 8);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_8 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[8] * _shfl_sync_565);
                int row_c_9 = q_row0 + 1 * num_q_heads + 1;
                float _shfl_sync_566 = __shfl_sync(0xFFFFFFFF, inv, 9);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_9 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[9] * _shfl_sync_566);
                int row_c_10 = q_row0 + 1 * num_q_heads + 2;
                float _shfl_sync_567 = __shfl_sync(0xFFFFFFFF, inv, 10);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_10 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[10] * _shfl_sync_567);
                int row_c_11 = q_row0 + 1 * num_q_heads + 3;
                float _shfl_sync_568 = __shfl_sync(0xFFFFFFFF, inv, 11);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_11 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[11] * _shfl_sync_568);
                int row_c_12 = q_row0 + 1 * num_q_heads + 4;
                float _shfl_sync_569 = __shfl_sync(0xFFFFFFFF, inv, 12);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_12 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[12] * _shfl_sync_569);
                int row_c_13 = q_row0 + 1 * num_q_heads + 5;
                float _shfl_sync_570 = __shfl_sync(0xFFFFFFFF, inv, 13);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_13 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[13] * _shfl_sync_570);
                int row_c_14 = q_row0 + 1 * num_q_heads + 6;
                float _shfl_sync_571 = __shfl_sync(0xFFFFFFFF, inv, 14);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_14 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[14] * _shfl_sync_571);
                int row_c_15 = q_row0 + 1 * num_q_heads + 7;
                float _shfl_sync_572 = __shfl_sync(0xFFFFFFFF, inv, 15);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_15 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[15] * _shfl_sync_572);
                int row_c_16 = q_row0 + 2 * num_q_heads + 0;
                float _shfl_sync_573 = __shfl_sync(0xFFFFFFFF, inv, 16);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_16 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[16] * _shfl_sync_573);
                int row_c_17 = q_row0 + 2 * num_q_heads + 1;
                float _shfl_sync_574 = __shfl_sync(0xFFFFFFFF, inv, 17);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_17 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[17] * _shfl_sync_574);
                int row_c_18 = q_row0 + 2 * num_q_heads + 2;
                float _shfl_sync_575 = __shfl_sync(0xFFFFFFFF, inv, 18);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_18 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[18] * _shfl_sync_575);
                int row_c_19 = q_row0 + 2 * num_q_heads + 3;
                float _shfl_sync_576 = __shfl_sync(0xFFFFFFFF, inv, 19);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_19 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[19] * _shfl_sync_576);
                int row_c_20 = q_row0 + 2 * num_q_heads + 4;
                float _shfl_sync_577 = __shfl_sync(0xFFFFFFFF, inv, 20);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_20 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[20] * _shfl_sync_577);
                int row_c_21 = q_row0 + 2 * num_q_heads + 5;
                float _shfl_sync_578 = __shfl_sync(0xFFFFFFFF, inv, 21);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_21 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[21] * _shfl_sync_578);
                int row_c_22 = q_row0 + 2 * num_q_heads + 6;
                float _shfl_sync_579 = __shfl_sync(0xFFFFFFFF, inv, 22);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_22 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[22] * _shfl_sync_579);
                int row_c_23 = q_row0 + 2 * num_q_heads + 7;
                float _shfl_sync_580 = __shfl_sync(0xFFFFFFFF, inv, 23);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_23 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[23] * _shfl_sync_580);
                int row_c_24 = q_row0 + 3 * num_q_heads + 0;
                float _shfl_sync_581 = __shfl_sync(0xFFFFFFFF, inv, 24);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_24 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[24] * _shfl_sync_581);
                int row_c_25 = q_row0 + 3 * num_q_heads + 1;
                float _shfl_sync_582 = __shfl_sync(0xFFFFFFFF, inv, 25);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_25 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[25] * _shfl_sync_582);
                int row_c_26 = q_row0 + 3 * num_q_heads + 2;
                float _shfl_sync_583 = __shfl_sync(0xFFFFFFFF, inv, 26);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_26 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[26] * _shfl_sync_583);
                int row_c_27 = q_row0 + 3 * num_q_heads + 3;
                float _shfl_sync_584 = __shfl_sync(0xFFFFFFFF, inv, 27);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_27 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[27] * _shfl_sync_584);
                int row_c_28 = q_row0 + 3 * num_q_heads + 4;
                float _shfl_sync_585 = __shfl_sync(0xFFFFFFFF, inv, 28);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_28 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[28] * _shfl_sync_585);
                int row_c_29 = q_row0 + 3 * num_q_heads + 5;
                float _shfl_sync_586 = __shfl_sync(0xFFFFFFFF, inv, 29);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_29 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[29] * _shfl_sync_586);
                int row_c_30 = q_row0 + 3 * num_q_heads + 6;
                float _shfl_sync_587 = __shfl_sync(0xFFFFFFFF, inv, 30);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_30 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[30] * _shfl_sync_587);
                int row_c_31 = q_row0 + 3 * num_q_heads + 7;
                float _shfl_sync_588 = __shfl_sync(0xFFFFFFFF, inv, 31);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_31 * 128 + key_row)) + (0)) = (__nv_bfloat16)(o_epi[31] * _shfl_sync_588);
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(decode_done_addr);
                }
                item_ord += 1;
            }
        }
    }
    // ---- Role: softmax_1 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // softmax_1_main
            const int lane_base_1 = warp % 4 * 32 << 16;
            const int warp_in_role_1 = warp % 4;
            const int key_row_1 = warp_in_role_1 * 32 + lane;
            const int own_half_1 = warp_in_role_1 / 2;
            const int grp_shift_1 = lane / 8 * 8;
            const unsigned int grp_mask_1 = 255;
            float tau_1 = 8.0f;
            int g_real_w_1 = num_q_heads / num_kv_heads;
            int heads_per_req_w_1 = num_kv_heads * (g_real_w_1 / 8);
            unsigned int total_items_1 = total_q / 8 * heads_per_req_w_1;
            int item_stage_1 = 0;
            int s_stage_1 = 0;
            int p_stage_1 = 0;
            int item_ord_1 = 0;
            int gpair_1 = 0;
            unsigned int _phase_meta_full_1 = 0;
            unsigned int _phase_s_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = blockIdx.x; work_idx_1 < total_items_1; work_idx_1 += gridDim.x) {
                int item_1 = work_idx_1;
                int g_real_1 = num_q_heads / num_kv_heads;
                int halves_1 = g_real_1 / 8;
                int heads_per_req_1 = num_kv_heads * halves_1;
                int request_1 = item_1 / heads_per_req_1;
                int sub_1 = item_1 % heads_per_req_1;
                int kv_head_1 = sub_1 / halves_1;
                int head0_1 = kv_head_1 * g_real_1 + sub_1 % halves_1 * 8;
                mbarrier_wait(meta_full_addr + (item_stage_1) * 8, _phase_meta_full_1);
                int npairs_1 = smem_npairs[item_stage_1];
                int kvlen_1 = smem_kvlen[item_stage_1];
                int meta_base_1 = item_stage_1 * 50;
                item_stage_1 += 1;
                if (item_stage_1 == 2) { item_stage_1 = 0; _phase_meta_full_1 ^= 1; }
                float o_hold_1[4];
                float ms_hold_1[4];
                o_hold_1[0] = -CUDART_INF_F;
                ms_hold_1[0] = 0.0f;
                o_hold_1[1] = -CUDART_INF_F;
                ms_hold_1[1] = 0.0f;
                o_hold_1[2] = -CUDART_INF_F;
                ms_hold_1[2] = 0.0f;
                o_hold_1[3] = -CUDART_INF_F;
                ms_hold_1[3] = 0.0f;
                float lane_sum_1[64];
                lane_sum_1[0] = 0.0f;
                lane_sum_1[1] = 0.0f;
                lane_sum_1[2] = 0.0f;
                lane_sum_1[3] = 0.0f;
                lane_sum_1[4] = 0.0f;
                lane_sum_1[5] = 0.0f;
                lane_sum_1[6] = 0.0f;
                lane_sum_1[7] = 0.0f;
                lane_sum_1[8] = 0.0f;
                lane_sum_1[9] = 0.0f;
                lane_sum_1[10] = 0.0f;
                lane_sum_1[11] = 0.0f;
                lane_sum_1[12] = 0.0f;
                lane_sum_1[13] = 0.0f;
                lane_sum_1[14] = 0.0f;
                lane_sum_1[15] = 0.0f;
                lane_sum_1[16] = 0.0f;
                lane_sum_1[17] = 0.0f;
                lane_sum_1[18] = 0.0f;
                lane_sum_1[19] = 0.0f;
                lane_sum_1[20] = 0.0f;
                lane_sum_1[21] = 0.0f;
                lane_sum_1[22] = 0.0f;
                lane_sum_1[23] = 0.0f;
                lane_sum_1[24] = 0.0f;
                lane_sum_1[25] = 0.0f;
                lane_sum_1[26] = 0.0f;
                lane_sum_1[27] = 0.0f;
                lane_sum_1[28] = 0.0f;
                lane_sum_1[29] = 0.0f;
                lane_sum_1[30] = 0.0f;
                lane_sum_1[31] = 0.0f;
                lane_sum_1[32] = 0.0f;
                lane_sum_1[33] = 0.0f;
                lane_sum_1[34] = 0.0f;
                lane_sum_1[35] = 0.0f;
                lane_sum_1[36] = 0.0f;
                lane_sum_1[37] = 0.0f;
                lane_sum_1[38] = 0.0f;
                lane_sum_1[39] = 0.0f;
                lane_sum_1[40] = 0.0f;
                lane_sum_1[41] = 0.0f;
                lane_sum_1[42] = 0.0f;
                lane_sum_1[43] = 0.0f;
                lane_sum_1[44] = 0.0f;
                lane_sum_1[45] = 0.0f;
                lane_sum_1[46] = 0.0f;
                lane_sum_1[47] = 0.0f;
                lane_sum_1[48] = 0.0f;
                lane_sum_1[49] = 0.0f;
                lane_sum_1[50] = 0.0f;
                lane_sum_1[51] = 0.0f;
                lane_sum_1[52] = 0.0f;
                lane_sum_1[53] = 0.0f;
                lane_sum_1[54] = 0.0f;
                lane_sum_1[55] = 0.0f;
                lane_sum_1[56] = 0.0f;
                lane_sum_1[57] = 0.0f;
                lane_sum_1[58] = 0.0f;
                lane_sum_1[59] = 0.0f;
                lane_sum_1[60] = 0.0f;
                lane_sum_1[61] = 0.0f;
                lane_sum_1[62] = 0.0f;
                lane_sum_1[63] = 0.0f;
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 25; pair_1++) {
                    if (pair_1 >= npairs_1) {
                        break;
                    }
                    mbarrier_wait(s_full_addr + (s_stage_1) * 8, _phase_s_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr_1 = taddr + (unsigned int)(s_stage_1 * 128) + 32 + (unsigned int)lane_base_1;
                    float sv0_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv0_1[0]), "=f"(sv0_1[1]), "=f"(sv0_1[2]), "=f"(sv0_1[3]), "=f"(sv0_1[4]), "=f"(sv0_1[5]), "=f"(sv0_1[6]), "=f"(sv0_1[7]), "=f"(sv0_1[8]), "=f"(sv0_1[9]), "=f"(sv0_1[10]), "=f"(sv0_1[11]), "=f"(sv0_1[12]), "=f"(sv0_1[13]), "=f"(sv0_1[14]), "=f"(sv0_1[15]), "=f"(sv0_1[16]), "=f"(sv0_1[17]), "=f"(sv0_1[18]), "=f"(sv0_1[19]), "=f"(sv0_1[20]), "=f"(sv0_1[21]), "=f"(sv0_1[22]), "=f"(sv0_1[23]), "=f"(sv0_1[24]), "=f"(sv0_1[25]), "=f"(sv0_1[26]), "=f"(sv0_1[27]), "=f"(sv0_1[28]), "=f"(sv0_1[29]), "=f"(sv0_1[30]), "=f"(sv0_1[31])
                        : "r"(s_addr_1));
                    float sv1_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv1_1[0]), "=f"(sv1_1[1]), "=f"(sv1_1[2]), "=f"(sv1_1[3]), "=f"(sv1_1[4]), "=f"(sv1_1[5]), "=f"(sv1_1[6]), "=f"(sv1_1[7]), "=f"(sv1_1[8]), "=f"(sv1_1[9]), "=f"(sv1_1[10]), "=f"(sv1_1[11]), "=f"(sv1_1[12]), "=f"(sv1_1[13]), "=f"(sv1_1[14]), "=f"(sv1_1[15]), "=f"(sv1_1[16]), "=f"(sv1_1[17]), "=f"(sv1_1[18]), "=f"(sv1_1[19]), "=f"(sv1_1[20]), "=f"(sv1_1[21]), "=f"(sv1_1[22]), "=f"(sv1_1[23]), "=f"(sv1_1[24]), "=f"(sv1_1[25]), "=f"(sv1_1[26]), "=f"(sv1_1[27]), "=f"(sv1_1[28]), "=f"(sv1_1[29]), "=f"(sv1_1[30]), "=f"(sv1_1[31])
                        : "r"(s_addr_1 + 64));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage_1) * 8);
                    }
                    int blk_t_1 = smem_block[meta_base_1 + pair_1 * 2 + 0];
                    unsigned int tm_t_1 = (unsigned int)smem_tokmask[meta_base_1 + pair_1 * 2 + 0];
                    int vbase_1 = kvlen_1 - 7 - blk_t_1 * 128;
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[0] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[0] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[1] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[1] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[2] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[2] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[3] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[3] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[4] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[4] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[5] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[5] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[6] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[6] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 4) {
                        sv0_1[7] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 4 & 1) == 0) {
                        sv0_1[7] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[8] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[8] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[9] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[9] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[10] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[10] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[11] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[11] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[12] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[12] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[13] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[13] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[14] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[14] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 5) {
                        sv0_1[15] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 5 & 1) == 0) {
                        sv0_1[15] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[16] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[16] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[17] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[17] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[18] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[18] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[19] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[19] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[20] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[20] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[21] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[21] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[22] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[22] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 6) {
                        sv0_1[23] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 6 & 1) == 0) {
                        sv0_1[23] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[24] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[24] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[25] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[25] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[26] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[26] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[27] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[27] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[28] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[28] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[29] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[29] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[30] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[30] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1 + 7) {
                        sv0_1[31] = -CUDART_INF_F;
                    }
                    if ((tm_t_1 >> 7 & 1) == 0) {
                        sv0_1[31] = -CUDART_INF_F;
                    }
                    float hold_1 = -CUDART_INF_F;
                    float mx_64;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_64) : "f"(sv0_1[0]));
                    if (lane == 0) {
                        hold_1 = mx_64;
                    }
                    float mx_65;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_65) : "f"(sv0_1[1]));
                    if (lane == 1) {
                        hold_1 = mx_65;
                    }
                    float mx_66;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_66) : "f"(sv0_1[2]));
                    if (lane == 2) {
                        hold_1 = mx_66;
                    }
                    float mx_67;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_67) : "f"(sv0_1[3]));
                    if (lane == 3) {
                        hold_1 = mx_67;
                    }
                    float mx_68;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_68) : "f"(sv0_1[4]));
                    if (lane == 4) {
                        hold_1 = mx_68;
                    }
                    float mx_69;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_69) : "f"(sv0_1[5]));
                    if (lane == 5) {
                        hold_1 = mx_69;
                    }
                    float mx_70;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_70) : "f"(sv0_1[6]));
                    if (lane == 6) {
                        hold_1 = mx_70;
                    }
                    float mx_71;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_71) : "f"(sv0_1[7]));
                    if (lane == 7) {
                        hold_1 = mx_71;
                    }
                    float mx_72;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_72) : "f"(sv0_1[8]));
                    if (lane == 8) {
                        hold_1 = mx_72;
                    }
                    float mx_73;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_73) : "f"(sv0_1[9]));
                    if (lane == 9) {
                        hold_1 = mx_73;
                    }
                    float mx_74;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_74) : "f"(sv0_1[10]));
                    if (lane == 10) {
                        hold_1 = mx_74;
                    }
                    float mx_75;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_75) : "f"(sv0_1[11]));
                    if (lane == 11) {
                        hold_1 = mx_75;
                    }
                    float mx_76;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_76) : "f"(sv0_1[12]));
                    if (lane == 12) {
                        hold_1 = mx_76;
                    }
                    float mx_77;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_77) : "f"(sv0_1[13]));
                    if (lane == 13) {
                        hold_1 = mx_77;
                    }
                    float mx_78;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_78) : "f"(sv0_1[14]));
                    if (lane == 14) {
                        hold_1 = mx_78;
                    }
                    float mx_79;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_79) : "f"(sv0_1[15]));
                    if (lane == 15) {
                        hold_1 = mx_79;
                    }
                    float mx_80;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_80) : "f"(sv0_1[16]));
                    if (lane == 16) {
                        hold_1 = mx_80;
                    }
                    float mx_81;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_81) : "f"(sv0_1[17]));
                    if (lane == 17) {
                        hold_1 = mx_81;
                    }
                    float mx_82;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_82) : "f"(sv0_1[18]));
                    if (lane == 18) {
                        hold_1 = mx_82;
                    }
                    float mx_83;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_83) : "f"(sv0_1[19]));
                    if (lane == 19) {
                        hold_1 = mx_83;
                    }
                    float mx_84;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_84) : "f"(sv0_1[20]));
                    if (lane == 20) {
                        hold_1 = mx_84;
                    }
                    float mx_85;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_85) : "f"(sv0_1[21]));
                    if (lane == 21) {
                        hold_1 = mx_85;
                    }
                    float mx_86;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_86) : "f"(sv0_1[22]));
                    if (lane == 22) {
                        hold_1 = mx_86;
                    }
                    float mx_87;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_87) : "f"(sv0_1[23]));
                    if (lane == 23) {
                        hold_1 = mx_87;
                    }
                    float mx_88;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_88) : "f"(sv0_1[24]));
                    if (lane == 24) {
                        hold_1 = mx_88;
                    }
                    float mx_89;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_89) : "f"(sv0_1[25]));
                    if (lane == 25) {
                        hold_1 = mx_89;
                    }
                    float mx_90;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_90) : "f"(sv0_1[26]));
                    if (lane == 26) {
                        hold_1 = mx_90;
                    }
                    float mx_91;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_91) : "f"(sv0_1[27]));
                    if (lane == 27) {
                        hold_1 = mx_91;
                    }
                    float mx_92;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_92) : "f"(sv0_1[28]));
                    if (lane == 28) {
                        hold_1 = mx_92;
                    }
                    float mx_93;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_93) : "f"(sv0_1[29]));
                    if (lane == 29) {
                        hold_1 = mx_93;
                    }
                    float mx_94;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_94) : "f"(sv0_1[30]));
                    if (lane == 30) {
                        hold_1 = mx_94;
                    }
                    float mx_95;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_95) : "f"(sv0_1[31]));
                    if (lane == 31) {
                        hold_1 = mx_95;
                    }
                    int exch_slot_1 = ((2 + pair_1 % 2) * 2 + 0) * 4 * 32 + warp_in_role_1 * 32;
                    smem_exch[exch_slot_1 + lane] = hold_1;
                    int blk_t_1_1 = smem_block[meta_base_1 + pair_1 * 2 + 1];
                    unsigned int tm_t_1_1 = (unsigned int)smem_tokmask[meta_base_1 + pair_1 * 2 + 1];
                    int vbase_1_1 = kvlen_1 - 7 - blk_t_1_1 * 128;
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[0] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[0] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[1] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[1] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[2] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[2] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[3] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[3] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[4] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[4] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[5] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[5] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[6] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[6] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 4) {
                        sv1_1[7] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 4 & 1) == 0) {
                        sv1_1[7] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[8] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[8] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[9] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[9] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[10] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[10] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[11] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[11] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[12] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[12] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[13] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[13] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[14] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[14] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 5) {
                        sv1_1[15] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 5 & 1) == 0) {
                        sv1_1[15] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[16] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[16] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[17] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[17] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[18] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[18] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[19] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[19] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[20] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[20] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[21] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[21] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[22] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[22] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 6) {
                        sv1_1[23] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 6 & 1) == 0) {
                        sv1_1[23] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[24] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[24] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[25] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[25] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[26] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[26] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[27] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[27] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[28] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[28] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[29] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[29] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[30] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[30] = -CUDART_INF_F;
                    }
                    if (key_row_1 >= vbase_1_1 + 7) {
                        sv1_1[31] = -CUDART_INF_F;
                    }
                    if ((tm_t_1_1 >> 7 & 1) == 0) {
                        sv1_1[31] = -CUDART_INF_F;
                    }
                    float hold_1_1 = -CUDART_INF_F;
                    float mx_96;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_96) : "f"(sv1_1[0]));
                    if (lane == 0) {
                        hold_1_1 = mx_96;
                    }
                    float mx_97;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_97) : "f"(sv1_1[1]));
                    if (lane == 1) {
                        hold_1_1 = mx_97;
                    }
                    float mx_98;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_98) : "f"(sv1_1[2]));
                    if (lane == 2) {
                        hold_1_1 = mx_98;
                    }
                    float mx_99;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_99) : "f"(sv1_1[3]));
                    if (lane == 3) {
                        hold_1_1 = mx_99;
                    }
                    float mx_100;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_100) : "f"(sv1_1[4]));
                    if (lane == 4) {
                        hold_1_1 = mx_100;
                    }
                    float mx_101;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_101) : "f"(sv1_1[5]));
                    if (lane == 5) {
                        hold_1_1 = mx_101;
                    }
                    float mx_102;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_102) : "f"(sv1_1[6]));
                    if (lane == 6) {
                        hold_1_1 = mx_102;
                    }
                    float mx_103;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_103) : "f"(sv1_1[7]));
                    if (lane == 7) {
                        hold_1_1 = mx_103;
                    }
                    float mx_104;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_104) : "f"(sv1_1[8]));
                    if (lane == 8) {
                        hold_1_1 = mx_104;
                    }
                    float mx_105;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_105) : "f"(sv1_1[9]));
                    if (lane == 9) {
                        hold_1_1 = mx_105;
                    }
                    float mx_106;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_106) : "f"(sv1_1[10]));
                    if (lane == 10) {
                        hold_1_1 = mx_106;
                    }
                    float mx_107;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_107) : "f"(sv1_1[11]));
                    if (lane == 11) {
                        hold_1_1 = mx_107;
                    }
                    float mx_108;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_108) : "f"(sv1_1[12]));
                    if (lane == 12) {
                        hold_1_1 = mx_108;
                    }
                    float mx_109;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_109) : "f"(sv1_1[13]));
                    if (lane == 13) {
                        hold_1_1 = mx_109;
                    }
                    float mx_110;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_110) : "f"(sv1_1[14]));
                    if (lane == 14) {
                        hold_1_1 = mx_110;
                    }
                    float mx_111;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_111) : "f"(sv1_1[15]));
                    if (lane == 15) {
                        hold_1_1 = mx_111;
                    }
                    float mx_112;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_112) : "f"(sv1_1[16]));
                    if (lane == 16) {
                        hold_1_1 = mx_112;
                    }
                    float mx_113;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_113) : "f"(sv1_1[17]));
                    if (lane == 17) {
                        hold_1_1 = mx_113;
                    }
                    float mx_114;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_114) : "f"(sv1_1[18]));
                    if (lane == 18) {
                        hold_1_1 = mx_114;
                    }
                    float mx_115;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_115) : "f"(sv1_1[19]));
                    if (lane == 19) {
                        hold_1_1 = mx_115;
                    }
                    float mx_116;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_116) : "f"(sv1_1[20]));
                    if (lane == 20) {
                        hold_1_1 = mx_116;
                    }
                    float mx_117;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_117) : "f"(sv1_1[21]));
                    if (lane == 21) {
                        hold_1_1 = mx_117;
                    }
                    float mx_118;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_118) : "f"(sv1_1[22]));
                    if (lane == 22) {
                        hold_1_1 = mx_118;
                    }
                    float mx_119;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_119) : "f"(sv1_1[23]));
                    if (lane == 23) {
                        hold_1_1 = mx_119;
                    }
                    float mx_120;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_120) : "f"(sv1_1[24]));
                    if (lane == 24) {
                        hold_1_1 = mx_120;
                    }
                    float mx_121;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_121) : "f"(sv1_1[25]));
                    if (lane == 25) {
                        hold_1_1 = mx_121;
                    }
                    float mx_122;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_122) : "f"(sv1_1[26]));
                    if (lane == 26) {
                        hold_1_1 = mx_122;
                    }
                    float mx_123;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_123) : "f"(sv1_1[27]));
                    if (lane == 27) {
                        hold_1_1 = mx_123;
                    }
                    float mx_124;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_124) : "f"(sv1_1[28]));
                    if (lane == 28) {
                        hold_1_1 = mx_124;
                    }
                    float mx_125;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_125) : "f"(sv1_1[29]));
                    if (lane == 29) {
                        hold_1_1 = mx_125;
                    }
                    float mx_126;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_126) : "f"(sv1_1[30]));
                    if (lane == 30) {
                        hold_1_1 = mx_126;
                    }
                    float mx_127;
                    asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(mx_127) : "f"(sv1_1[31]));
                    if (lane == 31) {
                        hold_1_1 = mx_127;
                    }
                    int exch_slot_1_1 = ((2 + pair_1 % 2) * 2 + 1) * 4 * 32 + warp_in_role_1 * 32;
                    smem_exch[exch_slot_1_1 + lane] = hold_1_1;
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int grow_any_1 = 0;
                    float alpha_1[4];
                    unsigned int grown_mask_1[4];
                    alpha_1[0] = 1.0f;
                    grown_mask_1[0] = 0;
                    alpha_1[1] = 1.0f;
                    grown_mask_1[1] = 0;
                    alpha_1[2] = 1.0f;
                    grown_mask_1[2] = 0;
                    alpha_1[3] = 1.0f;
                    grown_mask_1[3] = 0;
                    int base_4 = ((2 + pair_1 % 2) * 2 + 0) * 4 * 32 + 0;
                    float m_s_4;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_4) : "f"(smem_exch[base_4 + lane]), "f"(smem_exch[base_4 + 32 + lane]));
                    float cand_4;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_4) : "f"(o_hold_1[0]), "f"(m_s_4));
                    int moved_4 = 0;
                    if ((cand_4 - o_hold_1[0]) * softmax_scale_log2 > tau_1) {
                        moved_4 = 1;
                    }
                    unsigned int bal_4 = __ballot_sync(0xFFFFFFFF, moved_4 != 0);
                    if ((bal_4 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0) {
                        if (o_hold_1[0] > -CUDART_INF_F) {
                            float _exp2_0_1 = approx_exp2(softmax_scale_log2 * (o_hold_1[0] - cand_4));
                            alpha_1[0] = _exp2_0_1;
                        }
                        o_hold_1[0] = cand_4;
                        float safe_4 = ((cand_4 == -CUDART_INF_F) ? 0.0f : cand_4);
                        ms_hold_1[0] = safe_4 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_1_1 = __ballot_sync(0xFFFFFFFF, (bal_4 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0);
                    grown_mask_1[0] = _warp_vote_1_1;
                    if (grown_mask_1[0] != 0) {
                        grow_any_1 = 1;
                    }
                    int base_5 = ((2 + pair_1 % 2) * 2 + 0) * 4 * 32 + 64;
                    float m_s_5;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_5) : "f"(smem_exch[base_5 + lane]), "f"(smem_exch[base_5 + 32 + lane]));
                    float cand_5;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_5) : "f"(o_hold_1[1]), "f"(m_s_5));
                    int moved_5 = 0;
                    if ((cand_5 - o_hold_1[1]) * softmax_scale_log2 > tau_1) {
                        moved_5 = 1;
                    }
                    unsigned int bal_5 = __ballot_sync(0xFFFFFFFF, moved_5 != 0);
                    if ((bal_5 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0) {
                        if (o_hold_1[1] > -CUDART_INF_F) {
                            float _exp2_2_1 = approx_exp2(softmax_scale_log2 * (o_hold_1[1] - cand_5));
                            alpha_1[1] = _exp2_2_1;
                        }
                        o_hold_1[1] = cand_5;
                        float safe_5 = ((cand_5 == -CUDART_INF_F) ? 0.0f : cand_5);
                        ms_hold_1[1] = safe_5 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_3_1 = __ballot_sync(0xFFFFFFFF, (bal_5 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0);
                    grown_mask_1[1] = _warp_vote_3_1;
                    if (grown_mask_1[1] != 0) {
                        grow_any_1 = 1;
                    }
                    int base_6 = ((2 + pair_1 % 2) * 2 + 1) * 4 * 32 + 0;
                    float m_s_6;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_6) : "f"(smem_exch[base_6 + lane]), "f"(smem_exch[base_6 + 32 + lane]));
                    float cand_6;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_6) : "f"(o_hold_1[2]), "f"(m_s_6));
                    int moved_6 = 0;
                    if ((cand_6 - o_hold_1[2]) * softmax_scale_log2 > tau_1) {
                        moved_6 = 1;
                    }
                    unsigned int bal_6 = __ballot_sync(0xFFFFFFFF, moved_6 != 0);
                    if ((bal_6 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0) {
                        if (o_hold_1[2] > -CUDART_INF_F) {
                            float _exp2_4_1 = approx_exp2(softmax_scale_log2 * (o_hold_1[2] - cand_6));
                            alpha_1[2] = _exp2_4_1;
                        }
                        o_hold_1[2] = cand_6;
                        float safe_6 = ((cand_6 == -CUDART_INF_F) ? 0.0f : cand_6);
                        ms_hold_1[2] = safe_6 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_5_1 = __ballot_sync(0xFFFFFFFF, (bal_6 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0);
                    grown_mask_1[2] = _warp_vote_5_1;
                    if (grown_mask_1[2] != 0) {
                        grow_any_1 = 1;
                    }
                    int base_7 = ((2 + pair_1 % 2) * 2 + 1) * 4 * 32 + 64;
                    float m_s_7;
                    asm("max.f32 %0, %1, %2;" : "=f"(m_s_7) : "f"(smem_exch[base_7 + lane]), "f"(smem_exch[base_7 + 32 + lane]));
                    float cand_7;
                    asm("max.f32 %0, %1, %2;" : "=f"(cand_7) : "f"(o_hold_1[3]), "f"(m_s_7));
                    int moved_7 = 0;
                    if ((cand_7 - o_hold_1[3]) * softmax_scale_log2 > tau_1) {
                        moved_7 = 1;
                    }
                    unsigned int bal_7 = __ballot_sync(0xFFFFFFFF, moved_7 != 0);
                    if ((bal_7 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0) {
                        if (o_hold_1[3] > -CUDART_INF_F) {
                            float _exp2_6_1 = approx_exp2(softmax_scale_log2 * (o_hold_1[3] - cand_7));
                            alpha_1[3] = _exp2_6_1;
                        }
                        o_hold_1[3] = cand_7;
                        float safe_7 = ((cand_7 == -CUDART_INF_F) ? 0.0f : cand_7);
                        ms_hold_1[3] = safe_7 * softmax_scale_log2;
                    }
                    unsigned int _warp_vote_7_1 = __ballot_sync(0xFFFFFFFF, (bal_7 >> (unsigned int)grp_shift_1 & grp_mask_1) != 0);
                    grown_mask_1[3] = _warp_vote_7_1;
                    if (grown_mask_1[3] != 0) {
                        grow_any_1 = 1;
                    }
                    if (grow_any_1 != 0) {
                        float a_sel_2 = ((own_half_1 == 0) ? alpha_1[0] : alpha_1[1]);
                        float _shfl_sync_8_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 0);
                        lane_sum_1[0] = lane_sum_1[0] * _shfl_sync_8_1;
                        float _shfl_sync_9_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 1);
                        lane_sum_1[1] = lane_sum_1[1] * _shfl_sync_9_1;
                        float _shfl_sync_10_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 2);
                        lane_sum_1[2] = lane_sum_1[2] * _shfl_sync_10_1;
                        float _shfl_sync_11_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 3);
                        lane_sum_1[3] = lane_sum_1[3] * _shfl_sync_11_1;
                        float _shfl_sync_12_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 4);
                        lane_sum_1[4] = lane_sum_1[4] * _shfl_sync_12_1;
                        float _shfl_sync_13_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 5);
                        lane_sum_1[5] = lane_sum_1[5] * _shfl_sync_13_1;
                        float _shfl_sync_14_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 6);
                        lane_sum_1[6] = lane_sum_1[6] * _shfl_sync_14_1;
                        float _shfl_sync_15_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 7);
                        lane_sum_1[7] = lane_sum_1[7] * _shfl_sync_15_1;
                        float _shfl_sync_16_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 8);
                        lane_sum_1[8] = lane_sum_1[8] * _shfl_sync_16_1;
                        float _shfl_sync_17_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 9);
                        lane_sum_1[9] = lane_sum_1[9] * _shfl_sync_17_1;
                        float _shfl_sync_18_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 10);
                        lane_sum_1[10] = lane_sum_1[10] * _shfl_sync_18_1;
                        float _shfl_sync_19_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 11);
                        lane_sum_1[11] = lane_sum_1[11] * _shfl_sync_19_1;
                        float _shfl_sync_20_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 12);
                        lane_sum_1[12] = lane_sum_1[12] * _shfl_sync_20_1;
                        float _shfl_sync_21_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 13);
                        lane_sum_1[13] = lane_sum_1[13] * _shfl_sync_21_1;
                        float _shfl_sync_22_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 14);
                        lane_sum_1[14] = lane_sum_1[14] * _shfl_sync_22_1;
                        float _shfl_sync_23_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 15);
                        lane_sum_1[15] = lane_sum_1[15] * _shfl_sync_23_1;
                        float _shfl_sync_24_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 16);
                        lane_sum_1[16] = lane_sum_1[16] * _shfl_sync_24_1;
                        float _shfl_sync_25_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 17);
                        lane_sum_1[17] = lane_sum_1[17] * _shfl_sync_25_1;
                        float _shfl_sync_26_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 18);
                        lane_sum_1[18] = lane_sum_1[18] * _shfl_sync_26_1;
                        float _shfl_sync_27_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 19);
                        lane_sum_1[19] = lane_sum_1[19] * _shfl_sync_27_1;
                        float _shfl_sync_28_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 20);
                        lane_sum_1[20] = lane_sum_1[20] * _shfl_sync_28_1;
                        float _shfl_sync_29_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 21);
                        lane_sum_1[21] = lane_sum_1[21] * _shfl_sync_29_1;
                        float _shfl_sync_30_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 22);
                        lane_sum_1[22] = lane_sum_1[22] * _shfl_sync_30_1;
                        float _shfl_sync_31_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 23);
                        lane_sum_1[23] = lane_sum_1[23] * _shfl_sync_31_1;
                        float _shfl_sync_32_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 24);
                        lane_sum_1[24] = lane_sum_1[24] * _shfl_sync_32_1;
                        float _shfl_sync_33_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 25);
                        lane_sum_1[25] = lane_sum_1[25] * _shfl_sync_33_1;
                        float _shfl_sync_34_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 26);
                        lane_sum_1[26] = lane_sum_1[26] * _shfl_sync_34_1;
                        float _shfl_sync_35_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 27);
                        lane_sum_1[27] = lane_sum_1[27] * _shfl_sync_35_1;
                        float _shfl_sync_36_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 28);
                        lane_sum_1[28] = lane_sum_1[28] * _shfl_sync_36_1;
                        float _shfl_sync_37_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 29);
                        lane_sum_1[29] = lane_sum_1[29] * _shfl_sync_37_1;
                        float _shfl_sync_38_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 30);
                        lane_sum_1[30] = lane_sum_1[30] * _shfl_sync_38_1;
                        float _shfl_sync_39_1 = __shfl_sync(0xFFFFFFFF, a_sel_2, 31);
                        lane_sum_1[31] = lane_sum_1[31] * _shfl_sync_39_1;
                        float a_sel_3 = ((own_half_1 == 0) ? alpha_1[2] : alpha_1[3]);
                        float _shfl_sync_40_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 0);
                        lane_sum_1[32] = lane_sum_1[32] * _shfl_sync_40_1;
                        float _shfl_sync_41_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 1);
                        lane_sum_1[33] = lane_sum_1[33] * _shfl_sync_41_1;
                        float _shfl_sync_42_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 2);
                        lane_sum_1[34] = lane_sum_1[34] * _shfl_sync_42_1;
                        float _shfl_sync_43_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 3);
                        lane_sum_1[35] = lane_sum_1[35] * _shfl_sync_43_1;
                        float _shfl_sync_44_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 4);
                        lane_sum_1[36] = lane_sum_1[36] * _shfl_sync_44_1;
                        float _shfl_sync_45_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 5);
                        lane_sum_1[37] = lane_sum_1[37] * _shfl_sync_45_1;
                        float _shfl_sync_46_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 6);
                        lane_sum_1[38] = lane_sum_1[38] * _shfl_sync_46_1;
                        float _shfl_sync_47_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 7);
                        lane_sum_1[39] = lane_sum_1[39] * _shfl_sync_47_1;
                        float _shfl_sync_48_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 8);
                        lane_sum_1[40] = lane_sum_1[40] * _shfl_sync_48_1;
                        float _shfl_sync_49_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 9);
                        lane_sum_1[41] = lane_sum_1[41] * _shfl_sync_49_1;
                        float _shfl_sync_50_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 10);
                        lane_sum_1[42] = lane_sum_1[42] * _shfl_sync_50_1;
                        float _shfl_sync_51_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 11);
                        lane_sum_1[43] = lane_sum_1[43] * _shfl_sync_51_1;
                        float _shfl_sync_52_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 12);
                        lane_sum_1[44] = lane_sum_1[44] * _shfl_sync_52_1;
                        float _shfl_sync_53_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 13);
                        lane_sum_1[45] = lane_sum_1[45] * _shfl_sync_53_1;
                        float _shfl_sync_54_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 14);
                        lane_sum_1[46] = lane_sum_1[46] * _shfl_sync_54_1;
                        float _shfl_sync_55_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 15);
                        lane_sum_1[47] = lane_sum_1[47] * _shfl_sync_55_1;
                        float _shfl_sync_56_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 16);
                        lane_sum_1[48] = lane_sum_1[48] * _shfl_sync_56_1;
                        float _shfl_sync_57_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 17);
                        lane_sum_1[49] = lane_sum_1[49] * _shfl_sync_57_1;
                        float _shfl_sync_58_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 18);
                        lane_sum_1[50] = lane_sum_1[50] * _shfl_sync_58_1;
                        float _shfl_sync_59_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 19);
                        lane_sum_1[51] = lane_sum_1[51] * _shfl_sync_59_1;
                        float _shfl_sync_60_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 20);
                        lane_sum_1[52] = lane_sum_1[52] * _shfl_sync_60_1;
                        float _shfl_sync_61_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 21);
                        lane_sum_1[53] = lane_sum_1[53] * _shfl_sync_61_1;
                        float _shfl_sync_62_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 22);
                        lane_sum_1[54] = lane_sum_1[54] * _shfl_sync_62_1;
                        float _shfl_sync_63_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 23);
                        lane_sum_1[55] = lane_sum_1[55] * _shfl_sync_63_1;
                        float _shfl_sync_64_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 24);
                        lane_sum_1[56] = lane_sum_1[56] * _shfl_sync_64_1;
                        float _shfl_sync_65_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 25);
                        lane_sum_1[57] = lane_sum_1[57] * _shfl_sync_65_1;
                        float _shfl_sync_66_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 26);
                        lane_sum_1[58] = lane_sum_1[58] * _shfl_sync_66_1;
                        float _shfl_sync_67_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 27);
                        lane_sum_1[59] = lane_sum_1[59] * _shfl_sync_67_1;
                        float _shfl_sync_68_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 28);
                        lane_sum_1[60] = lane_sum_1[60] * _shfl_sync_68_1;
                        float _shfl_sync_69_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 29);
                        lane_sum_1[61] = lane_sum_1[61] * _shfl_sync_69_1;
                        float _shfl_sync_70_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 30);
                        lane_sum_1[62] = lane_sum_1[62] * _shfl_sync_70_1;
                        float _shfl_sync_71_1 = __shfl_sync(0xFFFFFFFF, a_sel_3, 31);
                        lane_sum_1[63] = lane_sum_1[63] * _shfl_sync_71_1;
                        if (pair_1 > 0) {
                            int _stage_pv_done_6 = (int)((unsigned int)(gpair_1 - 1) % 8);
                            int _parity_pv_done_6 = (int)((unsigned int)(gpair_1 - 1) / 8 % 2);
                            mbarrier_wait(pv_done_addr + (_stage_pv_done_6) * 8, _parity_pv_done_6);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (grown_mask_1[0] != 0) {
                                int o_addr_4 = taddr + 288 + (unsigned int)lane_base_1;
                                float o_acc_4[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_4[0]), "=f"(o_acc_4[1]), "=f"(o_acc_4[2]), "=f"(o_acc_4[3]), "=f"(o_acc_4[4]), "=f"(o_acc_4[5]), "=f"(o_acc_4[6]), "=f"(o_acc_4[7]), "=f"(o_acc_4[8]), "=f"(o_acc_4[9]), "=f"(o_acc_4[10]), "=f"(o_acc_4[11]), "=f"(o_acc_4[12]), "=f"(o_acc_4[13]), "=f"(o_acc_4[14]), "=f"(o_acc_4[15]), "=f"(o_acc_4[16]), "=f"(o_acc_4[17]), "=f"(o_acc_4[18]), "=f"(o_acc_4[19]), "=f"(o_acc_4[20]), "=f"(o_acc_4[21]), "=f"(o_acc_4[22]), "=f"(o_acc_4[23]), "=f"(o_acc_4[24]), "=f"(o_acc_4[25]), "=f"(o_acc_4[26]), "=f"(o_acc_4[27]), "=f"(o_acc_4[28]), "=f"(o_acc_4[29]), "=f"(o_acc_4[30]), "=f"(o_acc_4[31])
                                    : "r"(o_addr_4));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_72_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 0);
                                o_acc_4[0] = o_acc_4[0] * _shfl_sync_72_1;
                                float _shfl_sync_73_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 1);
                                o_acc_4[1] = o_acc_4[1] * _shfl_sync_73_1;
                                float _shfl_sync_74_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 2);
                                o_acc_4[2] = o_acc_4[2] * _shfl_sync_74_1;
                                float _shfl_sync_75_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 3);
                                o_acc_4[3] = o_acc_4[3] * _shfl_sync_75_1;
                                float _shfl_sync_76_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 4);
                                o_acc_4[4] = o_acc_4[4] * _shfl_sync_76_1;
                                float _shfl_sync_77_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 5);
                                o_acc_4[5] = o_acc_4[5] * _shfl_sync_77_1;
                                float _shfl_sync_78_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 6);
                                o_acc_4[6] = o_acc_4[6] * _shfl_sync_78_1;
                                float _shfl_sync_79_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 7);
                                o_acc_4[7] = o_acc_4[7] * _shfl_sync_79_1;
                                float _shfl_sync_80_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 8);
                                o_acc_4[8] = o_acc_4[8] * _shfl_sync_80_1;
                                float _shfl_sync_81_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 9);
                                o_acc_4[9] = o_acc_4[9] * _shfl_sync_81_1;
                                float _shfl_sync_82_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 10);
                                o_acc_4[10] = o_acc_4[10] * _shfl_sync_82_1;
                                float _shfl_sync_83_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 11);
                                o_acc_4[11] = o_acc_4[11] * _shfl_sync_83_1;
                                float _shfl_sync_84_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 12);
                                o_acc_4[12] = o_acc_4[12] * _shfl_sync_84_1;
                                float _shfl_sync_85_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 13);
                                o_acc_4[13] = o_acc_4[13] * _shfl_sync_85_1;
                                float _shfl_sync_86_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 14);
                                o_acc_4[14] = o_acc_4[14] * _shfl_sync_86_1;
                                float _shfl_sync_87_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 15);
                                o_acc_4[15] = o_acc_4[15] * _shfl_sync_87_1;
                                float _shfl_sync_88_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 16);
                                o_acc_4[16] = o_acc_4[16] * _shfl_sync_88_1;
                                float _shfl_sync_89_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 17);
                                o_acc_4[17] = o_acc_4[17] * _shfl_sync_89_1;
                                float _shfl_sync_90_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 18);
                                o_acc_4[18] = o_acc_4[18] * _shfl_sync_90_1;
                                float _shfl_sync_91_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 19);
                                o_acc_4[19] = o_acc_4[19] * _shfl_sync_91_1;
                                float _shfl_sync_92_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 20);
                                o_acc_4[20] = o_acc_4[20] * _shfl_sync_92_1;
                                float _shfl_sync_93_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 21);
                                o_acc_4[21] = o_acc_4[21] * _shfl_sync_93_1;
                                float _shfl_sync_94_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 22);
                                o_acc_4[22] = o_acc_4[22] * _shfl_sync_94_1;
                                float _shfl_sync_95_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 23);
                                o_acc_4[23] = o_acc_4[23] * _shfl_sync_95_1;
                                float _shfl_sync_96_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 24);
                                o_acc_4[24] = o_acc_4[24] * _shfl_sync_96_1;
                                float _shfl_sync_97_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 25);
                                o_acc_4[25] = o_acc_4[25] * _shfl_sync_97_1;
                                float _shfl_sync_98_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 26);
                                o_acc_4[26] = o_acc_4[26] * _shfl_sync_98_1;
                                float _shfl_sync_99_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 27);
                                o_acc_4[27] = o_acc_4[27] * _shfl_sync_99_1;
                                float _shfl_sync_100_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 28);
                                o_acc_4[28] = o_acc_4[28] * _shfl_sync_100_1;
                                float _shfl_sync_101_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 29);
                                o_acc_4[29] = o_acc_4[29] * _shfl_sync_101_1;
                                float _shfl_sync_102_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 30);
                                o_acc_4[30] = o_acc_4[30] * _shfl_sync_102_1;
                                float _shfl_sync_103_1 = __shfl_sync(0xFFFFFFFF, alpha_1[0], 31);
                                o_acc_4[31] = o_acc_4[31] * _shfl_sync_103_1;
                                tmem_st_x32_f32(o_addr_4, o_acc_4);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if (grown_mask_1[1] != 0) {
                                int o_addr_5 = taddr + 352 + (unsigned int)lane_base_1;
                                float o_acc_5[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_5[0]), "=f"(o_acc_5[1]), "=f"(o_acc_5[2]), "=f"(o_acc_5[3]), "=f"(o_acc_5[4]), "=f"(o_acc_5[5]), "=f"(o_acc_5[6]), "=f"(o_acc_5[7]), "=f"(o_acc_5[8]), "=f"(o_acc_5[9]), "=f"(o_acc_5[10]), "=f"(o_acc_5[11]), "=f"(o_acc_5[12]), "=f"(o_acc_5[13]), "=f"(o_acc_5[14]), "=f"(o_acc_5[15]), "=f"(o_acc_5[16]), "=f"(o_acc_5[17]), "=f"(o_acc_5[18]), "=f"(o_acc_5[19]), "=f"(o_acc_5[20]), "=f"(o_acc_5[21]), "=f"(o_acc_5[22]), "=f"(o_acc_5[23]), "=f"(o_acc_5[24]), "=f"(o_acc_5[25]), "=f"(o_acc_5[26]), "=f"(o_acc_5[27]), "=f"(o_acc_5[28]), "=f"(o_acc_5[29]), "=f"(o_acc_5[30]), "=f"(o_acc_5[31])
                                    : "r"(o_addr_5));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_104_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 0);
                                o_acc_5[0] = o_acc_5[0] * _shfl_sync_104_1;
                                float _shfl_sync_105_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 1);
                                o_acc_5[1] = o_acc_5[1] * _shfl_sync_105_1;
                                float _shfl_sync_106_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 2);
                                o_acc_5[2] = o_acc_5[2] * _shfl_sync_106_1;
                                float _shfl_sync_107_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 3);
                                o_acc_5[3] = o_acc_5[3] * _shfl_sync_107_1;
                                float _shfl_sync_108_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 4);
                                o_acc_5[4] = o_acc_5[4] * _shfl_sync_108_1;
                                float _shfl_sync_109_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 5);
                                o_acc_5[5] = o_acc_5[5] * _shfl_sync_109_1;
                                float _shfl_sync_110_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 6);
                                o_acc_5[6] = o_acc_5[6] * _shfl_sync_110_1;
                                float _shfl_sync_111_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 7);
                                o_acc_5[7] = o_acc_5[7] * _shfl_sync_111_1;
                                float _shfl_sync_112_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 8);
                                o_acc_5[8] = o_acc_5[8] * _shfl_sync_112_1;
                                float _shfl_sync_113_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 9);
                                o_acc_5[9] = o_acc_5[9] * _shfl_sync_113_1;
                                float _shfl_sync_114_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 10);
                                o_acc_5[10] = o_acc_5[10] * _shfl_sync_114_1;
                                float _shfl_sync_115_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 11);
                                o_acc_5[11] = o_acc_5[11] * _shfl_sync_115_1;
                                float _shfl_sync_116_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 12);
                                o_acc_5[12] = o_acc_5[12] * _shfl_sync_116_1;
                                float _shfl_sync_117_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 13);
                                o_acc_5[13] = o_acc_5[13] * _shfl_sync_117_1;
                                float _shfl_sync_118_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 14);
                                o_acc_5[14] = o_acc_5[14] * _shfl_sync_118_1;
                                float _shfl_sync_119_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 15);
                                o_acc_5[15] = o_acc_5[15] * _shfl_sync_119_1;
                                float _shfl_sync_120_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 16);
                                o_acc_5[16] = o_acc_5[16] * _shfl_sync_120_1;
                                float _shfl_sync_121_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 17);
                                o_acc_5[17] = o_acc_5[17] * _shfl_sync_121_1;
                                float _shfl_sync_122_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 18);
                                o_acc_5[18] = o_acc_5[18] * _shfl_sync_122_1;
                                float _shfl_sync_123_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 19);
                                o_acc_5[19] = o_acc_5[19] * _shfl_sync_123_1;
                                float _shfl_sync_124_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 20);
                                o_acc_5[20] = o_acc_5[20] * _shfl_sync_124_1;
                                float _shfl_sync_125_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 21);
                                o_acc_5[21] = o_acc_5[21] * _shfl_sync_125_1;
                                float _shfl_sync_126_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 22);
                                o_acc_5[22] = o_acc_5[22] * _shfl_sync_126_1;
                                float _shfl_sync_127_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 23);
                                o_acc_5[23] = o_acc_5[23] * _shfl_sync_127_1;
                                float _shfl_sync_128_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 24);
                                o_acc_5[24] = o_acc_5[24] * _shfl_sync_128_1;
                                float _shfl_sync_129_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 25);
                                o_acc_5[25] = o_acc_5[25] * _shfl_sync_129_1;
                                float _shfl_sync_130_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 26);
                                o_acc_5[26] = o_acc_5[26] * _shfl_sync_130_1;
                                float _shfl_sync_131_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 27);
                                o_acc_5[27] = o_acc_5[27] * _shfl_sync_131_1;
                                float _shfl_sync_132_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 28);
                                o_acc_5[28] = o_acc_5[28] * _shfl_sync_132_1;
                                float _shfl_sync_133_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 29);
                                o_acc_5[29] = o_acc_5[29] * _shfl_sync_133_1;
                                float _shfl_sync_134_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 30);
                                o_acc_5[30] = o_acc_5[30] * _shfl_sync_134_1;
                                float _shfl_sync_135_1 = __shfl_sync(0xFFFFFFFF, alpha_1[1], 31);
                                o_acc_5[31] = o_acc_5[31] * _shfl_sync_135_1;
                                tmem_st_x32_f32(o_addr_5, o_acc_5);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if (grown_mask_1[2] != 0) {
                                int o_addr_6 = taddr + 416 + (unsigned int)lane_base_1;
                                float o_acc_6[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_6[0]), "=f"(o_acc_6[1]), "=f"(o_acc_6[2]), "=f"(o_acc_6[3]), "=f"(o_acc_6[4]), "=f"(o_acc_6[5]), "=f"(o_acc_6[6]), "=f"(o_acc_6[7]), "=f"(o_acc_6[8]), "=f"(o_acc_6[9]), "=f"(o_acc_6[10]), "=f"(o_acc_6[11]), "=f"(o_acc_6[12]), "=f"(o_acc_6[13]), "=f"(o_acc_6[14]), "=f"(o_acc_6[15]), "=f"(o_acc_6[16]), "=f"(o_acc_6[17]), "=f"(o_acc_6[18]), "=f"(o_acc_6[19]), "=f"(o_acc_6[20]), "=f"(o_acc_6[21]), "=f"(o_acc_6[22]), "=f"(o_acc_6[23]), "=f"(o_acc_6[24]), "=f"(o_acc_6[25]), "=f"(o_acc_6[26]), "=f"(o_acc_6[27]), "=f"(o_acc_6[28]), "=f"(o_acc_6[29]), "=f"(o_acc_6[30]), "=f"(o_acc_6[31])
                                    : "r"(o_addr_6));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_136_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 0);
                                o_acc_6[0] = o_acc_6[0] * _shfl_sync_136_1;
                                float _shfl_sync_137_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 1);
                                o_acc_6[1] = o_acc_6[1] * _shfl_sync_137_1;
                                float _shfl_sync_138_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 2);
                                o_acc_6[2] = o_acc_6[2] * _shfl_sync_138_1;
                                float _shfl_sync_139_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 3);
                                o_acc_6[3] = o_acc_6[3] * _shfl_sync_139_1;
                                float _shfl_sync_140_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 4);
                                o_acc_6[4] = o_acc_6[4] * _shfl_sync_140_1;
                                float _shfl_sync_141_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 5);
                                o_acc_6[5] = o_acc_6[5] * _shfl_sync_141_1;
                                float _shfl_sync_142_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 6);
                                o_acc_6[6] = o_acc_6[6] * _shfl_sync_142_1;
                                float _shfl_sync_143_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 7);
                                o_acc_6[7] = o_acc_6[7] * _shfl_sync_143_1;
                                float _shfl_sync_144_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 8);
                                o_acc_6[8] = o_acc_6[8] * _shfl_sync_144_1;
                                float _shfl_sync_145_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 9);
                                o_acc_6[9] = o_acc_6[9] * _shfl_sync_145_1;
                                float _shfl_sync_146_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 10);
                                o_acc_6[10] = o_acc_6[10] * _shfl_sync_146_1;
                                float _shfl_sync_147_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 11);
                                o_acc_6[11] = o_acc_6[11] * _shfl_sync_147_1;
                                float _shfl_sync_148_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 12);
                                o_acc_6[12] = o_acc_6[12] * _shfl_sync_148_1;
                                float _shfl_sync_149_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 13);
                                o_acc_6[13] = o_acc_6[13] * _shfl_sync_149_1;
                                float _shfl_sync_150_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 14);
                                o_acc_6[14] = o_acc_6[14] * _shfl_sync_150_1;
                                float _shfl_sync_151_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 15);
                                o_acc_6[15] = o_acc_6[15] * _shfl_sync_151_1;
                                float _shfl_sync_152_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 16);
                                o_acc_6[16] = o_acc_6[16] * _shfl_sync_152_1;
                                float _shfl_sync_153_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 17);
                                o_acc_6[17] = o_acc_6[17] * _shfl_sync_153_1;
                                float _shfl_sync_154_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 18);
                                o_acc_6[18] = o_acc_6[18] * _shfl_sync_154_1;
                                float _shfl_sync_155_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 19);
                                o_acc_6[19] = o_acc_6[19] * _shfl_sync_155_1;
                                float _shfl_sync_156_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 20);
                                o_acc_6[20] = o_acc_6[20] * _shfl_sync_156_1;
                                float _shfl_sync_157_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 21);
                                o_acc_6[21] = o_acc_6[21] * _shfl_sync_157_1;
                                float _shfl_sync_158_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 22);
                                o_acc_6[22] = o_acc_6[22] * _shfl_sync_158_1;
                                float _shfl_sync_159_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 23);
                                o_acc_6[23] = o_acc_6[23] * _shfl_sync_159_1;
                                float _shfl_sync_160_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 24);
                                o_acc_6[24] = o_acc_6[24] * _shfl_sync_160_1;
                                float _shfl_sync_161_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 25);
                                o_acc_6[25] = o_acc_6[25] * _shfl_sync_161_1;
                                float _shfl_sync_162_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 26);
                                o_acc_6[26] = o_acc_6[26] * _shfl_sync_162_1;
                                float _shfl_sync_163_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 27);
                                o_acc_6[27] = o_acc_6[27] * _shfl_sync_163_1;
                                float _shfl_sync_164_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 28);
                                o_acc_6[28] = o_acc_6[28] * _shfl_sync_164_1;
                                float _shfl_sync_165_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 29);
                                o_acc_6[29] = o_acc_6[29] * _shfl_sync_165_1;
                                float _shfl_sync_166_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 30);
                                o_acc_6[30] = o_acc_6[30] * _shfl_sync_166_1;
                                float _shfl_sync_167_1 = __shfl_sync(0xFFFFFFFF, alpha_1[2], 31);
                                o_acc_6[31] = o_acc_6[31] * _shfl_sync_167_1;
                                tmem_st_x32_f32(o_addr_6, o_acc_6);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                            if (grown_mask_1[3] != 0) {
                                int o_addr_7 = taddr + 480 + (unsigned int)lane_base_1;
                                float o_acc_7[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(o_acc_7[0]), "=f"(o_acc_7[1]), "=f"(o_acc_7[2]), "=f"(o_acc_7[3]), "=f"(o_acc_7[4]), "=f"(o_acc_7[5]), "=f"(o_acc_7[6]), "=f"(o_acc_7[7]), "=f"(o_acc_7[8]), "=f"(o_acc_7[9]), "=f"(o_acc_7[10]), "=f"(o_acc_7[11]), "=f"(o_acc_7[12]), "=f"(o_acc_7[13]), "=f"(o_acc_7[14]), "=f"(o_acc_7[15]), "=f"(o_acc_7[16]), "=f"(o_acc_7[17]), "=f"(o_acc_7[18]), "=f"(o_acc_7[19]), "=f"(o_acc_7[20]), "=f"(o_acc_7[21]), "=f"(o_acc_7[22]), "=f"(o_acc_7[23]), "=f"(o_acc_7[24]), "=f"(o_acc_7[25]), "=f"(o_acc_7[26]), "=f"(o_acc_7[27]), "=f"(o_acc_7[28]), "=f"(o_acc_7[29]), "=f"(o_acc_7[30]), "=f"(o_acc_7[31])
                                    : "r"(o_addr_7));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float _shfl_sync_168_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 0);
                                o_acc_7[0] = o_acc_7[0] * _shfl_sync_168_1;
                                float _shfl_sync_169_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 1);
                                o_acc_7[1] = o_acc_7[1] * _shfl_sync_169_1;
                                float _shfl_sync_170_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 2);
                                o_acc_7[2] = o_acc_7[2] * _shfl_sync_170_1;
                                float _shfl_sync_171_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 3);
                                o_acc_7[3] = o_acc_7[3] * _shfl_sync_171_1;
                                float _shfl_sync_172_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 4);
                                o_acc_7[4] = o_acc_7[4] * _shfl_sync_172_1;
                                float _shfl_sync_173_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 5);
                                o_acc_7[5] = o_acc_7[5] * _shfl_sync_173_1;
                                float _shfl_sync_174_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 6);
                                o_acc_7[6] = o_acc_7[6] * _shfl_sync_174_1;
                                float _shfl_sync_175_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 7);
                                o_acc_7[7] = o_acc_7[7] * _shfl_sync_175_1;
                                float _shfl_sync_176_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 8);
                                o_acc_7[8] = o_acc_7[8] * _shfl_sync_176_1;
                                float _shfl_sync_177_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 9);
                                o_acc_7[9] = o_acc_7[9] * _shfl_sync_177_1;
                                float _shfl_sync_178_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 10);
                                o_acc_7[10] = o_acc_7[10] * _shfl_sync_178_1;
                                float _shfl_sync_179_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 11);
                                o_acc_7[11] = o_acc_7[11] * _shfl_sync_179_1;
                                float _shfl_sync_180_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 12);
                                o_acc_7[12] = o_acc_7[12] * _shfl_sync_180_1;
                                float _shfl_sync_181_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 13);
                                o_acc_7[13] = o_acc_7[13] * _shfl_sync_181_1;
                                float _shfl_sync_182_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 14);
                                o_acc_7[14] = o_acc_7[14] * _shfl_sync_182_1;
                                float _shfl_sync_183_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 15);
                                o_acc_7[15] = o_acc_7[15] * _shfl_sync_183_1;
                                float _shfl_sync_184_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 16);
                                o_acc_7[16] = o_acc_7[16] * _shfl_sync_184_1;
                                float _shfl_sync_185_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 17);
                                o_acc_7[17] = o_acc_7[17] * _shfl_sync_185_1;
                                float _shfl_sync_186_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 18);
                                o_acc_7[18] = o_acc_7[18] * _shfl_sync_186_1;
                                float _shfl_sync_187_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 19);
                                o_acc_7[19] = o_acc_7[19] * _shfl_sync_187_1;
                                float _shfl_sync_188_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 20);
                                o_acc_7[20] = o_acc_7[20] * _shfl_sync_188_1;
                                float _shfl_sync_189_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 21);
                                o_acc_7[21] = o_acc_7[21] * _shfl_sync_189_1;
                                float _shfl_sync_190_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 22);
                                o_acc_7[22] = o_acc_7[22] * _shfl_sync_190_1;
                                float _shfl_sync_191_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 23);
                                o_acc_7[23] = o_acc_7[23] * _shfl_sync_191_1;
                                float _shfl_sync_192_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 24);
                                o_acc_7[24] = o_acc_7[24] * _shfl_sync_192_1;
                                float _shfl_sync_193_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 25);
                                o_acc_7[25] = o_acc_7[25] * _shfl_sync_193_1;
                                float _shfl_sync_194_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 26);
                                o_acc_7[26] = o_acc_7[26] * _shfl_sync_194_1;
                                float _shfl_sync_195_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 27);
                                o_acc_7[27] = o_acc_7[27] * _shfl_sync_195_1;
                                float _shfl_sync_196_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 28);
                                o_acc_7[28] = o_acc_7[28] * _shfl_sync_196_1;
                                float _shfl_sync_197_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 29);
                                o_acc_7[29] = o_acc_7[29] * _shfl_sync_197_1;
                                float _shfl_sync_198_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 30);
                                o_acc_7[30] = o_acc_7[30] * _shfl_sync_198_1;
                                float _shfl_sync_199_1 = __shfl_sync(0xFFFFFFFF, alpha_1[3], 31);
                                o_acc_7[31] = o_acc_7[31] * _shfl_sync_199_1;
                                tmem_st_x32_f32(o_addr_7, o_acc_7);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                    }
                    if (gpair_1 >= 4) {
                        int _stage_pv_done_7 = (int)((unsigned int)(gpair_1 - 4) % 8);
                        int _parity_pv_done_7 = (int)((unsigned int)(gpair_1 - 4) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_7) * 8, _parity_pv_done_7);
                    }
                    float ms_sel_1 = ((own_half_1 == 0) ? ms_hold_1[0] : ms_hold_1[1]);
                    float _shfl_sync_201_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 0);
                    float _exp2_200_1 = approx_exp2(__fmaf_rn(sv0_1[0], softmax_scale_log2, -_shfl_sync_201_1));
                    sv0_1[0] = _exp2_200_1;
                    lane_sum_1[0] = lane_sum_1[0] + sv0_1[0];
                    float _shfl_sync_203_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 1);
                    float _exp2_202_1 = approx_exp2(__fmaf_rn(sv0_1[1], softmax_scale_log2, -_shfl_sync_203_1));
                    sv0_1[1] = _exp2_202_1;
                    lane_sum_1[1] = lane_sum_1[1] + sv0_1[1];
                    float _shfl_sync_205_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 2);
                    float _exp2_204_1 = approx_exp2(__fmaf_rn(sv0_1[2], softmax_scale_log2, -_shfl_sync_205_1));
                    sv0_1[2] = _exp2_204_1;
                    lane_sum_1[2] = lane_sum_1[2] + sv0_1[2];
                    float _shfl_sync_207_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 3);
                    float _exp2_206_1 = approx_exp2(__fmaf_rn(sv0_1[3], softmax_scale_log2, -_shfl_sync_207_1));
                    sv0_1[3] = _exp2_206_1;
                    lane_sum_1[3] = lane_sum_1[3] + sv0_1[3];
                    float _shfl_sync_209_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 4);
                    float _exp2_208_1 = approx_exp2(__fmaf_rn(sv0_1[4], softmax_scale_log2, -_shfl_sync_209_1));
                    sv0_1[4] = _exp2_208_1;
                    lane_sum_1[4] = lane_sum_1[4] + sv0_1[4];
                    float _shfl_sync_211_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 5);
                    float _exp2_210_1 = approx_exp2(__fmaf_rn(sv0_1[5], softmax_scale_log2, -_shfl_sync_211_1));
                    sv0_1[5] = _exp2_210_1;
                    lane_sum_1[5] = lane_sum_1[5] + sv0_1[5];
                    float _shfl_sync_213_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 6);
                    float _exp2_212_1 = approx_exp2(__fmaf_rn(sv0_1[6], softmax_scale_log2, -_shfl_sync_213_1));
                    sv0_1[6] = _exp2_212_1;
                    lane_sum_1[6] = lane_sum_1[6] + sv0_1[6];
                    float _shfl_sync_215_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 7);
                    float _exp2_214_1 = approx_exp2(__fmaf_rn(sv0_1[7], softmax_scale_log2, -_shfl_sync_215_1));
                    sv0_1[7] = _exp2_214_1;
                    lane_sum_1[7] = lane_sum_1[7] + sv0_1[7];
                    float _shfl_sync_217_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 8);
                    float _exp2_216_1 = approx_exp2(__fmaf_rn(sv0_1[8], softmax_scale_log2, -_shfl_sync_217_1));
                    sv0_1[8] = _exp2_216_1;
                    lane_sum_1[8] = lane_sum_1[8] + sv0_1[8];
                    float _shfl_sync_219_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 9);
                    float _exp2_218_1 = approx_exp2(__fmaf_rn(sv0_1[9], softmax_scale_log2, -_shfl_sync_219_1));
                    sv0_1[9] = _exp2_218_1;
                    lane_sum_1[9] = lane_sum_1[9] + sv0_1[9];
                    float _shfl_sync_221_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 10);
                    float _exp2_220_1 = approx_exp2(__fmaf_rn(sv0_1[10], softmax_scale_log2, -_shfl_sync_221_1));
                    sv0_1[10] = _exp2_220_1;
                    lane_sum_1[10] = lane_sum_1[10] + sv0_1[10];
                    float _shfl_sync_223_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 11);
                    float _exp2_222_1 = approx_exp2(__fmaf_rn(sv0_1[11], softmax_scale_log2, -_shfl_sync_223_1));
                    sv0_1[11] = _exp2_222_1;
                    lane_sum_1[11] = lane_sum_1[11] + sv0_1[11];
                    float _shfl_sync_225_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 12);
                    float _exp2_224_1 = approx_exp2(__fmaf_rn(sv0_1[12], softmax_scale_log2, -_shfl_sync_225_1));
                    sv0_1[12] = _exp2_224_1;
                    lane_sum_1[12] = lane_sum_1[12] + sv0_1[12];
                    float _shfl_sync_227_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 13);
                    float _exp2_226_1 = approx_exp2(__fmaf_rn(sv0_1[13], softmax_scale_log2, -_shfl_sync_227_1));
                    sv0_1[13] = _exp2_226_1;
                    lane_sum_1[13] = lane_sum_1[13] + sv0_1[13];
                    float _shfl_sync_229_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 14);
                    float _exp2_228_1 = approx_exp2(__fmaf_rn(sv0_1[14], softmax_scale_log2, -_shfl_sync_229_1));
                    sv0_1[14] = _exp2_228_1;
                    lane_sum_1[14] = lane_sum_1[14] + sv0_1[14];
                    float _shfl_sync_231_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 15);
                    float _exp2_230_1 = approx_exp2(__fmaf_rn(sv0_1[15], softmax_scale_log2, -_shfl_sync_231_1));
                    sv0_1[15] = _exp2_230_1;
                    lane_sum_1[15] = lane_sum_1[15] + sv0_1[15];
                    float _shfl_sync_233_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 16);
                    float _exp2_232_1 = approx_exp2(__fmaf_rn(sv0_1[16], softmax_scale_log2, -_shfl_sync_233_1));
                    sv0_1[16] = _exp2_232_1;
                    lane_sum_1[16] = lane_sum_1[16] + sv0_1[16];
                    float _shfl_sync_235_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 17);
                    float _exp2_234_1 = approx_exp2(__fmaf_rn(sv0_1[17], softmax_scale_log2, -_shfl_sync_235_1));
                    sv0_1[17] = _exp2_234_1;
                    lane_sum_1[17] = lane_sum_1[17] + sv0_1[17];
                    float _shfl_sync_237_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 18);
                    float _exp2_236_1 = approx_exp2(__fmaf_rn(sv0_1[18], softmax_scale_log2, -_shfl_sync_237_1));
                    sv0_1[18] = _exp2_236_1;
                    lane_sum_1[18] = lane_sum_1[18] + sv0_1[18];
                    float _shfl_sync_239_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 19);
                    float _exp2_238_1 = approx_exp2(__fmaf_rn(sv0_1[19], softmax_scale_log2, -_shfl_sync_239_1));
                    sv0_1[19] = _exp2_238_1;
                    lane_sum_1[19] = lane_sum_1[19] + sv0_1[19];
                    float _shfl_sync_241_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 20);
                    float _exp2_240_1 = approx_exp2(__fmaf_rn(sv0_1[20], softmax_scale_log2, -_shfl_sync_241_1));
                    sv0_1[20] = _exp2_240_1;
                    lane_sum_1[20] = lane_sum_1[20] + sv0_1[20];
                    float _shfl_sync_243_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 21);
                    float _exp2_242_1 = approx_exp2(__fmaf_rn(sv0_1[21], softmax_scale_log2, -_shfl_sync_243_1));
                    sv0_1[21] = _exp2_242_1;
                    lane_sum_1[21] = lane_sum_1[21] + sv0_1[21];
                    float _shfl_sync_245_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 22);
                    float _exp2_244_1 = approx_exp2(__fmaf_rn(sv0_1[22], softmax_scale_log2, -_shfl_sync_245_1));
                    sv0_1[22] = _exp2_244_1;
                    lane_sum_1[22] = lane_sum_1[22] + sv0_1[22];
                    float _shfl_sync_247_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 23);
                    float _exp2_246_1 = approx_exp2(__fmaf_rn(sv0_1[23], softmax_scale_log2, -_shfl_sync_247_1));
                    sv0_1[23] = _exp2_246_1;
                    lane_sum_1[23] = lane_sum_1[23] + sv0_1[23];
                    float _shfl_sync_249_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 24);
                    float _exp2_248_1 = approx_exp2(__fmaf_rn(sv0_1[24], softmax_scale_log2, -_shfl_sync_249_1));
                    sv0_1[24] = _exp2_248_1;
                    lane_sum_1[24] = lane_sum_1[24] + sv0_1[24];
                    float _shfl_sync_251_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 25);
                    float _exp2_250_1 = approx_exp2(__fmaf_rn(sv0_1[25], softmax_scale_log2, -_shfl_sync_251_1));
                    sv0_1[25] = _exp2_250_1;
                    lane_sum_1[25] = lane_sum_1[25] + sv0_1[25];
                    float _shfl_sync_253_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 26);
                    float _exp2_252_1 = approx_exp2(__fmaf_rn(sv0_1[26], softmax_scale_log2, -_shfl_sync_253_1));
                    sv0_1[26] = _exp2_252_1;
                    lane_sum_1[26] = lane_sum_1[26] + sv0_1[26];
                    float _shfl_sync_255_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 27);
                    float _exp2_254_1 = approx_exp2(__fmaf_rn(sv0_1[27], softmax_scale_log2, -_shfl_sync_255_1));
                    sv0_1[27] = _exp2_254_1;
                    lane_sum_1[27] = lane_sum_1[27] + sv0_1[27];
                    float _shfl_sync_257_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 28);
                    float _exp2_256_1 = approx_exp2(__fmaf_rn(sv0_1[28], softmax_scale_log2, -_shfl_sync_257_1));
                    sv0_1[28] = _exp2_256_1;
                    lane_sum_1[28] = lane_sum_1[28] + sv0_1[28];
                    float _shfl_sync_259_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 29);
                    float _exp2_258_1 = approx_exp2(__fmaf_rn(sv0_1[29], softmax_scale_log2, -_shfl_sync_259_1));
                    sv0_1[29] = _exp2_258_1;
                    lane_sum_1[29] = lane_sum_1[29] + sv0_1[29];
                    float _shfl_sync_261_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 30);
                    float _exp2_260_1 = approx_exp2(__fmaf_rn(sv0_1[30], softmax_scale_log2, -_shfl_sync_261_1));
                    sv0_1[30] = _exp2_260_1;
                    lane_sum_1[30] = lane_sum_1[30] + sv0_1[30];
                    float _shfl_sync_263_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1, 31);
                    float _exp2_262_1 = approx_exp2(__fmaf_rn(sv0_1[31], softmax_scale_log2, -_shfl_sync_263_1));
                    sv0_1[31] = _exp2_262_1;
                    lane_sum_1[31] = lane_sum_1[31] + sv0_1[31];
                    unsigned int words_1[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[0]), "f"(sv0_1[1]),
                                               "f"(sv0_1[2]), "f"(sv0_1[3]));
                        words_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[4]), "f"(sv0_1[5]),
                                               "f"(sv0_1[6]), "f"(sv0_1[7]));
                        words_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[8]), "f"(sv0_1[9]),
                                               "f"(sv0_1[10]), "f"(sv0_1[11]));
                        words_1[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[12]), "f"(sv0_1[13]),
                                               "f"(sv0_1[14]), "f"(sv0_1[15]));
                        words_1[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[16]), "f"(sv0_1[17]),
                                               "f"(sv0_1[18]), "f"(sv0_1[19]));
                        words_1[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[20]), "f"(sv0_1[21]),
                                               "f"(sv0_1[22]), "f"(sv0_1[23]));
                        words_1[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[24]), "f"(sv0_1[25]),
                                               "f"(sv0_1[26]), "f"(sv0_1[27]));
                        words_1[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv0_1[28]), "f"(sv0_1[29]),
                                               "f"(sv0_1[30]), "f"(sv0_1[31]));
                        words_1[7] = _packed;
                    }
                    int p_base_1 = smem_p_addr + (unsigned int)((p_stage_1 * 2 + 0) * 8192);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_1 + (key_row_1 * 64 + 32 ^ (key_row_1 * 64 + 32 >> 7 & 3) << 4))), "r"(words_1[0]), "r"(words_1[1]), "r"(words_1[2]), "r"(words_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_1 + (key_row_1 * 64 + 48 ^ (key_row_1 * 64 + 48 >> 7 & 3) << 4))), "r"(words_1[4]), "r"(words_1[5]), "r"(words_1[6]), "r"(words_1[7]) : "memory");
                    float ms_sel_1_1 = ((own_half_1 == 0) ? ms_hold_1[2] : ms_hold_1[3]);
                    float _shfl_sync_265_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 0);
                    float _exp2_264_1 = approx_exp2(__fmaf_rn(sv1_1[0], softmax_scale_log2, -_shfl_sync_265_1));
                    sv1_1[0] = _exp2_264_1;
                    lane_sum_1[32] = lane_sum_1[32] + sv1_1[0];
                    float _shfl_sync_267_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 1);
                    float _exp2_266_1 = approx_exp2(__fmaf_rn(sv1_1[1], softmax_scale_log2, -_shfl_sync_267_1));
                    sv1_1[1] = _exp2_266_1;
                    lane_sum_1[33] = lane_sum_1[33] + sv1_1[1];
                    float _shfl_sync_269_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 2);
                    float _exp2_268_1 = approx_exp2(__fmaf_rn(sv1_1[2], softmax_scale_log2, -_shfl_sync_269_1));
                    sv1_1[2] = _exp2_268_1;
                    lane_sum_1[34] = lane_sum_1[34] + sv1_1[2];
                    float _shfl_sync_271_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 3);
                    float _exp2_270_1 = approx_exp2(__fmaf_rn(sv1_1[3], softmax_scale_log2, -_shfl_sync_271_1));
                    sv1_1[3] = _exp2_270_1;
                    lane_sum_1[35] = lane_sum_1[35] + sv1_1[3];
                    float _shfl_sync_273_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 4);
                    float _exp2_272_1 = approx_exp2(__fmaf_rn(sv1_1[4], softmax_scale_log2, -_shfl_sync_273_1));
                    sv1_1[4] = _exp2_272_1;
                    lane_sum_1[36] = lane_sum_1[36] + sv1_1[4];
                    float _shfl_sync_275_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 5);
                    float _exp2_274_1 = approx_exp2(__fmaf_rn(sv1_1[5], softmax_scale_log2, -_shfl_sync_275_1));
                    sv1_1[5] = _exp2_274_1;
                    lane_sum_1[37] = lane_sum_1[37] + sv1_1[5];
                    float _shfl_sync_277_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 6);
                    float _exp2_276_1 = approx_exp2(__fmaf_rn(sv1_1[6], softmax_scale_log2, -_shfl_sync_277_1));
                    sv1_1[6] = _exp2_276_1;
                    lane_sum_1[38] = lane_sum_1[38] + sv1_1[6];
                    float _shfl_sync_279_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 7);
                    float _exp2_278_1 = approx_exp2(__fmaf_rn(sv1_1[7], softmax_scale_log2, -_shfl_sync_279_1));
                    sv1_1[7] = _exp2_278_1;
                    lane_sum_1[39] = lane_sum_1[39] + sv1_1[7];
                    float _shfl_sync_281_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 8);
                    float _exp2_280_1 = approx_exp2(__fmaf_rn(sv1_1[8], softmax_scale_log2, -_shfl_sync_281_1));
                    sv1_1[8] = _exp2_280_1;
                    lane_sum_1[40] = lane_sum_1[40] + sv1_1[8];
                    float _shfl_sync_283_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 9);
                    float _exp2_282_1 = approx_exp2(__fmaf_rn(sv1_1[9], softmax_scale_log2, -_shfl_sync_283_1));
                    sv1_1[9] = _exp2_282_1;
                    lane_sum_1[41] = lane_sum_1[41] + sv1_1[9];
                    float _shfl_sync_285_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 10);
                    float _exp2_284_1 = approx_exp2(__fmaf_rn(sv1_1[10], softmax_scale_log2, -_shfl_sync_285_1));
                    sv1_1[10] = _exp2_284_1;
                    lane_sum_1[42] = lane_sum_1[42] + sv1_1[10];
                    float _shfl_sync_287_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 11);
                    float _exp2_286_1 = approx_exp2(__fmaf_rn(sv1_1[11], softmax_scale_log2, -_shfl_sync_287_1));
                    sv1_1[11] = _exp2_286_1;
                    lane_sum_1[43] = lane_sum_1[43] + sv1_1[11];
                    float _shfl_sync_289_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 12);
                    float _exp2_288_1 = approx_exp2(__fmaf_rn(sv1_1[12], softmax_scale_log2, -_shfl_sync_289_1));
                    sv1_1[12] = _exp2_288_1;
                    lane_sum_1[44] = lane_sum_1[44] + sv1_1[12];
                    float _shfl_sync_291_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 13);
                    float _exp2_290_1 = approx_exp2(__fmaf_rn(sv1_1[13], softmax_scale_log2, -_shfl_sync_291_1));
                    sv1_1[13] = _exp2_290_1;
                    lane_sum_1[45] = lane_sum_1[45] + sv1_1[13];
                    float _shfl_sync_293_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 14);
                    float _exp2_292_1 = approx_exp2(__fmaf_rn(sv1_1[14], softmax_scale_log2, -_shfl_sync_293_1));
                    sv1_1[14] = _exp2_292_1;
                    lane_sum_1[46] = lane_sum_1[46] + sv1_1[14];
                    float _shfl_sync_295_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 15);
                    float _exp2_294_1 = approx_exp2(__fmaf_rn(sv1_1[15], softmax_scale_log2, -_shfl_sync_295_1));
                    sv1_1[15] = _exp2_294_1;
                    lane_sum_1[47] = lane_sum_1[47] + sv1_1[15];
                    float _shfl_sync_297_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 16);
                    float _exp2_296_1 = approx_exp2(__fmaf_rn(sv1_1[16], softmax_scale_log2, -_shfl_sync_297_1));
                    sv1_1[16] = _exp2_296_1;
                    lane_sum_1[48] = lane_sum_1[48] + sv1_1[16];
                    float _shfl_sync_299_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 17);
                    float _exp2_298_1 = approx_exp2(__fmaf_rn(sv1_1[17], softmax_scale_log2, -_shfl_sync_299_1));
                    sv1_1[17] = _exp2_298_1;
                    lane_sum_1[49] = lane_sum_1[49] + sv1_1[17];
                    float _shfl_sync_301_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 18);
                    float _exp2_300_1 = approx_exp2(__fmaf_rn(sv1_1[18], softmax_scale_log2, -_shfl_sync_301_1));
                    sv1_1[18] = _exp2_300_1;
                    lane_sum_1[50] = lane_sum_1[50] + sv1_1[18];
                    float _shfl_sync_303_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 19);
                    float _exp2_302_1 = approx_exp2(__fmaf_rn(sv1_1[19], softmax_scale_log2, -_shfl_sync_303_1));
                    sv1_1[19] = _exp2_302_1;
                    lane_sum_1[51] = lane_sum_1[51] + sv1_1[19];
                    float _shfl_sync_305_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 20);
                    float _exp2_304_1 = approx_exp2(__fmaf_rn(sv1_1[20], softmax_scale_log2, -_shfl_sync_305_1));
                    sv1_1[20] = _exp2_304_1;
                    lane_sum_1[52] = lane_sum_1[52] + sv1_1[20];
                    float _shfl_sync_307_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 21);
                    float _exp2_306_1 = approx_exp2(__fmaf_rn(sv1_1[21], softmax_scale_log2, -_shfl_sync_307_1));
                    sv1_1[21] = _exp2_306_1;
                    lane_sum_1[53] = lane_sum_1[53] + sv1_1[21];
                    float _shfl_sync_309_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 22);
                    float _exp2_308_1 = approx_exp2(__fmaf_rn(sv1_1[22], softmax_scale_log2, -_shfl_sync_309_1));
                    sv1_1[22] = _exp2_308_1;
                    lane_sum_1[54] = lane_sum_1[54] + sv1_1[22];
                    float _shfl_sync_311_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 23);
                    float _exp2_310_1 = approx_exp2(__fmaf_rn(sv1_1[23], softmax_scale_log2, -_shfl_sync_311_1));
                    sv1_1[23] = _exp2_310_1;
                    lane_sum_1[55] = lane_sum_1[55] + sv1_1[23];
                    float _shfl_sync_313_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 24);
                    float _exp2_312_1 = approx_exp2(__fmaf_rn(sv1_1[24], softmax_scale_log2, -_shfl_sync_313_1));
                    sv1_1[24] = _exp2_312_1;
                    lane_sum_1[56] = lane_sum_1[56] + sv1_1[24];
                    float _shfl_sync_315_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 25);
                    float _exp2_314_1 = approx_exp2(__fmaf_rn(sv1_1[25], softmax_scale_log2, -_shfl_sync_315_1));
                    sv1_1[25] = _exp2_314_1;
                    lane_sum_1[57] = lane_sum_1[57] + sv1_1[25];
                    float _shfl_sync_317_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 26);
                    float _exp2_316_1 = approx_exp2(__fmaf_rn(sv1_1[26], softmax_scale_log2, -_shfl_sync_317_1));
                    sv1_1[26] = _exp2_316_1;
                    lane_sum_1[58] = lane_sum_1[58] + sv1_1[26];
                    float _shfl_sync_319_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 27);
                    float _exp2_318_1 = approx_exp2(__fmaf_rn(sv1_1[27], softmax_scale_log2, -_shfl_sync_319_1));
                    sv1_1[27] = _exp2_318_1;
                    lane_sum_1[59] = lane_sum_1[59] + sv1_1[27];
                    float _shfl_sync_321_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 28);
                    float _exp2_320_1 = approx_exp2(__fmaf_rn(sv1_1[28], softmax_scale_log2, -_shfl_sync_321_1));
                    sv1_1[28] = _exp2_320_1;
                    lane_sum_1[60] = lane_sum_1[60] + sv1_1[28];
                    float _shfl_sync_323_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 29);
                    float _exp2_322_1 = approx_exp2(__fmaf_rn(sv1_1[29], softmax_scale_log2, -_shfl_sync_323_1));
                    sv1_1[29] = _exp2_322_1;
                    lane_sum_1[61] = lane_sum_1[61] + sv1_1[29];
                    float _shfl_sync_325_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 30);
                    float _exp2_324_1 = approx_exp2(__fmaf_rn(sv1_1[30], softmax_scale_log2, -_shfl_sync_325_1));
                    sv1_1[30] = _exp2_324_1;
                    lane_sum_1[62] = lane_sum_1[62] + sv1_1[30];
                    float _shfl_sync_327_1 = __shfl_sync(0xFFFFFFFF, ms_sel_1_1, 31);
                    float _exp2_326_1 = approx_exp2(__fmaf_rn(sv1_1[31], softmax_scale_log2, -_shfl_sync_327_1));
                    sv1_1[31] = _exp2_326_1;
                    lane_sum_1[63] = lane_sum_1[63] + sv1_1[31];
                    unsigned int words_1_1[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[0]), "f"(sv1_1[1]),
                                               "f"(sv1_1[2]), "f"(sv1_1[3]));
                        words_1_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[4]), "f"(sv1_1[5]),
                                               "f"(sv1_1[6]), "f"(sv1_1[7]));
                        words_1_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[8]), "f"(sv1_1[9]),
                                               "f"(sv1_1[10]), "f"(sv1_1[11]));
                        words_1_1[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[12]), "f"(sv1_1[13]),
                                               "f"(sv1_1[14]), "f"(sv1_1[15]));
                        words_1_1[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[16]), "f"(sv1_1[17]),
                                               "f"(sv1_1[18]), "f"(sv1_1[19]));
                        words_1_1[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[20]), "f"(sv1_1[21]),
                                               "f"(sv1_1[22]), "f"(sv1_1[23]));
                        words_1_1[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[24]), "f"(sv1_1[25]),
                                               "f"(sv1_1[26]), "f"(sv1_1[27]));
                        words_1_1[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sv1_1[28]), "f"(sv1_1[29]),
                                               "f"(sv1_1[30]), "f"(sv1_1[31]));
                        words_1_1[7] = _packed;
                    }
                    int p_base_1_1 = smem_p_addr + (unsigned int)((p_stage_1 * 2 + 1) * 8192);
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_1_1 + (key_row_1 * 64 + 32 ^ (key_row_1 * 64 + 32 >> 7 & 3) << 4))), "r"(words_1_1[0]), "r"(words_1_1[1]), "r"(words_1_1[2]), "r"(words_1_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_1_1 + (key_row_1 * 64 + 48 ^ (key_row_1 * 64 + 48 >> 7 & 3) << 4))), "r"(words_1_1[4]), "r"(words_1_1[5]), "r"(words_1_1[6]), "r"(words_1_1[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage_1) * 8);
                    }
                    s_stage_1 += 1;
                    if (s_stage_1 == 2) { s_stage_1 = 0; _phase_s_full_1 ^= 1; }
                    p_stage_1 += 1;
                    if (p_stage_1 == 4) { p_stage_1 = 0; }
                    gpair_1 += 1;
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)item_ord_1 % 2)) * 8);
                }
                float m_fin_1 = o_hold_1[0];
                m_fin_1 = max_noftz(m_fin_1, o_hold_1[1]);
                m_fin_1 = max_noftz(m_fin_1, o_hold_1[2]);
                m_fin_1 = max_noftz(m_fin_1, o_hold_1[3]);
                float w_hold_1[4];
                float _exp2_328_1 = approx_exp2((o_hold_1[0] - m_fin_1) * softmax_scale_log2);
                w_hold_1[0] = ((o_hold_1[0] == -CUDART_INF_F) ? 0.0f : _exp2_328_1);
                float _exp2_329_1 = approx_exp2((o_hold_1[1] - m_fin_1) * softmax_scale_log2);
                w_hold_1[1] = ((o_hold_1[1] == -CUDART_INF_F) ? 0.0f : _exp2_329_1);
                float _exp2_330_1 = approx_exp2((o_hold_1[2] - m_fin_1) * softmax_scale_log2);
                w_hold_1[2] = ((o_hold_1[2] == -CUDART_INF_F) ? 0.0f : _exp2_330_1);
                float _exp2_331_1 = approx_exp2((o_hold_1[3] - m_fin_1) * softmax_scale_log2);
                w_hold_1[3] = ((o_hold_1[3] == -CUDART_INF_F) ? 0.0f : _exp2_331_1);
                float colsum_1[32];
                float acc_32 = 0.0f;
                float w_sel_64 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_332_1 = __shfl_sync(0xFFFFFFFF, w_sel_64, 0);
                acc_32 = __fmaf_rn(lane_sum_1[0], _shfl_sync_332_1, acc_32);
                float w_sel_65 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_333_1 = __shfl_sync(0xFFFFFFFF, w_sel_65, 0);
                acc_32 = __fmaf_rn(lane_sum_1[32], _shfl_sync_333_1, acc_32);
                colsum_1[0] = acc_32;
                float acc_33 = 0.0f;
                float w_sel_66 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_334_1 = __shfl_sync(0xFFFFFFFF, w_sel_66, 1);
                acc_33 = __fmaf_rn(lane_sum_1[1], _shfl_sync_334_1, acc_33);
                float w_sel_67 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_335_1 = __shfl_sync(0xFFFFFFFF, w_sel_67, 1);
                acc_33 = __fmaf_rn(lane_sum_1[33], _shfl_sync_335_1, acc_33);
                colsum_1[1] = acc_33;
                float acc_34 = 0.0f;
                float w_sel_68 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_336_1 = __shfl_sync(0xFFFFFFFF, w_sel_68, 2);
                acc_34 = __fmaf_rn(lane_sum_1[2], _shfl_sync_336_1, acc_34);
                float w_sel_69 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_337_1 = __shfl_sync(0xFFFFFFFF, w_sel_69, 2);
                acc_34 = __fmaf_rn(lane_sum_1[34], _shfl_sync_337_1, acc_34);
                colsum_1[2] = acc_34;
                float acc_35 = 0.0f;
                float w_sel_70 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_338_1 = __shfl_sync(0xFFFFFFFF, w_sel_70, 3);
                acc_35 = __fmaf_rn(lane_sum_1[3], _shfl_sync_338_1, acc_35);
                float w_sel_71 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_339_1 = __shfl_sync(0xFFFFFFFF, w_sel_71, 3);
                acc_35 = __fmaf_rn(lane_sum_1[35], _shfl_sync_339_1, acc_35);
                colsum_1[3] = acc_35;
                float acc_36 = 0.0f;
                float w_sel_72 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_340_1 = __shfl_sync(0xFFFFFFFF, w_sel_72, 4);
                acc_36 = __fmaf_rn(lane_sum_1[4], _shfl_sync_340_1, acc_36);
                float w_sel_73 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_341_1 = __shfl_sync(0xFFFFFFFF, w_sel_73, 4);
                acc_36 = __fmaf_rn(lane_sum_1[36], _shfl_sync_341_1, acc_36);
                colsum_1[4] = acc_36;
                float acc_37 = 0.0f;
                float w_sel_74 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_342_1 = __shfl_sync(0xFFFFFFFF, w_sel_74, 5);
                acc_37 = __fmaf_rn(lane_sum_1[5], _shfl_sync_342_1, acc_37);
                float w_sel_75 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_343_1 = __shfl_sync(0xFFFFFFFF, w_sel_75, 5);
                acc_37 = __fmaf_rn(lane_sum_1[37], _shfl_sync_343_1, acc_37);
                colsum_1[5] = acc_37;
                float acc_38 = 0.0f;
                float w_sel_76 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_344_1 = __shfl_sync(0xFFFFFFFF, w_sel_76, 6);
                acc_38 = __fmaf_rn(lane_sum_1[6], _shfl_sync_344_1, acc_38);
                float w_sel_77 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_345_1 = __shfl_sync(0xFFFFFFFF, w_sel_77, 6);
                acc_38 = __fmaf_rn(lane_sum_1[38], _shfl_sync_345_1, acc_38);
                colsum_1[6] = acc_38;
                float acc_39 = 0.0f;
                float w_sel_78 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_346_1 = __shfl_sync(0xFFFFFFFF, w_sel_78, 7);
                acc_39 = __fmaf_rn(lane_sum_1[7], _shfl_sync_346_1, acc_39);
                float w_sel_79 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_347_1 = __shfl_sync(0xFFFFFFFF, w_sel_79, 7);
                acc_39 = __fmaf_rn(lane_sum_1[39], _shfl_sync_347_1, acc_39);
                colsum_1[7] = acc_39;
                float acc_40 = 0.0f;
                float w_sel_80 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_348_1 = __shfl_sync(0xFFFFFFFF, w_sel_80, 8);
                acc_40 = __fmaf_rn(lane_sum_1[8], _shfl_sync_348_1, acc_40);
                float w_sel_81 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_349_1 = __shfl_sync(0xFFFFFFFF, w_sel_81, 8);
                acc_40 = __fmaf_rn(lane_sum_1[40], _shfl_sync_349_1, acc_40);
                colsum_1[8] = acc_40;
                float acc_41 = 0.0f;
                float w_sel_82 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_350_1 = __shfl_sync(0xFFFFFFFF, w_sel_82, 9);
                acc_41 = __fmaf_rn(lane_sum_1[9], _shfl_sync_350_1, acc_41);
                float w_sel_83 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_351_1 = __shfl_sync(0xFFFFFFFF, w_sel_83, 9);
                acc_41 = __fmaf_rn(lane_sum_1[41], _shfl_sync_351_1, acc_41);
                colsum_1[9] = acc_41;
                float acc_42 = 0.0f;
                float w_sel_84 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_352_1 = __shfl_sync(0xFFFFFFFF, w_sel_84, 10);
                acc_42 = __fmaf_rn(lane_sum_1[10], _shfl_sync_352_1, acc_42);
                float w_sel_85 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_353_1 = __shfl_sync(0xFFFFFFFF, w_sel_85, 10);
                acc_42 = __fmaf_rn(lane_sum_1[42], _shfl_sync_353_1, acc_42);
                colsum_1[10] = acc_42;
                float acc_43 = 0.0f;
                float w_sel_86 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_354_1 = __shfl_sync(0xFFFFFFFF, w_sel_86, 11);
                acc_43 = __fmaf_rn(lane_sum_1[11], _shfl_sync_354_1, acc_43);
                float w_sel_87 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_355_1 = __shfl_sync(0xFFFFFFFF, w_sel_87, 11);
                acc_43 = __fmaf_rn(lane_sum_1[43], _shfl_sync_355_1, acc_43);
                colsum_1[11] = acc_43;
                float acc_44 = 0.0f;
                float w_sel_88 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_356_1 = __shfl_sync(0xFFFFFFFF, w_sel_88, 12);
                acc_44 = __fmaf_rn(lane_sum_1[12], _shfl_sync_356_1, acc_44);
                float w_sel_89 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_357_1 = __shfl_sync(0xFFFFFFFF, w_sel_89, 12);
                acc_44 = __fmaf_rn(lane_sum_1[44], _shfl_sync_357_1, acc_44);
                colsum_1[12] = acc_44;
                float acc_45 = 0.0f;
                float w_sel_90 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_358_1 = __shfl_sync(0xFFFFFFFF, w_sel_90, 13);
                acc_45 = __fmaf_rn(lane_sum_1[13], _shfl_sync_358_1, acc_45);
                float w_sel_91 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_359_1 = __shfl_sync(0xFFFFFFFF, w_sel_91, 13);
                acc_45 = __fmaf_rn(lane_sum_1[45], _shfl_sync_359_1, acc_45);
                colsum_1[13] = acc_45;
                float acc_46 = 0.0f;
                float w_sel_92 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_360_1 = __shfl_sync(0xFFFFFFFF, w_sel_92, 14);
                acc_46 = __fmaf_rn(lane_sum_1[14], _shfl_sync_360_1, acc_46);
                float w_sel_93 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_361_1 = __shfl_sync(0xFFFFFFFF, w_sel_93, 14);
                acc_46 = __fmaf_rn(lane_sum_1[46], _shfl_sync_361_1, acc_46);
                colsum_1[14] = acc_46;
                float acc_47 = 0.0f;
                float w_sel_94 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_362_1 = __shfl_sync(0xFFFFFFFF, w_sel_94, 15);
                acc_47 = __fmaf_rn(lane_sum_1[15], _shfl_sync_362_1, acc_47);
                float w_sel_95 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_363_1 = __shfl_sync(0xFFFFFFFF, w_sel_95, 15);
                acc_47 = __fmaf_rn(lane_sum_1[47], _shfl_sync_363_1, acc_47);
                colsum_1[15] = acc_47;
                float acc_48 = 0.0f;
                float w_sel_96 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_364_1 = __shfl_sync(0xFFFFFFFF, w_sel_96, 16);
                acc_48 = __fmaf_rn(lane_sum_1[16], _shfl_sync_364_1, acc_48);
                float w_sel_97 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_365_1 = __shfl_sync(0xFFFFFFFF, w_sel_97, 16);
                acc_48 = __fmaf_rn(lane_sum_1[48], _shfl_sync_365_1, acc_48);
                colsum_1[16] = acc_48;
                float acc_49 = 0.0f;
                float w_sel_98 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_366_1 = __shfl_sync(0xFFFFFFFF, w_sel_98, 17);
                acc_49 = __fmaf_rn(lane_sum_1[17], _shfl_sync_366_1, acc_49);
                float w_sel_99 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_367_1 = __shfl_sync(0xFFFFFFFF, w_sel_99, 17);
                acc_49 = __fmaf_rn(lane_sum_1[49], _shfl_sync_367_1, acc_49);
                colsum_1[17] = acc_49;
                float acc_50 = 0.0f;
                float w_sel_100 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_368_1 = __shfl_sync(0xFFFFFFFF, w_sel_100, 18);
                acc_50 = __fmaf_rn(lane_sum_1[18], _shfl_sync_368_1, acc_50);
                float w_sel_101 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_369_1 = __shfl_sync(0xFFFFFFFF, w_sel_101, 18);
                acc_50 = __fmaf_rn(lane_sum_1[50], _shfl_sync_369_1, acc_50);
                colsum_1[18] = acc_50;
                float acc_51 = 0.0f;
                float w_sel_102 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_370_1 = __shfl_sync(0xFFFFFFFF, w_sel_102, 19);
                acc_51 = __fmaf_rn(lane_sum_1[19], _shfl_sync_370_1, acc_51);
                float w_sel_103 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_371_1 = __shfl_sync(0xFFFFFFFF, w_sel_103, 19);
                acc_51 = __fmaf_rn(lane_sum_1[51], _shfl_sync_371_1, acc_51);
                colsum_1[19] = acc_51;
                float acc_52 = 0.0f;
                float w_sel_104 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_372_1 = __shfl_sync(0xFFFFFFFF, w_sel_104, 20);
                acc_52 = __fmaf_rn(lane_sum_1[20], _shfl_sync_372_1, acc_52);
                float w_sel_105 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_373_1 = __shfl_sync(0xFFFFFFFF, w_sel_105, 20);
                acc_52 = __fmaf_rn(lane_sum_1[52], _shfl_sync_373_1, acc_52);
                colsum_1[20] = acc_52;
                float acc_53 = 0.0f;
                float w_sel_106 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_374_1 = __shfl_sync(0xFFFFFFFF, w_sel_106, 21);
                acc_53 = __fmaf_rn(lane_sum_1[21], _shfl_sync_374_1, acc_53);
                float w_sel_107 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_375_1 = __shfl_sync(0xFFFFFFFF, w_sel_107, 21);
                acc_53 = __fmaf_rn(lane_sum_1[53], _shfl_sync_375_1, acc_53);
                colsum_1[21] = acc_53;
                float acc_54 = 0.0f;
                float w_sel_108 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_376_1 = __shfl_sync(0xFFFFFFFF, w_sel_108, 22);
                acc_54 = __fmaf_rn(lane_sum_1[22], _shfl_sync_376_1, acc_54);
                float w_sel_109 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_377_1 = __shfl_sync(0xFFFFFFFF, w_sel_109, 22);
                acc_54 = __fmaf_rn(lane_sum_1[54], _shfl_sync_377_1, acc_54);
                colsum_1[22] = acc_54;
                float acc_55 = 0.0f;
                float w_sel_110 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_378_1 = __shfl_sync(0xFFFFFFFF, w_sel_110, 23);
                acc_55 = __fmaf_rn(lane_sum_1[23], _shfl_sync_378_1, acc_55);
                float w_sel_111 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_379_1 = __shfl_sync(0xFFFFFFFF, w_sel_111, 23);
                acc_55 = __fmaf_rn(lane_sum_1[55], _shfl_sync_379_1, acc_55);
                colsum_1[23] = acc_55;
                float acc_56 = 0.0f;
                float w_sel_112 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_380_1 = __shfl_sync(0xFFFFFFFF, w_sel_112, 24);
                acc_56 = __fmaf_rn(lane_sum_1[24], _shfl_sync_380_1, acc_56);
                float w_sel_113 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_381_1 = __shfl_sync(0xFFFFFFFF, w_sel_113, 24);
                acc_56 = __fmaf_rn(lane_sum_1[56], _shfl_sync_381_1, acc_56);
                colsum_1[24] = acc_56;
                float acc_57 = 0.0f;
                float w_sel_114 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_382_1 = __shfl_sync(0xFFFFFFFF, w_sel_114, 25);
                acc_57 = __fmaf_rn(lane_sum_1[25], _shfl_sync_382_1, acc_57);
                float w_sel_115 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_383_1 = __shfl_sync(0xFFFFFFFF, w_sel_115, 25);
                acc_57 = __fmaf_rn(lane_sum_1[57], _shfl_sync_383_1, acc_57);
                colsum_1[25] = acc_57;
                float acc_58 = 0.0f;
                float w_sel_116 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_384_1 = __shfl_sync(0xFFFFFFFF, w_sel_116, 26);
                acc_58 = __fmaf_rn(lane_sum_1[26], _shfl_sync_384_1, acc_58);
                float w_sel_117 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_385_1 = __shfl_sync(0xFFFFFFFF, w_sel_117, 26);
                acc_58 = __fmaf_rn(lane_sum_1[58], _shfl_sync_385_1, acc_58);
                colsum_1[26] = acc_58;
                float acc_59 = 0.0f;
                float w_sel_118 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_386_1 = __shfl_sync(0xFFFFFFFF, w_sel_118, 27);
                acc_59 = __fmaf_rn(lane_sum_1[27], _shfl_sync_386_1, acc_59);
                float w_sel_119 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_387_1 = __shfl_sync(0xFFFFFFFF, w_sel_119, 27);
                acc_59 = __fmaf_rn(lane_sum_1[59], _shfl_sync_387_1, acc_59);
                colsum_1[27] = acc_59;
                float acc_60 = 0.0f;
                float w_sel_120 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_388_1 = __shfl_sync(0xFFFFFFFF, w_sel_120, 28);
                acc_60 = __fmaf_rn(lane_sum_1[28], _shfl_sync_388_1, acc_60);
                float w_sel_121 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_389_1 = __shfl_sync(0xFFFFFFFF, w_sel_121, 28);
                acc_60 = __fmaf_rn(lane_sum_1[60], _shfl_sync_389_1, acc_60);
                colsum_1[28] = acc_60;
                float acc_61 = 0.0f;
                float w_sel_122 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_390_1 = __shfl_sync(0xFFFFFFFF, w_sel_122, 29);
                acc_61 = __fmaf_rn(lane_sum_1[29], _shfl_sync_390_1, acc_61);
                float w_sel_123 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_391_1 = __shfl_sync(0xFFFFFFFF, w_sel_123, 29);
                acc_61 = __fmaf_rn(lane_sum_1[61], _shfl_sync_391_1, acc_61);
                colsum_1[29] = acc_61;
                float acc_62 = 0.0f;
                float w_sel_124 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_392_1 = __shfl_sync(0xFFFFFFFF, w_sel_124, 30);
                acc_62 = __fmaf_rn(lane_sum_1[30], _shfl_sync_392_1, acc_62);
                float w_sel_125 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_393_1 = __shfl_sync(0xFFFFFFFF, w_sel_125, 30);
                acc_62 = __fmaf_rn(lane_sum_1[62], _shfl_sync_393_1, acc_62);
                colsum_1[30] = acc_62;
                float acc_63 = 0.0f;
                float w_sel_126 = ((own_half_1 == 0) ? w_hold_1[0] : w_hold_1[1]);
                float _shfl_sync_394_1 = __shfl_sync(0xFFFFFFFF, w_sel_126, 31);
                acc_63 = __fmaf_rn(lane_sum_1[31], _shfl_sync_394_1, acc_63);
                float w_sel_127 = ((own_half_1 == 0) ? w_hold_1[2] : w_hold_1[3]);
                float _shfl_sync_395_1 = __shfl_sync(0xFFFFFFFF, w_sel_127, 31);
                acc_63 = __fmaf_rn(lane_sum_1[63], _shfl_sync_395_1, acc_63);
                colsum_1[31] = acc_63;
                int b4_1 = lane / 16 % 2;
                int b3_1 = lane / 8 % 2;
                int b2_1 = lane / 4 % 2;
                int b1_1 = lane / 2 % 2;
                int b0_1 = lane % 2;
                float r16_1[16];
                float snd_16 = ((b4_1 == 0) ? colsum_1[16] : colsum_1[0]);
                float kp_16 = ((b4_1 == 0) ? colsum_1[0] : colsum_1[16]);
                float _shfl_xor_sync_396_1 = __shfl_xor_sync(0xFFFFFFFF, snd_16, 16);
                r16_1[0] = kp_16 + _shfl_xor_sync_396_1;
                float snd_17 = ((b4_1 == 0) ? colsum_1[17] : colsum_1[1]);
                float kp_17 = ((b4_1 == 0) ? colsum_1[1] : colsum_1[17]);
                float _shfl_xor_sync_397_1 = __shfl_xor_sync(0xFFFFFFFF, snd_17, 16);
                r16_1[1] = kp_17 + _shfl_xor_sync_397_1;
                float snd_18 = ((b4_1 == 0) ? colsum_1[18] : colsum_1[2]);
                float kp_18 = ((b4_1 == 0) ? colsum_1[2] : colsum_1[18]);
                float _shfl_xor_sync_398_1 = __shfl_xor_sync(0xFFFFFFFF, snd_18, 16);
                r16_1[2] = kp_18 + _shfl_xor_sync_398_1;
                float snd_19 = ((b4_1 == 0) ? colsum_1[19] : colsum_1[3]);
                float kp_19 = ((b4_1 == 0) ? colsum_1[3] : colsum_1[19]);
                float _shfl_xor_sync_399_1 = __shfl_xor_sync(0xFFFFFFFF, snd_19, 16);
                r16_1[3] = kp_19 + _shfl_xor_sync_399_1;
                float snd_20 = ((b4_1 == 0) ? colsum_1[20] : colsum_1[4]);
                float kp_20 = ((b4_1 == 0) ? colsum_1[4] : colsum_1[20]);
                float _shfl_xor_sync_400_1 = __shfl_xor_sync(0xFFFFFFFF, snd_20, 16);
                r16_1[4] = kp_20 + _shfl_xor_sync_400_1;
                float snd_21 = ((b4_1 == 0) ? colsum_1[21] : colsum_1[5]);
                float kp_21 = ((b4_1 == 0) ? colsum_1[5] : colsum_1[21]);
                float _shfl_xor_sync_401_1 = __shfl_xor_sync(0xFFFFFFFF, snd_21, 16);
                r16_1[5] = kp_21 + _shfl_xor_sync_401_1;
                float snd_22 = ((b4_1 == 0) ? colsum_1[22] : colsum_1[6]);
                float kp_22 = ((b4_1 == 0) ? colsum_1[6] : colsum_1[22]);
                float _shfl_xor_sync_402_1 = __shfl_xor_sync(0xFFFFFFFF, snd_22, 16);
                r16_1[6] = kp_22 + _shfl_xor_sync_402_1;
                float snd_23 = ((b4_1 == 0) ? colsum_1[23] : colsum_1[7]);
                float kp_23 = ((b4_1 == 0) ? colsum_1[7] : colsum_1[23]);
                float _shfl_xor_sync_403_1 = __shfl_xor_sync(0xFFFFFFFF, snd_23, 16);
                r16_1[7] = kp_23 + _shfl_xor_sync_403_1;
                float snd_24 = ((b4_1 == 0) ? colsum_1[24] : colsum_1[8]);
                float kp_24 = ((b4_1 == 0) ? colsum_1[8] : colsum_1[24]);
                float _shfl_xor_sync_404_1 = __shfl_xor_sync(0xFFFFFFFF, snd_24, 16);
                r16_1[8] = kp_24 + _shfl_xor_sync_404_1;
                float snd_25 = ((b4_1 == 0) ? colsum_1[25] : colsum_1[9]);
                float kp_25 = ((b4_1 == 0) ? colsum_1[9] : colsum_1[25]);
                float _shfl_xor_sync_405_1 = __shfl_xor_sync(0xFFFFFFFF, snd_25, 16);
                r16_1[9] = kp_25 + _shfl_xor_sync_405_1;
                float snd_26 = ((b4_1 == 0) ? colsum_1[26] : colsum_1[10]);
                float kp_26 = ((b4_1 == 0) ? colsum_1[10] : colsum_1[26]);
                float _shfl_xor_sync_406_1 = __shfl_xor_sync(0xFFFFFFFF, snd_26, 16);
                r16_1[10] = kp_26 + _shfl_xor_sync_406_1;
                float snd_27 = ((b4_1 == 0) ? colsum_1[27] : colsum_1[11]);
                float kp_27 = ((b4_1 == 0) ? colsum_1[11] : colsum_1[27]);
                float _shfl_xor_sync_407_1 = __shfl_xor_sync(0xFFFFFFFF, snd_27, 16);
                r16_1[11] = kp_27 + _shfl_xor_sync_407_1;
                float snd_28 = ((b4_1 == 0) ? colsum_1[28] : colsum_1[12]);
                float kp_28 = ((b4_1 == 0) ? colsum_1[12] : colsum_1[28]);
                float _shfl_xor_sync_408_1 = __shfl_xor_sync(0xFFFFFFFF, snd_28, 16);
                r16_1[12] = kp_28 + _shfl_xor_sync_408_1;
                float snd_29 = ((b4_1 == 0) ? colsum_1[29] : colsum_1[13]);
                float kp_29 = ((b4_1 == 0) ? colsum_1[13] : colsum_1[29]);
                float _shfl_xor_sync_409_1 = __shfl_xor_sync(0xFFFFFFFF, snd_29, 16);
                r16_1[13] = kp_29 + _shfl_xor_sync_409_1;
                float snd_30 = ((b4_1 == 0) ? colsum_1[30] : colsum_1[14]);
                float kp_30 = ((b4_1 == 0) ? colsum_1[14] : colsum_1[30]);
                float _shfl_xor_sync_410_1 = __shfl_xor_sync(0xFFFFFFFF, snd_30, 16);
                r16_1[14] = kp_30 + _shfl_xor_sync_410_1;
                float snd_31 = ((b4_1 == 0) ? colsum_1[31] : colsum_1[15]);
                float kp_31 = ((b4_1 == 0) ? colsum_1[15] : colsum_1[31]);
                float _shfl_xor_sync_411_1 = __shfl_xor_sync(0xFFFFFFFF, snd_31, 16);
                r16_1[15] = kp_31 + _shfl_xor_sync_411_1;
                float r8_1[8];
                float snd8_8 = ((b3_1 == 0) ? r16_1[8] : r16_1[0]);
                float kp8_8 = ((b3_1 == 0) ? r16_1[0] : r16_1[8]);
                float _shfl_xor_sync_412_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_8, 8);
                r8_1[0] = kp8_8 + _shfl_xor_sync_412_1;
                float snd8_9 = ((b3_1 == 0) ? r16_1[9] : r16_1[1]);
                float kp8_9 = ((b3_1 == 0) ? r16_1[1] : r16_1[9]);
                float _shfl_xor_sync_413_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_9, 8);
                r8_1[1] = kp8_9 + _shfl_xor_sync_413_1;
                float snd8_10 = ((b3_1 == 0) ? r16_1[10] : r16_1[2]);
                float kp8_10 = ((b3_1 == 0) ? r16_1[2] : r16_1[10]);
                float _shfl_xor_sync_414_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_10, 8);
                r8_1[2] = kp8_10 + _shfl_xor_sync_414_1;
                float snd8_11 = ((b3_1 == 0) ? r16_1[11] : r16_1[3]);
                float kp8_11 = ((b3_1 == 0) ? r16_1[3] : r16_1[11]);
                float _shfl_xor_sync_415_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_11, 8);
                r8_1[3] = kp8_11 + _shfl_xor_sync_415_1;
                float snd8_12 = ((b3_1 == 0) ? r16_1[12] : r16_1[4]);
                float kp8_12 = ((b3_1 == 0) ? r16_1[4] : r16_1[12]);
                float _shfl_xor_sync_416_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_12, 8);
                r8_1[4] = kp8_12 + _shfl_xor_sync_416_1;
                float snd8_13 = ((b3_1 == 0) ? r16_1[13] : r16_1[5]);
                float kp8_13 = ((b3_1 == 0) ? r16_1[5] : r16_1[13]);
                float _shfl_xor_sync_417_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_13, 8);
                r8_1[5] = kp8_13 + _shfl_xor_sync_417_1;
                float snd8_14 = ((b3_1 == 0) ? r16_1[14] : r16_1[6]);
                float kp8_14 = ((b3_1 == 0) ? r16_1[6] : r16_1[14]);
                float _shfl_xor_sync_418_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_14, 8);
                r8_1[6] = kp8_14 + _shfl_xor_sync_418_1;
                float snd8_15 = ((b3_1 == 0) ? r16_1[15] : r16_1[7]);
                float kp8_15 = ((b3_1 == 0) ? r16_1[7] : r16_1[15]);
                float _shfl_xor_sync_419_1 = __shfl_xor_sync(0xFFFFFFFF, snd8_15, 8);
                r8_1[7] = kp8_15 + _shfl_xor_sync_419_1;
                float r4_1[4];
                float snd4_4 = ((b2_1 == 0) ? r8_1[4] : r8_1[0]);
                float kp4_4 = ((b2_1 == 0) ? r8_1[0] : r8_1[4]);
                float _shfl_xor_sync_420_1 = __shfl_xor_sync(0xFFFFFFFF, snd4_4, 4);
                r4_1[0] = kp4_4 + _shfl_xor_sync_420_1;
                float snd4_5 = ((b2_1 == 0) ? r8_1[5] : r8_1[1]);
                float kp4_5 = ((b2_1 == 0) ? r8_1[1] : r8_1[5]);
                float _shfl_xor_sync_421_1 = __shfl_xor_sync(0xFFFFFFFF, snd4_5, 4);
                r4_1[1] = kp4_5 + _shfl_xor_sync_421_1;
                float snd4_6 = ((b2_1 == 0) ? r8_1[6] : r8_1[2]);
                float kp4_6 = ((b2_1 == 0) ? r8_1[2] : r8_1[6]);
                float _shfl_xor_sync_422_1 = __shfl_xor_sync(0xFFFFFFFF, snd4_6, 4);
                r4_1[2] = kp4_6 + _shfl_xor_sync_422_1;
                float snd4_7 = ((b2_1 == 0) ? r8_1[7] : r8_1[3]);
                float kp4_7 = ((b2_1 == 0) ? r8_1[3] : r8_1[7]);
                float _shfl_xor_sync_423_1 = __shfl_xor_sync(0xFFFFFFFF, snd4_7, 4);
                r4_1[3] = kp4_7 + _shfl_xor_sync_423_1;
                float r2_1[2];
                float snd2_2 = ((b1_1 == 0) ? r4_1[2] : r4_1[0]);
                float kp2_2 = ((b1_1 == 0) ? r4_1[0] : r4_1[2]);
                float _shfl_xor_sync_424_1 = __shfl_xor_sync(0xFFFFFFFF, snd2_2, 2);
                r2_1[0] = kp2_2 + _shfl_xor_sync_424_1;
                float snd2_3 = ((b1_1 == 0) ? r4_1[3] : r4_1[1]);
                float kp2_3 = ((b1_1 == 0) ? r4_1[1] : r4_1[3]);
                float _shfl_xor_sync_425_1 = __shfl_xor_sync(0xFFFFFFFF, snd2_3, 2);
                r2_1[1] = kp2_3 + _shfl_xor_sync_425_1;
                float snd1_1 = ((b0_1 == 0) ? r2_1[1] : r2_1[0]);
                float kp1_1 = ((b0_1 == 0) ? r2_1[0] : r2_1[1]);
                float _shfl_xor_sync_426_1 = __shfl_xor_sync(0xFFFFFFFF, snd1_1, 1);
                float r1_1 = kp1_1 + _shfl_xor_sync_426_1;
                float xs_1 = r1_1;
                if (npairs_1 == 0) {
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                }
                smem_sum[(4 + warp_in_role_1) * 32 + lane] = xs_1;
                asm volatile("barrier.sync 2, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float head_sum_1 = 0.0f;
                head_sum_1 = head_sum_1 + smem_sum[128 + lane];
                head_sum_1 = head_sum_1 + smem_sum[160 + lane];
                head_sum_1 = head_sum_1 + smem_sum[192 + lane];
                head_sum_1 = head_sum_1 + smem_sum[224 + lane];
                float _rcp_427_1 = approx_rcp(head_sum_1);
                float inv_1 = ((head_sum_1 > 0.0f) ? output_scale * _rcp_427_1 : 0.0f);
                int _stage_item_tick_1 = (int)((unsigned int)item_ord_1 % 2);
                int _parity_item_tick_1 = (int)((unsigned int)item_ord_1 / 2 % 2);
                mbarrier_wait(item_tick_addr + (_stage_item_tick_1) * 8, _parity_item_tick_1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (npairs_1 > 0) {
                    int _stage_pv_done_8 = (int)((unsigned int)(gpair_1 - 1) % 8);
                    int _parity_pv_done_8 = (int)((unsigned int)(gpair_1 - 1) / 8 % 2);
                    mbarrier_wait(pv_done_addr + (_stage_pv_done_8) * 8, _parity_pv_done_8);
                    if (npairs_1 > 1) {
                        int _stage_pv_done_9 = (int)((unsigned int)(gpair_1 - 1 - 1) % 8);
                        int _parity_pv_done_9 = (int)((unsigned int)(gpair_1 - 1 - 1) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_9) * 8, _parity_pv_done_9);
                    }
                    if (npairs_1 > 2) {
                        int _stage_pv_done_10 = (int)((unsigned int)(gpair_1 - 1 - 2) % 8);
                        int _parity_pv_done_10 = (int)((unsigned int)(gpair_1 - 1 - 2) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_10) * 8, _parity_pv_done_10);
                    }
                    if (npairs_1 > 3) {
                        int _stage_pv_done_11 = (int)((unsigned int)(gpair_1 - 1 - 3) % 8);
                        int _parity_pv_done_11 = (int)((unsigned int)(gpair_1 - 1 - 3) / 8 % 2);
                        mbarrier_wait(pv_done_addr + (_stage_pv_done_11) * 8, _parity_pv_done_11);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_epi_1[32];
                o_epi_1[0] = 0.0f;
                o_epi_1[1] = 0.0f;
                o_epi_1[2] = 0.0f;
                o_epi_1[3] = 0.0f;
                o_epi_1[4] = 0.0f;
                o_epi_1[5] = 0.0f;
                o_epi_1[6] = 0.0f;
                o_epi_1[7] = 0.0f;
                o_epi_1[8] = 0.0f;
                o_epi_1[9] = 0.0f;
                o_epi_1[10] = 0.0f;
                o_epi_1[11] = 0.0f;
                o_epi_1[12] = 0.0f;
                o_epi_1[13] = 0.0f;
                o_epi_1[14] = 0.0f;
                o_epi_1[15] = 0.0f;
                o_epi_1[16] = 0.0f;
                o_epi_1[17] = 0.0f;
                o_epi_1[18] = 0.0f;
                o_epi_1[19] = 0.0f;
                o_epi_1[20] = 0.0f;
                o_epi_1[21] = 0.0f;
                o_epi_1[22] = 0.0f;
                o_epi_1[23] = 0.0f;
                o_epi_1[24] = 0.0f;
                o_epi_1[25] = 0.0f;
                o_epi_1[26] = 0.0f;
                o_epi_1[27] = 0.0f;
                o_epi_1[28] = 0.0f;
                o_epi_1[29] = 0.0f;
                o_epi_1[30] = 0.0f;
                o_epi_1[31] = 0.0f;
                if (npairs_1 > 0) {
                    float o_read_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_4[0]), "=f"(o_read_4[1]), "=f"(o_read_4[2]), "=f"(o_read_4[3]), "=f"(o_read_4[4]), "=f"(o_read_4[5]), "=f"(o_read_4[6]), "=f"(o_read_4[7]), "=f"(o_read_4[8]), "=f"(o_read_4[9]), "=f"(o_read_4[10]), "=f"(o_read_4[11]), "=f"(o_read_4[12]), "=f"(o_read_4[13]), "=f"(o_read_4[14]), "=f"(o_read_4[15]), "=f"(o_read_4[16]), "=f"(o_read_4[17]), "=f"(o_read_4[18]), "=f"(o_read_4[19]), "=f"(o_read_4[20]), "=f"(o_read_4[21]), "=f"(o_read_4[22]), "=f"(o_read_4[23]), "=f"(o_read_4[24]), "=f"(o_read_4[25]), "=f"(o_read_4[26]), "=f"(o_read_4[27]), "=f"(o_read_4[28]), "=f"(o_read_4[29]), "=f"(o_read_4[30]), "=f"(o_read_4[31])
                        : "r"(taddr + 288 + (unsigned int)lane_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_428_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 0);
                    o_epi_1[0] = __fmaf_rn(o_read_4[0], _shfl_sync_428_1, o_epi_1[0]);
                    float _shfl_sync_429_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 1);
                    o_epi_1[1] = __fmaf_rn(o_read_4[1], _shfl_sync_429_1, o_epi_1[1]);
                    float _shfl_sync_430_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 2);
                    o_epi_1[2] = __fmaf_rn(o_read_4[2], _shfl_sync_430_1, o_epi_1[2]);
                    float _shfl_sync_431_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 3);
                    o_epi_1[3] = __fmaf_rn(o_read_4[3], _shfl_sync_431_1, o_epi_1[3]);
                    float _shfl_sync_432_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 4);
                    o_epi_1[4] = __fmaf_rn(o_read_4[4], _shfl_sync_432_1, o_epi_1[4]);
                    float _shfl_sync_433_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 5);
                    o_epi_1[5] = __fmaf_rn(o_read_4[5], _shfl_sync_433_1, o_epi_1[5]);
                    float _shfl_sync_434_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 6);
                    o_epi_1[6] = __fmaf_rn(o_read_4[6], _shfl_sync_434_1, o_epi_1[6]);
                    float _shfl_sync_435_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 7);
                    o_epi_1[7] = __fmaf_rn(o_read_4[7], _shfl_sync_435_1, o_epi_1[7]);
                    float _shfl_sync_436_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 8);
                    o_epi_1[8] = __fmaf_rn(o_read_4[8], _shfl_sync_436_1, o_epi_1[8]);
                    float _shfl_sync_437_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 9);
                    o_epi_1[9] = __fmaf_rn(o_read_4[9], _shfl_sync_437_1, o_epi_1[9]);
                    float _shfl_sync_438_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 10);
                    o_epi_1[10] = __fmaf_rn(o_read_4[10], _shfl_sync_438_1, o_epi_1[10]);
                    float _shfl_sync_439_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 11);
                    o_epi_1[11] = __fmaf_rn(o_read_4[11], _shfl_sync_439_1, o_epi_1[11]);
                    float _shfl_sync_440_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 12);
                    o_epi_1[12] = __fmaf_rn(o_read_4[12], _shfl_sync_440_1, o_epi_1[12]);
                    float _shfl_sync_441_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 13);
                    o_epi_1[13] = __fmaf_rn(o_read_4[13], _shfl_sync_441_1, o_epi_1[13]);
                    float _shfl_sync_442_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 14);
                    o_epi_1[14] = __fmaf_rn(o_read_4[14], _shfl_sync_442_1, o_epi_1[14]);
                    float _shfl_sync_443_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 15);
                    o_epi_1[15] = __fmaf_rn(o_read_4[15], _shfl_sync_443_1, o_epi_1[15]);
                    float _shfl_sync_444_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 16);
                    o_epi_1[16] = __fmaf_rn(o_read_4[16], _shfl_sync_444_1, o_epi_1[16]);
                    float _shfl_sync_445_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 17);
                    o_epi_1[17] = __fmaf_rn(o_read_4[17], _shfl_sync_445_1, o_epi_1[17]);
                    float _shfl_sync_446_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 18);
                    o_epi_1[18] = __fmaf_rn(o_read_4[18], _shfl_sync_446_1, o_epi_1[18]);
                    float _shfl_sync_447_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 19);
                    o_epi_1[19] = __fmaf_rn(o_read_4[19], _shfl_sync_447_1, o_epi_1[19]);
                    float _shfl_sync_448_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 20);
                    o_epi_1[20] = __fmaf_rn(o_read_4[20], _shfl_sync_448_1, o_epi_1[20]);
                    float _shfl_sync_449_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 21);
                    o_epi_1[21] = __fmaf_rn(o_read_4[21], _shfl_sync_449_1, o_epi_1[21]);
                    float _shfl_sync_450_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 22);
                    o_epi_1[22] = __fmaf_rn(o_read_4[22], _shfl_sync_450_1, o_epi_1[22]);
                    float _shfl_sync_451_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 23);
                    o_epi_1[23] = __fmaf_rn(o_read_4[23], _shfl_sync_451_1, o_epi_1[23]);
                    float _shfl_sync_452_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 24);
                    o_epi_1[24] = __fmaf_rn(o_read_4[24], _shfl_sync_452_1, o_epi_1[24]);
                    float _shfl_sync_453_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 25);
                    o_epi_1[25] = __fmaf_rn(o_read_4[25], _shfl_sync_453_1, o_epi_1[25]);
                    float _shfl_sync_454_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 26);
                    o_epi_1[26] = __fmaf_rn(o_read_4[26], _shfl_sync_454_1, o_epi_1[26]);
                    float _shfl_sync_455_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 27);
                    o_epi_1[27] = __fmaf_rn(o_read_4[27], _shfl_sync_455_1, o_epi_1[27]);
                    float _shfl_sync_456_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 28);
                    o_epi_1[28] = __fmaf_rn(o_read_4[28], _shfl_sync_456_1, o_epi_1[28]);
                    float _shfl_sync_457_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 29);
                    o_epi_1[29] = __fmaf_rn(o_read_4[29], _shfl_sync_457_1, o_epi_1[29]);
                    float _shfl_sync_458_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 30);
                    o_epi_1[30] = __fmaf_rn(o_read_4[30], _shfl_sync_458_1, o_epi_1[30]);
                    float _shfl_sync_459_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[0], 31);
                    o_epi_1[31] = __fmaf_rn(o_read_4[31], _shfl_sync_459_1, o_epi_1[31]);
                    float o_read_5[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_5[0]), "=f"(o_read_5[1]), "=f"(o_read_5[2]), "=f"(o_read_5[3]), "=f"(o_read_5[4]), "=f"(o_read_5[5]), "=f"(o_read_5[6]), "=f"(o_read_5[7]), "=f"(o_read_5[8]), "=f"(o_read_5[9]), "=f"(o_read_5[10]), "=f"(o_read_5[11]), "=f"(o_read_5[12]), "=f"(o_read_5[13]), "=f"(o_read_5[14]), "=f"(o_read_5[15]), "=f"(o_read_5[16]), "=f"(o_read_5[17]), "=f"(o_read_5[18]), "=f"(o_read_5[19]), "=f"(o_read_5[20]), "=f"(o_read_5[21]), "=f"(o_read_5[22]), "=f"(o_read_5[23]), "=f"(o_read_5[24]), "=f"(o_read_5[25]), "=f"(o_read_5[26]), "=f"(o_read_5[27]), "=f"(o_read_5[28]), "=f"(o_read_5[29]), "=f"(o_read_5[30]), "=f"(o_read_5[31])
                        : "r"(taddr + 352 + (unsigned int)lane_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_460_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 0);
                    o_epi_1[0] = __fmaf_rn(o_read_5[0], _shfl_sync_460_1, o_epi_1[0]);
                    float _shfl_sync_461_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 1);
                    o_epi_1[1] = __fmaf_rn(o_read_5[1], _shfl_sync_461_1, o_epi_1[1]);
                    float _shfl_sync_462_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 2);
                    o_epi_1[2] = __fmaf_rn(o_read_5[2], _shfl_sync_462_1, o_epi_1[2]);
                    float _shfl_sync_463_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 3);
                    o_epi_1[3] = __fmaf_rn(o_read_5[3], _shfl_sync_463_1, o_epi_1[3]);
                    float _shfl_sync_464_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 4);
                    o_epi_1[4] = __fmaf_rn(o_read_5[4], _shfl_sync_464_1, o_epi_1[4]);
                    float _shfl_sync_465_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 5);
                    o_epi_1[5] = __fmaf_rn(o_read_5[5], _shfl_sync_465_1, o_epi_1[5]);
                    float _shfl_sync_466_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 6);
                    o_epi_1[6] = __fmaf_rn(o_read_5[6], _shfl_sync_466_1, o_epi_1[6]);
                    float _shfl_sync_467_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 7);
                    o_epi_1[7] = __fmaf_rn(o_read_5[7], _shfl_sync_467_1, o_epi_1[7]);
                    float _shfl_sync_468_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 8);
                    o_epi_1[8] = __fmaf_rn(o_read_5[8], _shfl_sync_468_1, o_epi_1[8]);
                    float _shfl_sync_469_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 9);
                    o_epi_1[9] = __fmaf_rn(o_read_5[9], _shfl_sync_469_1, o_epi_1[9]);
                    float _shfl_sync_470_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 10);
                    o_epi_1[10] = __fmaf_rn(o_read_5[10], _shfl_sync_470_1, o_epi_1[10]);
                    float _shfl_sync_471_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 11);
                    o_epi_1[11] = __fmaf_rn(o_read_5[11], _shfl_sync_471_1, o_epi_1[11]);
                    float _shfl_sync_472_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 12);
                    o_epi_1[12] = __fmaf_rn(o_read_5[12], _shfl_sync_472_1, o_epi_1[12]);
                    float _shfl_sync_473_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 13);
                    o_epi_1[13] = __fmaf_rn(o_read_5[13], _shfl_sync_473_1, o_epi_1[13]);
                    float _shfl_sync_474_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 14);
                    o_epi_1[14] = __fmaf_rn(o_read_5[14], _shfl_sync_474_1, o_epi_1[14]);
                    float _shfl_sync_475_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 15);
                    o_epi_1[15] = __fmaf_rn(o_read_5[15], _shfl_sync_475_1, o_epi_1[15]);
                    float _shfl_sync_476_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 16);
                    o_epi_1[16] = __fmaf_rn(o_read_5[16], _shfl_sync_476_1, o_epi_1[16]);
                    float _shfl_sync_477_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 17);
                    o_epi_1[17] = __fmaf_rn(o_read_5[17], _shfl_sync_477_1, o_epi_1[17]);
                    float _shfl_sync_478_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 18);
                    o_epi_1[18] = __fmaf_rn(o_read_5[18], _shfl_sync_478_1, o_epi_1[18]);
                    float _shfl_sync_479_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 19);
                    o_epi_1[19] = __fmaf_rn(o_read_5[19], _shfl_sync_479_1, o_epi_1[19]);
                    float _shfl_sync_480_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 20);
                    o_epi_1[20] = __fmaf_rn(o_read_5[20], _shfl_sync_480_1, o_epi_1[20]);
                    float _shfl_sync_481_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 21);
                    o_epi_1[21] = __fmaf_rn(o_read_5[21], _shfl_sync_481_1, o_epi_1[21]);
                    float _shfl_sync_482_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 22);
                    o_epi_1[22] = __fmaf_rn(o_read_5[22], _shfl_sync_482_1, o_epi_1[22]);
                    float _shfl_sync_483_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 23);
                    o_epi_1[23] = __fmaf_rn(o_read_5[23], _shfl_sync_483_1, o_epi_1[23]);
                    float _shfl_sync_484_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 24);
                    o_epi_1[24] = __fmaf_rn(o_read_5[24], _shfl_sync_484_1, o_epi_1[24]);
                    float _shfl_sync_485_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 25);
                    o_epi_1[25] = __fmaf_rn(o_read_5[25], _shfl_sync_485_1, o_epi_1[25]);
                    float _shfl_sync_486_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 26);
                    o_epi_1[26] = __fmaf_rn(o_read_5[26], _shfl_sync_486_1, o_epi_1[26]);
                    float _shfl_sync_487_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 27);
                    o_epi_1[27] = __fmaf_rn(o_read_5[27], _shfl_sync_487_1, o_epi_1[27]);
                    float _shfl_sync_488_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 28);
                    o_epi_1[28] = __fmaf_rn(o_read_5[28], _shfl_sync_488_1, o_epi_1[28]);
                    float _shfl_sync_489_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 29);
                    o_epi_1[29] = __fmaf_rn(o_read_5[29], _shfl_sync_489_1, o_epi_1[29]);
                    float _shfl_sync_490_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 30);
                    o_epi_1[30] = __fmaf_rn(o_read_5[30], _shfl_sync_490_1, o_epi_1[30]);
                    float _shfl_sync_491_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[1], 31);
                    o_epi_1[31] = __fmaf_rn(o_read_5[31], _shfl_sync_491_1, o_epi_1[31]);
                    float o_read_6[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_6[0]), "=f"(o_read_6[1]), "=f"(o_read_6[2]), "=f"(o_read_6[3]), "=f"(o_read_6[4]), "=f"(o_read_6[5]), "=f"(o_read_6[6]), "=f"(o_read_6[7]), "=f"(o_read_6[8]), "=f"(o_read_6[9]), "=f"(o_read_6[10]), "=f"(o_read_6[11]), "=f"(o_read_6[12]), "=f"(o_read_6[13]), "=f"(o_read_6[14]), "=f"(o_read_6[15]), "=f"(o_read_6[16]), "=f"(o_read_6[17]), "=f"(o_read_6[18]), "=f"(o_read_6[19]), "=f"(o_read_6[20]), "=f"(o_read_6[21]), "=f"(o_read_6[22]), "=f"(o_read_6[23]), "=f"(o_read_6[24]), "=f"(o_read_6[25]), "=f"(o_read_6[26]), "=f"(o_read_6[27]), "=f"(o_read_6[28]), "=f"(o_read_6[29]), "=f"(o_read_6[30]), "=f"(o_read_6[31])
                        : "r"(taddr + 416 + (unsigned int)lane_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_492_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 0);
                    o_epi_1[0] = __fmaf_rn(o_read_6[0], _shfl_sync_492_1, o_epi_1[0]);
                    float _shfl_sync_493_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 1);
                    o_epi_1[1] = __fmaf_rn(o_read_6[1], _shfl_sync_493_1, o_epi_1[1]);
                    float _shfl_sync_494_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 2);
                    o_epi_1[2] = __fmaf_rn(o_read_6[2], _shfl_sync_494_1, o_epi_1[2]);
                    float _shfl_sync_495_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 3);
                    o_epi_1[3] = __fmaf_rn(o_read_6[3], _shfl_sync_495_1, o_epi_1[3]);
                    float _shfl_sync_496_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 4);
                    o_epi_1[4] = __fmaf_rn(o_read_6[4], _shfl_sync_496_1, o_epi_1[4]);
                    float _shfl_sync_497_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 5);
                    o_epi_1[5] = __fmaf_rn(o_read_6[5], _shfl_sync_497_1, o_epi_1[5]);
                    float _shfl_sync_498_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 6);
                    o_epi_1[6] = __fmaf_rn(o_read_6[6], _shfl_sync_498_1, o_epi_1[6]);
                    float _shfl_sync_499_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 7);
                    o_epi_1[7] = __fmaf_rn(o_read_6[7], _shfl_sync_499_1, o_epi_1[7]);
                    float _shfl_sync_500_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 8);
                    o_epi_1[8] = __fmaf_rn(o_read_6[8], _shfl_sync_500_1, o_epi_1[8]);
                    float _shfl_sync_501_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 9);
                    o_epi_1[9] = __fmaf_rn(o_read_6[9], _shfl_sync_501_1, o_epi_1[9]);
                    float _shfl_sync_502_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 10);
                    o_epi_1[10] = __fmaf_rn(o_read_6[10], _shfl_sync_502_1, o_epi_1[10]);
                    float _shfl_sync_503_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 11);
                    o_epi_1[11] = __fmaf_rn(o_read_6[11], _shfl_sync_503_1, o_epi_1[11]);
                    float _shfl_sync_504_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 12);
                    o_epi_1[12] = __fmaf_rn(o_read_6[12], _shfl_sync_504_1, o_epi_1[12]);
                    float _shfl_sync_505_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 13);
                    o_epi_1[13] = __fmaf_rn(o_read_6[13], _shfl_sync_505_1, o_epi_1[13]);
                    float _shfl_sync_506_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 14);
                    o_epi_1[14] = __fmaf_rn(o_read_6[14], _shfl_sync_506_1, o_epi_1[14]);
                    float _shfl_sync_507_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 15);
                    o_epi_1[15] = __fmaf_rn(o_read_6[15], _shfl_sync_507_1, o_epi_1[15]);
                    float _shfl_sync_508_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 16);
                    o_epi_1[16] = __fmaf_rn(o_read_6[16], _shfl_sync_508_1, o_epi_1[16]);
                    float _shfl_sync_509_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 17);
                    o_epi_1[17] = __fmaf_rn(o_read_6[17], _shfl_sync_509_1, o_epi_1[17]);
                    float _shfl_sync_510_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 18);
                    o_epi_1[18] = __fmaf_rn(o_read_6[18], _shfl_sync_510_1, o_epi_1[18]);
                    float _shfl_sync_511_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 19);
                    o_epi_1[19] = __fmaf_rn(o_read_6[19], _shfl_sync_511_1, o_epi_1[19]);
                    float _shfl_sync_512_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 20);
                    o_epi_1[20] = __fmaf_rn(o_read_6[20], _shfl_sync_512_1, o_epi_1[20]);
                    float _shfl_sync_513_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 21);
                    o_epi_1[21] = __fmaf_rn(o_read_6[21], _shfl_sync_513_1, o_epi_1[21]);
                    float _shfl_sync_514_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 22);
                    o_epi_1[22] = __fmaf_rn(o_read_6[22], _shfl_sync_514_1, o_epi_1[22]);
                    float _shfl_sync_515_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 23);
                    o_epi_1[23] = __fmaf_rn(o_read_6[23], _shfl_sync_515_1, o_epi_1[23]);
                    float _shfl_sync_516_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 24);
                    o_epi_1[24] = __fmaf_rn(o_read_6[24], _shfl_sync_516_1, o_epi_1[24]);
                    float _shfl_sync_517_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 25);
                    o_epi_1[25] = __fmaf_rn(o_read_6[25], _shfl_sync_517_1, o_epi_1[25]);
                    float _shfl_sync_518_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 26);
                    o_epi_1[26] = __fmaf_rn(o_read_6[26], _shfl_sync_518_1, o_epi_1[26]);
                    float _shfl_sync_519_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 27);
                    o_epi_1[27] = __fmaf_rn(o_read_6[27], _shfl_sync_519_1, o_epi_1[27]);
                    float _shfl_sync_520_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 28);
                    o_epi_1[28] = __fmaf_rn(o_read_6[28], _shfl_sync_520_1, o_epi_1[28]);
                    float _shfl_sync_521_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 29);
                    o_epi_1[29] = __fmaf_rn(o_read_6[29], _shfl_sync_521_1, o_epi_1[29]);
                    float _shfl_sync_522_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 30);
                    o_epi_1[30] = __fmaf_rn(o_read_6[30], _shfl_sync_522_1, o_epi_1[30]);
                    float _shfl_sync_523_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[2], 31);
                    o_epi_1[31] = __fmaf_rn(o_read_6[31], _shfl_sync_523_1, o_epi_1[31]);
                    float o_read_7[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(o_read_7[0]), "=f"(o_read_7[1]), "=f"(o_read_7[2]), "=f"(o_read_7[3]), "=f"(o_read_7[4]), "=f"(o_read_7[5]), "=f"(o_read_7[6]), "=f"(o_read_7[7]), "=f"(o_read_7[8]), "=f"(o_read_7[9]), "=f"(o_read_7[10]), "=f"(o_read_7[11]), "=f"(o_read_7[12]), "=f"(o_read_7[13]), "=f"(o_read_7[14]), "=f"(o_read_7[15]), "=f"(o_read_7[16]), "=f"(o_read_7[17]), "=f"(o_read_7[18]), "=f"(o_read_7[19]), "=f"(o_read_7[20]), "=f"(o_read_7[21]), "=f"(o_read_7[22]), "=f"(o_read_7[23]), "=f"(o_read_7[24]), "=f"(o_read_7[25]), "=f"(o_read_7[26]), "=f"(o_read_7[27]), "=f"(o_read_7[28]), "=f"(o_read_7[29]), "=f"(o_read_7[30]), "=f"(o_read_7[31])
                        : "r"(taddr + 480 + (unsigned int)lane_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float _shfl_sync_524_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 0);
                    o_epi_1[0] = __fmaf_rn(o_read_7[0], _shfl_sync_524_1, o_epi_1[0]);
                    float _shfl_sync_525_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 1);
                    o_epi_1[1] = __fmaf_rn(o_read_7[1], _shfl_sync_525_1, o_epi_1[1]);
                    float _shfl_sync_526_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 2);
                    o_epi_1[2] = __fmaf_rn(o_read_7[2], _shfl_sync_526_1, o_epi_1[2]);
                    float _shfl_sync_527_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 3);
                    o_epi_1[3] = __fmaf_rn(o_read_7[3], _shfl_sync_527_1, o_epi_1[3]);
                    float _shfl_sync_528_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 4);
                    o_epi_1[4] = __fmaf_rn(o_read_7[4], _shfl_sync_528_1, o_epi_1[4]);
                    float _shfl_sync_529_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 5);
                    o_epi_1[5] = __fmaf_rn(o_read_7[5], _shfl_sync_529_1, o_epi_1[5]);
                    float _shfl_sync_530_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 6);
                    o_epi_1[6] = __fmaf_rn(o_read_7[6], _shfl_sync_530_1, o_epi_1[6]);
                    float _shfl_sync_531_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 7);
                    o_epi_1[7] = __fmaf_rn(o_read_7[7], _shfl_sync_531_1, o_epi_1[7]);
                    float _shfl_sync_532_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 8);
                    o_epi_1[8] = __fmaf_rn(o_read_7[8], _shfl_sync_532_1, o_epi_1[8]);
                    float _shfl_sync_533_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 9);
                    o_epi_1[9] = __fmaf_rn(o_read_7[9], _shfl_sync_533_1, o_epi_1[9]);
                    float _shfl_sync_534_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 10);
                    o_epi_1[10] = __fmaf_rn(o_read_7[10], _shfl_sync_534_1, o_epi_1[10]);
                    float _shfl_sync_535_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 11);
                    o_epi_1[11] = __fmaf_rn(o_read_7[11], _shfl_sync_535_1, o_epi_1[11]);
                    float _shfl_sync_536_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 12);
                    o_epi_1[12] = __fmaf_rn(o_read_7[12], _shfl_sync_536_1, o_epi_1[12]);
                    float _shfl_sync_537_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 13);
                    o_epi_1[13] = __fmaf_rn(o_read_7[13], _shfl_sync_537_1, o_epi_1[13]);
                    float _shfl_sync_538_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 14);
                    o_epi_1[14] = __fmaf_rn(o_read_7[14], _shfl_sync_538_1, o_epi_1[14]);
                    float _shfl_sync_539_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 15);
                    o_epi_1[15] = __fmaf_rn(o_read_7[15], _shfl_sync_539_1, o_epi_1[15]);
                    float _shfl_sync_540_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 16);
                    o_epi_1[16] = __fmaf_rn(o_read_7[16], _shfl_sync_540_1, o_epi_1[16]);
                    float _shfl_sync_541_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 17);
                    o_epi_1[17] = __fmaf_rn(o_read_7[17], _shfl_sync_541_1, o_epi_1[17]);
                    float _shfl_sync_542_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 18);
                    o_epi_1[18] = __fmaf_rn(o_read_7[18], _shfl_sync_542_1, o_epi_1[18]);
                    float _shfl_sync_543_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 19);
                    o_epi_1[19] = __fmaf_rn(o_read_7[19], _shfl_sync_543_1, o_epi_1[19]);
                    float _shfl_sync_544_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 20);
                    o_epi_1[20] = __fmaf_rn(o_read_7[20], _shfl_sync_544_1, o_epi_1[20]);
                    float _shfl_sync_545_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 21);
                    o_epi_1[21] = __fmaf_rn(o_read_7[21], _shfl_sync_545_1, o_epi_1[21]);
                    float _shfl_sync_546_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 22);
                    o_epi_1[22] = __fmaf_rn(o_read_7[22], _shfl_sync_546_1, o_epi_1[22]);
                    float _shfl_sync_547_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 23);
                    o_epi_1[23] = __fmaf_rn(o_read_7[23], _shfl_sync_547_1, o_epi_1[23]);
                    float _shfl_sync_548_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 24);
                    o_epi_1[24] = __fmaf_rn(o_read_7[24], _shfl_sync_548_1, o_epi_1[24]);
                    float _shfl_sync_549_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 25);
                    o_epi_1[25] = __fmaf_rn(o_read_7[25], _shfl_sync_549_1, o_epi_1[25]);
                    float _shfl_sync_550_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 26);
                    o_epi_1[26] = __fmaf_rn(o_read_7[26], _shfl_sync_550_1, o_epi_1[26]);
                    float _shfl_sync_551_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 27);
                    o_epi_1[27] = __fmaf_rn(o_read_7[27], _shfl_sync_551_1, o_epi_1[27]);
                    float _shfl_sync_552_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 28);
                    o_epi_1[28] = __fmaf_rn(o_read_7[28], _shfl_sync_552_1, o_epi_1[28]);
                    float _shfl_sync_553_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 29);
                    o_epi_1[29] = __fmaf_rn(o_read_7[29], _shfl_sync_553_1, o_epi_1[29]);
                    float _shfl_sync_554_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 30);
                    o_epi_1[30] = __fmaf_rn(o_read_7[30], _shfl_sync_554_1, o_epi_1[30]);
                    float _shfl_sync_555_1 = __shfl_sync(0xFFFFFFFF, w_hold_1[3], 31);
                    o_epi_1[31] = __fmaf_rn(o_read_7[31], _shfl_sync_555_1, o_epi_1[31]);
                }
                if (work_idx_1 + (unsigned int)gridDim.x >= total_items_1) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                int q_row0_1 = request_1 * 8 * num_q_heads + head0_1;
                if (warp_in_role_1 == 0) {
                    int gc_1 = 32 + lane;
                    int lse_row_1 = q_row0_1 + gc_1 / 8 * num_q_heads + gc_1 % 8;
                    float lse_1 = -CUDART_INF_F;
                    if (head_sum_1 > 0.0f) {
                        float _log2_556_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_556_1) : "f"(head_sum_1));
                        lse_1 = __fmaf_rn(m_fin_1 * softmax_scale_log2, 0.6931472f, _log2_556_1 * 0.6931472f);
                    }
                    *(reinterpret_cast<float*>(msa_lse + lse_row_1) + (0)) = lse_1;
                }
                int row_c_32 = q_row0_1 + 4 * num_q_heads + 0;
                float _shfl_sync_557_1 = __shfl_sync(0xFFFFFFFF, inv_1, 0);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_32 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[0] * _shfl_sync_557_1);
                int row_c_33 = q_row0_1 + 4 * num_q_heads + 1;
                float _shfl_sync_558_1 = __shfl_sync(0xFFFFFFFF, inv_1, 1);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_33 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[1] * _shfl_sync_558_1);
                int row_c_34 = q_row0_1 + 4 * num_q_heads + 2;
                float _shfl_sync_559_1 = __shfl_sync(0xFFFFFFFF, inv_1, 2);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_34 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[2] * _shfl_sync_559_1);
                int row_c_35 = q_row0_1 + 4 * num_q_heads + 3;
                float _shfl_sync_560_1 = __shfl_sync(0xFFFFFFFF, inv_1, 3);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_35 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[3] * _shfl_sync_560_1);
                int row_c_36 = q_row0_1 + 4 * num_q_heads + 4;
                float _shfl_sync_561_1 = __shfl_sync(0xFFFFFFFF, inv_1, 4);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_36 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[4] * _shfl_sync_561_1);
                int row_c_37 = q_row0_1 + 4 * num_q_heads + 5;
                float _shfl_sync_562_1 = __shfl_sync(0xFFFFFFFF, inv_1, 5);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_37 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[5] * _shfl_sync_562_1);
                int row_c_38 = q_row0_1 + 4 * num_q_heads + 6;
                float _shfl_sync_563_1 = __shfl_sync(0xFFFFFFFF, inv_1, 6);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_38 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[6] * _shfl_sync_563_1);
                int row_c_39 = q_row0_1 + 4 * num_q_heads + 7;
                float _shfl_sync_564_1 = __shfl_sync(0xFFFFFFFF, inv_1, 7);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_39 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[7] * _shfl_sync_564_1);
                int row_c_40 = q_row0_1 + 5 * num_q_heads + 0;
                float _shfl_sync_565_1 = __shfl_sync(0xFFFFFFFF, inv_1, 8);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_40 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[8] * _shfl_sync_565_1);
                int row_c_41 = q_row0_1 + 5 * num_q_heads + 1;
                float _shfl_sync_566_1 = __shfl_sync(0xFFFFFFFF, inv_1, 9);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_41 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[9] * _shfl_sync_566_1);
                int row_c_42 = q_row0_1 + 5 * num_q_heads + 2;
                float _shfl_sync_567_1 = __shfl_sync(0xFFFFFFFF, inv_1, 10);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_42 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[10] * _shfl_sync_567_1);
                int row_c_43 = q_row0_1 + 5 * num_q_heads + 3;
                float _shfl_sync_568_1 = __shfl_sync(0xFFFFFFFF, inv_1, 11);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_43 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[11] * _shfl_sync_568_1);
                int row_c_44 = q_row0_1 + 5 * num_q_heads + 4;
                float _shfl_sync_569_1 = __shfl_sync(0xFFFFFFFF, inv_1, 12);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_44 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[12] * _shfl_sync_569_1);
                int row_c_45 = q_row0_1 + 5 * num_q_heads + 5;
                float _shfl_sync_570_1 = __shfl_sync(0xFFFFFFFF, inv_1, 13);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_45 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[13] * _shfl_sync_570_1);
                int row_c_46 = q_row0_1 + 5 * num_q_heads + 6;
                float _shfl_sync_571_1 = __shfl_sync(0xFFFFFFFF, inv_1, 14);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_46 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[14] * _shfl_sync_571_1);
                int row_c_47 = q_row0_1 + 5 * num_q_heads + 7;
                float _shfl_sync_572_1 = __shfl_sync(0xFFFFFFFF, inv_1, 15);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_47 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[15] * _shfl_sync_572_1);
                int row_c_48 = q_row0_1 + 6 * num_q_heads + 0;
                float _shfl_sync_573_1 = __shfl_sync(0xFFFFFFFF, inv_1, 16);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_48 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[16] * _shfl_sync_573_1);
                int row_c_49 = q_row0_1 + 6 * num_q_heads + 1;
                float _shfl_sync_574_1 = __shfl_sync(0xFFFFFFFF, inv_1, 17);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_49 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[17] * _shfl_sync_574_1);
                int row_c_50 = q_row0_1 + 6 * num_q_heads + 2;
                float _shfl_sync_575_1 = __shfl_sync(0xFFFFFFFF, inv_1, 18);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_50 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[18] * _shfl_sync_575_1);
                int row_c_51 = q_row0_1 + 6 * num_q_heads + 3;
                float _shfl_sync_576_1 = __shfl_sync(0xFFFFFFFF, inv_1, 19);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_51 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[19] * _shfl_sync_576_1);
                int row_c_52 = q_row0_1 + 6 * num_q_heads + 4;
                float _shfl_sync_577_1 = __shfl_sync(0xFFFFFFFF, inv_1, 20);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_52 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[20] * _shfl_sync_577_1);
                int row_c_53 = q_row0_1 + 6 * num_q_heads + 5;
                float _shfl_sync_578_1 = __shfl_sync(0xFFFFFFFF, inv_1, 21);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_53 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[21] * _shfl_sync_578_1);
                int row_c_54 = q_row0_1 + 6 * num_q_heads + 6;
                float _shfl_sync_579_1 = __shfl_sync(0xFFFFFFFF, inv_1, 22);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_54 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[22] * _shfl_sync_579_1);
                int row_c_55 = q_row0_1 + 6 * num_q_heads + 7;
                float _shfl_sync_580_1 = __shfl_sync(0xFFFFFFFF, inv_1, 23);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_55 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[23] * _shfl_sync_580_1);
                int row_c_56 = q_row0_1 + 7 * num_q_heads + 0;
                float _shfl_sync_581_1 = __shfl_sync(0xFFFFFFFF, inv_1, 24);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_56 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[24] * _shfl_sync_581_1);
                int row_c_57 = q_row0_1 + 7 * num_q_heads + 1;
                float _shfl_sync_582_1 = __shfl_sync(0xFFFFFFFF, inv_1, 25);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_57 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[25] * _shfl_sync_582_1);
                int row_c_58 = q_row0_1 + 7 * num_q_heads + 2;
                float _shfl_sync_583_1 = __shfl_sync(0xFFFFFFFF, inv_1, 26);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_58 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[26] * _shfl_sync_583_1);
                int row_c_59 = q_row0_1 + 7 * num_q_heads + 3;
                float _shfl_sync_584_1 = __shfl_sync(0xFFFFFFFF, inv_1, 27);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_59 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[27] * _shfl_sync_584_1);
                int row_c_60 = q_row0_1 + 7 * num_q_heads + 4;
                float _shfl_sync_585_1 = __shfl_sync(0xFFFFFFFF, inv_1, 28);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_60 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[28] * _shfl_sync_585_1);
                int row_c_61 = q_row0_1 + 7 * num_q_heads + 5;
                float _shfl_sync_586_1 = __shfl_sync(0xFFFFFFFF, inv_1, 29);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_61 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[29] * _shfl_sync_586_1);
                int row_c_62 = q_row0_1 + 7 * num_q_heads + 6;
                float _shfl_sync_587_1 = __shfl_sync(0xFFFFFFFF, inv_1, 30);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_62 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[30] * _shfl_sync_587_1);
                int row_c_63 = q_row0_1 + 7 * num_q_heads + 7;
                float _shfl_sync_588_1 = __shfl_sync(0xFFFFFFFF, inv_1, 31);
                *(reinterpret_cast<__nv_bfloat16*>(O + (row_c_63 * 128 + key_row_1)) + (0)) = (__nv_bfloat16)(o_epi_1[31] * _shfl_sync_588_1);
                asm volatile("tcgen05.fence::before_thread_sync;");
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(decode_done_addr);
                }
                item_ord_1 += 1;
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            int g_real_l = num_q_heads / num_kv_heads;
            int heads_per_req_l = num_kv_heads * (g_real_l / 8);
            unsigned int total_items_l = total_q / 8 * heads_per_req_l;
            int load_stage = 0;
            int item_par = 0;
            int cur_count = 0;
            int cur_kvlen = 0;
            int cur_has_a = 0;
            int cur_pos_a = 0;
            int cur_page_a = 0;
            int cur_tm_a = 0;
            int cur_has_b = 0;
            int cur_pos_b = 0;
            int cur_page_b = 0;
            int cur_tm_b = 0;
            unsigned int first_idx = blockIdx.x;
            if (first_idx < total_items_l) {
                int item_f = first_idx;
                int g_real_2 = num_q_heads / num_kv_heads;
                int halves_2 = g_real_2 / 8;
                int heads_per_req_2 = num_kv_heads * halves_2;
                int request_2 = item_f / heads_per_req_2;
                int sub_2 = item_f % heads_per_req_2;
                int kv_head_2 = sub_2 / halves_2;
                int head0_2 = kv_head_2 * g_real_2 + sub_2 % halves_2 * 8;
                int tok = lane / 4;
                int kv_len = task_kv_head[request_2];
                int query = request_2 * 8 + tok;
                int qpos = kv_len - 8 + tok;
                unsigned int my_lo = 0;
                unsigned int my_hi = 0;
                unsigned int one = 1;
                int slot_0 = lane % 4 + 0;
                int blk_0 = task_kind[(kv_head_2 * total_q + query) * 16 + slot_0];
                if (blk_0 >= 0) {
                    if (blk_0 * 128 <= qpos) {
                        if (blk_0 < 32) {
                            my_lo = my_lo | one << (unsigned int)blk_0;
                        } else if (blk_0 < 64) {
                            my_hi = my_hi | one << (unsigned int)(blk_0 - 32);
                        }
                    }
                }
                int slot_1 = lane % 4 + 4;
                int blk_1 = task_kind[(kv_head_2 * total_q + query) * 16 + slot_1];
                if (blk_1 >= 0) {
                    if (blk_1 * 128 <= qpos) {
                        if (blk_1 < 32) {
                            my_lo = my_lo | one << (unsigned int)blk_1;
                        } else if (blk_1 < 64) {
                            my_hi = my_hi | one << (unsigned int)(blk_1 - 32);
                        }
                    }
                }
                int slot_2 = lane % 4 + 8;
                int blk_2 = task_kind[(kv_head_2 * total_q + query) * 16 + slot_2];
                if (blk_2 >= 0) {
                    if (blk_2 * 128 <= qpos) {
                        if (blk_2 < 32) {
                            my_lo = my_lo | one << (unsigned int)blk_2;
                        } else if (blk_2 < 64) {
                            my_hi = my_hi | one << (unsigned int)(blk_2 - 32);
                        }
                    }
                }
                int slot_3 = lane % 4 + 12;
                int blk_3 = task_kind[(kv_head_2 * total_q + query) * 16 + slot_3];
                if (blk_3 >= 0) {
                    if (blk_3 * 128 <= qpos) {
                        if (blk_3 < 32) {
                            my_lo = my_lo | one << (unsigned int)blk_3;
                        } else if (blk_3 < 64) {
                            my_hi = my_hi | one << (unsigned int)(blk_3 - 32);
                        }
                    }
                }
                unsigned int tok_lo[8];
                unsigned int tok_hi[8];
                unsigned int u_lo = 0;
                unsigned int u_hi = 0;
                unsigned int v_lo_0 = ((tok == 0) ? my_lo : 0);
                unsigned int v_hi_0 = ((tok == 0) ? my_hi : 0);
                unsigned int _warp_redux_0;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_0) : "r"(v_lo_0));
                tok_lo[0] = _warp_redux_0;
                unsigned int _warp_redux_1;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_1) : "r"(v_hi_0));
                tok_hi[0] = _warp_redux_1;
                u_lo = u_lo | tok_lo[0];
                u_hi = u_hi | tok_hi[0];
                unsigned int v_lo_1 = ((tok == 1) ? my_lo : 0);
                unsigned int v_hi_1 = ((tok == 1) ? my_hi : 0);
                unsigned int _warp_redux_2;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_2) : "r"(v_lo_1));
                tok_lo[1] = _warp_redux_2;
                unsigned int _warp_redux_3;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_3) : "r"(v_hi_1));
                tok_hi[1] = _warp_redux_3;
                u_lo = u_lo | tok_lo[1];
                u_hi = u_hi | tok_hi[1];
                unsigned int v_lo_2 = ((tok == 2) ? my_lo : 0);
                unsigned int v_hi_2 = ((tok == 2) ? my_hi : 0);
                unsigned int _warp_redux_4;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_4) : "r"(v_lo_2));
                tok_lo[2] = _warp_redux_4;
                unsigned int _warp_redux_5;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_5) : "r"(v_hi_2));
                tok_hi[2] = _warp_redux_5;
                u_lo = u_lo | tok_lo[2];
                u_hi = u_hi | tok_hi[2];
                unsigned int v_lo_3 = ((tok == 3) ? my_lo : 0);
                unsigned int v_hi_3 = ((tok == 3) ? my_hi : 0);
                unsigned int _warp_redux_6;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_6) : "r"(v_lo_3));
                tok_lo[3] = _warp_redux_6;
                unsigned int _warp_redux_7;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_7) : "r"(v_hi_3));
                tok_hi[3] = _warp_redux_7;
                u_lo = u_lo | tok_lo[3];
                u_hi = u_hi | tok_hi[3];
                unsigned int v_lo_4 = ((tok == 4) ? my_lo : 0);
                unsigned int v_hi_4 = ((tok == 4) ? my_hi : 0);
                unsigned int _warp_redux_8;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_8) : "r"(v_lo_4));
                tok_lo[4] = _warp_redux_8;
                unsigned int _warp_redux_9;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_9) : "r"(v_hi_4));
                tok_hi[4] = _warp_redux_9;
                u_lo = u_lo | tok_lo[4];
                u_hi = u_hi | tok_hi[4];
                unsigned int v_lo_5 = ((tok == 5) ? my_lo : 0);
                unsigned int v_hi_5 = ((tok == 5) ? my_hi : 0);
                unsigned int _warp_redux_10;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_10) : "r"(v_lo_5));
                tok_lo[5] = _warp_redux_10;
                unsigned int _warp_redux_11;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_11) : "r"(v_hi_5));
                tok_hi[5] = _warp_redux_11;
                u_lo = u_lo | tok_lo[5];
                u_hi = u_hi | tok_hi[5];
                unsigned int v_lo_6 = ((tok == 6) ? my_lo : 0);
                unsigned int v_hi_6 = ((tok == 6) ? my_hi : 0);
                unsigned int _warp_redux_12;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_12) : "r"(v_lo_6));
                tok_lo[6] = _warp_redux_12;
                unsigned int _warp_redux_13;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_13) : "r"(v_hi_6));
                tok_hi[6] = _warp_redux_13;
                u_lo = u_lo | tok_lo[6];
                u_hi = u_hi | tok_hi[6];
                unsigned int v_lo_7 = ((tok == 7) ? my_lo : 0);
                unsigned int v_hi_7 = ((tok == 7) ? my_hi : 0);
                unsigned int _warp_redux_14;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_14) : "r"(v_lo_7));
                tok_lo[7] = _warp_redux_14;
                unsigned int _warp_redux_15;
                asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_15) : "r"(v_hi_7));
                tok_hi[7] = _warp_redux_15;
                u_lo = u_lo | tok_lo[7];
                u_hi = u_hi | tok_hi[7];
                int count = (unsigned int)__popc(u_lo) + (unsigned int)__popc(u_hi);
                if (count > 50) {
                    count = 50;
                }
                unsigned int below = (one << (unsigned int)lane) - 1;
                unsigned int lane_bit = one << (unsigned int)lane;
                int has_a = (((u_lo & lane_bit) != 0) ? 1 : 0);
                int has_b = (((u_hi & lane_bit) != 0) ? 1 : 0);
                int pos_a = (unsigned int)__popc(u_lo & below);
                int pos_b = (unsigned int)__popc(u_lo) + (unsigned int)__popc(u_hi & below);
                int tm_a = 0;
                int tm_b = 0;
                if ((tok_lo[0] & lane_bit) != 0) {
                    tm_a = tm_a | 1;
                }
                if ((tok_hi[0] & lane_bit) != 0) {
                    tm_b = tm_b | 1;
                }
                if ((tok_lo[1] & lane_bit) != 0) {
                    tm_a = tm_a | 2;
                }
                if ((tok_hi[1] & lane_bit) != 0) {
                    tm_b = tm_b | 2;
                }
                if ((tok_lo[2] & lane_bit) != 0) {
                    tm_a = tm_a | 4;
                }
                if ((tok_hi[2] & lane_bit) != 0) {
                    tm_b = tm_b | 4;
                }
                if ((tok_lo[3] & lane_bit) != 0) {
                    tm_a = tm_a | 8;
                }
                if ((tok_hi[3] & lane_bit) != 0) {
                    tm_b = tm_b | 8;
                }
                if ((tok_lo[4] & lane_bit) != 0) {
                    tm_a = tm_a | 16;
                }
                if ((tok_hi[4] & lane_bit) != 0) {
                    tm_b = tm_b | 16;
                }
                if ((tok_lo[5] & lane_bit) != 0) {
                    tm_a = tm_a | 32;
                }
                if ((tok_hi[5] & lane_bit) != 0) {
                    tm_b = tm_b | 32;
                }
                if ((tok_lo[6] & lane_bit) != 0) {
                    tm_a = tm_a | 64;
                }
                if ((tok_hi[6] & lane_bit) != 0) {
                    tm_b = tm_b | 64;
                }
                if ((tok_lo[7] & lane_bit) != 0) {
                    tm_a = tm_a | 128;
                }
                if ((tok_hi[7] & lane_bit) != 0) {
                    tm_b = tm_b | 128;
                }
                int page_a = 0;
                int page_b = 0;
                if (has_a != 0) {
                    page_a = kv_indices[request_2 * msa_max_pages + lane];
                }
                if (has_b != 0) {
                    page_b = kv_indices[request_2 * msa_max_pages + 32 + lane];
                }
                cur_count = count;
                cur_kvlen = kv_len;
                cur_has_a = has_a;
                cur_pos_a = pos_a;
                cur_page_a = page_a;
                cur_tm_a = tm_a;
                cur_has_b = has_b;
                cur_pos_b = pos_b;
                cur_page_b = page_b;
                cur_tm_b = tm_b;
            }
            unsigned int _phase_meta_empty = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int work_idx_l = blockIdx.x; work_idx_l < total_items_l; work_idx_l += gridDim.x) {
                int item_l = work_idx_l;
                int g_real_3 = num_q_heads / num_kv_heads;
                int halves_3 = g_real_3 / 8;
                int heads_per_req_3 = num_kv_heads * halves_3;
                int request_3 = item_l / heads_per_req_3;
                int sub_3 = item_l % heads_per_req_3;
                int kv_head_3 = sub_3 / halves_3;
                int head0_3 = kv_head_3 * g_real_3 + sub_3 % halves_3 * 8;
                int count_l = cur_count;
                __syncwarp();
                mbarrier_wait(meta_empty_addr + (item_par) * 8, _phase_meta_empty);
                int meta_base_l = item_par * 50;
                if (cur_has_a != 0) {
                    int slot_a = count_l - 1 - cur_pos_a;
                    smem_block[meta_base_l + slot_a] = lane;
                    smem_page[meta_base_l + slot_a] = cur_page_a;
                    smem_tokmask[meta_base_l + slot_a] = cur_tm_a;
                    if (cur_pos_a == 0) {
                        if (count_l % 2 == 1) {
                            smem_block[meta_base_l + count_l] = lane;
                            smem_page[meta_base_l + count_l] = cur_page_a;
                            smem_tokmask[meta_base_l + count_l] = 0;
                        }
                    }
                }
                if (cur_has_b != 0) {
                    int slot_b = count_l - 1 - cur_pos_b;
                    smem_block[meta_base_l + slot_b] = 32 + lane;
                    smem_page[meta_base_l + slot_b] = cur_page_b;
                    smem_tokmask[meta_base_l + slot_b] = cur_tm_b;
                    if (cur_pos_b == 0) {
                        if (count_l % 2 == 1) {
                            smem_block[meta_base_l + count_l] = 32 + lane;
                            smem_page[meta_base_l + count_l] = cur_page_b;
                            smem_tokmask[meta_base_l + count_l] = 0;
                        }
                    }
                }
                int npairs_l = (count_l + count_l % 2) / 2;
                __syncwarp();
                if (elect_sync()) {
                    if (npairs_l > 0) {
                        #pragma unroll
                        for (int k_tile = 0; k_tile < 2; k_tile++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[meta_base_l + k_tile] * num_kv_heads + kv_head_3, kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 8192);
                    tma_3d_gmem2smem(smem_qt_addr, (&Q), 0, head0_3, request_3 * 8, q_full_addr);
                }
                if (elect_sync()) {
                    if (npairs_l > 1) {
                        #pragma unroll
                        for (int k_tile_1 = 0; k_tile_1 < 2; k_tile_1++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[meta_base_l + 2 + k_tile_1] * num_kv_heads + kv_head_3, kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                if (elect_sync()) {
                    smem_npairs[item_par] = npairs_l;
                    smem_kvlen[item_par] = cur_kvlen;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(meta_full_addr + (item_par) * 8);
                }
                if (elect_sync()) {
                    if (npairs_l > 0) {
                        if (npairs_l > 2) {
                            #pragma unroll
                            for (int k_tile_2 = 0; k_tile_2 < 2; k_tile_2++) {
                                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                                tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[meta_base_l + 4 + k_tile_2] * num_kv_heads + kv_head_3, kv_full_addr + (load_stage) * 8);
                                load_stage += 1;
                                if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                            }
                        }
                        #pragma unroll
                        for (int v_tile = 0; v_tile < 2; v_tile++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&V), 0, 0, smem_page[meta_base_l + v_tile] * num_kv_heads + kv_head_3, kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                unsigned int nxt_idx = work_idx_l + (unsigned int)gridDim.x;
                cur_count = 0;
                cur_has_a = 0;
                cur_has_b = 0;
                if (nxt_idx < total_items_l) {
                    int item_n = nxt_idx;
                    int g_real_0 = num_q_heads / num_kv_heads;
                    int halves_0 = g_real_0 / 8;
                    int heads_per_req_0 = num_kv_heads * halves_0;
                    int request_0 = item_n / heads_per_req_0;
                    int sub_0 = item_n % heads_per_req_0;
                    int kv_head_0 = sub_0 / halves_0;
                    int head0_0 = kv_head_0 * g_real_0 + sub_0 % halves_0 * 8;
                    int tok_1 = lane / 4;
                    int kv_len_1 = task_kv_head[request_0];
                    int query_1 = request_0 * 8 + tok_1;
                    int qpos_1 = kv_len_1 - 8 + tok_1;
                    unsigned int my_lo_1 = 0;
                    unsigned int my_hi_1 = 0;
                    unsigned int one_1 = 1;
                    int slot_4 = lane % 4 + 0;
                    int blk_4 = task_kind[(kv_head_0 * total_q + query_1) * 16 + slot_4];
                    if (blk_4 >= 0) {
                        if (blk_4 * 128 <= qpos_1) {
                            if (blk_4 < 32) {
                                my_lo_1 = my_lo_1 | one_1 << (unsigned int)blk_4;
                            } else if (blk_4 < 64) {
                                my_hi_1 = my_hi_1 | one_1 << (unsigned int)(blk_4 - 32);
                            }
                        }
                    }
                    int slot_5 = lane % 4 + 4;
                    int blk_5 = task_kind[(kv_head_0 * total_q + query_1) * 16 + slot_5];
                    if (blk_5 >= 0) {
                        if (blk_5 * 128 <= qpos_1) {
                            if (blk_5 < 32) {
                                my_lo_1 = my_lo_1 | one_1 << (unsigned int)blk_5;
                            } else if (blk_5 < 64) {
                                my_hi_1 = my_hi_1 | one_1 << (unsigned int)(blk_5 - 32);
                            }
                        }
                    }
                    int slot_6 = lane % 4 + 8;
                    int blk_6 = task_kind[(kv_head_0 * total_q + query_1) * 16 + slot_6];
                    if (blk_6 >= 0) {
                        if (blk_6 * 128 <= qpos_1) {
                            if (blk_6 < 32) {
                                my_lo_1 = my_lo_1 | one_1 << (unsigned int)blk_6;
                            } else if (blk_6 < 64) {
                                my_hi_1 = my_hi_1 | one_1 << (unsigned int)(blk_6 - 32);
                            }
                        }
                    }
                    int slot_7 = lane % 4 + 12;
                    int blk_7 = task_kind[(kv_head_0 * total_q + query_1) * 16 + slot_7];
                    if (blk_7 >= 0) {
                        if (blk_7 * 128 <= qpos_1) {
                            if (blk_7 < 32) {
                                my_lo_1 = my_lo_1 | one_1 << (unsigned int)blk_7;
                            } else if (blk_7 < 64) {
                                my_hi_1 = my_hi_1 | one_1 << (unsigned int)(blk_7 - 32);
                            }
                        }
                    }
                    unsigned int tok_lo_1[8];
                    unsigned int tok_hi_1[8];
                    unsigned int u_lo_1 = 0;
                    unsigned int u_hi_1 = 0;
                    unsigned int v_lo_8 = ((tok_1 == 0) ? my_lo_1 : 0);
                    unsigned int v_hi_8 = ((tok_1 == 0) ? my_hi_1 : 0);
                    unsigned int _warp_redux_16;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_16) : "r"(v_lo_8));
                    tok_lo_1[0] = _warp_redux_16;
                    unsigned int _warp_redux_17;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_17) : "r"(v_hi_8));
                    tok_hi_1[0] = _warp_redux_17;
                    u_lo_1 = u_lo_1 | tok_lo_1[0];
                    u_hi_1 = u_hi_1 | tok_hi_1[0];
                    unsigned int v_lo_9 = ((tok_1 == 1) ? my_lo_1 : 0);
                    unsigned int v_hi_9 = ((tok_1 == 1) ? my_hi_1 : 0);
                    unsigned int _warp_redux_18;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_18) : "r"(v_lo_9));
                    tok_lo_1[1] = _warp_redux_18;
                    unsigned int _warp_redux_19;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_19) : "r"(v_hi_9));
                    tok_hi_1[1] = _warp_redux_19;
                    u_lo_1 = u_lo_1 | tok_lo_1[1];
                    u_hi_1 = u_hi_1 | tok_hi_1[1];
                    unsigned int v_lo_10 = ((tok_1 == 2) ? my_lo_1 : 0);
                    unsigned int v_hi_10 = ((tok_1 == 2) ? my_hi_1 : 0);
                    unsigned int _warp_redux_20;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_20) : "r"(v_lo_10));
                    tok_lo_1[2] = _warp_redux_20;
                    unsigned int _warp_redux_21;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_21) : "r"(v_hi_10));
                    tok_hi_1[2] = _warp_redux_21;
                    u_lo_1 = u_lo_1 | tok_lo_1[2];
                    u_hi_1 = u_hi_1 | tok_hi_1[2];
                    unsigned int v_lo_11 = ((tok_1 == 3) ? my_lo_1 : 0);
                    unsigned int v_hi_11 = ((tok_1 == 3) ? my_hi_1 : 0);
                    unsigned int _warp_redux_22;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_22) : "r"(v_lo_11));
                    tok_lo_1[3] = _warp_redux_22;
                    unsigned int _warp_redux_23;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_23) : "r"(v_hi_11));
                    tok_hi_1[3] = _warp_redux_23;
                    u_lo_1 = u_lo_1 | tok_lo_1[3];
                    u_hi_1 = u_hi_1 | tok_hi_1[3];
                    unsigned int v_lo_12 = ((tok_1 == 4) ? my_lo_1 : 0);
                    unsigned int v_hi_12 = ((tok_1 == 4) ? my_hi_1 : 0);
                    unsigned int _warp_redux_24;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_24) : "r"(v_lo_12));
                    tok_lo_1[4] = _warp_redux_24;
                    unsigned int _warp_redux_25;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_25) : "r"(v_hi_12));
                    tok_hi_1[4] = _warp_redux_25;
                    u_lo_1 = u_lo_1 | tok_lo_1[4];
                    u_hi_1 = u_hi_1 | tok_hi_1[4];
                    unsigned int v_lo_13 = ((tok_1 == 5) ? my_lo_1 : 0);
                    unsigned int v_hi_13 = ((tok_1 == 5) ? my_hi_1 : 0);
                    unsigned int _warp_redux_26;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_26) : "r"(v_lo_13));
                    tok_lo_1[5] = _warp_redux_26;
                    unsigned int _warp_redux_27;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_27) : "r"(v_hi_13));
                    tok_hi_1[5] = _warp_redux_27;
                    u_lo_1 = u_lo_1 | tok_lo_1[5];
                    u_hi_1 = u_hi_1 | tok_hi_1[5];
                    unsigned int v_lo_14 = ((tok_1 == 6) ? my_lo_1 : 0);
                    unsigned int v_hi_14 = ((tok_1 == 6) ? my_hi_1 : 0);
                    unsigned int _warp_redux_28;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_28) : "r"(v_lo_14));
                    tok_lo_1[6] = _warp_redux_28;
                    unsigned int _warp_redux_29;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_29) : "r"(v_hi_14));
                    tok_hi_1[6] = _warp_redux_29;
                    u_lo_1 = u_lo_1 | tok_lo_1[6];
                    u_hi_1 = u_hi_1 | tok_hi_1[6];
                    unsigned int v_lo_15 = ((tok_1 == 7) ? my_lo_1 : 0);
                    unsigned int v_hi_15 = ((tok_1 == 7) ? my_hi_1 : 0);
                    unsigned int _warp_redux_30;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_30) : "r"(v_lo_15));
                    tok_lo_1[7] = _warp_redux_30;
                    unsigned int _warp_redux_31;
                    asm volatile("redux.sync.or.b32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_31) : "r"(v_hi_15));
                    tok_hi_1[7] = _warp_redux_31;
                    u_lo_1 = u_lo_1 | tok_lo_1[7];
                    u_hi_1 = u_hi_1 | tok_hi_1[7];
                    int count_1 = (unsigned int)__popc(u_lo_1) + (unsigned int)__popc(u_hi_1);
                    if (count_1 > 50) {
                        count_1 = 50;
                    }
                    unsigned int below_1 = (one_1 << (unsigned int)lane) - 1;
                    unsigned int lane_bit_1 = one_1 << (unsigned int)lane;
                    int has_a_1 = (((u_lo_1 & lane_bit_1) != 0) ? 1 : 0);
                    int has_b_1 = (((u_hi_1 & lane_bit_1) != 0) ? 1 : 0);
                    int pos_a_1 = (unsigned int)__popc(u_lo_1 & below_1);
                    int pos_b_1 = (unsigned int)__popc(u_lo_1) + (unsigned int)__popc(u_hi_1 & below_1);
                    int tm_a_1 = 0;
                    int tm_b_1 = 0;
                    if ((tok_lo_1[0] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 1;
                    }
                    if ((tok_hi_1[0] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 1;
                    }
                    if ((tok_lo_1[1] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 2;
                    }
                    if ((tok_hi_1[1] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 2;
                    }
                    if ((tok_lo_1[2] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 4;
                    }
                    if ((tok_hi_1[2] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 4;
                    }
                    if ((tok_lo_1[3] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 8;
                    }
                    if ((tok_hi_1[3] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 8;
                    }
                    if ((tok_lo_1[4] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 16;
                    }
                    if ((tok_hi_1[4] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 16;
                    }
                    if ((tok_lo_1[5] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 32;
                    }
                    if ((tok_hi_1[5] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 32;
                    }
                    if ((tok_lo_1[6] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 64;
                    }
                    if ((tok_hi_1[6] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 64;
                    }
                    if ((tok_lo_1[7] & lane_bit_1) != 0) {
                        tm_a_1 = tm_a_1 | 128;
                    }
                    if ((tok_hi_1[7] & lane_bit_1) != 0) {
                        tm_b_1 = tm_b_1 | 128;
                    }
                    int page_a_1 = 0;
                    int page_b_1 = 0;
                    if (has_a_1 != 0) {
                        page_a_1 = kv_indices[request_0 * msa_max_pages + lane];
                    }
                    if (has_b_1 != 0) {
                        page_b_1 = kv_indices[request_0 * msa_max_pages + 32 + lane];
                    }
                    cur_count = count_1;
                    cur_kvlen = kv_len_1;
                    cur_has_a = has_a_1;
                    cur_pos_a = pos_a_1;
                    cur_page_a = page_a_1;
                    cur_tm_a = tm_a_1;
                    cur_has_b = has_b_1;
                    cur_pos_b = pos_b_1;
                    cur_page_b = page_b_1;
                    cur_tm_b = tm_b_1;
                }
                __syncwarp();
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int pair_2 = 1; pair_2 < 25; pair_2++) {
                        if (pair_2 >= npairs_l) {
                            break;
                        }
                        if (pair_2 + 2 < npairs_l) {
                            #pragma unroll
                            for (int k_tile_3 = 0; k_tile_3 < 2; k_tile_3++) {
                                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                                tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&K), 0, 0, smem_page[meta_base_l + pair_2 * 2 + 4 + k_tile_3] * num_kv_heads + kv_head_3, kv_full_addr + (load_stage) * 8);
                                load_stage += 1;
                                if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                            }
                        }
                        #pragma unroll
                        for (int v_tile_1 = 0; v_tile_1 < 2; v_tile_1++) {
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 16384);
                            tma_3d_gmem2smem(smem_kv_addr + (unsigned int)(load_stage * 16384), (&V), 0, 0, smem_page[meta_base_l + pair_2 * 2 + v_tile_1] * num_kv_heads + kv_head_3, kv_full_addr + (load_stage) * 8);
                            load_stage += 1;
                            if (load_stage == 9) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                }
                item_par += 1;
                if (item_par == 2) { item_par = 0; _phase_meta_empty ^= 1; }
            }
        }
    }
    // ---- Role: mma_s ----
    if (warp == 9) {
        { // mma_s_main
            int g_real_s = num_q_heads / num_kv_heads;
            int heads_per_req_s = num_kv_heads * (g_real_s / 8);
            unsigned int total_items_s = total_q / 8 * heads_per_req_s;
            int item_stage_s = 0;
            int item_ord_s = 0;
            int s_stage_s = 0;
            int k_tile_s = 0;
            int s_pairs_s = 0;
            int pv_tile_c = 0;
            int pv_pairs_c = 0;
            unsigned int _phase_meta_full_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_s_empty = 1;
            #pragma unroll 1
            for (unsigned int work_idx_s = blockIdx.x; work_idx_s < total_items_s; work_idx_s += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_s) * 8, _phase_meta_full_2);
                int npairs_s = smem_npairs[item_stage_s];
                item_stage_s += 1;
                if (item_stage_s == 2) { item_stage_s = 0; _phase_meta_full_2 ^= 1; }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)item_ord_s % 2)) * 8);
                }
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                if (npairs_s == 0) {
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(q_empty_addr);
                    }
                }
                #pragma unroll 1
                for (int s_pair = 0; s_pair < npairs_s; s_pair++) {
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr)), "r"(static_cast<uint32_t>((unsigned int)k_tile_s)) : "memory");
                    }
                    int need_tile_s = k_tile_s + 2 - 9;
                    int need_pairs_s = s_pairs_s + 1 - 2;
                    #pragma unroll 1
                    for (int spin_s = 0; spin_s < 4194304; spin_s++) {
                        int ok_s = 0;
                        if (pv_tile_c >= need_tile_s) {
                            if (pv_pairs_c >= need_pairs_s) {
                                ok_s = 1;
                            }
                        }
                        unsigned int ready_s = __ballot_sync(0xFFFFFFFF, ok_s != 0);
                        if (ready_s == 4294967295u) {
                            break;
                        }
                        uint32_t pv_tile_w;
                        asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(pv_tile_w) : "r"(static_cast<uint32_t>(smem_issue_addr + 4)) : "memory");
                        pv_tile_c = (int)pv_tile_w;
                        uint32_t pv_pairs_w;
                        asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(pv_pairs_w) : "r"(static_cast<uint32_t>(smem_issue_addr + 8)) : "memory");
                        pv_pairs_c = (int)pv_pairs_w;
                    }
                    int k0 = k_tile_s % 9;
                    int k1 = (k_tile_s + 1) % 9;
                    int _parity_kv_full = (int)((unsigned int)k_tile_s / 9 % 2);
                    mbarrier_wait(kv_full_addr + (k0) * 8, _parity_kv_full);
                    int _parity_kv_full_1 = (int)((unsigned int)(k_tile_s + 1) / 9 % 2);
                    mbarrier_wait(kv_full_addr + (k1) * 8, _parity_kv_full_1);
                    mbarrier_wait(s_empty_addr + (s_stage_s) * 8, _phase_s_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k0) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform(((smem_qt_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (k1) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128 + 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135266320, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128 + 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128 + 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_stage_s * 128 + 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135266320, 1);
                        }
                    }
                    if (npairs_s == s_pair + 1) {
                        elect_commit(q_empty_addr);
                    }
                    elect_commit(s_full_addr + (s_stage_s) * 8);
                    elect_commit(kv_empty_addr + (k0) * 8);
                    elect_commit(kv_empty_addr + (k1) * 8);
                    s_stage_s += 1;
                    if (s_stage_s == 2) { s_stage_s = 0; _phase_s_empty ^= 1; }
                    s_pairs_s = s_pairs_s + 1;
                    if (s_pair < 2) {
                        k_tile_s += 2;
                    } else {
                        k_tile_s += 4;
                    }
                }
                if (npairs_s == 1) {
                    k_tile_s += 2;
                }
                if (npairs_s > 1) {
                    k_tile_s += 4;
                }
                if (elect_sync()) {
                    asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr)), "r"(static_cast<uint32_t>((unsigned int)k_tile_s)) : "memory");
                }
                item_ord_s += 1;
            }
        }
    }
    // ---- Role: mma_pv ----
    if (warp == 10) {
        { // mma_pv_main
            int g_real_v = num_q_heads / num_kv_heads;
            int heads_per_req_v = num_kv_heads * (g_real_v / 8);
            unsigned int total_items_v = total_q / 8 * heads_per_req_v;
            int item_stage_v = 0;
            int item_ord_v = 0;
            int pf_stage_v = 0;
            int have_prev_v = 0;
            int v_tile_v = 0;
            int gpair_v = 0;
            int s_tile_c = 0;
            unsigned int _phase_meta_full_3 = 0;
            unsigned int _phase_decode_done_0 = 0;
            unsigned int _phase_p_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx_v = blockIdx.x; work_idx_v < total_items_v; work_idx_v += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_v) * 8, _phase_meta_full_3);
                int npairs_v = smem_npairs[item_stage_v];
                item_stage_v += 1;
                if (item_stage_v == 2) { item_stage_v = 0; _phase_meta_full_3 ^= 1; }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(meta_empty_addr + ((int)((unsigned int)item_ord_v % 2)) * 8);
                }
                if (npairs_v > 0) {
                    v_tile_v += 2;
                }
                if (npairs_v > 1) {
                    v_tile_v += 2;
                }
                if (npairs_v > 2) {
                    v_tile_v += 2;
                }
                if (npairs_v > 0) {
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 4)), "r"(static_cast<uint32_t>((unsigned int)v_tile_v)) : "memory");
                    }
                }
                if (have_prev_v != 0) {
                    mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                    _phase_decode_done_0 ^= 1;
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(item_tick_addr + ((int)((unsigned int)item_ord_v % 2)) * 8);
                }
                int first_pv_v = 1;
                #pragma unroll 1
                for (int pv_pair = 0; pv_pair < npairs_v; pv_pair++) {
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 4)), "r"(static_cast<uint32_t>((unsigned int)v_tile_v)) : "memory");
                    }
                    int need_tile_v = v_tile_v + 2 - 9;
                    #pragma unroll 1
                    for (int spin_v = 0; spin_v < 4194304; spin_v++) {
                        int ok_v = 0;
                        if (s_tile_c >= need_tile_v) {
                            ok_v = 1;
                        }
                        unsigned int ready_v = __ballot_sync(0xFFFFFFFF, ok_v != 0);
                        if (ready_v == 4294967295u) {
                            break;
                        }
                        uint32_t s_tile_w;
                        asm volatile("ld.acquire.cta.shared::cta.b32 %0, [%1];" : "=r"(s_tile_w) : "r"(static_cast<uint32_t>(smem_issue_addr)) : "memory");
                        s_tile_c = (int)s_tile_w;
                    }
                    int v0 = v_tile_v % 9;
                    int v1 = (v_tile_v + 1) % 9;
                    int _parity_kv_full_2 = (int)((unsigned int)v_tile_v / 9 % 2);
                    mbarrier_wait(kv_full_addr + (v0) * 8, _parity_kv_full_2);
                    int _parity_kv_full_3 = (int)((unsigned int)(v_tile_v + 1) / 9 % 2);
                    mbarrier_wait(kv_full_addr + (v1) * 8, _parity_kv_full_3);
                    mbarrier_wait(p_full_addr + (pf_stage_v) * 8, _phase_p_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v0) * 1024);
                    int _mma_b_lo_0_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2 + 0) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135364624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0_1), "r"(_mma_b_lo_0_1), "r"(tmem_tmem_o), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135364624;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0_1), "r"(_mma_b_lo_0_1), "r"((tmem_tmem_o + (64))), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    int _mma_a_lo_1_1 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v1) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000) + (pf_stage_v * 2 + 1) * 512);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135364624;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_o + (128))), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 135364624;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 256;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_o + (192))), "r"(((first_pv_v) ? 0 : 1)) : "memory");
                    elect_commit(pv_done_addr + ((int)((unsigned int)gpair_v % 8)) * 8);
                    elect_commit(kv_empty_addr + (v0) * 8);
                    elect_commit(kv_empty_addr + (v1) * 8);
                    pf_stage_v += 1;
                    if (pf_stage_v == 4) { pf_stage_v = 0; _phase_p_full ^= 1; }
                    first_pv_v = 0;
                    if (npairs_v > pv_pair + 3) {
                        v_tile_v += 4;
                    } else {
                        v_tile_v += 2;
                    }
                    gpair_v += 1;
                    if (elect_sync()) {
                        asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 8)), "r"(static_cast<uint32_t>((unsigned int)gpair_v)) : "memory");
                    }
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                have_prev_v = 1;
                if (elect_sync()) {
                    asm volatile("st.release.cta.shared::cta.b32 [%0], %1;" :: "r"(static_cast<uint32_t>(smem_issue_addr + 4)), "r"(static_cast<uint32_t>((unsigned int)(v_tile_v + 2))) : "memory");
                }
                item_ord_v += 1;
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 11) {
        { // idle_main
            unsigned int _phase_tmem_dead_0 = 0;
            mbarrier_wait(tmem_dead_addr, _phase_tmem_dead_0);
            _phase_tmem_dead_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }

    // Cleanup
}

} // extern "C"
