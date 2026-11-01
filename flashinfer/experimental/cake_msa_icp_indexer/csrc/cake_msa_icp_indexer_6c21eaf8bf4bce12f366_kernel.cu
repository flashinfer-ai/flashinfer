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
#define TMEM_NCOLS 384
#define TMEM_ACC_OFFSET 0
#define TMEM_TQ_0_OFFSET 256
#define TMEM_TQ_1_OFFSET 288
#define TMEM_TQ_2_OFFSET 320
#define TMEM_TQ_3_OFFSET 352
#define NUM_Q_PIPE_STAGES 2
#define NUM_K_PIPE_STAGES 8
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 66560
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_STAGE_OFF 197632
#define SMEM_STAGE_STAGE_BYTES 33792
#define SMEM_STAGE_STRIDE 33792
#define SMEM_TOTAL 231424
#define THREADS 320

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


__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}




__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_msa_icp_indexer_6c21eaf8bf4bce12f366(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, int* __restrict__ block_table, float* __restrict__ max_score, int* __restrict__ qo_lens, int* __restrict__ kv_lens, int* __restrict__ qo_indptr, int* __restrict__ qo_offset, long long* __restrict__ work_range, long long* __restrict__ work_info, int rank, int compact, int bt_stride, long long score_stride_t, long long score_stride_h, int score_cols)
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
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 32)
    #define k_empty_addr (mbar_base + 96)
    #define acc_full_addr (mbar_base + 160)
    #define acc_empty_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;
    float* stage = reinterpret_cast<float*>(smem_raw + 197632);
    const int stage_addr = smem + 197632;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 8 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // k_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tq_0 = taddr + 256;
    const int tq_1 = taddr + 288;
    const int tq_2 = taddr + 320;
    const int tq_3 = taddr + 352;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epi ----
    if (warp <= 7) {
        { // epi_main
            long long wr = work_range[blockIdx.x];
            int j0 = (int)(wr & 65535);
            int j1 = (int)(wr >> 16 & 65535);
            int c0 = (int)(wr >> 32 & 65535);
            int c1 = (int)(wr >> 48 & 65535);
            int j_end = ((c1 > 0) ? j1 + 1 : j1);
            int wg = warp / 4;
            int quarter = warp % 4;
            int lane_1 = lane;
            int lrow = quarter * 32 + lane_1;
            int head = lrow % 4;
            unsigned int lane_base = (unsigned int)(quarter * 32) << 16;
            int own = rank * 2 / 4;
            int wcols = 4;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int j = j0; j < j_end; j++) {
                long long wi = work_info[j];
                long long prev = 0;
                if (j > 0) {
                    prev = work_info[j - 1] >> 32;
                }
                int b = (int)(wi & 65535);
                int it = (int)(wi >> 16 & 65535);
                int ce = (int)((wi >> 32) - prev);
                if (j == j1) {
                    ce = c1;
                }
                int cb = ((j == j0) ? c0 : 0);
                int kt0 = cb / 4;
                int kt1 = (ce + 4 - 1) / 4;
                if (kt1 > kt0) {
                    int q_len = qo_lens[b];
                    int kv_len = kv_lens[b];
                    int q_off = qo_offset[b];
                    int t_base = qo_indptr[b];
                    int c_first = kt0 * wcols;
                    int use_stage = 0;
                    if ((kt1 - kt0) * wcols >= 64) {
                        use_stage = 1;
                    }
                    int windows[2];
                    int r_0 = it * 64 + lrow / 4;
                    int u_0 = 0;
                    if (r_0 < q_len) {
                        int _min_0 = ((kv_len) < (q_off + r_0 + 1) ? (kv_len) : (q_off + r_0 + 1));
                        u_0 = _min_0;
                    }
                    int cpre_0 = (((u_0 + 127) / 128) < (score_cols) ? ((u_0 + 127) / 128) : (score_cols));
                    windows[0] = c_first - c_first % 32;
                    long long off_0 = (long long)(t_base + r_0) * score_stride_t + (long long)head * score_stride_h;
                    int r_1 = it * 64 + 32 + lrow / 4;
                    int u_1 = 0;
                    if (r_1 < q_len) {
                        int _min_1 = ((kv_len) < (q_off + r_1 + 1) ? (kv_len) : (q_off + r_1 + 1));
                        u_1 = _min_1;
                    }
                    int cpre_1 = (((u_1 + 127) / 128) < (score_cols) ? ((u_1 + 127) / 128) : (score_cols));
                    windows[1] = c_first - c_first % 32;
                    long long off_1 = (long long)(t_base + r_1) * score_stride_t + (long long)head * score_stride_h;
                    if (use_stage == 1) {
                        #pragma unroll 1
                        for (int kt_0 = kt0; kt_0 < kt1; kt_0++) {
                            int mine_0 = 1;
                            mine_0 = ((0 == wg) ? 1 : 0);
                            if (mine_0 != 0) {
                                mbarrier_wait(acc_full_addr + (wg) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float v_0[128];
                                unsigned int tile_addr_0 = taddr + lane_base + (unsigned int)wg * 128;
                                float pm_0 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_0[0]), "=f"(v_0[1]), "=f"(v_0[2]), "=f"(v_0[3]), "=f"(v_0[4]), "=f"(v_0[5]), "=f"(v_0[6]), "=f"(v_0[7]), "=f"(v_0[8]), "=f"(v_0[9]), "=f"(v_0[10]), "=f"(v_0[11]), "=f"(v_0[12]), "=f"(v_0[13]), "=f"(v_0[14]), "=f"(v_0[15]), "=f"(v_0[16]), "=f"(v_0[17]), "=f"(v_0[18]), "=f"(v_0[19]), "=f"(v_0[20]), "=f"(v_0[21]), "=f"(v_0[22]), "=f"(v_0[23]), "=f"(v_0[24]), "=f"(v_0[25]), "=f"(v_0[26]), "=f"(v_0[27]), "=f"(v_0[28]), "=f"(v_0[29]), "=f"(v_0[30]), "=f"(v_0[31]), "=f"(pm_0)
                                    : "r"(tile_addr_0 + 0));
                                float pm_1 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_0[32]), "=f"(v_0[33]), "=f"(v_0[34]), "=f"(v_0[35]), "=f"(v_0[36]), "=f"(v_0[37]), "=f"(v_0[38]), "=f"(v_0[39]), "=f"(v_0[40]), "=f"(v_0[41]), "=f"(v_0[42]), "=f"(v_0[43]), "=f"(v_0[44]), "=f"(v_0[45]), "=f"(v_0[46]), "=f"(v_0[47]), "=f"(v_0[48]), "=f"(v_0[49]), "=f"(v_0[50]), "=f"(v_0[51]), "=f"(v_0[52]), "=f"(v_0[53]), "=f"(v_0[54]), "=f"(v_0[55]), "=f"(v_0[56]), "=f"(v_0[57]), "=f"(v_0[58]), "=f"(v_0[59]), "=f"(v_0[60]), "=f"(v_0[61]), "=f"(v_0[62]), "=f"(v_0[63]), "=f"(pm_1)
                                    : "r"(tile_addr_0 + 32));
                                float pm_2 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_0[64]), "=f"(v_0[65]), "=f"(v_0[66]), "=f"(v_0[67]), "=f"(v_0[68]), "=f"(v_0[69]), "=f"(v_0[70]), "=f"(v_0[71]), "=f"(v_0[72]), "=f"(v_0[73]), "=f"(v_0[74]), "=f"(v_0[75]), "=f"(v_0[76]), "=f"(v_0[77]), "=f"(v_0[78]), "=f"(v_0[79]), "=f"(v_0[80]), "=f"(v_0[81]), "=f"(v_0[82]), "=f"(v_0[83]), "=f"(v_0[84]), "=f"(v_0[85]), "=f"(v_0[86]), "=f"(v_0[87]), "=f"(v_0[88]), "=f"(v_0[89]), "=f"(v_0[90]), "=f"(v_0[91]), "=f"(v_0[92]), "=f"(v_0[93]), "=f"(v_0[94]), "=f"(v_0[95]), "=f"(pm_2)
                                    : "r"(tile_addr_0 + 64));
                                float pm_3 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_0[96]), "=f"(v_0[97]), "=f"(v_0[98]), "=f"(v_0[99]), "=f"(v_0[100]), "=f"(v_0[101]), "=f"(v_0[102]), "=f"(v_0[103]), "=f"(v_0[104]), "=f"(v_0[105]), "=f"(v_0[106]), "=f"(v_0[107]), "=f"(v_0[108]), "=f"(v_0[109]), "=f"(v_0[110]), "=f"(v_0[111]), "=f"(v_0[112]), "=f"(v_0[113]), "=f"(v_0[114]), "=f"(v_0[115]), "=f"(v_0[116]), "=f"(v_0[117]), "=f"(v_0[118]), "=f"(v_0[119]), "=f"(v_0[120]), "=f"(v_0[121]), "=f"(v_0[122]), "=f"(v_0[123]), "=f"(v_0[124]), "=f"(v_0[125]), "=f"(v_0[126]), "=f"(v_0[127]), "=f"(pm_3)
                                    : "r"(tile_addr_0 + 96));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + (wg) * 8);
                                _phase_acc_full ^= 1;
                                int row_stage_0 = (quarter * 32 + lane_1) * 33 - windows[0];
                                int lim_0 = cpre_0;
                                int p_0 = kt_0 * 4;
                                int nvis_0 = (((u_0 - 128 * p_0 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_0 - rank + 4 - 1) / 4) : (0));
                                float m_0 = -CUDART_INF_F;
                                int masked_0 = 1;
                                m_0 = pm_0;
                                int _warp_vote_2 = __any_sync(0xFFFFFFFF, nvis_0 < 32);
                                masked_0 = _warp_vote_2;
                                if (masked_0 != 0) {
                                    m_0 = -CUDART_INF_F;
                                    int _warp_vote_3 = __any_sync(0xFFFFFFFF, nvis_0 < 32);
                                    if (_warp_vote_3 != 0) {
                                        uint32_t _thresh_mask_4;
                                        {
                                            int _lim_5 = nvis_0;
                                            if (_lim_5 <= 0) { _thresh_mask_4 = 0u; }
                                            else if (_lim_5 >= 32) { _thresh_mask_4 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_4) : "r"(_lim_5));
                                            }
                                        }
                                        if (!(_thresh_mask_4 & (1u << 0))) v_0[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 1))) v_0[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 2))) v_0[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 3))) v_0[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 4))) v_0[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 5))) v_0[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 6))) v_0[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 7))) v_0[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 8))) v_0[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 9))) v_0[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 10))) v_0[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 11))) v_0[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 12))) v_0[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 13))) v_0[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 14))) v_0[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 15))) v_0[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 16))) v_0[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 17))) v_0[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 18))) v_0[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 19))) v_0[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 20))) v_0[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 21))) v_0[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 22))) v_0[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 23))) v_0[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 24))) v_0[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 25))) v_0[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 26))) v_0[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 27))) v_0[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 28))) v_0[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 29))) v_0[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 30))) v_0[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_4 & (1u << 31))) v_0[31] = -CUDART_INF_F;
                                    }
                                    float part_0 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_7 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[0], v_0[1]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[2], v_0[3]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[4], v_0[5]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[6], v_0[7]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[8], v_0[9]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[10], v_0[11]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[12], v_0[13]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[14], v_0[15]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[16], v_0[17]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[18], v_0[19]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[20], v_0[21]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[22], v_0[23]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[24], v_0[25]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[26], v_0[27]));
                                    _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(v_0[28], v_0[29]));
                                    _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(v_0[30], v_0[31]));
                                    float _reg_array_reduce_6 = row_max_reduce(_reg_reduce_max2_7);
                                    part_0 = _reg_array_reduce_6;
                                    m_0 = fmaxf(m_0, part_0);
                                }
                                m_0 = ((m_0 == 0.0f) ? 0.0f : m_0);
                                stage[row_stage_0 + p_0] = m_0;
                                int p_1 = kt_0 * 4 + 1;
                                int nvis_1 = (((u_0 - 128 * p_1 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_1 - rank + 4 - 1) / 4) : (0));
                                float m_1 = -CUDART_INF_F;
                                int masked_1 = 1;
                                m_1 = pm_1;
                                int _warp_vote_8 = __any_sync(0xFFFFFFFF, nvis_1 < 32);
                                masked_1 = _warp_vote_8;
                                if (masked_1 != 0) {
                                    m_1 = -CUDART_INF_F;
                                    int _warp_vote_9 = __any_sync(0xFFFFFFFF, nvis_1 < 32);
                                    if (_warp_vote_9 != 0) {
                                        uint32_t _thresh_mask_10;
                                        {
                                            int _lim_11 = nvis_1;
                                            if (_lim_11 <= 0) { _thresh_mask_10 = 0u; }
                                            else if (_lim_11 >= 32) { _thresh_mask_10 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_10) : "r"(_lim_11));
                                            }
                                        }
                                        if (!(_thresh_mask_10 & (1u << 0))) v_0[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 1))) v_0[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 2))) v_0[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 3))) v_0[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 4))) v_0[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 5))) v_0[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 6))) v_0[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 7))) v_0[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 8))) v_0[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 9))) v_0[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 10))) v_0[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 11))) v_0[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 12))) v_0[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 13))) v_0[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 14))) v_0[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 15))) v_0[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 16))) v_0[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 17))) v_0[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 18))) v_0[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 19))) v_0[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 20))) v_0[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 21))) v_0[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 22))) v_0[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 23))) v_0[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 24))) v_0[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 25))) v_0[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 26))) v_0[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 27))) v_0[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 28))) v_0[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 29))) v_0[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 30))) v_0[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_10 & (1u << 31))) v_0[63] = -CUDART_INF_F;
                                    }
                                    float part_1 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_13 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(32) + 0], _reg_reduce_max2_13);
                                    float _reg_array_reduce_12 = row_max_reduce(_reg_reduce_max2_13);
                                    part_1 = _reg_array_reduce_12;
                                    m_1 = fmaxf(m_1, part_1);
                                }
                                m_1 = ((m_1 == 0.0f) ? 0.0f : m_1);
                                stage[row_stage_0 + p_1] = m_1;
                                int p_2 = kt_0 * 4 + 2;
                                int nvis_2 = (((u_0 - 128 * p_2 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_2 - rank + 4 - 1) / 4) : (0));
                                float m_2 = -CUDART_INF_F;
                                int masked_2 = 1;
                                m_2 = pm_2;
                                int _warp_vote_14 = __any_sync(0xFFFFFFFF, nvis_2 < 32);
                                masked_2 = _warp_vote_14;
                                if (masked_2 != 0) {
                                    m_2 = -CUDART_INF_F;
                                    int _warp_vote_15 = __any_sync(0xFFFFFFFF, nvis_2 < 32);
                                    if (_warp_vote_15 != 0) {
                                        uint32_t _thresh_mask_16;
                                        {
                                            int _lim_17 = nvis_2;
                                            if (_lim_17 <= 0) { _thresh_mask_16 = 0u; }
                                            else if (_lim_17 >= 32) { _thresh_mask_16 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_16) : "r"(_lim_17));
                                            }
                                        }
                                        if (!(_thresh_mask_16 & (1u << 0))) v_0[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 1))) v_0[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 2))) v_0[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 3))) v_0[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 4))) v_0[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 5))) v_0[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 6))) v_0[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 7))) v_0[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 8))) v_0[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 9))) v_0[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 10))) v_0[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 11))) v_0[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 12))) v_0[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 13))) v_0[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 14))) v_0[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 15))) v_0[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 16))) v_0[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 17))) v_0[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 18))) v_0[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 19))) v_0[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 20))) v_0[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 21))) v_0[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 22))) v_0[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 23))) v_0[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 24))) v_0[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 25))) v_0[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 26))) v_0[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 27))) v_0[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 28))) v_0[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 29))) v_0[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 30))) v_0[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_16 & (1u << 31))) v_0[95] = -CUDART_INF_F;
                                    }
                                    float part_2 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_19 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(64) + 0], _reg_reduce_max2_19);
                                    float _reg_array_reduce_18 = row_max_reduce(_reg_reduce_max2_19);
                                    part_2 = _reg_array_reduce_18;
                                    m_2 = fmaxf(m_2, part_2);
                                }
                                m_2 = ((m_2 == 0.0f) ? 0.0f : m_2);
                                stage[row_stage_0 + p_2] = m_2;
                                int p_3 = kt_0 * 4 + 3;
                                int nvis_3 = (((u_0 - 128 * p_3 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_3 - rank + 4 - 1) / 4) : (0));
                                float m_3 = -CUDART_INF_F;
                                int masked_3 = 1;
                                m_3 = pm_3;
                                int _warp_vote_20 = __any_sync(0xFFFFFFFF, nvis_3 < 32);
                                masked_3 = _warp_vote_20;
                                if (masked_3 != 0) {
                                    m_3 = -CUDART_INF_F;
                                    int _warp_vote_21 = __any_sync(0xFFFFFFFF, nvis_3 < 32);
                                    if (_warp_vote_21 != 0) {
                                        uint32_t _thresh_mask_22;
                                        {
                                            int _lim_23 = nvis_3;
                                            if (_lim_23 <= 0) { _thresh_mask_22 = 0u; }
                                            else if (_lim_23 >= 32) { _thresh_mask_22 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_22) : "r"(_lim_23));
                                            }
                                        }
                                        if (!(_thresh_mask_22 & (1u << 0))) v_0[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 1))) v_0[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 2))) v_0[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 3))) v_0[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 4))) v_0[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 5))) v_0[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 6))) v_0[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 7))) v_0[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 8))) v_0[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 9))) v_0[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 10))) v_0[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 11))) v_0[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 12))) v_0[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 13))) v_0[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 14))) v_0[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 15))) v_0[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 16))) v_0[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 17))) v_0[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 18))) v_0[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 19))) v_0[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 20))) v_0[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 21))) v_0[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 22))) v_0[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 23))) v_0[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 24))) v_0[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 25))) v_0[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 26))) v_0[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 27))) v_0[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 28))) v_0[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 29))) v_0[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 30))) v_0[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_22 & (1u << 31))) v_0[127] = -CUDART_INF_F;
                                    }
                                    float part_3 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_25 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_0[(96) + 0], _reg_reduce_max2_25);
                                    float _reg_array_reduce_24 = row_max_reduce(_reg_reduce_max2_25);
                                    part_3 = _reg_array_reduce_24;
                                    m_3 = fmaxf(m_3, part_3);
                                }
                                m_3 = ((m_3 == 0.0f) ? 0.0f : m_3);
                                stage[row_stage_0 + p_3] = m_3;
                                int c_end_0 = (kt_0 + 1) * wcols;
                                if (c_end_0 % 32 == 0 || kt_0 == kt1 - 1) {
                                    __syncwarp();
                                    int col_0 = windows[0] + lane_1;
                                    int in_slice_0 = ((col_0 >= c_first && col_0 < c_end_0) ? 1 : 0);
                                    #pragma unroll 4
                                    for (int rr_0 = 0; rr_0 < 32; rr_0++) {
                                        int lim_1 = __shfl_sync(0xFFFFFFFF, cpre_0, rr_0);
                                        float val_0 = stage[(quarter * 32 + rr_0) * 33 + lane_1];
                                        if (in_slice_0 != 0 && col_0 < lim_1) {
                                            int srow_0 = quarter * 32 + rr_0;
                                            int t_0 = t_base + it * 64 + srow_0 / 4;
                                            long long off_2 = (long long)t_0 * score_stride_t + (long long)(srow_0 % 4) * score_stride_h + (long long)col_0;
                                            max_score[off_2] = val_0;
                                        }
                                    }
                                    __syncwarp();
                                    windows[0] = c_end_0;
                                }
                            }
                            int mine_1 = 1;
                            mine_1 = ((1 == wg) ? 1 : 0);
                            if (mine_1 != 0) {
                                mbarrier_wait(acc_full_addr + (wg) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float v_1[128];
                                unsigned int tile_addr_1 = taddr + lane_base + (unsigned int)wg * 128;
                                float pm_4 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_1[0]), "=f"(v_1[1]), "=f"(v_1[2]), "=f"(v_1[3]), "=f"(v_1[4]), "=f"(v_1[5]), "=f"(v_1[6]), "=f"(v_1[7]), "=f"(v_1[8]), "=f"(v_1[9]), "=f"(v_1[10]), "=f"(v_1[11]), "=f"(v_1[12]), "=f"(v_1[13]), "=f"(v_1[14]), "=f"(v_1[15]), "=f"(v_1[16]), "=f"(v_1[17]), "=f"(v_1[18]), "=f"(v_1[19]), "=f"(v_1[20]), "=f"(v_1[21]), "=f"(v_1[22]), "=f"(v_1[23]), "=f"(v_1[24]), "=f"(v_1[25]), "=f"(v_1[26]), "=f"(v_1[27]), "=f"(v_1[28]), "=f"(v_1[29]), "=f"(v_1[30]), "=f"(v_1[31]), "=f"(pm_4)
                                    : "r"(tile_addr_1 + 0));
                                float pm_5 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_1[32]), "=f"(v_1[33]), "=f"(v_1[34]), "=f"(v_1[35]), "=f"(v_1[36]), "=f"(v_1[37]), "=f"(v_1[38]), "=f"(v_1[39]), "=f"(v_1[40]), "=f"(v_1[41]), "=f"(v_1[42]), "=f"(v_1[43]), "=f"(v_1[44]), "=f"(v_1[45]), "=f"(v_1[46]), "=f"(v_1[47]), "=f"(v_1[48]), "=f"(v_1[49]), "=f"(v_1[50]), "=f"(v_1[51]), "=f"(v_1[52]), "=f"(v_1[53]), "=f"(v_1[54]), "=f"(v_1[55]), "=f"(v_1[56]), "=f"(v_1[57]), "=f"(v_1[58]), "=f"(v_1[59]), "=f"(v_1[60]), "=f"(v_1[61]), "=f"(v_1[62]), "=f"(v_1[63]), "=f"(pm_5)
                                    : "r"(tile_addr_1 + 32));
                                float pm_6 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_1[64]), "=f"(v_1[65]), "=f"(v_1[66]), "=f"(v_1[67]), "=f"(v_1[68]), "=f"(v_1[69]), "=f"(v_1[70]), "=f"(v_1[71]), "=f"(v_1[72]), "=f"(v_1[73]), "=f"(v_1[74]), "=f"(v_1[75]), "=f"(v_1[76]), "=f"(v_1[77]), "=f"(v_1[78]), "=f"(v_1[79]), "=f"(v_1[80]), "=f"(v_1[81]), "=f"(v_1[82]), "=f"(v_1[83]), "=f"(v_1[84]), "=f"(v_1[85]), "=f"(v_1[86]), "=f"(v_1[87]), "=f"(v_1[88]), "=f"(v_1[89]), "=f"(v_1[90]), "=f"(v_1[91]), "=f"(v_1[92]), "=f"(v_1[93]), "=f"(v_1[94]), "=f"(v_1[95]), "=f"(pm_6)
                                    : "r"(tile_addr_1 + 64));
                                float pm_7 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_1[96]), "=f"(v_1[97]), "=f"(v_1[98]), "=f"(v_1[99]), "=f"(v_1[100]), "=f"(v_1[101]), "=f"(v_1[102]), "=f"(v_1[103]), "=f"(v_1[104]), "=f"(v_1[105]), "=f"(v_1[106]), "=f"(v_1[107]), "=f"(v_1[108]), "=f"(v_1[109]), "=f"(v_1[110]), "=f"(v_1[111]), "=f"(v_1[112]), "=f"(v_1[113]), "=f"(v_1[114]), "=f"(v_1[115]), "=f"(v_1[116]), "=f"(v_1[117]), "=f"(v_1[118]), "=f"(v_1[119]), "=f"(v_1[120]), "=f"(v_1[121]), "=f"(v_1[122]), "=f"(v_1[123]), "=f"(v_1[124]), "=f"(v_1[125]), "=f"(v_1[126]), "=f"(v_1[127]), "=f"(pm_7)
                                    : "r"(tile_addr_1 + 96));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + (wg) * 8);
                                _phase_acc_full ^= 1;
                                int row_stage_1 = ((4 + quarter) * 32 + lane_1) * 33 - windows[1];
                                int lim_2 = cpre_1;
                                int p_4 = kt_0 * 4;
                                int nvis_4 = (((u_1 - 128 * p_4 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_4 - rank + 4 - 1) / 4) : (0));
                                float m_4 = -CUDART_INF_F;
                                int masked_4 = 1;
                                m_4 = pm_4;
                                int _warp_vote_26 = __any_sync(0xFFFFFFFF, nvis_4 < 32);
                                masked_4 = _warp_vote_26;
                                if (masked_4 != 0) {
                                    m_4 = -CUDART_INF_F;
                                    int _warp_vote_27 = __any_sync(0xFFFFFFFF, nvis_4 < 32);
                                    if (_warp_vote_27 != 0) {
                                        uint32_t _thresh_mask_28;
                                        {
                                            int _lim_29 = nvis_4;
                                            if (_lim_29 <= 0) { _thresh_mask_28 = 0u; }
                                            else if (_lim_29 >= 32) { _thresh_mask_28 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_28) : "r"(_lim_29));
                                            }
                                        }
                                        if (!(_thresh_mask_28 & (1u << 0))) v_1[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 1))) v_1[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 2))) v_1[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 3))) v_1[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 4))) v_1[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 5))) v_1[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 6))) v_1[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 7))) v_1[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 8))) v_1[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 9))) v_1[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 10))) v_1[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 11))) v_1[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 12))) v_1[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 13))) v_1[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 14))) v_1[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 15))) v_1[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 16))) v_1[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 17))) v_1[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 18))) v_1[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 19))) v_1[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 20))) v_1[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 21))) v_1[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 22))) v_1[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 23))) v_1[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 24))) v_1[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 25))) v_1[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 26))) v_1[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 27))) v_1[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 28))) v_1[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 29))) v_1[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 30))) v_1[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_28 & (1u << 31))) v_1[31] = -CUDART_INF_F;
                                    }
                                    float part_4 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_31 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[0], v_1[1]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[2], v_1[3]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[4], v_1[5]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[6], v_1[7]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[8], v_1[9]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[10], v_1[11]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[12], v_1[13]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[14], v_1[15]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[16], v_1[17]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[18], v_1[19]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[20], v_1[21]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[22], v_1[23]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[24], v_1[25]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[26], v_1[27]));
                                    _reg_reduce_max2_31.x = max_noftz(_reg_reduce_max2_31.x, max_noftz(v_1[28], v_1[29]));
                                    _reg_reduce_max2_31.y = max_noftz(_reg_reduce_max2_31.y, max_noftz(v_1[30], v_1[31]));
                                    float _reg_array_reduce_30 = row_max_reduce(_reg_reduce_max2_31);
                                    part_4 = _reg_array_reduce_30;
                                    m_4 = fmaxf(m_4, part_4);
                                }
                                m_4 = ((m_4 == 0.0f) ? 0.0f : m_4);
                                stage[row_stage_1 + p_4] = m_4;
                                int p_5 = kt_0 * 4 + 1;
                                int nvis_5 = (((u_1 - 128 * p_5 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_5 - rank + 4 - 1) / 4) : (0));
                                float m_5 = -CUDART_INF_F;
                                int masked_5 = 1;
                                m_5 = pm_5;
                                int _warp_vote_32 = __any_sync(0xFFFFFFFF, nvis_5 < 32);
                                masked_5 = _warp_vote_32;
                                if (masked_5 != 0) {
                                    m_5 = -CUDART_INF_F;
                                    int _warp_vote_33 = __any_sync(0xFFFFFFFF, nvis_5 < 32);
                                    if (_warp_vote_33 != 0) {
                                        uint32_t _thresh_mask_34;
                                        {
                                            int _lim_35 = nvis_5;
                                            if (_lim_35 <= 0) { _thresh_mask_34 = 0u; }
                                            else if (_lim_35 >= 32) { _thresh_mask_34 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_34) : "r"(_lim_35));
                                            }
                                        }
                                        if (!(_thresh_mask_34 & (1u << 0))) v_1[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 1))) v_1[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 2))) v_1[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 3))) v_1[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 4))) v_1[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 5))) v_1[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 6))) v_1[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 7))) v_1[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 8))) v_1[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 9))) v_1[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 10))) v_1[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 11))) v_1[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 12))) v_1[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 13))) v_1[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 14))) v_1[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 15))) v_1[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 16))) v_1[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 17))) v_1[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 18))) v_1[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 19))) v_1[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 20))) v_1[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 21))) v_1[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 22))) v_1[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 23))) v_1[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 24))) v_1[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 25))) v_1[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 26))) v_1[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 27))) v_1[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 28))) v_1[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 29))) v_1[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 30))) v_1[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_34 & (1u << 31))) v_1[63] = -CUDART_INF_F;
                                    }
                                    float part_5 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_37 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(32) + 0], _reg_reduce_max2_37);
                                    float _reg_array_reduce_36 = row_max_reduce(_reg_reduce_max2_37);
                                    part_5 = _reg_array_reduce_36;
                                    m_5 = fmaxf(m_5, part_5);
                                }
                                m_5 = ((m_5 == 0.0f) ? 0.0f : m_5);
                                stage[row_stage_1 + p_5] = m_5;
                                int p_6 = kt_0 * 4 + 2;
                                int nvis_6 = (((u_1 - 128 * p_6 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_6 - rank + 4 - 1) / 4) : (0));
                                float m_6 = -CUDART_INF_F;
                                int masked_6 = 1;
                                m_6 = pm_6;
                                int _warp_vote_38 = __any_sync(0xFFFFFFFF, nvis_6 < 32);
                                masked_6 = _warp_vote_38;
                                if (masked_6 != 0) {
                                    m_6 = -CUDART_INF_F;
                                    int _warp_vote_39 = __any_sync(0xFFFFFFFF, nvis_6 < 32);
                                    if (_warp_vote_39 != 0) {
                                        uint32_t _thresh_mask_40;
                                        {
                                            int _lim_41 = nvis_6;
                                            if (_lim_41 <= 0) { _thresh_mask_40 = 0u; }
                                            else if (_lim_41 >= 32) { _thresh_mask_40 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_40) : "r"(_lim_41));
                                            }
                                        }
                                        if (!(_thresh_mask_40 & (1u << 0))) v_1[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 1))) v_1[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 2))) v_1[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 3))) v_1[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 4))) v_1[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 5))) v_1[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 6))) v_1[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 7))) v_1[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 8))) v_1[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 9))) v_1[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 10))) v_1[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 11))) v_1[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 12))) v_1[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 13))) v_1[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 14))) v_1[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 15))) v_1[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 16))) v_1[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 17))) v_1[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 18))) v_1[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 19))) v_1[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 20))) v_1[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 21))) v_1[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 22))) v_1[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 23))) v_1[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 24))) v_1[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 25))) v_1[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 26))) v_1[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 27))) v_1[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 28))) v_1[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 29))) v_1[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 30))) v_1[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_40 & (1u << 31))) v_1[95] = -CUDART_INF_F;
                                    }
                                    float part_6 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_43 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(64) + 0], _reg_reduce_max2_43);
                                    float _reg_array_reduce_42 = row_max_reduce(_reg_reduce_max2_43);
                                    part_6 = _reg_array_reduce_42;
                                    m_6 = fmaxf(m_6, part_6);
                                }
                                m_6 = ((m_6 == 0.0f) ? 0.0f : m_6);
                                stage[row_stage_1 + p_6] = m_6;
                                int p_7 = kt_0 * 4 + 3;
                                int nvis_7 = (((u_1 - 128 * p_7 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_7 - rank + 4 - 1) / 4) : (0));
                                float m_7 = -CUDART_INF_F;
                                int masked_7 = 1;
                                m_7 = pm_7;
                                int _warp_vote_44 = __any_sync(0xFFFFFFFF, nvis_7 < 32);
                                masked_7 = _warp_vote_44;
                                if (masked_7 != 0) {
                                    m_7 = -CUDART_INF_F;
                                    int _warp_vote_45 = __any_sync(0xFFFFFFFF, nvis_7 < 32);
                                    if (_warp_vote_45 != 0) {
                                        uint32_t _thresh_mask_46;
                                        {
                                            int _lim_47 = nvis_7;
                                            if (_lim_47 <= 0) { _thresh_mask_46 = 0u; }
                                            else if (_lim_47 >= 32) { _thresh_mask_46 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_46) : "r"(_lim_47));
                                            }
                                        }
                                        if (!(_thresh_mask_46 & (1u << 0))) v_1[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 1))) v_1[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 2))) v_1[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 3))) v_1[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 4))) v_1[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 5))) v_1[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 6))) v_1[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 7))) v_1[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 8))) v_1[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 9))) v_1[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 10))) v_1[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 11))) v_1[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 12))) v_1[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 13))) v_1[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 14))) v_1[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 15))) v_1[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 16))) v_1[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 17))) v_1[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 18))) v_1[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 19))) v_1[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 20))) v_1[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 21))) v_1[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 22))) v_1[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 23))) v_1[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 24))) v_1[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 25))) v_1[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 26))) v_1[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 27))) v_1[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 28))) v_1[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 29))) v_1[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 30))) v_1[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_46 & (1u << 31))) v_1[127] = -CUDART_INF_F;
                                    }
                                    float part_7 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_49 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_1[(96) + 0], _reg_reduce_max2_49);
                                    float _reg_array_reduce_48 = row_max_reduce(_reg_reduce_max2_49);
                                    part_7 = _reg_array_reduce_48;
                                    m_7 = fmaxf(m_7, part_7);
                                }
                                m_7 = ((m_7 == 0.0f) ? 0.0f : m_7);
                                stage[row_stage_1 + p_7] = m_7;
                                int c_end_1 = (kt_0 + 1) * wcols;
                                if (c_end_1 % 32 == 0 || kt_0 == kt1 - 1) {
                                    __syncwarp();
                                    int col_1 = windows[1] + lane_1;
                                    int in_slice_1 = ((col_1 >= c_first && col_1 < c_end_1) ? 1 : 0);
                                    #pragma unroll 4
                                    for (int rr_1 = 0; rr_1 < 32; rr_1++) {
                                        int lim_3 = __shfl_sync(0xFFFFFFFF, cpre_1, rr_1);
                                        float val_1 = stage[((4 + quarter) * 32 + rr_1) * 33 + lane_1];
                                        if (in_slice_1 != 0 && col_1 < lim_3) {
                                            int srow_1 = quarter * 32 + rr_1;
                                            int t_1 = t_base + it * 64 + 32 + srow_1 / 4;
                                            long long off_3 = (long long)t_1 * score_stride_t + (long long)(srow_1 % 4) * score_stride_h + (long long)col_1;
                                            max_score[off_3] = val_1;
                                        }
                                    }
                                    __syncwarp();
                                    windows[1] = c_end_1;
                                }
                            }
                        }
                    }
                    if (use_stage == 0) {
                        #pragma unroll 1
                        for (int kt_1 = kt0; kt_1 < kt1; kt_1++) {
                            int mine_2 = 1;
                            mine_2 = ((0 == wg) ? 1 : 0);
                            if (mine_2 != 0) {
                                mbarrier_wait(acc_full_addr + (wg) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float v_2[128];
                                unsigned int tile_addr_2 = taddr + lane_base + (unsigned int)wg * 128;
                                float pm_8 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_2[0]), "=f"(v_2[1]), "=f"(v_2[2]), "=f"(v_2[3]), "=f"(v_2[4]), "=f"(v_2[5]), "=f"(v_2[6]), "=f"(v_2[7]), "=f"(v_2[8]), "=f"(v_2[9]), "=f"(v_2[10]), "=f"(v_2[11]), "=f"(v_2[12]), "=f"(v_2[13]), "=f"(v_2[14]), "=f"(v_2[15]), "=f"(v_2[16]), "=f"(v_2[17]), "=f"(v_2[18]), "=f"(v_2[19]), "=f"(v_2[20]), "=f"(v_2[21]), "=f"(v_2[22]), "=f"(v_2[23]), "=f"(v_2[24]), "=f"(v_2[25]), "=f"(v_2[26]), "=f"(v_2[27]), "=f"(v_2[28]), "=f"(v_2[29]), "=f"(v_2[30]), "=f"(v_2[31]), "=f"(pm_8)
                                    : "r"(tile_addr_2 + 0));
                                float pm_9 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_2[32]), "=f"(v_2[33]), "=f"(v_2[34]), "=f"(v_2[35]), "=f"(v_2[36]), "=f"(v_2[37]), "=f"(v_2[38]), "=f"(v_2[39]), "=f"(v_2[40]), "=f"(v_2[41]), "=f"(v_2[42]), "=f"(v_2[43]), "=f"(v_2[44]), "=f"(v_2[45]), "=f"(v_2[46]), "=f"(v_2[47]), "=f"(v_2[48]), "=f"(v_2[49]), "=f"(v_2[50]), "=f"(v_2[51]), "=f"(v_2[52]), "=f"(v_2[53]), "=f"(v_2[54]), "=f"(v_2[55]), "=f"(v_2[56]), "=f"(v_2[57]), "=f"(v_2[58]), "=f"(v_2[59]), "=f"(v_2[60]), "=f"(v_2[61]), "=f"(v_2[62]), "=f"(v_2[63]), "=f"(pm_9)
                                    : "r"(tile_addr_2 + 32));
                                float pm_10 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_2[64]), "=f"(v_2[65]), "=f"(v_2[66]), "=f"(v_2[67]), "=f"(v_2[68]), "=f"(v_2[69]), "=f"(v_2[70]), "=f"(v_2[71]), "=f"(v_2[72]), "=f"(v_2[73]), "=f"(v_2[74]), "=f"(v_2[75]), "=f"(v_2[76]), "=f"(v_2[77]), "=f"(v_2[78]), "=f"(v_2[79]), "=f"(v_2[80]), "=f"(v_2[81]), "=f"(v_2[82]), "=f"(v_2[83]), "=f"(v_2[84]), "=f"(v_2[85]), "=f"(v_2[86]), "=f"(v_2[87]), "=f"(v_2[88]), "=f"(v_2[89]), "=f"(v_2[90]), "=f"(v_2[91]), "=f"(v_2[92]), "=f"(v_2[93]), "=f"(v_2[94]), "=f"(v_2[95]), "=f"(pm_10)
                                    : "r"(tile_addr_2 + 64));
                                float pm_11 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_2[96]), "=f"(v_2[97]), "=f"(v_2[98]), "=f"(v_2[99]), "=f"(v_2[100]), "=f"(v_2[101]), "=f"(v_2[102]), "=f"(v_2[103]), "=f"(v_2[104]), "=f"(v_2[105]), "=f"(v_2[106]), "=f"(v_2[107]), "=f"(v_2[108]), "=f"(v_2[109]), "=f"(v_2[110]), "=f"(v_2[111]), "=f"(v_2[112]), "=f"(v_2[113]), "=f"(v_2[114]), "=f"(v_2[115]), "=f"(v_2[116]), "=f"(v_2[117]), "=f"(v_2[118]), "=f"(v_2[119]), "=f"(v_2[120]), "=f"(v_2[121]), "=f"(v_2[122]), "=f"(v_2[123]), "=f"(v_2[124]), "=f"(v_2[125]), "=f"(v_2[126]), "=f"(v_2[127]), "=f"(pm_11)
                                    : "r"(tile_addr_2 + 96));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + (wg) * 8);
                                _phase_acc_full ^= 1;
                                int row_stage_2 = (quarter * 32 + lane_1) * 33 - windows[0];
                                int lim_4 = cpre_0;
                                int p_8 = kt_1 * 4;
                                int nvis_8 = (((u_0 - 128 * p_8 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_8 - rank + 4 - 1) / 4) : (0));
                                float m_8 = -CUDART_INF_F;
                                int masked_8 = 1;
                                m_8 = pm_8;
                                int _warp_vote_50 = __any_sync(0xFFFFFFFF, nvis_8 < 32);
                                masked_8 = _warp_vote_50;
                                if (masked_8 != 0) {
                                    m_8 = -CUDART_INF_F;
                                    int _warp_vote_51 = __any_sync(0xFFFFFFFF, nvis_8 < 32);
                                    if (_warp_vote_51 != 0) {
                                        uint32_t _thresh_mask_52;
                                        {
                                            int _lim_53 = nvis_8;
                                            if (_lim_53 <= 0) { _thresh_mask_52 = 0u; }
                                            else if (_lim_53 >= 32) { _thresh_mask_52 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_52) : "r"(_lim_53));
                                            }
                                        }
                                        if (!(_thresh_mask_52 & (1u << 0))) v_2[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 1))) v_2[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 2))) v_2[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 3))) v_2[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 4))) v_2[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 5))) v_2[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 6))) v_2[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 7))) v_2[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 8))) v_2[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 9))) v_2[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 10))) v_2[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 11))) v_2[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 12))) v_2[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 13))) v_2[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 14))) v_2[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 15))) v_2[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 16))) v_2[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 17))) v_2[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 18))) v_2[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 19))) v_2[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 20))) v_2[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 21))) v_2[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 22))) v_2[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 23))) v_2[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 24))) v_2[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 25))) v_2[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 26))) v_2[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 27))) v_2[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 28))) v_2[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 29))) v_2[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 30))) v_2[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_52 & (1u << 31))) v_2[31] = -CUDART_INF_F;
                                    }
                                    float part_8 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_55 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[0], v_2[1]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[2], v_2[3]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[4], v_2[5]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[6], v_2[7]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[8], v_2[9]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[10], v_2[11]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[12], v_2[13]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[14], v_2[15]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[16], v_2[17]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[18], v_2[19]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[20], v_2[21]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[22], v_2[23]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[24], v_2[25]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[26], v_2[27]));
                                    _reg_reduce_max2_55.x = max_noftz(_reg_reduce_max2_55.x, max_noftz(v_2[28], v_2[29]));
                                    _reg_reduce_max2_55.y = max_noftz(_reg_reduce_max2_55.y, max_noftz(v_2[30], v_2[31]));
                                    float _reg_array_reduce_54 = row_max_reduce(_reg_reduce_max2_55);
                                    part_8 = _reg_array_reduce_54;
                                    m_8 = fmaxf(m_8, part_8);
                                }
                                m_8 = ((m_8 == 0.0f) ? 0.0f : m_8);
                                if (p_8 < lim_4) {
                                    max_score[off_0 + (long long)p_8] = m_8;
                                }
                                int p_9 = kt_1 * 4 + 1;
                                int nvis_9 = (((u_0 - 128 * p_9 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_9 - rank + 4 - 1) / 4) : (0));
                                float m_9 = -CUDART_INF_F;
                                int masked_9 = 1;
                                m_9 = pm_9;
                                int _warp_vote_56 = __any_sync(0xFFFFFFFF, nvis_9 < 32);
                                masked_9 = _warp_vote_56;
                                if (masked_9 != 0) {
                                    m_9 = -CUDART_INF_F;
                                    int _warp_vote_57 = __any_sync(0xFFFFFFFF, nvis_9 < 32);
                                    if (_warp_vote_57 != 0) {
                                        uint32_t _thresh_mask_58;
                                        {
                                            int _lim_59 = nvis_9;
                                            if (_lim_59 <= 0) { _thresh_mask_58 = 0u; }
                                            else if (_lim_59 >= 32) { _thresh_mask_58 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_58) : "r"(_lim_59));
                                            }
                                        }
                                        if (!(_thresh_mask_58 & (1u << 0))) v_2[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 1))) v_2[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 2))) v_2[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 3))) v_2[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 4))) v_2[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 5))) v_2[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 6))) v_2[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 7))) v_2[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 8))) v_2[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 9))) v_2[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 10))) v_2[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 11))) v_2[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 12))) v_2[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 13))) v_2[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 14))) v_2[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 15))) v_2[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 16))) v_2[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 17))) v_2[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 18))) v_2[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 19))) v_2[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 20))) v_2[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 21))) v_2[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 22))) v_2[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 23))) v_2[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 24))) v_2[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 25))) v_2[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 26))) v_2[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 27))) v_2[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 28))) v_2[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 29))) v_2[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 30))) v_2[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_58 & (1u << 31))) v_2[63] = -CUDART_INF_F;
                                    }
                                    float part_9 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_61 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_2[(32) + 0], _reg_reduce_max2_61);
                                    float _reg_array_reduce_60 = row_max_reduce(_reg_reduce_max2_61);
                                    part_9 = _reg_array_reduce_60;
                                    m_9 = fmaxf(m_9, part_9);
                                }
                                m_9 = ((m_9 == 0.0f) ? 0.0f : m_9);
                                if (p_9 < lim_4) {
                                    max_score[off_0 + (long long)p_9] = m_9;
                                }
                                int p_10 = kt_1 * 4 + 2;
                                int nvis_10 = (((u_0 - 128 * p_10 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_10 - rank + 4 - 1) / 4) : (0));
                                float m_10 = -CUDART_INF_F;
                                int masked_10 = 1;
                                m_10 = pm_10;
                                int _warp_vote_62 = __any_sync(0xFFFFFFFF, nvis_10 < 32);
                                masked_10 = _warp_vote_62;
                                if (masked_10 != 0) {
                                    m_10 = -CUDART_INF_F;
                                    int _warp_vote_63 = __any_sync(0xFFFFFFFF, nvis_10 < 32);
                                    if (_warp_vote_63 != 0) {
                                        uint32_t _thresh_mask_64;
                                        {
                                            int _lim_65 = nvis_10;
                                            if (_lim_65 <= 0) { _thresh_mask_64 = 0u; }
                                            else if (_lim_65 >= 32) { _thresh_mask_64 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_64) : "r"(_lim_65));
                                            }
                                        }
                                        if (!(_thresh_mask_64 & (1u << 0))) v_2[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 1))) v_2[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 2))) v_2[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 3))) v_2[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 4))) v_2[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 5))) v_2[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 6))) v_2[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 7))) v_2[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 8))) v_2[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 9))) v_2[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 10))) v_2[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 11))) v_2[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 12))) v_2[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 13))) v_2[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 14))) v_2[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 15))) v_2[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 16))) v_2[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 17))) v_2[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 18))) v_2[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 19))) v_2[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 20))) v_2[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 21))) v_2[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 22))) v_2[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 23))) v_2[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 24))) v_2[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 25))) v_2[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 26))) v_2[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 27))) v_2[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 28))) v_2[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 29))) v_2[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 30))) v_2[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_64 & (1u << 31))) v_2[95] = -CUDART_INF_F;
                                    }
                                    float part_10 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_67 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_2[(64) + 0], _reg_reduce_max2_67);
                                    float _reg_array_reduce_66 = row_max_reduce(_reg_reduce_max2_67);
                                    part_10 = _reg_array_reduce_66;
                                    m_10 = fmaxf(m_10, part_10);
                                }
                                m_10 = ((m_10 == 0.0f) ? 0.0f : m_10);
                                if (p_10 < lim_4) {
                                    max_score[off_0 + (long long)p_10] = m_10;
                                }
                                int p_11 = kt_1 * 4 + 3;
                                int nvis_11 = (((u_0 - 128 * p_11 - rank + 4 - 1) / 4) > (0) ? ((u_0 - 128 * p_11 - rank + 4 - 1) / 4) : (0));
                                float m_11 = -CUDART_INF_F;
                                int masked_11 = 1;
                                m_11 = pm_11;
                                int _warp_vote_68 = __any_sync(0xFFFFFFFF, nvis_11 < 32);
                                masked_11 = _warp_vote_68;
                                if (masked_11 != 0) {
                                    m_11 = -CUDART_INF_F;
                                    int _warp_vote_69 = __any_sync(0xFFFFFFFF, nvis_11 < 32);
                                    if (_warp_vote_69 != 0) {
                                        uint32_t _thresh_mask_70;
                                        {
                                            int _lim_71 = nvis_11;
                                            if (_lim_71 <= 0) { _thresh_mask_70 = 0u; }
                                            else if (_lim_71 >= 32) { _thresh_mask_70 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_70) : "r"(_lim_71));
                                            }
                                        }
                                        if (!(_thresh_mask_70 & (1u << 0))) v_2[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 1))) v_2[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 2))) v_2[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 3))) v_2[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 4))) v_2[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 5))) v_2[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 6))) v_2[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 7))) v_2[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 8))) v_2[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 9))) v_2[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 10))) v_2[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 11))) v_2[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 12))) v_2[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 13))) v_2[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 14))) v_2[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 15))) v_2[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 16))) v_2[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 17))) v_2[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 18))) v_2[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 19))) v_2[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 20))) v_2[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 21))) v_2[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 22))) v_2[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 23))) v_2[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 24))) v_2[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 25))) v_2[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 26))) v_2[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 27))) v_2[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 28))) v_2[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 29))) v_2[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 30))) v_2[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_70 & (1u << 31))) v_2[127] = -CUDART_INF_F;
                                    }
                                    float part_11 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_73 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_2[(96) + 0], _reg_reduce_max2_73);
                                    float _reg_array_reduce_72 = row_max_reduce(_reg_reduce_max2_73);
                                    part_11 = _reg_array_reduce_72;
                                    m_11 = fmaxf(m_11, part_11);
                                }
                                m_11 = ((m_11 == 0.0f) ? 0.0f : m_11);
                                if (p_11 < lim_4) {
                                    max_score[off_0 + (long long)p_11] = m_11;
                                }
                            }
                            int mine_3 = 1;
                            mine_3 = ((1 == wg) ? 1 : 0);
                            if (mine_3 != 0) {
                                mbarrier_wait(acc_full_addr + (wg) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float v_3[128];
                                unsigned int tile_addr_3 = taddr + lane_base + (unsigned int)wg * 128;
                                float pm_12 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_3[0]), "=f"(v_3[1]), "=f"(v_3[2]), "=f"(v_3[3]), "=f"(v_3[4]), "=f"(v_3[5]), "=f"(v_3[6]), "=f"(v_3[7]), "=f"(v_3[8]), "=f"(v_3[9]), "=f"(v_3[10]), "=f"(v_3[11]), "=f"(v_3[12]), "=f"(v_3[13]), "=f"(v_3[14]), "=f"(v_3[15]), "=f"(v_3[16]), "=f"(v_3[17]), "=f"(v_3[18]), "=f"(v_3[19]), "=f"(v_3[20]), "=f"(v_3[21]), "=f"(v_3[22]), "=f"(v_3[23]), "=f"(v_3[24]), "=f"(v_3[25]), "=f"(v_3[26]), "=f"(v_3[27]), "=f"(v_3[28]), "=f"(v_3[29]), "=f"(v_3[30]), "=f"(v_3[31]), "=f"(pm_12)
                                    : "r"(tile_addr_3 + 0));
                                float pm_13 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_3[32]), "=f"(v_3[33]), "=f"(v_3[34]), "=f"(v_3[35]), "=f"(v_3[36]), "=f"(v_3[37]), "=f"(v_3[38]), "=f"(v_3[39]), "=f"(v_3[40]), "=f"(v_3[41]), "=f"(v_3[42]), "=f"(v_3[43]), "=f"(v_3[44]), "=f"(v_3[45]), "=f"(v_3[46]), "=f"(v_3[47]), "=f"(v_3[48]), "=f"(v_3[49]), "=f"(v_3[50]), "=f"(v_3[51]), "=f"(v_3[52]), "=f"(v_3[53]), "=f"(v_3[54]), "=f"(v_3[55]), "=f"(v_3[56]), "=f"(v_3[57]), "=f"(v_3[58]), "=f"(v_3[59]), "=f"(v_3[60]), "=f"(v_3[61]), "=f"(v_3[62]), "=f"(v_3[63]), "=f"(pm_13)
                                    : "r"(tile_addr_3 + 32));
                                float pm_14 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_3[64]), "=f"(v_3[65]), "=f"(v_3[66]), "=f"(v_3[67]), "=f"(v_3[68]), "=f"(v_3[69]), "=f"(v_3[70]), "=f"(v_3[71]), "=f"(v_3[72]), "=f"(v_3[73]), "=f"(v_3[74]), "=f"(v_3[75]), "=f"(v_3[76]), "=f"(v_3[77]), "=f"(v_3[78]), "=f"(v_3[79]), "=f"(v_3[80]), "=f"(v_3[81]), "=f"(v_3[82]), "=f"(v_3[83]), "=f"(v_3[84]), "=f"(v_3[85]), "=f"(v_3[86]), "=f"(v_3[87]), "=f"(v_3[88]), "=f"(v_3[89]), "=f"(v_3[90]), "=f"(v_3[91]), "=f"(v_3[92]), "=f"(v_3[93]), "=f"(v_3[94]), "=f"(v_3[95]), "=f"(pm_14)
                                    : "r"(tile_addr_3 + 64));
                                float pm_15 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(v_3[96]), "=f"(v_3[97]), "=f"(v_3[98]), "=f"(v_3[99]), "=f"(v_3[100]), "=f"(v_3[101]), "=f"(v_3[102]), "=f"(v_3[103]), "=f"(v_3[104]), "=f"(v_3[105]), "=f"(v_3[106]), "=f"(v_3[107]), "=f"(v_3[108]), "=f"(v_3[109]), "=f"(v_3[110]), "=f"(v_3[111]), "=f"(v_3[112]), "=f"(v_3[113]), "=f"(v_3[114]), "=f"(v_3[115]), "=f"(v_3[116]), "=f"(v_3[117]), "=f"(v_3[118]), "=f"(v_3[119]), "=f"(v_3[120]), "=f"(v_3[121]), "=f"(v_3[122]), "=f"(v_3[123]), "=f"(v_3[124]), "=f"(v_3[125]), "=f"(v_3[126]), "=f"(v_3[127]), "=f"(pm_15)
                                    : "r"(tile_addr_3 + 96));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + (wg) * 8);
                                _phase_acc_full ^= 1;
                                int row_stage_3 = ((4 + quarter) * 32 + lane_1) * 33 - windows[1];
                                int lim_5 = cpre_1;
                                int p_12 = kt_1 * 4;
                                int nvis_12 = (((u_1 - 128 * p_12 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_12 - rank + 4 - 1) / 4) : (0));
                                float m_12 = -CUDART_INF_F;
                                int masked_12 = 1;
                                m_12 = pm_12;
                                int _warp_vote_74 = __any_sync(0xFFFFFFFF, nvis_12 < 32);
                                masked_12 = _warp_vote_74;
                                if (masked_12 != 0) {
                                    m_12 = -CUDART_INF_F;
                                    int _warp_vote_75 = __any_sync(0xFFFFFFFF, nvis_12 < 32);
                                    if (_warp_vote_75 != 0) {
                                        uint32_t _thresh_mask_76;
                                        {
                                            int _lim_77 = nvis_12;
                                            if (_lim_77 <= 0) { _thresh_mask_76 = 0u; }
                                            else if (_lim_77 >= 32) { _thresh_mask_76 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_76) : "r"(_lim_77));
                                            }
                                        }
                                        if (!(_thresh_mask_76 & (1u << 0))) v_3[0] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 1))) v_3[1] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 2))) v_3[2] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 3))) v_3[3] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 4))) v_3[4] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 5))) v_3[5] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 6))) v_3[6] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 7))) v_3[7] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 8))) v_3[8] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 9))) v_3[9] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 10))) v_3[10] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 11))) v_3[11] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 12))) v_3[12] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 13))) v_3[13] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 14))) v_3[14] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 15))) v_3[15] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 16))) v_3[16] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 17))) v_3[17] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 18))) v_3[18] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 19))) v_3[19] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 20))) v_3[20] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 21))) v_3[21] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 22))) v_3[22] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 23))) v_3[23] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 24))) v_3[24] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 25))) v_3[25] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 26))) v_3[26] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 27))) v_3[27] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 28))) v_3[28] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 29))) v_3[29] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 30))) v_3[30] = -CUDART_INF_F;
                                        if (!(_thresh_mask_76 & (1u << 31))) v_3[31] = -CUDART_INF_F;
                                    }
                                    float part_12 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_79 = {-CAKE_INF, -CAKE_INF};
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[0], v_3[1]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[2], v_3[3]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[4], v_3[5]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[6], v_3[7]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[8], v_3[9]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[10], v_3[11]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[12], v_3[13]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[14], v_3[15]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[16], v_3[17]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[18], v_3[19]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[20], v_3[21]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[22], v_3[23]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[24], v_3[25]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[26], v_3[27]));
                                    _reg_reduce_max2_79.x = max_noftz(_reg_reduce_max2_79.x, max_noftz(v_3[28], v_3[29]));
                                    _reg_reduce_max2_79.y = max_noftz(_reg_reduce_max2_79.y, max_noftz(v_3[30], v_3[31]));
                                    float _reg_array_reduce_78 = row_max_reduce(_reg_reduce_max2_79);
                                    part_12 = _reg_array_reduce_78;
                                    m_12 = fmaxf(m_12, part_12);
                                }
                                m_12 = ((m_12 == 0.0f) ? 0.0f : m_12);
                                if (p_12 < lim_5) {
                                    max_score[off_1 + (long long)p_12] = m_12;
                                }
                                int p_13 = kt_1 * 4 + 1;
                                int nvis_13 = (((u_1 - 128 * p_13 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_13 - rank + 4 - 1) / 4) : (0));
                                float m_13 = -CUDART_INF_F;
                                int masked_13 = 1;
                                m_13 = pm_13;
                                int _warp_vote_80 = __any_sync(0xFFFFFFFF, nvis_13 < 32);
                                masked_13 = _warp_vote_80;
                                if (masked_13 != 0) {
                                    m_13 = -CUDART_INF_F;
                                    int _warp_vote_81 = __any_sync(0xFFFFFFFF, nvis_13 < 32);
                                    if (_warp_vote_81 != 0) {
                                        uint32_t _thresh_mask_82;
                                        {
                                            int _lim_83 = nvis_13;
                                            if (_lim_83 <= 0) { _thresh_mask_82 = 0u; }
                                            else if (_lim_83 >= 32) { _thresh_mask_82 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_82) : "r"(_lim_83));
                                            }
                                        }
                                        if (!(_thresh_mask_82 & (1u << 0))) v_3[32] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 1))) v_3[33] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 2))) v_3[34] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 3))) v_3[35] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 4))) v_3[36] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 5))) v_3[37] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 6))) v_3[38] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 7))) v_3[39] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 8))) v_3[40] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 9))) v_3[41] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 10))) v_3[42] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 11))) v_3[43] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 12))) v_3[44] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 13))) v_3[45] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 14))) v_3[46] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 15))) v_3[47] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 16))) v_3[48] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 17))) v_3[49] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 18))) v_3[50] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 19))) v_3[51] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 20))) v_3[52] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 21))) v_3[53] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 22))) v_3[54] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 23))) v_3[55] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 24))) v_3[56] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 25))) v_3[57] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 26))) v_3[58] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 27))) v_3[59] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 28))) v_3[60] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 29))) v_3[61] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 30))) v_3[62] = -CUDART_INF_F;
                                        if (!(_thresh_mask_82 & (1u << 31))) v_3[63] = -CUDART_INF_F;
                                    }
                                    float part_13 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_85 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_3[(32) + 0], _reg_reduce_max2_85);
                                    float _reg_array_reduce_84 = row_max_reduce(_reg_reduce_max2_85);
                                    part_13 = _reg_array_reduce_84;
                                    m_13 = fmaxf(m_13, part_13);
                                }
                                m_13 = ((m_13 == 0.0f) ? 0.0f : m_13);
                                if (p_13 < lim_5) {
                                    max_score[off_1 + (long long)p_13] = m_13;
                                }
                                int p_14 = kt_1 * 4 + 2;
                                int nvis_14 = (((u_1 - 128 * p_14 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_14 - rank + 4 - 1) / 4) : (0));
                                float m_14 = -CUDART_INF_F;
                                int masked_14 = 1;
                                m_14 = pm_14;
                                int _warp_vote_86 = __any_sync(0xFFFFFFFF, nvis_14 < 32);
                                masked_14 = _warp_vote_86;
                                if (masked_14 != 0) {
                                    m_14 = -CUDART_INF_F;
                                    int _warp_vote_87 = __any_sync(0xFFFFFFFF, nvis_14 < 32);
                                    if (_warp_vote_87 != 0) {
                                        uint32_t _thresh_mask_88;
                                        {
                                            int _lim_89 = nvis_14;
                                            if (_lim_89 <= 0) { _thresh_mask_88 = 0u; }
                                            else if (_lim_89 >= 32) { _thresh_mask_88 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_88) : "r"(_lim_89));
                                            }
                                        }
                                        if (!(_thresh_mask_88 & (1u << 0))) v_3[64] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 1))) v_3[65] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 2))) v_3[66] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 3))) v_3[67] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 4))) v_3[68] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 5))) v_3[69] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 6))) v_3[70] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 7))) v_3[71] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 8))) v_3[72] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 9))) v_3[73] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 10))) v_3[74] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 11))) v_3[75] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 12))) v_3[76] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 13))) v_3[77] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 14))) v_3[78] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 15))) v_3[79] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 16))) v_3[80] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 17))) v_3[81] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 18))) v_3[82] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 19))) v_3[83] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 20))) v_3[84] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 21))) v_3[85] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 22))) v_3[86] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 23))) v_3[87] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 24))) v_3[88] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 25))) v_3[89] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 26))) v_3[90] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 27))) v_3[91] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 28))) v_3[92] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 29))) v_3[93] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 30))) v_3[94] = -CUDART_INF_F;
                                        if (!(_thresh_mask_88 & (1u << 31))) v_3[95] = -CUDART_INF_F;
                                    }
                                    float part_14 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_91 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_3[(64) + 0], _reg_reduce_max2_91);
                                    float _reg_array_reduce_90 = row_max_reduce(_reg_reduce_max2_91);
                                    part_14 = _reg_array_reduce_90;
                                    m_14 = fmaxf(m_14, part_14);
                                }
                                m_14 = ((m_14 == 0.0f) ? 0.0f : m_14);
                                if (p_14 < lim_5) {
                                    max_score[off_1 + (long long)p_14] = m_14;
                                }
                                int p_15 = kt_1 * 4 + 3;
                                int nvis_15 = (((u_1 - 128 * p_15 - rank + 4 - 1) / 4) > (0) ? ((u_1 - 128 * p_15 - rank + 4 - 1) / 4) : (0));
                                float m_15 = -CUDART_INF_F;
                                int masked_15 = 1;
                                m_15 = pm_15;
                                int _warp_vote_92 = __any_sync(0xFFFFFFFF, nvis_15 < 32);
                                masked_15 = _warp_vote_92;
                                if (masked_15 != 0) {
                                    m_15 = -CUDART_INF_F;
                                    int _warp_vote_93 = __any_sync(0xFFFFFFFF, nvis_15 < 32);
                                    if (_warp_vote_93 != 0) {
                                        uint32_t _thresh_mask_94;
                                        {
                                            int _lim_95 = nvis_15;
                                            if (_lim_95 <= 0) { _thresh_mask_94 = 0u; }
                                            else if (_lim_95 >= 32) { _thresh_mask_94 = 0xFFFFFFFFu; }
                                            else {
                                                asm volatile("{"
                                                    ".reg .u32 t;\n\t"
                                                    "shl.b32 t, 1, %1;\n\t"
                                                    "add.u32 %0, t, -1;\n\t"
                                                    "}" : "=r"(_thresh_mask_94) : "r"(_lim_95));
                                            }
                                        }
                                        if (!(_thresh_mask_94 & (1u << 0))) v_3[96] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 1))) v_3[97] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 2))) v_3[98] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 3))) v_3[99] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 4))) v_3[100] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 5))) v_3[101] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 6))) v_3[102] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 7))) v_3[103] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 8))) v_3[104] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 9))) v_3[105] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 10))) v_3[106] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 11))) v_3[107] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 12))) v_3[108] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 13))) v_3[109] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 14))) v_3[110] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 15))) v_3[111] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 16))) v_3[112] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 17))) v_3[113] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 18))) v_3[114] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 19))) v_3[115] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 20))) v_3[116] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 21))) v_3[117] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 22))) v_3[118] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 23))) v_3[119] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 24))) v_3[120] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 25))) v_3[121] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 26))) v_3[122] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 27))) v_3[123] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 28))) v_3[124] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 29))) v_3[125] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 30))) v_3[126] = -CUDART_INF_F;
                                        if (!(_thresh_mask_94 & (1u << 31))) v_3[127] = -CUDART_INF_F;
                                    }
                                    float part_15 = -CUDART_INF_F;
                                    float2 _reg_reduce_max2_97 = {-CAKE_INF, -CAKE_INF};
                                    row_max_x32_accum(&v_3[(96) + 0], _reg_reduce_max2_97);
                                    float _reg_array_reduce_96 = row_max_reduce(_reg_reduce_max2_97);
                                    part_15 = _reg_array_reduce_96;
                                    m_15 = fmaxf(m_15, part_15);
                                }
                                m_15 = ((m_15 == 0.0f) ? 0.0f : m_15);
                                if (p_15 < lim_5) {
                                    max_score[off_1 + (long long)p_15] = m_15;
                                }
                            }
                        }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            long long wr_1 = work_range[blockIdx.x];
            int j0_1 = (int)(wr_1 & 65535);
            int j1_1 = (int)(wr_1 >> 16 & 65535);
            int c0_1 = (int)(wr_1 >> 32 & 65535);
            int c1_1 = (int)(wr_1 >> 48 & 65535);
            int j_end_1 = ((c1_1 > 0) ? j1_1 + 1 : j1_1);
            unsigned int q_slot = 0;
            unsigned int k_stage = 0;
            int lane_0 = lane;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (int j_1 = j0_1; j_1 < j_end_1; j_1++) {
                long long wi_1 = work_info[j_1];
                long long prev_1 = 0;
                if (j_1 > 0) {
                    prev_1 = work_info[j_1 - 1] >> 32;
                }
                int b_1 = (int)(wi_1 & 65535);
                int it_1 = (int)(wi_1 >> 16 & 65535);
                int ce_1 = (int)((wi_1 >> 32) - prev_1);
                if (j_1 == j1_1) {
                    ce_1 = c1_1;
                }
                int cb_1 = ((j_1 == j0_1) ? c0_1 : 0);
                int kt0_1 = cb_1 / 4;
                int kt1_1 = (ce_1 + 4 - 1) / 4;
                if (kt1_1 > kt0_1) {
                    int pages_end = ce_1;
                    int q_row = (qo_indptr[b_1] + it_1 * 64) * 4;
                    mbarrier_wait(q_empty_addr + (q_slot) * 8, _phase_q_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_slot) * 8, 32768);
                        tma_2d_gmem2smem(smem_q_addr + q_slot * 2 * 16384, (&q), 0, q_row, q_full_addr + (q_slot) * 8);
                        tma_2d_gmem2smem(smem_q_addr + (q_slot * 2 + 1) * 16384, (&q), 0, q_row + 128, q_full_addr + (q_slot) * 8);
                    }
                    q_slot += 1;
                    if (q_slot == 2) { q_slot = 0; _phase_q_empty ^= 1; }
                    int bt_base = b_1 * bt_stride;
                    int p0 = kt0_1 * 4;
                    int ids = 0;
                    int ids_next = 0;
                    if (p0 + lane_0 < pages_end) {
                        ids = block_table[bt_base + p0 + lane_0];
                    }
                    if (p0 + 32 + lane_0 < pages_end) {
                        ids_next = block_table[bt_base + p0 + 32 + lane_0];
                    }
                    #pragma unroll 1
                    for (int kt = kt0_1; kt < kt1_1; kt++) {
                        int pl = kt * 4 - p0;
                        if (pl % 32 == 0 && pl > 0) {
                            ids = ids_next;
                            ids_next = 0;
                            if (p0 + pl + 32 + lane_0 < pages_end) {
                                ids_next = block_table[bt_base + p0 + pl + 32 + lane_0];
                            }
                        }
                        int npg = ((pages_end - kt * 4) < (4) ? (pages_end - kt * 4) : (4));
                        int pid_0 = __shfl_sync(0xFFFFFFFF, ids, pl % 32);
                        int pid_1 = __shfl_sync(0xFFFFFFFF, ids, pl % 32 + 1);
                        int pid_2 = __shfl_sync(0xFFFFFFFF, ids, pl % 32 + 2);
                        int pid_3 = __shfl_sync(0xFFFFFFFF, ids, pl % 32 + 3);
                        mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                        if (elect_sync()) {
                            if (npg > 0) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 4096);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384, (&k), 0, 0, pid_0, k_full_addr + (k_stage) * 8);
                            }
                            if (npg > 1) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 4096);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + 4096, (&k), 0, 0, pid_1, k_full_addr + (k_stage) * 8);
                            }
                            if (npg > 2) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 4096);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + 8192, (&k), 0, 0, pid_2, k_full_addr + (k_stage) * 8);
                            }
                            if (npg > 3) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 4096);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + 12288, (&k), 0, 0, pid_3, k_full_addr + (k_stage) * 8);
                            }
                            mbarrier_arrive(k_full_addr + (k_stage) * 8);
                        }
                        k_stage += 1;
                        if (k_stage == 8) { k_stage = 0; _phase_k_empty ^= 1; }
                    }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            long long wr_2 = work_range[blockIdx.x];
            int j0_2 = (int)(wr_2 & 65535);
            int j1_2 = (int)(wr_2 >> 16 & 65535);
            int c0_2 = (int)(wr_2 >> 32 & 65535);
            int c1_2 = (int)(wr_2 >> 48 & 65535);
            int j_end_2 = ((c1_2 > 0) ? j1_2 + 1 : j1_2);
            unsigned int q_slot_1 = 0;
            unsigned int k_stage_1 = 0;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_acc_empty_0 = 1;
            unsigned int _phase_acc_empty_1 = 1;
            #pragma unroll 1
            for (int j_2 = j0_2; j_2 < j_end_2; j_2++) {
                long long wi_2 = work_info[j_2];
                long long prev_2 = 0;
                if (j_2 > 0) {
                    prev_2 = work_info[j_2 - 1] >> 32;
                }
                int b_2 = (int)(wi_2 & 65535);
                int it_2 = (int)(wi_2 >> 16 & 65535);
                int ce_2 = (int)((wi_2 >> 32) - prev_2);
                if (j_2 == j1_2) {
                    ce_2 = c1_2;
                }
                int cb_2 = ((j_2 == j0_2) ? c0_2 : 0);
                int kt0_2 = cb_2 / 4;
                int kt1_2 = (ce_2 + 4 - 1) / 4;
                if (kt1_2 > kt0_2) {
                    mbarrier_wait(q_full_addr + (q_slot_1) * 8, _phase_q_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t _elected_0 = elect_sync();
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_q_addr + q_slot_1 * 2 * 16384 + 0)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 256 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_1), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_q_addr + q_slot_1 * 2 * 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 264 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_2), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_q_addr + q_slot_1 * 2 * 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 272 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_3), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_q_addr + q_slot_1 * 2 * 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 280 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_4), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_q_addr + (q_slot_1 * 2 + 1) * 16384 + 0)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 288 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_5), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_q_addr + (q_slot_1 * 2 + 1) * 16384 + 32)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 296 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_6), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_q_addr + (q_slot_1 * 2 + 1) * 16384 + 64)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 304 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_7), "r"(_elected_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_q_addr + (q_slot_1 * 2 + 1) * 16384 + 96)) & 0x7FFFFULL) >> 4ULL) | (((((uint64_t)(16)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(1024)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
                        asm volatile(
                            "{\n\t.reg .pred leader;\n\tsetp.ne.b32 leader, %2, 0;\n\t"
                            "@leader tcgen05.cp.cta_group::1.128x256b [%0], %1;\n\t}\n"
                            :: "r"((uint32_t)(taddr + 312 + q_slot_1 * 64)), "l"(_tcgen05_cp_desc_8), "r"(_elected_0)
                            : "memory");
                    }
                    #pragma unroll 1
                    for (int kt_2 = kt0_2; kt_2 < kt1_2; kt_2++) {
                        mbarrier_wait(k_full_addr + (k_stage_1) * 8, _phase_k_full);
                        int b_lo = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (k_stage_1) * 1024);
                        mbarrier_wait(acc_empty_addr, _phase_acc_empty_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (q_slot_1 == 0) {
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
                    "mov.b32 id, 673185808;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_acc + (0 * 128))), "r"(b_lo), "r"(tq_0), "r"(0) : "memory");
                            }
                            if (q_slot_1 == 1) {
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
                    "mov.b32 id, 673185808;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_acc + (0 * 128))), "r"(b_lo), "r"(tq_2), "r"(0) : "memory");
                            }
                        }
                        elect_commit(acc_full_addr);
                        mbarrier_wait(acc_empty_addr + 8, _phase_acc_empty_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (q_slot_1 == 0) {
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
                    "mov.b32 id, 673185808;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_acc + (1 * 128))), "r"(b_lo), "r"(tq_1), "r"(0) : "memory");
                            }
                            if (q_slot_1 == 1) {
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
                    "mov.b32 id, 673185808;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 16;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_acc + (1 * 128))), "r"(b_lo), "r"(tq_3), "r"(0) : "memory");
                            }
                        }
                        elect_commit(acc_full_addr + 8);
                        elect_commit(k_empty_addr + (k_stage_1) * 8);
                        k_stage_1 += 1;
                        if (k_stage_1 == 8) { k_stage_1 = 0; _phase_k_full ^= 1; }
                        _phase_acc_empty_0 ^= 1;
                        _phase_acc_empty_1 ^= 1;
                    }
                    elect_commit(q_empty_addr + (q_slot_1) * 8);
                    q_slot_1 += 1;
                    if (q_slot_1 == 2) { q_slot_1 = 0; _phase_q_full ^= 1; }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
