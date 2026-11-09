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
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xFFFFFFFF;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_msa_icp_indexer_3e2eb2447ae81260b606(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, int* __restrict__ block_table, unsigned long long* __restrict__ pool, int* __restrict__ qo_lens, int* __restrict__ kv_lens, int* __restrict__ qo_indptr, int* __restrict__ qo_offset, long long* __restrict__ work_range, long long* __restrict__ work_info, int* __restrict__ num_valid_pages, int rank, int compact, int bt_stride, long long pool_piece_stride, int score_cols, int force_begin, int force_end)
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

    // Prologue
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_k_addr = smem + 66560;
    unsigned long long* sbuf = reinterpret_cast<unsigned long long*>(smem_raw + 197632);
    const int sbuf_addr = smem + 197632;

    // Mbarrier init (24 barriers at smem_raw[0..192))

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

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;
    const unsigned int tq_0 = taddr + 256;
    const unsigned int tq_1 = taddr + 288;
    const unsigned int tq_2 = taddr + 320;
    const unsigned int tq_3 = taddr + 352;

    // Kernel post-init ops
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
            int own = rank * 2 / 2;
            int wcols = 2;
            int te = warp * 32 + lane_1;
            int lpc_n = ((force_begin) > (0) ? (force_begin) : (0));
            int lfb = lpc_n;
            unsigned long long top[16];
            unsigned long long buf[8];
            int cnt = 0;
            unsigned long long cand = 0;
            int need = 0;
            unsigned int mb = 0;
            unsigned int ok = 0;
            int gid = 0;
            unsigned long long forced = 0;
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
                int kt0 = cb / 2;
                int kt1 = (ce + 2 - 1) / 2;
                if (kt1 > kt0) {
                    int q_len = qo_lens[b];
                    int kv_len = kv_lens[b];
                    int q_off = qo_offset[b];
                    int t_base = qo_indptr[b];
                    int c_first = kt0 * wcols;
                    int pend = ce;
                    top[0] = 0;
                    top[1] = 0;
                    top[2] = 0;
                    top[3] = 0;
                    top[4] = 0;
                    top[5] = 0;
                    top[6] = 0;
                    top[7] = 0;
                    top[8] = 0;
                    top[9] = 0;
                    top[10] = 0;
                    top[11] = 0;
                    top[12] = 0;
                    top[13] = 0;
                    top[14] = 0;
                    top[15] = 0;
                    cnt = 0;
                    int use_stage = 0;
                    int windows[2];
                    int r_0 = it * 64 + wg * 32 + lrow / 4;
                    int u_0 = 0;
                    if (r_0 < q_len) {
                        int _min_0 = ((kv_len) < (q_off + r_0 + 1) ? (kv_len) : (q_off + r_0 + 1));
                        u_0 = _min_0;
                    }
                    int cpre_0 = (((u_0 + 127) / 128) < (score_cols) ? ((u_0 + 127) / 128) : (score_cols));
                    windows[0] = c_first - c_first % 32;
                    int nvp_0 = 0;
                    if (r_0 < q_len) {
                        nvp_0 = num_valid_pages[t_base + r_0];
                    }
                    int _max_1 = ((nvp_0) > (0) ? (nvp_0) : (0));
                    int ng_0 = ((_max_1) < (8192) ? (_max_1) : (8192));
                    int _min_2 = ((ng_0) < (score_cols) ? (ng_0) : (score_cols));
                    ng_0 = _min_2;
                    int lpc_n_0 = ((ng_0) > (0) ? (ng_0) : (0));
                    int lc_0 = ((lpc_n_0) < (pend) ? (lpc_n_0) : (pend));
                    int _max_3 = ((ng_0 - force_end) > (0) ? (ng_0 - force_end) : (0));
                    int lpc_n_1 = ((_max_3) > (0) ? (_max_3) : (0));
                    int fe_0 = lpc_n_1;
                    int live_0 = ((r_0 < q_len) ? 1 : 0);
                    if (use_stage == 0) {
                        #pragma unroll 1
                        for (int kt_0 = kt0; kt_0 < kt1; kt_0++) {
                            int mine_0 = 1;
                            if (mine_0 != 0) {
                                mbarrier_wait(acc_full_addr + (wg) * 8, _phase_acc_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                unsigned int tile_addr_0 = taddr + lane_base + (unsigned int)wg * 128;
                                float vg_0[64];
                                float mrun_0 = -CUDART_INF_F;
                                int pl_0 = kt_0 * 2 + 2 - 1;
                                int nvl_0 = (((u_0 - 128 * pl_0 - rank + 2 - 1) / 2) > (0) ? ((u_0 - 128 * pl_0 - rank + 2 - 1) / 2) : (0));
                                int partial_0 = __any_sync(0xFFFFFFFF, nvl_0 < 64);
                                float pm_0 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                    : "=f"(vg_0[0]), "=f"(vg_0[1]), "=f"(vg_0[2]), "=f"(vg_0[3]), "=f"(vg_0[4]), "=f"(vg_0[5]), "=f"(vg_0[6]), "=f"(vg_0[7]), "=f"(vg_0[8]), "=f"(vg_0[9]), "=f"(vg_0[10]), "=f"(vg_0[11]), "=f"(vg_0[12]), "=f"(vg_0[13]), "=f"(vg_0[14]), "=f"(vg_0[15]), "=f"(vg_0[16]), "=f"(vg_0[17]), "=f"(vg_0[18]), "=f"(vg_0[19]), "=f"(vg_0[20]), "=f"(vg_0[21]), "=f"(vg_0[22]), "=f"(vg_0[23]), "=f"(vg_0[24]), "=f"(vg_0[25]), "=f"(vg_0[26]), "=f"(vg_0[27]), "=f"(vg_0[28]), "=f"(vg_0[29]), "=f"(vg_0[30]), "=f"(vg_0[31]), "=f"(vg_0[32]), "=f"(vg_0[33]), "=f"(vg_0[34]), "=f"(vg_0[35]), "=f"(vg_0[36]), "=f"(vg_0[37]), "=f"(vg_0[38]), "=f"(vg_0[39]), "=f"(vg_0[40]), "=f"(vg_0[41]), "=f"(vg_0[42]), "=f"(vg_0[43]), "=f"(vg_0[44]), "=f"(vg_0[45]), "=f"(vg_0[46]), "=f"(vg_0[47]), "=f"(vg_0[48]), "=f"(vg_0[49]), "=f"(vg_0[50]), "=f"(vg_0[51]), "=f"(vg_0[52]), "=f"(vg_0[53]), "=f"(vg_0[54]), "=f"(vg_0[55]), "=f"(vg_0[56]), "=f"(vg_0[57]), "=f"(vg_0[58]), "=f"(vg_0[59]), "=f"(vg_0[60]), "=f"(vg_0[61]), "=f"(vg_0[62]), "=f"(vg_0[63]), "=f"(pm_0)
                                    : "r"(tile_addr_0 + 0));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                float mp_0 = -CUDART_INF_F;
                                mp_0 = pm_0;
                                if (partial_0 != 0) {
                                    int p_0 = kt_0 * 2;
                                    int nvis_0 = (((u_0 - 128 * p_0 - rank + 2 - 1) / 2) > (0) ? ((u_0 - 128 * p_0 - rank + 2 - 1) / 2) : (0));
                                    int masked_0 = 1;
                                    int _warp_vote_4 = __any_sync(0xFFFFFFFF, nvis_0 < 64);
                                    masked_0 = _warp_vote_4;
                                    if (masked_0 != 0) {
                                        mp_0 = -CUDART_INF_F;
                                        int _warp_vote_5 = __any_sync(0xFFFFFFFF, nvis_0 < 32);
                                        if (_warp_vote_5 != 0) {
                                            uint32_t _thresh_mask_6;
                                            {
                                                int _lim_7 = nvis_0;
                                                if (_lim_7 <= 0) { _thresh_mask_6 = 0u; }
                                                else if (_lim_7 >= 32) { _thresh_mask_6 = 0xFFFFFFFFu; }
                                                else {
                                                    asm volatile("{"
                                                        ".reg .u32 t;\n\t"
                                                        "shl.b32 t, 1, %1;\n\t"
                                                        "add.u32 %0, t, -1;\n\t"
                                                        "}" : "=r"(_thresh_mask_6) : "r"(_lim_7));
                                                }
                                            }
                                            if (!(_thresh_mask_6 & (1u << 0))) vg_0[0] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 1))) vg_0[1] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 2))) vg_0[2] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 3))) vg_0[3] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 4))) vg_0[4] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 5))) vg_0[5] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 6))) vg_0[6] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 7))) vg_0[7] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 8))) vg_0[8] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 9))) vg_0[9] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 10))) vg_0[10] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 11))) vg_0[11] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 12))) vg_0[12] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 13))) vg_0[13] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 14))) vg_0[14] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 15))) vg_0[15] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 16))) vg_0[16] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 17))) vg_0[17] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 18))) vg_0[18] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 19))) vg_0[19] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 20))) vg_0[20] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 21))) vg_0[21] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 22))) vg_0[22] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 23))) vg_0[23] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 24))) vg_0[24] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 25))) vg_0[25] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 26))) vg_0[26] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 27))) vg_0[27] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 28))) vg_0[28] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 29))) vg_0[29] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 30))) vg_0[30] = -CUDART_INF_F;
                                            if (!(_thresh_mask_6 & (1u << 31))) vg_0[31] = -CUDART_INF_F;
                                        }
                                        float part_0 = -CUDART_INF_F;
                                        float2 _reg_reduce_max2_9 = {-CUDART_INF_F, -CUDART_INF_F};
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[0], vg_0[1]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[2], vg_0[3]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[4], vg_0[5]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[6], vg_0[7]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[8], vg_0[9]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[10], vg_0[11]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[12], vg_0[13]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[14], vg_0[15]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[16], vg_0[17]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[18], vg_0[19]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[20], vg_0[21]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[22], vg_0[23]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[24], vg_0[25]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[26], vg_0[27]));
                                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(vg_0[28], vg_0[29]));
                                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(vg_0[30], vg_0[31]));
                                        float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                                        part_0 = _reg_array_reduce_8;
                                        mp_0 = fmaxf(mp_0, part_0);
                                        int _warp_vote_10 = __any_sync(0xFFFFFFFF, nvis_0 < 64);
                                        if (_warp_vote_10 != 0) {
                                            uint32_t _thresh_mask_11;
                                            {
                                                int _lim_12 = nvis_0 - 32;
                                                if (_lim_12 <= 0) { _thresh_mask_11 = 0u; }
                                                else if (_lim_12 >= 32) { _thresh_mask_11 = 0xFFFFFFFFu; }
                                                else {
                                                    asm volatile("{"
                                                        ".reg .u32 t;\n\t"
                                                        "shl.b32 t, 1, %1;\n\t"
                                                        "add.u32 %0, t, -1;\n\t"
                                                        "}" : "=r"(_thresh_mask_11) : "r"(_lim_12));
                                                }
                                            }
                                            if (!(_thresh_mask_11 & (1u << 0))) vg_0[32] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 1))) vg_0[33] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 2))) vg_0[34] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 3))) vg_0[35] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 4))) vg_0[36] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 5))) vg_0[37] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 6))) vg_0[38] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 7))) vg_0[39] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 8))) vg_0[40] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 9))) vg_0[41] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 10))) vg_0[42] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 11))) vg_0[43] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 12))) vg_0[44] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 13))) vg_0[45] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 14))) vg_0[46] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 15))) vg_0[47] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 16))) vg_0[48] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 17))) vg_0[49] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 18))) vg_0[50] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 19))) vg_0[51] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 20))) vg_0[52] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 21))) vg_0[53] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 22))) vg_0[54] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 23))) vg_0[55] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 24))) vg_0[56] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 25))) vg_0[57] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 26))) vg_0[58] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 27))) vg_0[59] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 28))) vg_0[60] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 29))) vg_0[61] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 30))) vg_0[62] = -CUDART_INF_F;
                                            if (!(_thresh_mask_11 & (1u << 31))) vg_0[63] = -CUDART_INF_F;
                                        }
                                        float part_1 = -CUDART_INF_F;
                                        float2 _reg_reduce_max2_14 = {-CUDART_INF_F, -CUDART_INF_F};
                                        row_max_x32_accum(&vg_0[(32) + 0], _reg_reduce_max2_14);
                                        float _reg_array_reduce_13 = row_max_reduce(_reg_reduce_max2_14);
                                        part_1 = _reg_array_reduce_13;
                                        mp_0 = fmaxf(mp_0, part_1);
                                    }
                                }
                                mrun_0 = mp_0;
                                float pmx_0 = mrun_0;
                                if (pmx_0 != pmx_0) {
                                    pmx_0 = -CUDART_INF_F;
                                }
                                pmx_0 = ((pmx_0 == 0.0f) ? 0.0f : pmx_0);
                                float pm_1 = -CUDART_INF_F;
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                    : "=f"(vg_0[0]), "=f"(vg_0[1]), "=f"(vg_0[2]), "=f"(vg_0[3]), "=f"(vg_0[4]), "=f"(vg_0[5]), "=f"(vg_0[6]), "=f"(vg_0[7]), "=f"(vg_0[8]), "=f"(vg_0[9]), "=f"(vg_0[10]), "=f"(vg_0[11]), "=f"(vg_0[12]), "=f"(vg_0[13]), "=f"(vg_0[14]), "=f"(vg_0[15]), "=f"(vg_0[16]), "=f"(vg_0[17]), "=f"(vg_0[18]), "=f"(vg_0[19]), "=f"(vg_0[20]), "=f"(vg_0[21]), "=f"(vg_0[22]), "=f"(vg_0[23]), "=f"(vg_0[24]), "=f"(vg_0[25]), "=f"(vg_0[26]), "=f"(vg_0[27]), "=f"(vg_0[28]), "=f"(vg_0[29]), "=f"(vg_0[30]), "=f"(vg_0[31]), "=f"(vg_0[32]), "=f"(vg_0[33]), "=f"(vg_0[34]), "=f"(vg_0[35]), "=f"(vg_0[36]), "=f"(vg_0[37]), "=f"(vg_0[38]), "=f"(vg_0[39]), "=f"(vg_0[40]), "=f"(vg_0[41]), "=f"(vg_0[42]), "=f"(vg_0[43]), "=f"(vg_0[44]), "=f"(vg_0[45]), "=f"(vg_0[46]), "=f"(vg_0[47]), "=f"(vg_0[48]), "=f"(vg_0[49]), "=f"(vg_0[50]), "=f"(vg_0[51]), "=f"(vg_0[52]), "=f"(vg_0[53]), "=f"(vg_0[54]), "=f"(vg_0[55]), "=f"(vg_0[56]), "=f"(vg_0[57]), "=f"(vg_0[58]), "=f"(vg_0[59]), "=f"(vg_0[60]), "=f"(vg_0[61]), "=f"(vg_0[62]), "=f"(vg_0[63]), "=f"(pm_1)
                                    : "r"(tile_addr_0 + 64));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(acc_empty_addr + (wg) * 8);
                                _phase_acc_full ^= 1;
                                float mp_1 = -CUDART_INF_F;
                                mp_1 = pm_1;
                                if (partial_0 != 0) {
                                    int p_1 = kt_0 * 2 + 1;
                                    int nvis_1 = (((u_0 - 128 * p_1 - rank + 2 - 1) / 2) > (0) ? ((u_0 - 128 * p_1 - rank + 2 - 1) / 2) : (0));
                                    int masked_1 = 1;
                                    int _warp_vote_15 = __any_sync(0xFFFFFFFF, nvis_1 < 64);
                                    masked_1 = _warp_vote_15;
                                    if (masked_1 != 0) {
                                        mp_1 = -CUDART_INF_F;
                                        int _warp_vote_16 = __any_sync(0xFFFFFFFF, nvis_1 < 32);
                                        if (_warp_vote_16 != 0) {
                                            uint32_t _thresh_mask_17;
                                            {
                                                int _lim_18 = nvis_1;
                                                if (_lim_18 <= 0) { _thresh_mask_17 = 0u; }
                                                else if (_lim_18 >= 32) { _thresh_mask_17 = 0xFFFFFFFFu; }
                                                else {
                                                    asm volatile("{"
                                                        ".reg .u32 t;\n\t"
                                                        "shl.b32 t, 1, %1;\n\t"
                                                        "add.u32 %0, t, -1;\n\t"
                                                        "}" : "=r"(_thresh_mask_17) : "r"(_lim_18));
                                                }
                                            }
                                            if (!(_thresh_mask_17 & (1u << 0))) vg_0[0] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 1))) vg_0[1] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 2))) vg_0[2] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 3))) vg_0[3] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 4))) vg_0[4] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 5))) vg_0[5] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 6))) vg_0[6] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 7))) vg_0[7] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 8))) vg_0[8] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 9))) vg_0[9] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 10))) vg_0[10] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 11))) vg_0[11] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 12))) vg_0[12] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 13))) vg_0[13] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 14))) vg_0[14] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 15))) vg_0[15] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 16))) vg_0[16] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 17))) vg_0[17] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 18))) vg_0[18] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 19))) vg_0[19] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 20))) vg_0[20] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 21))) vg_0[21] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 22))) vg_0[22] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 23))) vg_0[23] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 24))) vg_0[24] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 25))) vg_0[25] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 26))) vg_0[26] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 27))) vg_0[27] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 28))) vg_0[28] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 29))) vg_0[29] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 30))) vg_0[30] = -CUDART_INF_F;
                                            if (!(_thresh_mask_17 & (1u << 31))) vg_0[31] = -CUDART_INF_F;
                                        }
                                        float part_2 = -CUDART_INF_F;
                                        float2 _reg_reduce_max2_20 = {-CUDART_INF_F, -CUDART_INF_F};
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[0], vg_0[1]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[2], vg_0[3]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[4], vg_0[5]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[6], vg_0[7]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[8], vg_0[9]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[10], vg_0[11]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[12], vg_0[13]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[14], vg_0[15]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[16], vg_0[17]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[18], vg_0[19]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[20], vg_0[21]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[22], vg_0[23]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[24], vg_0[25]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[26], vg_0[27]));
                                        _reg_reduce_max2_20.x = max_noftz(_reg_reduce_max2_20.x, max_noftz(vg_0[28], vg_0[29]));
                                        _reg_reduce_max2_20.y = max_noftz(_reg_reduce_max2_20.y, max_noftz(vg_0[30], vg_0[31]));
                                        float _reg_array_reduce_19 = row_max_reduce(_reg_reduce_max2_20);
                                        part_2 = _reg_array_reduce_19;
                                        mp_1 = fmaxf(mp_1, part_2);
                                        int _warp_vote_21 = __any_sync(0xFFFFFFFF, nvis_1 < 64);
                                        if (_warp_vote_21 != 0) {
                                            uint32_t _thresh_mask_22;
                                            {
                                                int _lim_23 = nvis_1 - 32;
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
                                            if (!(_thresh_mask_22 & (1u << 0))) vg_0[32] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 1))) vg_0[33] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 2))) vg_0[34] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 3))) vg_0[35] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 4))) vg_0[36] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 5))) vg_0[37] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 6))) vg_0[38] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 7))) vg_0[39] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 8))) vg_0[40] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 9))) vg_0[41] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 10))) vg_0[42] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 11))) vg_0[43] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 12))) vg_0[44] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 13))) vg_0[45] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 14))) vg_0[46] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 15))) vg_0[47] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 16))) vg_0[48] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 17))) vg_0[49] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 18))) vg_0[50] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 19))) vg_0[51] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 20))) vg_0[52] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 21))) vg_0[53] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 22))) vg_0[54] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 23))) vg_0[55] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 24))) vg_0[56] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 25))) vg_0[57] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 26))) vg_0[58] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 27))) vg_0[59] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 28))) vg_0[60] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 29))) vg_0[61] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 30))) vg_0[62] = -CUDART_INF_F;
                                            if (!(_thresh_mask_22 & (1u << 31))) vg_0[63] = -CUDART_INF_F;
                                        }
                                        float part_3 = -CUDART_INF_F;
                                        float2 _reg_reduce_max2_25 = {-CUDART_INF_F, -CUDART_INF_F};
                                        row_max_x32_accum(&vg_0[(32) + 0], _reg_reduce_max2_25);
                                        float _reg_array_reduce_24 = row_max_reduce(_reg_reduce_max2_25);
                                        part_3 = _reg_array_reduce_24;
                                        mp_1 = fmaxf(mp_1, part_3);
                                    }
                                }
                                mrun_0 = mp_1;
                                float pmx_1 = mrun_0;
                                if (pmx_1 != pmx_1) {
                                    pmx_1 = -CUDART_INF_F;
                                }
                                pmx_1 = ((pmx_1 == 0.0f) ? 0.0f : pmx_1);
                                int pf_0 = kt_0 * 2;
                                int plain_0 = 0;
                                if (pf_0 + 2 <= lc_0) {
                                    if (pf_0 >= lfb && pf_0 + 2 <= fe_0) {
                                        plain_0 = 1;
                                    }
                                }
                                int all_plain_0 = __all_sync(0xFFFFFFFF, plain_0 != 0);
                                if (all_plain_0 != 0) {
                                    gid = pf_0;
                                    mb = reinterpret_cast<unsigned int*>(&pmx_0)[0];
                                    ok = (((mb & 2147483648u) != 0) ? ~mb : mb | 2147483648u);
                                    cand = (unsigned long long)ok << 31 | (unsigned long long)(2147483647 - gid);
                                    if (cand > top[15]) {
                                        sbuf[cnt * 256 + te] = cand;
                                        cnt = cnt + 1;
                                    }
                                    gid = pf_0 + 1;
                                    mb = reinterpret_cast<unsigned int*>(&pmx_1)[0];
                                    ok = (((mb & 2147483648u) != 0) ? ~mb : mb | 2147483648u);
                                    cand = (unsigned long long)ok << 31 | (unsigned long long)(2147483647 - gid);
                                    if (cand > top[15]) {
                                        sbuf[cnt * 256 + te] = cand;
                                        cnt = cnt + 1;
                                    }
                                }
                                if (all_plain_0 == 0) {
                                    int p_2 = kt_0 * 2;
                                    need = 0;
                                    cand = 0;
                                    if (p_2 < lc_0) {
                                        gid = p_2;
                                        forced = 0;
                                        if (p_2 < lfb || p_2 >= fe_0) {
                                            forced = 1;
                                        }
                                        mb = reinterpret_cast<unsigned int*>(&pmx_0)[0];
                                        ok = (((mb & 2147483648u) != 0) ? ~mb : mb | 2147483648u);
                                        cand = forced << 63 | (unsigned long long)ok << 31 | (unsigned long long)(2147483647 - gid);
                                        if (cand > top[15]) {
                                            need = 1;
                                        }
                                    }
                                    if (need != 0) {
                                        sbuf[cnt * 256 + te] = cand;
                                        cnt = cnt + 1;
                                    }
                                    int p_3 = kt_0 * 2 + 1;
                                    need = 0;
                                    cand = 0;
                                    if (p_3 < lc_0) {
                                        gid = p_3;
                                        forced = 0;
                                        if (p_3 < lfb || p_3 >= fe_0) {
                                            forced = 1;
                                        }
                                        mb = reinterpret_cast<unsigned int*>(&pmx_1)[0];
                                        ok = (((mb & 2147483648u) != 0) ? ~mb : mb | 2147483648u);
                                        cand = forced << 63 | (unsigned long long)ok << 31 | (unsigned long long)(2147483647 - gid);
                                        if (cand > top[15]) {
                                            need = 1;
                                        }
                                    }
                                    if (need != 0) {
                                        sbuf[cnt * 256 + te] = cand;
                                        cnt = cnt + 1;
                                    }
                                }
                                int _warp_vote_26 = __any_sync(0xFFFFFFFF, cnt > 14);
                                if (_warp_vote_26 != 0) {
                                    unsigned long long y0_0 = sbuf[te];
                                    buf[0] = ((0 < cnt) ? y0_0 : 0);
                                    unsigned long long y0_1 = sbuf[256 + te];
                                    buf[1] = ((1 < cnt) ? y0_1 : 0);
                                    unsigned long long y0_2 = sbuf[512 + te];
                                    buf[2] = ((2 < cnt) ? y0_2 : 0);
                                    unsigned long long y0_3 = sbuf[768 + te];
                                    buf[3] = ((3 < cnt) ? y0_3 : 0);
                                    unsigned long long y0_4 = sbuf[1024 + te];
                                    buf[4] = ((4 < cnt) ? y0_4 : 0);
                                    unsigned long long y0_5 = sbuf[1280 + te];
                                    buf[5] = ((5 < cnt) ? y0_5 : 0);
                                    unsigned long long y0_6 = sbuf[1536 + te];
                                    buf[6] = ((6 < cnt) ? y0_6 : 0);
                                    unsigned long long y0_7 = sbuf[1792 + te];
                                    buf[7] = ((7 < cnt) ? y0_7 : 0);
                                    if (8 < cnt) {
                                        unsigned long long y1_0 = sbuf[2048 + te];
                                        sbuf[te] = y1_0;
                                    }
                                    if (9 < cnt) {
                                        unsigned long long y1_1 = sbuf[2304 + te];
                                        sbuf[256 + te] = y1_1;
                                    }
                                    if (10 < cnt) {
                                        unsigned long long y1_2 = sbuf[2560 + te];
                                        sbuf[512 + te] = y1_2;
                                    }
                                    if (11 < cnt) {
                                        unsigned long long y1_3 = sbuf[2816 + te];
                                        sbuf[768 + te] = y1_3;
                                    }
                                    if (12 < cnt) {
                                        unsigned long long y1_4 = sbuf[3072 + te];
                                        sbuf[1024 + te] = y1_4;
                                    }
                                    if (13 < cnt) {
                                        unsigned long long y1_5 = sbuf[3328 + te];
                                        sbuf[1280 + te] = y1_5;
                                    }
                                    if (14 < cnt) {
                                        unsigned long long y1_6 = sbuf[3584 + te];
                                        sbuf[1536 + te] = y1_6;
                                    }
                                    if (15 < cnt) {
                                        unsigned long long y1_7 = sbuf[3840 + te];
                                        sbuf[1792 + te] = y1_7;
                                    }
                                    unsigned long long x = buf[0];
                                    unsigned long long y = buf[1];
                                    buf[0] = ((y > x) ? y : x);
                                    buf[1] = ((y > x) ? x : y);
                                    unsigned long long x_0 = buf[2];
                                    unsigned long long y_0 = buf[3];
                                    buf[2] = ((y_0 > x_0) ? y_0 : x_0);
                                    buf[3] = ((y_0 > x_0) ? x_0 : y_0);
                                    unsigned long long x_1 = buf[4];
                                    unsigned long long y_1 = buf[5];
                                    buf[4] = ((y_1 > x_1) ? y_1 : x_1);
                                    buf[5] = ((y_1 > x_1) ? x_1 : y_1);
                                    unsigned long long x_2 = buf[6];
                                    unsigned long long y_2 = buf[7];
                                    buf[6] = ((y_2 > x_2) ? y_2 : x_2);
                                    buf[7] = ((y_2 > x_2) ? x_2 : y_2);
                                    unsigned long long x_3 = buf[0];
                                    unsigned long long y_3 = buf[2];
                                    buf[0] = ((y_3 > x_3) ? y_3 : x_3);
                                    buf[2] = ((y_3 > x_3) ? x_3 : y_3);
                                    unsigned long long x_4 = buf[1];
                                    unsigned long long y_4 = buf[3];
                                    buf[1] = ((y_4 > x_4) ? y_4 : x_4);
                                    buf[3] = ((y_4 > x_4) ? x_4 : y_4);
                                    unsigned long long x_5 = buf[4];
                                    unsigned long long y_5 = buf[6];
                                    buf[4] = ((y_5 > x_5) ? y_5 : x_5);
                                    buf[6] = ((y_5 > x_5) ? x_5 : y_5);
                                    unsigned long long x_6 = buf[5];
                                    unsigned long long y_6 = buf[7];
                                    buf[5] = ((y_6 > x_6) ? y_6 : x_6);
                                    buf[7] = ((y_6 > x_6) ? x_6 : y_6);
                                    unsigned long long x_7 = buf[1];
                                    unsigned long long y_7 = buf[2];
                                    buf[1] = ((y_7 > x_7) ? y_7 : x_7);
                                    buf[2] = ((y_7 > x_7) ? x_7 : y_7);
                                    unsigned long long x_8 = buf[5];
                                    unsigned long long y_8 = buf[6];
                                    buf[5] = ((y_8 > x_8) ? y_8 : x_8);
                                    buf[6] = ((y_8 > x_8) ? x_8 : y_8);
                                    unsigned long long x_9 = buf[0];
                                    unsigned long long y_9 = buf[4];
                                    buf[0] = ((y_9 > x_9) ? y_9 : x_9);
                                    buf[4] = ((y_9 > x_9) ? x_9 : y_9);
                                    unsigned long long x_10 = buf[1];
                                    unsigned long long y_10 = buf[5];
                                    buf[1] = ((y_10 > x_10) ? y_10 : x_10);
                                    buf[5] = ((y_10 > x_10) ? x_10 : y_10);
                                    unsigned long long x_11 = buf[2];
                                    unsigned long long y_11 = buf[6];
                                    buf[2] = ((y_11 > x_11) ? y_11 : x_11);
                                    buf[6] = ((y_11 > x_11) ? x_11 : y_11);
                                    unsigned long long x_12 = buf[3];
                                    unsigned long long y_12 = buf[7];
                                    buf[3] = ((y_12 > x_12) ? y_12 : x_12);
                                    buf[7] = ((y_12 > x_12) ? x_12 : y_12);
                                    unsigned long long x_13 = buf[2];
                                    unsigned long long y_13 = buf[4];
                                    buf[2] = ((y_13 > x_13) ? y_13 : x_13);
                                    buf[4] = ((y_13 > x_13) ? x_13 : y_13);
                                    unsigned long long x_14 = buf[3];
                                    unsigned long long y_14 = buf[5];
                                    buf[3] = ((y_14 > x_14) ? y_14 : x_14);
                                    buf[5] = ((y_14 > x_14) ? x_14 : y_14);
                                    unsigned long long x_15 = buf[1];
                                    unsigned long long y_15 = buf[2];
                                    buf[1] = ((y_15 > x_15) ? y_15 : x_15);
                                    buf[2] = ((y_15 > x_15) ? x_15 : y_15);
                                    unsigned long long x_16 = buf[3];
                                    unsigned long long y_16 = buf[4];
                                    buf[3] = ((y_16 > x_16) ? y_16 : x_16);
                                    buf[4] = ((y_16 > x_16) ? x_16 : y_16);
                                    unsigned long long x_17 = buf[5];
                                    unsigned long long y_17 = buf[6];
                                    buf[5] = ((y_17 > x_17) ? y_17 : x_17);
                                    buf[6] = ((y_17 > x_17) ? x_17 : y_17);
                                    unsigned long long y_18 = buf[0];
                                    top[15] = ((y_18 > top[15]) ? y_18 : top[15]);
                                    unsigned long long y_19 = buf[1];
                                    top[14] = ((y_19 > top[14]) ? y_19 : top[14]);
                                    unsigned long long y_20 = buf[2];
                                    top[13] = ((y_20 > top[13]) ? y_20 : top[13]);
                                    unsigned long long y_21 = buf[3];
                                    top[12] = ((y_21 > top[12]) ? y_21 : top[12]);
                                    unsigned long long y_22 = buf[4];
                                    top[11] = ((y_22 > top[11]) ? y_22 : top[11]);
                                    unsigned long long y_23 = buf[5];
                                    top[10] = ((y_23 > top[10]) ? y_23 : top[10]);
                                    unsigned long long y_24 = buf[6];
                                    top[9] = ((y_24 > top[9]) ? y_24 : top[9]);
                                    unsigned long long y_25 = buf[7];
                                    top[8] = ((y_25 > top[8]) ? y_25 : top[8]);
                                    unsigned long long x_18 = top[0];
                                    unsigned long long y_26 = top[8];
                                    top[0] = ((y_26 > x_18) ? y_26 : x_18);
                                    top[8] = ((y_26 > x_18) ? x_18 : y_26);
                                    unsigned long long x_19 = top[1];
                                    unsigned long long y_27 = top[9];
                                    top[1] = ((y_27 > x_19) ? y_27 : x_19);
                                    top[9] = ((y_27 > x_19) ? x_19 : y_27);
                                    unsigned long long x_20 = top[2];
                                    unsigned long long y_28 = top[10];
                                    top[2] = ((y_28 > x_20) ? y_28 : x_20);
                                    top[10] = ((y_28 > x_20) ? x_20 : y_28);
                                    unsigned long long x_21 = top[3];
                                    unsigned long long y_29 = top[11];
                                    top[3] = ((y_29 > x_21) ? y_29 : x_21);
                                    top[11] = ((y_29 > x_21) ? x_21 : y_29);
                                    unsigned long long x_22 = top[4];
                                    unsigned long long y_30 = top[12];
                                    top[4] = ((y_30 > x_22) ? y_30 : x_22);
                                    top[12] = ((y_30 > x_22) ? x_22 : y_30);
                                    unsigned long long x_23 = top[5];
                                    unsigned long long y_31 = top[13];
                                    top[5] = ((y_31 > x_23) ? y_31 : x_23);
                                    top[13] = ((y_31 > x_23) ? x_23 : y_31);
                                    unsigned long long x_24 = top[6];
                                    unsigned long long y_32 = top[14];
                                    top[6] = ((y_32 > x_24) ? y_32 : x_24);
                                    top[14] = ((y_32 > x_24) ? x_24 : y_32);
                                    unsigned long long x_25 = top[7];
                                    unsigned long long y_33 = top[15];
                                    top[7] = ((y_33 > x_25) ? y_33 : x_25);
                                    top[15] = ((y_33 > x_25) ? x_25 : y_33);
                                    unsigned long long x_26 = top[0];
                                    unsigned long long y_34 = top[4];
                                    top[0] = ((y_34 > x_26) ? y_34 : x_26);
                                    top[4] = ((y_34 > x_26) ? x_26 : y_34);
                                    unsigned long long x_27 = top[1];
                                    unsigned long long y_35 = top[5];
                                    top[1] = ((y_35 > x_27) ? y_35 : x_27);
                                    top[5] = ((y_35 > x_27) ? x_27 : y_35);
                                    unsigned long long x_28 = top[2];
                                    unsigned long long y_36 = top[6];
                                    top[2] = ((y_36 > x_28) ? y_36 : x_28);
                                    top[6] = ((y_36 > x_28) ? x_28 : y_36);
                                    unsigned long long x_29 = top[3];
                                    unsigned long long y_37 = top[7];
                                    top[3] = ((y_37 > x_29) ? y_37 : x_29);
                                    top[7] = ((y_37 > x_29) ? x_29 : y_37);
                                    unsigned long long x_30 = top[8];
                                    unsigned long long y_38 = top[12];
                                    top[8] = ((y_38 > x_30) ? y_38 : x_30);
                                    top[12] = ((y_38 > x_30) ? x_30 : y_38);
                                    unsigned long long x_31 = top[9];
                                    unsigned long long y_39 = top[13];
                                    top[9] = ((y_39 > x_31) ? y_39 : x_31);
                                    top[13] = ((y_39 > x_31) ? x_31 : y_39);
                                    unsigned long long x_32 = top[10];
                                    unsigned long long y_40 = top[14];
                                    top[10] = ((y_40 > x_32) ? y_40 : x_32);
                                    top[14] = ((y_40 > x_32) ? x_32 : y_40);
                                    unsigned long long x_33 = top[11];
                                    unsigned long long y_41 = top[15];
                                    top[11] = ((y_41 > x_33) ? y_41 : x_33);
                                    top[15] = ((y_41 > x_33) ? x_33 : y_41);
                                    unsigned long long x_34 = top[0];
                                    unsigned long long y_42 = top[2];
                                    top[0] = ((y_42 > x_34) ? y_42 : x_34);
                                    top[2] = ((y_42 > x_34) ? x_34 : y_42);
                                    unsigned long long x_35 = top[1];
                                    unsigned long long y_43 = top[3];
                                    top[1] = ((y_43 > x_35) ? y_43 : x_35);
                                    top[3] = ((y_43 > x_35) ? x_35 : y_43);
                                    unsigned long long x_36 = top[4];
                                    unsigned long long y_44 = top[6];
                                    top[4] = ((y_44 > x_36) ? y_44 : x_36);
                                    top[6] = ((y_44 > x_36) ? x_36 : y_44);
                                    unsigned long long x_37 = top[5];
                                    unsigned long long y_45 = top[7];
                                    top[5] = ((y_45 > x_37) ? y_45 : x_37);
                                    top[7] = ((y_45 > x_37) ? x_37 : y_45);
                                    unsigned long long x_38 = top[8];
                                    unsigned long long y_46 = top[10];
                                    top[8] = ((y_46 > x_38) ? y_46 : x_38);
                                    top[10] = ((y_46 > x_38) ? x_38 : y_46);
                                    unsigned long long x_39 = top[9];
                                    unsigned long long y_47 = top[11];
                                    top[9] = ((y_47 > x_39) ? y_47 : x_39);
                                    top[11] = ((y_47 > x_39) ? x_39 : y_47);
                                    unsigned long long x_40 = top[12];
                                    unsigned long long y_48 = top[14];
                                    top[12] = ((y_48 > x_40) ? y_48 : x_40);
                                    top[14] = ((y_48 > x_40) ? x_40 : y_48);
                                    unsigned long long x_41 = top[13];
                                    unsigned long long y_49 = top[15];
                                    top[13] = ((y_49 > x_41) ? y_49 : x_41);
                                    top[15] = ((y_49 > x_41) ? x_41 : y_49);
                                    unsigned long long x_42 = top[0];
                                    unsigned long long y_50 = top[1];
                                    top[0] = ((y_50 > x_42) ? y_50 : x_42);
                                    top[1] = ((y_50 > x_42) ? x_42 : y_50);
                                    unsigned long long x_43 = top[2];
                                    unsigned long long y_51 = top[3];
                                    top[2] = ((y_51 > x_43) ? y_51 : x_43);
                                    top[3] = ((y_51 > x_43) ? x_43 : y_51);
                                    unsigned long long x_44 = top[4];
                                    unsigned long long y_52 = top[5];
                                    top[4] = ((y_52 > x_44) ? y_52 : x_44);
                                    top[5] = ((y_52 > x_44) ? x_44 : y_52);
                                    unsigned long long x_45 = top[6];
                                    unsigned long long y_53 = top[7];
                                    top[6] = ((y_53 > x_45) ? y_53 : x_45);
                                    top[7] = ((y_53 > x_45) ? x_45 : y_53);
                                    unsigned long long x_46 = top[8];
                                    unsigned long long y_54 = top[9];
                                    top[8] = ((y_54 > x_46) ? y_54 : x_46);
                                    top[9] = ((y_54 > x_46) ? x_46 : y_54);
                                    unsigned long long x_47 = top[10];
                                    unsigned long long y_55 = top[11];
                                    top[10] = ((y_55 > x_47) ? y_55 : x_47);
                                    top[11] = ((y_55 > x_47) ? x_47 : y_55);
                                    unsigned long long x_48 = top[12];
                                    unsigned long long y_56 = top[13];
                                    top[12] = ((y_56 > x_48) ? y_56 : x_48);
                                    top[13] = ((y_56 > x_48) ? x_48 : y_56);
                                    unsigned long long x_49 = top[14];
                                    unsigned long long y_57 = top[15];
                                    top[14] = ((y_57 > x_49) ? y_57 : x_49);
                                    top[15] = ((y_57 > x_49) ? x_49 : y_57);
                                    int _max_27 = ((cnt - 8) > (0) ? (cnt - 8) : (0));
                                    cnt = _max_27;
                                }
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int fl = 0; fl < 2; fl++) {
                        int _warp_vote_28 = __any_sync(0xFFFFFFFF, cnt != 0);
                        if (_warp_vote_28 != 0) {
                            unsigned long long y0_8 = sbuf[te];
                            buf[0] = ((0 < cnt) ? y0_8 : 0);
                            unsigned long long y0_9 = sbuf[256 + te];
                            buf[1] = ((1 < cnt) ? y0_9 : 0);
                            unsigned long long y0_10 = sbuf[512 + te];
                            buf[2] = ((2 < cnt) ? y0_10 : 0);
                            unsigned long long y0_11 = sbuf[768 + te];
                            buf[3] = ((3 < cnt) ? y0_11 : 0);
                            unsigned long long y0_12 = sbuf[1024 + te];
                            buf[4] = ((4 < cnt) ? y0_12 : 0);
                            unsigned long long y0_13 = sbuf[1280 + te];
                            buf[5] = ((5 < cnt) ? y0_13 : 0);
                            unsigned long long y0_14 = sbuf[1536 + te];
                            buf[6] = ((6 < cnt) ? y0_14 : 0);
                            unsigned long long y0_15 = sbuf[1792 + te];
                            buf[7] = ((7 < cnt) ? y0_15 : 0);
                            if (8 < cnt) {
                                unsigned long long y1_8 = sbuf[2048 + te];
                                sbuf[te] = y1_8;
                            }
                            if (9 < cnt) {
                                unsigned long long y1_9 = sbuf[2304 + te];
                                sbuf[256 + te] = y1_9;
                            }
                            if (10 < cnt) {
                                unsigned long long y1_10 = sbuf[2560 + te];
                                sbuf[512 + te] = y1_10;
                            }
                            if (11 < cnt) {
                                unsigned long long y1_11 = sbuf[2816 + te];
                                sbuf[768 + te] = y1_11;
                            }
                            if (12 < cnt) {
                                unsigned long long y1_12 = sbuf[3072 + te];
                                sbuf[1024 + te] = y1_12;
                            }
                            if (13 < cnt) {
                                unsigned long long y1_13 = sbuf[3328 + te];
                                sbuf[1280 + te] = y1_13;
                            }
                            if (14 < cnt) {
                                unsigned long long y1_14 = sbuf[3584 + te];
                                sbuf[1536 + te] = y1_14;
                            }
                            if (15 < cnt) {
                                unsigned long long y1_15 = sbuf[3840 + te];
                                sbuf[1792 + te] = y1_15;
                            }
                            unsigned long long x_50 = buf[0];
                            unsigned long long y_58 = buf[1];
                            buf[0] = ((y_58 > x_50) ? y_58 : x_50);
                            buf[1] = ((y_58 > x_50) ? x_50 : y_58);
                            unsigned long long x_50_1 = buf[2];
                            unsigned long long y_58_1 = buf[3];
                            buf[2] = ((y_58_1 > x_50_1) ? y_58_1 : x_50_1);
                            buf[3] = ((y_58_1 > x_50_1) ? x_50_1 : y_58_1);
                            unsigned long long x_51 = buf[4];
                            unsigned long long y_59 = buf[5];
                            buf[4] = ((y_59 > x_51) ? y_59 : x_51);
                            buf[5] = ((y_59 > x_51) ? x_51 : y_59);
                            unsigned long long x_52 = buf[6];
                            unsigned long long y_60 = buf[7];
                            buf[6] = ((y_60 > x_52) ? y_60 : x_52);
                            buf[7] = ((y_60 > x_52) ? x_52 : y_60);
                            unsigned long long x_53 = buf[0];
                            unsigned long long y_61 = buf[2];
                            buf[0] = ((y_61 > x_53) ? y_61 : x_53);
                            buf[2] = ((y_61 > x_53) ? x_53 : y_61);
                            unsigned long long x_54 = buf[1];
                            unsigned long long y_62 = buf[3];
                            buf[1] = ((y_62 > x_54) ? y_62 : x_54);
                            buf[3] = ((y_62 > x_54) ? x_54 : y_62);
                            unsigned long long x_55 = buf[4];
                            unsigned long long y_63 = buf[6];
                            buf[4] = ((y_63 > x_55) ? y_63 : x_55);
                            buf[6] = ((y_63 > x_55) ? x_55 : y_63);
                            unsigned long long x_56 = buf[5];
                            unsigned long long y_64 = buf[7];
                            buf[5] = ((y_64 > x_56) ? y_64 : x_56);
                            buf[7] = ((y_64 > x_56) ? x_56 : y_64);
                            unsigned long long x_57 = buf[1];
                            unsigned long long y_65 = buf[2];
                            buf[1] = ((y_65 > x_57) ? y_65 : x_57);
                            buf[2] = ((y_65 > x_57) ? x_57 : y_65);
                            unsigned long long x_58 = buf[5];
                            unsigned long long y_66 = buf[6];
                            buf[5] = ((y_66 > x_58) ? y_66 : x_58);
                            buf[6] = ((y_66 > x_58) ? x_58 : y_66);
                            unsigned long long x_59 = buf[0];
                            unsigned long long y_67 = buf[4];
                            buf[0] = ((y_67 > x_59) ? y_67 : x_59);
                            buf[4] = ((y_67 > x_59) ? x_59 : y_67);
                            unsigned long long x_60 = buf[1];
                            unsigned long long y_68 = buf[5];
                            buf[1] = ((y_68 > x_60) ? y_68 : x_60);
                            buf[5] = ((y_68 > x_60) ? x_60 : y_68);
                            unsigned long long x_61 = buf[2];
                            unsigned long long y_69 = buf[6];
                            buf[2] = ((y_69 > x_61) ? y_69 : x_61);
                            buf[6] = ((y_69 > x_61) ? x_61 : y_69);
                            unsigned long long x_62 = buf[3];
                            unsigned long long y_70 = buf[7];
                            buf[3] = ((y_70 > x_62) ? y_70 : x_62);
                            buf[7] = ((y_70 > x_62) ? x_62 : y_70);
                            unsigned long long x_63 = buf[2];
                            unsigned long long y_71 = buf[4];
                            buf[2] = ((y_71 > x_63) ? y_71 : x_63);
                            buf[4] = ((y_71 > x_63) ? x_63 : y_71);
                            unsigned long long x_64 = buf[3];
                            unsigned long long y_72 = buf[5];
                            buf[3] = ((y_72 > x_64) ? y_72 : x_64);
                            buf[5] = ((y_72 > x_64) ? x_64 : y_72);
                            unsigned long long x_65 = buf[1];
                            unsigned long long y_73 = buf[2];
                            buf[1] = ((y_73 > x_65) ? y_73 : x_65);
                            buf[2] = ((y_73 > x_65) ? x_65 : y_73);
                            unsigned long long x_66 = buf[3];
                            unsigned long long y_74 = buf[4];
                            buf[3] = ((y_74 > x_66) ? y_74 : x_66);
                            buf[4] = ((y_74 > x_66) ? x_66 : y_74);
                            unsigned long long x_67 = buf[5];
                            unsigned long long y_75 = buf[6];
                            buf[5] = ((y_75 > x_67) ? y_75 : x_67);
                            buf[6] = ((y_75 > x_67) ? x_67 : y_75);
                            unsigned long long y_76 = buf[0];
                            top[15] = ((y_76 > top[15]) ? y_76 : top[15]);
                            unsigned long long y_77 = buf[1];
                            top[14] = ((y_77 > top[14]) ? y_77 : top[14]);
                            unsigned long long y_78 = buf[2];
                            top[13] = ((y_78 > top[13]) ? y_78 : top[13]);
                            unsigned long long y_79 = buf[3];
                            top[12] = ((y_79 > top[12]) ? y_79 : top[12]);
                            unsigned long long y_80 = buf[4];
                            top[11] = ((y_80 > top[11]) ? y_80 : top[11]);
                            unsigned long long y_81 = buf[5];
                            top[10] = ((y_81 > top[10]) ? y_81 : top[10]);
                            unsigned long long y_82 = buf[6];
                            top[9] = ((y_82 > top[9]) ? y_82 : top[9]);
                            unsigned long long y_83 = buf[7];
                            top[8] = ((y_83 > top[8]) ? y_83 : top[8]);
                            unsigned long long x_68 = top[0];
                            unsigned long long y_84 = top[8];
                            top[0] = ((y_84 > x_68) ? y_84 : x_68);
                            top[8] = ((y_84 > x_68) ? x_68 : y_84);
                            unsigned long long x_69 = top[1];
                            unsigned long long y_85 = top[9];
                            top[1] = ((y_85 > x_69) ? y_85 : x_69);
                            top[9] = ((y_85 > x_69) ? x_69 : y_85);
                            unsigned long long x_70 = top[2];
                            unsigned long long y_86 = top[10];
                            top[2] = ((y_86 > x_70) ? y_86 : x_70);
                            top[10] = ((y_86 > x_70) ? x_70 : y_86);
                            unsigned long long x_71 = top[3];
                            unsigned long long y_87 = top[11];
                            top[3] = ((y_87 > x_71) ? y_87 : x_71);
                            top[11] = ((y_87 > x_71) ? x_71 : y_87);
                            unsigned long long x_72 = top[4];
                            unsigned long long y_88 = top[12];
                            top[4] = ((y_88 > x_72) ? y_88 : x_72);
                            top[12] = ((y_88 > x_72) ? x_72 : y_88);
                            unsigned long long x_73 = top[5];
                            unsigned long long y_89 = top[13];
                            top[5] = ((y_89 > x_73) ? y_89 : x_73);
                            top[13] = ((y_89 > x_73) ? x_73 : y_89);
                            unsigned long long x_74 = top[6];
                            unsigned long long y_90 = top[14];
                            top[6] = ((y_90 > x_74) ? y_90 : x_74);
                            top[14] = ((y_90 > x_74) ? x_74 : y_90);
                            unsigned long long x_75 = top[7];
                            unsigned long long y_91 = top[15];
                            top[7] = ((y_91 > x_75) ? y_91 : x_75);
                            top[15] = ((y_91 > x_75) ? x_75 : y_91);
                            unsigned long long x_76 = top[0];
                            unsigned long long y_92 = top[4];
                            top[0] = ((y_92 > x_76) ? y_92 : x_76);
                            top[4] = ((y_92 > x_76) ? x_76 : y_92);
                            unsigned long long x_77 = top[1];
                            unsigned long long y_93 = top[5];
                            top[1] = ((y_93 > x_77) ? y_93 : x_77);
                            top[5] = ((y_93 > x_77) ? x_77 : y_93);
                            unsigned long long x_78 = top[2];
                            unsigned long long y_94 = top[6];
                            top[2] = ((y_94 > x_78) ? y_94 : x_78);
                            top[6] = ((y_94 > x_78) ? x_78 : y_94);
                            unsigned long long x_79 = top[3];
                            unsigned long long y_95 = top[7];
                            top[3] = ((y_95 > x_79) ? y_95 : x_79);
                            top[7] = ((y_95 > x_79) ? x_79 : y_95);
                            unsigned long long x_80 = top[8];
                            unsigned long long y_96 = top[12];
                            top[8] = ((y_96 > x_80) ? y_96 : x_80);
                            top[12] = ((y_96 > x_80) ? x_80 : y_96);
                            unsigned long long x_81 = top[9];
                            unsigned long long y_97 = top[13];
                            top[9] = ((y_97 > x_81) ? y_97 : x_81);
                            top[13] = ((y_97 > x_81) ? x_81 : y_97);
                            unsigned long long x_82 = top[10];
                            unsigned long long y_98 = top[14];
                            top[10] = ((y_98 > x_82) ? y_98 : x_82);
                            top[14] = ((y_98 > x_82) ? x_82 : y_98);
                            unsigned long long x_83 = top[11];
                            unsigned long long y_99 = top[15];
                            top[11] = ((y_99 > x_83) ? y_99 : x_83);
                            top[15] = ((y_99 > x_83) ? x_83 : y_99);
                            unsigned long long x_84 = top[0];
                            unsigned long long y_100 = top[2];
                            top[0] = ((y_100 > x_84) ? y_100 : x_84);
                            top[2] = ((y_100 > x_84) ? x_84 : y_100);
                            unsigned long long x_85 = top[1];
                            unsigned long long y_101 = top[3];
                            top[1] = ((y_101 > x_85) ? y_101 : x_85);
                            top[3] = ((y_101 > x_85) ? x_85 : y_101);
                            unsigned long long x_86 = top[4];
                            unsigned long long y_102 = top[6];
                            top[4] = ((y_102 > x_86) ? y_102 : x_86);
                            top[6] = ((y_102 > x_86) ? x_86 : y_102);
                            unsigned long long x_87 = top[5];
                            unsigned long long y_103 = top[7];
                            top[5] = ((y_103 > x_87) ? y_103 : x_87);
                            top[7] = ((y_103 > x_87) ? x_87 : y_103);
                            unsigned long long x_88 = top[8];
                            unsigned long long y_104 = top[10];
                            top[8] = ((y_104 > x_88) ? y_104 : x_88);
                            top[10] = ((y_104 > x_88) ? x_88 : y_104);
                            unsigned long long x_89 = top[9];
                            unsigned long long y_105 = top[11];
                            top[9] = ((y_105 > x_89) ? y_105 : x_89);
                            top[11] = ((y_105 > x_89) ? x_89 : y_105);
                            unsigned long long x_90 = top[12];
                            unsigned long long y_106 = top[14];
                            top[12] = ((y_106 > x_90) ? y_106 : x_90);
                            top[14] = ((y_106 > x_90) ? x_90 : y_106);
                            unsigned long long x_91 = top[13];
                            unsigned long long y_107 = top[15];
                            top[13] = ((y_107 > x_91) ? y_107 : x_91);
                            top[15] = ((y_107 > x_91) ? x_91 : y_107);
                            unsigned long long x_92 = top[0];
                            unsigned long long y_108 = top[1];
                            top[0] = ((y_108 > x_92) ? y_108 : x_92);
                            top[1] = ((y_108 > x_92) ? x_92 : y_108);
                            unsigned long long x_93 = top[2];
                            unsigned long long y_109 = top[3];
                            top[2] = ((y_109 > x_93) ? y_109 : x_93);
                            top[3] = ((y_109 > x_93) ? x_93 : y_109);
                            unsigned long long x_94 = top[4];
                            unsigned long long y_110 = top[5];
                            top[4] = ((y_110 > x_94) ? y_110 : x_94);
                            top[5] = ((y_110 > x_94) ? x_94 : y_110);
                            unsigned long long x_95 = top[6];
                            unsigned long long y_111 = top[7];
                            top[6] = ((y_111 > x_95) ? y_111 : x_95);
                            top[7] = ((y_111 > x_95) ? x_95 : y_111);
                            unsigned long long x_96 = top[8];
                            unsigned long long y_112 = top[9];
                            top[8] = ((y_112 > x_96) ? y_112 : x_96);
                            top[9] = ((y_112 > x_96) ? x_96 : y_112);
                            unsigned long long x_97 = top[10];
                            unsigned long long y_113 = top[11];
                            top[10] = ((y_113 > x_97) ? y_113 : x_97);
                            top[11] = ((y_113 > x_97) ? x_97 : y_113);
                            unsigned long long x_98 = top[12];
                            unsigned long long y_114 = top[13];
                            top[12] = ((y_114 > x_98) ? y_114 : x_98);
                            top[13] = ((y_114 > x_98) ? x_98 : y_114);
                            unsigned long long x_99 = top[14];
                            unsigned long long y_115 = top[15];
                            top[14] = ((y_115 > x_99) ? y_115 : x_99);
                            top[15] = ((y_115 > x_99) ? x_99 : y_115);
                            int _max_29 = ((cnt - 8) > (0) ? (cnt - 8) : (0));
                            cnt = _max_29;
                        }
                    }
                    if (live_0 != 0) {
                        long long dst = (long long)((int)blockIdx.x + j) * pool_piece_stride + (long long)((wg * 128 + lrow) * 16);
                        pool[dst] = top[0];
                        pool[dst + 1] = top[1];
                        pool[dst + 2] = top[2];
                        pool[dst + 3] = top[3];
                        pool[dst + 4] = top[4];
                        pool[dst + 5] = top[5];
                        pool[dst + 6] = top[6];
                        pool[dst + 7] = top[7];
                        pool[dst + 8] = top[8];
                        pool[dst + 9] = top[9];
                        pool[dst + 10] = top[10];
                        pool[dst + 11] = top[11];
                        pool[dst + 12] = top[12];
                        pool[dst + 13] = top[13];
                        pool[dst + 14] = top[14];
                        pool[dst + 15] = top[15];
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
                int kt0_1 = cb_1 / 2;
                int kt1_1 = (ce_1 + 2 - 1) / 2;
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
                    int p0 = kt0_1 * 2;
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
                        int pl = kt * 2 - p0;
                        if (pl % 32 == 0 && pl > 0) {
                            ids = ids_next;
                            ids_next = 0;
                            if (p0 + pl + 32 + lane_0 < pages_end) {
                                ids_next = block_table[bt_base + p0 + pl + 32 + lane_0];
                            }
                        }
                        int npg = ((pages_end - kt * 2) < (2) ? (pages_end - kt * 2) : (2));
                        int pid_0 = __shfl_sync(0xFFFFFFFF, ids, pl % 32);
                        int pid_1 = __shfl_sync(0xFFFFFFFF, ids, pl % 32 + 1);
                        mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                        if (elect_sync()) {
                            if (npg > 0) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 8192);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384, (&k), 0, 0, pid_0, k_full_addr + (k_stage) * 8);
                            }
                            if (npg > 1) {
                                mbarrier_expect_tx(k_full_addr + (k_stage) * 8, 8192);
                                tma_3d_gmem2smem(smem_k_addr + k_stage * 16384 + 8192, (&k), 0, 0, pid_1, k_full_addr + (k_stage) * 8);
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
                int kt0_2 = cb_2 / 2;
                int kt1_2 = (ce_2 + 2 - 1) / 2;
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
                    for (int kt_1 = kt0_2; kt_1 < kt1_2; kt_1++) {
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
