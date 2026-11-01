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
#define TMEM_ACC_OFFSET 0
#define NUM_K_PIPE_STAGES 12
#define NUM_ACC_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 17408
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_LISTS_OFF 1024
#define SMEM_LISTS_STAGE_BYTES 16384
#define SMEM_LISTS_STRIDE 16384
#define SMEM_TOTAL 214016
#define THREADS 192

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



__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}



__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
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

__global__ __launch_bounds__(192) void
kernel_cake_msa_icp_indexer_d0a9cf9610e2e4cee0a3(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, int* __restrict__ block_table, int* __restrict__ seq_lens, int* __restrict__ num_valid_pages, float* __restrict__ scores, int bt_stride, int score_stride_t, int score_stride_h, int num_heads, int query_len, int columns, int splits, int rank, int world, int page_size, int init_blocks, int local_blocks)
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
    #define mma_done_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 112)
    #define acc_full_addr (mbar_base + 208)
    #define acc_empty_addr (mbar_base + 240)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_k_addr = smem + 17408;
    unsigned long long* lists = reinterpret_cast<unsigned long long*>(smem_raw + 1024);
    const int lists_addr = smem + 1024;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // mma_done: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 12 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // k_empty: 12 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 4 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // acc_empty: 4 barriers, init_count=32
            mbarrier_init(smem + 240, 32);
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 272);
    if (warp == 0) {
        int _tmem_hold = smem + 272;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc = taddr;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Role: epi ----
    if (warp <= 3) {
        { // epi_main
            int b = blockIdx.x;
            int split = blockIdx.y;
            int seqlen = seq_lens[b];
            int ncols = 0;
            if (seqlen > 0) {
                ncols = (seqlen + page_size - 1) / page_size;
            }
            if (ncols > columns) {
                ncols = columns;
            }
            int n_blk = 0;
            if (split < ncols) {
                n_blk = (ncols - split + splits - 1) / splits;
            }
            int n_tiles = (n_blk + 3) / 4;
            int w = (int)warp;
            int lane_0 = (int)lane;
            int j = lane_0 / num_heads;
            int h = lane_0 - j * num_heads;
            int live = 0;
            if (j < query_len) {
                live = 1;
            }
            int t = b * query_len + j;
            int q_pos = seqlen - query_len + j;
            int row_off = rank;
            int row_shift = 1;
            if (world == 4) {
                row_shift = 2;
            }
            int idx_shift = 0;
            int idx_off = 0;
            if (page_size != 128) {
                row_off = rank * 32;
                row_shift = 0;
                idx_shift = 1;
                idx_off = rank * 2 / world;
            }
            int row_base = t * score_stride_t + h * score_stride_h;
            unsigned int _phase_acc_full = 0;
            if (n_tiles > 0 && w < 4) {
                unsigned int slot_addr = taddr + (unsigned int)(w * 32 << 16) + (unsigned int)(w * 128);
                float acc[32];
                int i = 0;
                int c = 0;
                int d = 0;
                int nvis = 0;
                float m = 0.0f;
                #pragma unroll 1
                for (int tile = w; tile < n_tiles; tile += 4) {
                    mbarrier_wait(acc_full_addr + (w) * 8, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    i = tile * 4;
                    if (i < n_blk) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
                            : "r"(slot_addr + 0));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        c = split + i * splits;
                        d = q_pos - (c * page_size + row_off);
                        nvis = 0;
                        if (d >= 0) {
                            nvis = (d >> row_shift) + 1;
                            if (nvis > 32) {
                                nvis = 32;
                            }
                        }
                        if (nvis < 32) {
                            uint32_t _thresh_mask_0;
                            {
                                int _lim_1 = nvis;
                                if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                                else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                                }
                            }
                            if (!(_thresh_mask_0 & (1u << 0))) acc[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 1))) acc[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 2))) acc[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 3))) acc[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 4))) acc[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 5))) acc[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 6))) acc[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 7))) acc[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 8))) acc[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 9))) acc[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 10))) acc[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 11))) acc[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 12))) acc[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 13))) acc[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 14))) acc[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 15))) acc[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 16))) acc[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 17))) acc[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 18))) acc[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 19))) acc[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 20))) acc[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 21))) acc[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 22))) acc[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 23))) acc[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 24))) acc[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 25))) acc[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 26))) acc[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 27))) acc[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 28))) acc[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 29))) acc[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 30))) acc[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 31))) acc[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_3 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[0], acc[1]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[2], acc[3]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[4], acc[5]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[6], acc[7]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[8], acc[9]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[10], acc[11]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[12], acc[13]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[14], acc[15]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[16], acc[17]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[18], acc[19]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[20], acc[21]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[22], acc[23]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[24], acc[25]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[26], acc[27]));
                        _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(acc[28], acc[29]));
                        _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(acc[30], acc[31]));
                        float _reg_array_reduce_2 = row_max_reduce(_reg_reduce_max2_3);
                        m = _reg_array_reduce_2;
                        if (m != m) {
                            m = -CUDART_INF_F;
                        }
                        if (m == 0.0f) {
                            m = 0.0f;
                        }
                        if (live != 0) {
                            scores[row_base + c] = m;
                        }
                    }
                    i = tile * 4 + 1;
                    if (i < n_blk) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
                            : "r"(slot_addr + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        c = split + i * splits;
                        d = q_pos - (c * page_size + row_off);
                        nvis = 0;
                        if (d >= 0) {
                            nvis = (d >> row_shift) + 1;
                            if (nvis > 32) {
                                nvis = 32;
                            }
                        }
                        if (nvis < 32) {
                            uint32_t _thresh_mask_4;
                            {
                                int _lim_5 = nvis;
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
                            if (!(_thresh_mask_4 & (1u << 0))) acc[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 1))) acc[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 2))) acc[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 3))) acc[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 4))) acc[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 5))) acc[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 6))) acc[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 7))) acc[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 8))) acc[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 9))) acc[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 10))) acc[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 11))) acc[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 12))) acc[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 13))) acc[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 14))) acc[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 15))) acc[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 16))) acc[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 17))) acc[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 18))) acc[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 19))) acc[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 20))) acc[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 21))) acc[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 22))) acc[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 23))) acc[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 24))) acc[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 25))) acc[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 26))) acc[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 27))) acc[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 28))) acc[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 29))) acc[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 30))) acc[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 31))) acc[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_7 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[0], acc[1]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[2], acc[3]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[4], acc[5]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[6], acc[7]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[8], acc[9]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[10], acc[11]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[12], acc[13]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[14], acc[15]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[16], acc[17]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[18], acc[19]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[20], acc[21]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[22], acc[23]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[24], acc[25]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[26], acc[27]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(acc[28], acc[29]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(acc[30], acc[31]));
                        float _reg_array_reduce_6 = row_max_reduce(_reg_reduce_max2_7);
                        m = _reg_array_reduce_6;
                        if (m != m) {
                            m = -CUDART_INF_F;
                        }
                        if (m == 0.0f) {
                            m = 0.0f;
                        }
                        if (live != 0) {
                            scores[row_base + c] = m;
                        }
                    }
                    i = tile * 4 + 2;
                    if (i < n_blk) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
                            : "r"(slot_addr + 64));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        c = split + i * splits;
                        d = q_pos - (c * page_size + row_off);
                        nvis = 0;
                        if (d >= 0) {
                            nvis = (d >> row_shift) + 1;
                            if (nvis > 32) {
                                nvis = 32;
                            }
                        }
                        if (nvis < 32) {
                            uint32_t _thresh_mask_8;
                            {
                                int _lim_9 = nvis;
                                if (_lim_9 <= 0) { _thresh_mask_8 = 0u; }
                                else if (_lim_9 >= 32) { _thresh_mask_8 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_8) : "r"(_lim_9));
                                }
                            }
                            if (!(_thresh_mask_8 & (1u << 0))) acc[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 1))) acc[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 2))) acc[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 3))) acc[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 4))) acc[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 5))) acc[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 6))) acc[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 7))) acc[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 8))) acc[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 9))) acc[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 10))) acc[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 11))) acc[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 12))) acc[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 13))) acc[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 14))) acc[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 15))) acc[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 16))) acc[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 17))) acc[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 18))) acc[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 19))) acc[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 20))) acc[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 21))) acc[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 22))) acc[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 23))) acc[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 24))) acc[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 25))) acc[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 26))) acc[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 27))) acc[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 28))) acc[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 29))) acc[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 30))) acc[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 31))) acc[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_11 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[0], acc[1]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[2], acc[3]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[4], acc[5]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[6], acc[7]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[8], acc[9]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[10], acc[11]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[12], acc[13]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[14], acc[15]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[16], acc[17]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[18], acc[19]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[20], acc[21]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[22], acc[23]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[24], acc[25]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[26], acc[27]));
                        _reg_reduce_max2_11.x = max_noftz(_reg_reduce_max2_11.x, max_noftz(acc[28], acc[29]));
                        _reg_reduce_max2_11.y = max_noftz(_reg_reduce_max2_11.y, max_noftz(acc[30], acc[31]));
                        float _reg_array_reduce_10 = row_max_reduce(_reg_reduce_max2_11);
                        m = _reg_array_reduce_10;
                        if (m != m) {
                            m = -CUDART_INF_F;
                        }
                        if (m == 0.0f) {
                            m = 0.0f;
                        }
                        if (live != 0) {
                            scores[row_base + c] = m;
                        }
                    }
                    i = tile * 4 + 3;
                    if (i < n_blk) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3]), "=f"(acc[4]), "=f"(acc[5]), "=f"(acc[6]), "=f"(acc[7]), "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15]), "=f"(acc[16]), "=f"(acc[17]), "=f"(acc[18]), "=f"(acc[19]), "=f"(acc[20]), "=f"(acc[21]), "=f"(acc[22]), "=f"(acc[23]), "=f"(acc[24]), "=f"(acc[25]), "=f"(acc[26]), "=f"(acc[27]), "=f"(acc[28]), "=f"(acc[29]), "=f"(acc[30]), "=f"(acc[31])
                            : "r"(slot_addr + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        c = split + i * splits;
                        d = q_pos - (c * page_size + row_off);
                        nvis = 0;
                        if (d >= 0) {
                            nvis = (d >> row_shift) + 1;
                            if (nvis > 32) {
                                nvis = 32;
                            }
                        }
                        if (nvis < 32) {
                            uint32_t _thresh_mask_12;
                            {
                                int _lim_13 = nvis;
                                if (_lim_13 <= 0) { _thresh_mask_12 = 0u; }
                                else if (_lim_13 >= 32) { _thresh_mask_12 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_12) : "r"(_lim_13));
                                }
                            }
                            if (!(_thresh_mask_12 & (1u << 0))) acc[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 1))) acc[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 2))) acc[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 3))) acc[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 4))) acc[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 5))) acc[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 6))) acc[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 7))) acc[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 8))) acc[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 9))) acc[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 10))) acc[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 11))) acc[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 12))) acc[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 13))) acc[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 14))) acc[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 15))) acc[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 16))) acc[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 17))) acc[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 18))) acc[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 19))) acc[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 20))) acc[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 21))) acc[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 22))) acc[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 23))) acc[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 24))) acc[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 25))) acc[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 26))) acc[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 27))) acc[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 28))) acc[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 29))) acc[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 30))) acc[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_12 & (1u << 31))) acc[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_15 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[0], acc[1]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[2], acc[3]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[4], acc[5]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[6], acc[7]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[8], acc[9]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[10], acc[11]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[12], acc[13]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[14], acc[15]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[16], acc[17]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[18], acc[19]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[20], acc[21]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[22], acc[23]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[24], acc[25]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[26], acc[27]));
                        _reg_reduce_max2_15.x = max_noftz(_reg_reduce_max2_15.x, max_noftz(acc[28], acc[29]));
                        _reg_reduce_max2_15.y = max_noftz(_reg_reduce_max2_15.y, max_noftz(acc[30], acc[31]));
                        float _reg_array_reduce_14 = row_max_reduce(_reg_reduce_max2_15);
                        m = _reg_array_reduce_14;
                        if (m != m) {
                            m = -CUDART_INF_F;
                        }
                        if (m == 0.0f) {
                            m = 0.0f;
                        }
                        if (live != 0) {
                            scores[row_base + c] = m;
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(acc_empty_addr + (w) * 8);
                    _phase_acc_full ^= 1;
                }
            }
            if (w == 0 && live != 0) {
                int lim = (ncols + 3) / 4 * 4;
                if (lim > columns) {
                    lim = columns;
                }
                int cpad = 0;
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    cpad = split + (n_blk + p) * splits;
                    if (cpad < lim) {
                        scores[row_base + cpad] = -CUDART_INF_F;
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int b_1 = blockIdx.x;
            int split_1 = blockIdx.y;
            int seqlen_1 = seq_lens[b_1];
            int ncols_1 = 0;
            if (seqlen_1 > 0) {
                ncols_1 = (seqlen_1 + page_size - 1) / page_size;
            }
            if (ncols_1 > columns) {
                ncols_1 = columns;
            }
            int n_blk_1 = 0;
            if (split_1 < ncols_1) {
                n_blk_1 = (ncols_1 - split_1 + splits - 1) / splits;
            }
            int n_tiles_1 = (n_blk_1 + 3) / 4;
            int lane_0_1 = (int)lane;
            int bt_base = b_1 * bt_stride;
            int ids = 0;
            int ids_next = 0;
            int gi = 0;
            if (lane_0_1 < n_blk_1) {
                ids = block_table[bt_base + split_1 + lane_0_1 * splits];
            }
            unsigned int _phase_k_empty = 1;
            if (n_tiles_1 > 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    tma_2d_gmem2smem(smem_q_addr + 0, (&q), 0, b_1 * query_len * num_heads, q_full_addr);
                    tma_2d_gmem2smem(smem_q_addr + 4096, (&q), 0, b_1 * query_len * num_heads, q_full_addr);
                    tma_2d_gmem2smem(smem_q_addr + 8192, (&q), 0, b_1 * query_len * num_heads, q_full_addr);
                    tma_2d_gmem2smem(smem_q_addr + 12288, (&q), 0, b_1 * query_len * num_heads, q_full_addr);
                }
                unsigned int k_stage = 0;
                int i0 = 0;
                int nb = 0;
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < n_tiles_1; tile_1++) {
                    i0 = tile_1 * 4;
                    if (i0 % 32 == 0) {
                        gi = i0 + 32 + lane_0_1;
                        ids_next = 0;
                        if (gi < n_blk_1) {
                            ids_next = block_table[bt_base + split_1 + gi * splits];
                        }
                    }
                    nb = n_blk_1 - i0;
                    if (nb > 4) {
                        nb = 4;
                    }
                    int pg_0 = __shfl_sync(0xFFFFFFFF, ids, i0 % 32);
                    int pg_1 = __shfl_sync(0xFFFFFFFF, ids, i0 % 32 + 1);
                    int pg_2 = __shfl_sync(0xFFFFFFFF, ids, i0 % 32 + 2);
                    int pg_3 = __shfl_sync(0xFFFFFFFF, ids, i0 % 32 + 3);
                    mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (k_stage) * 8, (unsigned int)nb * 4096);
                        if (nb > 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 16384 + 0), "l"((&k)), "r"(0), "r"(0), "r"(pg_0),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (nb > 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 16384 + 4096), "l"((&k)), "r"(0), "r"(0), "r"(pg_1),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (nb > 2) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 16384 + 8192), "l"((&k)), "r"(0), "r"(0), "r"(pg_2),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (nb > 3) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 16384 + 12288), "l"((&k)), "r"(0), "r"(0), "r"(pg_3),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    if ((i0 + 4) % 32 == 0) {
                        ids = ids_next;
                    }
                    k_stage += 1;
                    if (k_stage == 12) { k_stage = 0; _phase_k_empty ^= 1; }
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int b_2 = blockIdx.x;
            int split_2 = blockIdx.y;
            int seqlen_2 = seq_lens[b_2];
            int ncols_2 = 0;
            if (seqlen_2 > 0) {
                ncols_2 = (seqlen_2 + page_size - 1) / page_size;
            }
            if (ncols_2 > columns) {
                ncols_2 = columns;
            }
            int n_blk_2 = 0;
            if (split_2 < ncols_2) {
                n_blk_2 = (ncols_2 - split_2 + splits - 1) / splits;
            }
            int n_tiles_2 = (n_blk_2 + 3) / 4;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_mma_done_0 = 0;
            if (n_tiles_2 > 0) {
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int k_stage_1 = 0;
                unsigned int slot = 0;
                #pragma unroll 1
                for (int tile_2 = 0; tile_2 < n_tiles_2; tile_2++) {
                    mbarrier_wait(k_full_addr + (k_stage_1) * 8, _phase_k_full);
                    mbarrier_wait(acc_empty_addr + (slot) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int b_lo = make_warp_uniform((((smem_k_addr) >> 4) & 0x7FFF) + (k_stage_1) * 1024);
                    int a_lo = make_warp_uniform(((smem_q_addr) >> 4) & 0x7FFF);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 673185808;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 4;\n\t"
                    "add.u32 blo, blo, 4;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(a_lo), "r"(b_lo), "r"((tmem_acc + (slot * 128))), "r"(0) : "memory");
                    elect_commit2(acc_full_addr + (slot) * 8, k_empty_addr + (k_stage_1) * 8);
                    k_stage_1 += 1;
                    if (k_stage_1 == 12) { k_stage_1 = 0; _phase_k_full ^= 1; }
                    slot += 1;
                    if (slot == 4) { slot = 0; _phase_acc_empty ^= 1; }
                }
                elect_commit(mma_done_addr);
                mbarrier_wait(mma_done_addr, _phase_mma_done_0);
                _phase_mma_done_0 ^= 1;
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
