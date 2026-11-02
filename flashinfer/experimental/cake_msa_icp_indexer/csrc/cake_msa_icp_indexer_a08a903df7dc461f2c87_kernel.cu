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
#define TMEM_NCOLS 256
#define TMEM_ACC_OFFSET 0
#define NUM_K_PIPE_STAGES 6
#define NUM_ACC_PIPE_STAGES 4
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 16384
#define SMEM_SMEM_Q_STRIDE 16384
#define SMEM_SMEM_K_OFF 17408
#define SMEM_SMEM_K_STAGE_BYTES 8192
#define SMEM_SMEM_K_STRIDE 8192
#define SMEM_LISTS_OFF 1024
#define SMEM_LISTS_STAGE_BYTES 16384
#define SMEM_LISTS_STRIDE 16384
#define SMEM_TOTAL 78848
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
kernel_cake_msa_icp_indexer_a08a903df7dc461f2c87(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, int* __restrict__ block_table, int* __restrict__ seq_lens, int* __restrict__ num_valid_pages, unsigned long long* __restrict__ pool, int bt_stride, int pool_row_stride, int num_heads, int query_len, int columns, int splits, int rank, int world, int page_size, int init_blocks, int local_blocks)
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
    #define k_empty_addr (mbar_base + 64)
    #define acc_full_addr (mbar_base + 112)
    #define acc_empty_addr (mbar_base + 144)

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

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 22 barriers)
    // Mbarriers at smem_raw[0..176)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // mma_done: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 6 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // k_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 4 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // acc_empty: 4 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 176);
    if (warp == 0) {
        int _tmem_hold = smem + 176;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
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
            int n_tiles = (n_blk + 1) / 2;
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
            int lc = 0;
            int fe0 = 0;
            if (live != 0) {
                int ng = num_valid_pages[t];
                if (ng < 0) {
                    ng = 0;
                }
                if (ng > 8192) {
                    ng = 8192;
                }
                if (ng > columns << idx_shift) {
                    ng = columns << idx_shift;
                }
                if (ng > idx_off) {
                    lc = (ng - 1 - idx_off >> idx_shift) + 1;
                }
                fe0 = ng - local_blocks;
                if (fe0 < 0) {
                    fe0 = 0;
                }
            }
            unsigned long long top[16];
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
            unsigned long long buf[8];
            buf[0] = 0;
            buf[1] = 0;
            buf[2] = 0;
            buf[3] = 0;
            buf[4] = 0;
            buf[5] = 0;
            buf[6] = 0;
            buf[7] = 0;
            int cnt = 0;
            unsigned long long cand = 0;
            int need = 0;
            unsigned int mb = 0;
            unsigned int ok = 0;
            int gid = 0;
            unsigned long long forced = 0;
            unsigned int _phase_acc_full = 0;
            if (n_tiles > 0 && w < 4) {
                unsigned int slot_addr = taddr + (unsigned int)(w * 32 << 16) + (unsigned int)(w * 64);
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
                    i = tile * 2;
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
                        need = 0;
                        cand = 0;
                        if (live != 0 && c < lc) {
                            gid = (c << idx_shift) + idx_off;
                            forced = 0;
                            if (gid < init_blocks || gid >= fe0) {
                                forced = 1;
                            }
                            mb = reinterpret_cast<unsigned int*>(&m)[0];
                            ok = (((mb & 2147483648u) != 0) ? ~mb : mb | 2147483648u);
                            cand = forced << 63 | (unsigned long long)ok << 31 | (unsigned long long)(2147483647 - gid);
                            if (cand > top[15]) {
                                need = 1;
                            }
                        }
                        int _warp_vote_4 = __any_sync(0xFFFFFFFF, need != 0);
                        if (_warp_vote_4 != 0) {
                            buf[0] = ((need != 0 && cnt == 0) ? cand : buf[0]);
                            buf[1] = ((need != 0 && cnt == 1) ? cand : buf[1]);
                            buf[2] = ((need != 0 && cnt == 2) ? cand : buf[2]);
                            buf[3] = ((need != 0 && cnt == 3) ? cand : buf[3]);
                            buf[4] = ((need != 0 && cnt == 4) ? cand : buf[4]);
                            buf[5] = ((need != 0 && cnt == 5) ? cand : buf[5]);
                            buf[6] = ((need != 0 && cnt == 6) ? cand : buf[6]);
                            buf[7] = ((need != 0 && cnt == 7) ? cand : buf[7]);
                            cnt = cnt + need;
                            int _warp_vote_5 = __any_sync(0xFFFFFFFF, cnt == 8);
                            if (_warp_vote_5 != 0) {
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
                                buf[0] = 0;
                                buf[1] = 0;
                                buf[2] = 0;
                                buf[3] = 0;
                                buf[4] = 0;
                                buf[5] = 0;
                                buf[6] = 0;
                                buf[7] = 0;
                                cnt = 0;
                            }
                        }
                    }
                    i = tile * 2 + 1;
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
                            uint32_t _thresh_mask_6;
                            {
                                int _lim_7 = nvis;
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
                            if (!(_thresh_mask_6 & (1u << 0))) acc[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 1))) acc[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 2))) acc[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 3))) acc[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 4))) acc[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 5))) acc[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 6))) acc[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 7))) acc[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 8))) acc[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 9))) acc[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 10))) acc[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 11))) acc[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 12))) acc[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 13))) acc[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 14))) acc[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 15))) acc[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 16))) acc[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 17))) acc[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 18))) acc[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 19))) acc[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 20))) acc[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 21))) acc[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 22))) acc[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 23))) acc[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 24))) acc[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 25))) acc[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 26))) acc[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 27))) acc[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 28))) acc[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 29))) acc[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 30))) acc[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 31))) acc[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_9 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[0], acc[1]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[2], acc[3]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[4], acc[5]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[6], acc[7]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[8], acc[9]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[10], acc[11]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[12], acc[13]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[14], acc[15]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[16], acc[17]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[18], acc[19]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[20], acc[21]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[22], acc[23]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[24], acc[25]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[26], acc[27]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(acc[28], acc[29]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(acc[30], acc[31]));
                        float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                        m = _reg_array_reduce_8;
                        if (m != m) {
                            m = -CUDART_INF_F;
                        }
                        if (m == 0.0f) {
                            m = 0.0f;
                        }
                        need = 0;
                        cand = 0;
                        if (live != 0 && c < lc) {
                            gid = (c << idx_shift) + idx_off;
                            forced = 0;
                            if (gid < init_blocks || gid >= fe0) {
                                forced = 1;
                            }
                            mb = reinterpret_cast<unsigned int*>(&m)[0];
                            ok = (((mb & 2147483648u) != 0) ? ~mb : mb | 2147483648u);
                            cand = forced << 63 | (unsigned long long)ok << 31 | (unsigned long long)(2147483647 - gid);
                            if (cand > top[15]) {
                                need = 1;
                            }
                        }
                        int _warp_vote_10 = __any_sync(0xFFFFFFFF, need != 0);
                        if (_warp_vote_10 != 0) {
                            buf[0] = ((need != 0 && cnt == 0) ? cand : buf[0]);
                            buf[1] = ((need != 0 && cnt == 1) ? cand : buf[1]);
                            buf[2] = ((need != 0 && cnt == 2) ? cand : buf[2]);
                            buf[3] = ((need != 0 && cnt == 3) ? cand : buf[3]);
                            buf[4] = ((need != 0 && cnt == 4) ? cand : buf[4]);
                            buf[5] = ((need != 0 && cnt == 5) ? cand : buf[5]);
                            buf[6] = ((need != 0 && cnt == 6) ? cand : buf[6]);
                            buf[7] = ((need != 0 && cnt == 7) ? cand : buf[7]);
                            cnt = cnt + need;
                            int _warp_vote_11 = __any_sync(0xFFFFFFFF, cnt == 8);
                            if (_warp_vote_11 != 0) {
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
                                buf[0] = 0;
                                buf[1] = 0;
                                buf[2] = 0;
                                buf[3] = 0;
                                buf[4] = 0;
                                buf[5] = 0;
                                buf[6] = 0;
                                buf[7] = 0;
                                cnt = 0;
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(acc_empty_addr + (w) * 8);
                    _phase_acc_full ^= 1;
                }
                int _warp_vote_12 = __any_sync(0xFFFFFFFF, cnt != 0);
                if (_warp_vote_12 != 0) {
                    unsigned long long x_100 = buf[0];
                    unsigned long long y_116 = buf[1];
                    buf[0] = ((y_116 > x_100) ? y_116 : x_100);
                    buf[1] = ((y_116 > x_100) ? x_100 : y_116);
                    unsigned long long x_100_1 = buf[2];
                    unsigned long long y_116_1 = buf[3];
                    buf[2] = ((y_116_1 > x_100_1) ? y_116_1 : x_100_1);
                    buf[3] = ((y_116_1 > x_100_1) ? x_100_1 : y_116_1);
                    unsigned long long x_101 = buf[4];
                    unsigned long long y_117 = buf[5];
                    buf[4] = ((y_117 > x_101) ? y_117 : x_101);
                    buf[5] = ((y_117 > x_101) ? x_101 : y_117);
                    unsigned long long x_102 = buf[6];
                    unsigned long long y_118 = buf[7];
                    buf[6] = ((y_118 > x_102) ? y_118 : x_102);
                    buf[7] = ((y_118 > x_102) ? x_102 : y_118);
                    unsigned long long x_103 = buf[0];
                    unsigned long long y_119 = buf[2];
                    buf[0] = ((y_119 > x_103) ? y_119 : x_103);
                    buf[2] = ((y_119 > x_103) ? x_103 : y_119);
                    unsigned long long x_104 = buf[1];
                    unsigned long long y_120 = buf[3];
                    buf[1] = ((y_120 > x_104) ? y_120 : x_104);
                    buf[3] = ((y_120 > x_104) ? x_104 : y_120);
                    unsigned long long x_105 = buf[4];
                    unsigned long long y_121 = buf[6];
                    buf[4] = ((y_121 > x_105) ? y_121 : x_105);
                    buf[6] = ((y_121 > x_105) ? x_105 : y_121);
                    unsigned long long x_106 = buf[5];
                    unsigned long long y_122 = buf[7];
                    buf[5] = ((y_122 > x_106) ? y_122 : x_106);
                    buf[7] = ((y_122 > x_106) ? x_106 : y_122);
                    unsigned long long x_107 = buf[1];
                    unsigned long long y_123 = buf[2];
                    buf[1] = ((y_123 > x_107) ? y_123 : x_107);
                    buf[2] = ((y_123 > x_107) ? x_107 : y_123);
                    unsigned long long x_108 = buf[5];
                    unsigned long long y_124 = buf[6];
                    buf[5] = ((y_124 > x_108) ? y_124 : x_108);
                    buf[6] = ((y_124 > x_108) ? x_108 : y_124);
                    unsigned long long x_109 = buf[0];
                    unsigned long long y_125 = buf[4];
                    buf[0] = ((y_125 > x_109) ? y_125 : x_109);
                    buf[4] = ((y_125 > x_109) ? x_109 : y_125);
                    unsigned long long x_110 = buf[1];
                    unsigned long long y_126 = buf[5];
                    buf[1] = ((y_126 > x_110) ? y_126 : x_110);
                    buf[5] = ((y_126 > x_110) ? x_110 : y_126);
                    unsigned long long x_111 = buf[2];
                    unsigned long long y_127 = buf[6];
                    buf[2] = ((y_127 > x_111) ? y_127 : x_111);
                    buf[6] = ((y_127 > x_111) ? x_111 : y_127);
                    unsigned long long x_112 = buf[3];
                    unsigned long long y_128 = buf[7];
                    buf[3] = ((y_128 > x_112) ? y_128 : x_112);
                    buf[7] = ((y_128 > x_112) ? x_112 : y_128);
                    unsigned long long x_113 = buf[2];
                    unsigned long long y_129 = buf[4];
                    buf[2] = ((y_129 > x_113) ? y_129 : x_113);
                    buf[4] = ((y_129 > x_113) ? x_113 : y_129);
                    unsigned long long x_114 = buf[3];
                    unsigned long long y_130 = buf[5];
                    buf[3] = ((y_130 > x_114) ? y_130 : x_114);
                    buf[5] = ((y_130 > x_114) ? x_114 : y_130);
                    unsigned long long x_115 = buf[1];
                    unsigned long long y_131 = buf[2];
                    buf[1] = ((y_131 > x_115) ? y_131 : x_115);
                    buf[2] = ((y_131 > x_115) ? x_115 : y_131);
                    unsigned long long x_116 = buf[3];
                    unsigned long long y_132 = buf[4];
                    buf[3] = ((y_132 > x_116) ? y_132 : x_116);
                    buf[4] = ((y_132 > x_116) ? x_116 : y_132);
                    unsigned long long x_117 = buf[5];
                    unsigned long long y_133 = buf[6];
                    buf[5] = ((y_133 > x_117) ? y_133 : x_117);
                    buf[6] = ((y_133 > x_117) ? x_117 : y_133);
                    unsigned long long y_134 = buf[0];
                    top[15] = ((y_134 > top[15]) ? y_134 : top[15]);
                    unsigned long long y_135 = buf[1];
                    top[14] = ((y_135 > top[14]) ? y_135 : top[14]);
                    unsigned long long y_136 = buf[2];
                    top[13] = ((y_136 > top[13]) ? y_136 : top[13]);
                    unsigned long long y_137 = buf[3];
                    top[12] = ((y_137 > top[12]) ? y_137 : top[12]);
                    unsigned long long y_138 = buf[4];
                    top[11] = ((y_138 > top[11]) ? y_138 : top[11]);
                    unsigned long long y_139 = buf[5];
                    top[10] = ((y_139 > top[10]) ? y_139 : top[10]);
                    unsigned long long y_140 = buf[6];
                    top[9] = ((y_140 > top[9]) ? y_140 : top[9]);
                    unsigned long long y_141 = buf[7];
                    top[8] = ((y_141 > top[8]) ? y_141 : top[8]);
                    unsigned long long x_118 = top[0];
                    unsigned long long y_142 = top[8];
                    top[0] = ((y_142 > x_118) ? y_142 : x_118);
                    top[8] = ((y_142 > x_118) ? x_118 : y_142);
                    unsigned long long x_119 = top[1];
                    unsigned long long y_143 = top[9];
                    top[1] = ((y_143 > x_119) ? y_143 : x_119);
                    top[9] = ((y_143 > x_119) ? x_119 : y_143);
                    unsigned long long x_120 = top[2];
                    unsigned long long y_144 = top[10];
                    top[2] = ((y_144 > x_120) ? y_144 : x_120);
                    top[10] = ((y_144 > x_120) ? x_120 : y_144);
                    unsigned long long x_121 = top[3];
                    unsigned long long y_145 = top[11];
                    top[3] = ((y_145 > x_121) ? y_145 : x_121);
                    top[11] = ((y_145 > x_121) ? x_121 : y_145);
                    unsigned long long x_122 = top[4];
                    unsigned long long y_146 = top[12];
                    top[4] = ((y_146 > x_122) ? y_146 : x_122);
                    top[12] = ((y_146 > x_122) ? x_122 : y_146);
                    unsigned long long x_123 = top[5];
                    unsigned long long y_147 = top[13];
                    top[5] = ((y_147 > x_123) ? y_147 : x_123);
                    top[13] = ((y_147 > x_123) ? x_123 : y_147);
                    unsigned long long x_124 = top[6];
                    unsigned long long y_148 = top[14];
                    top[6] = ((y_148 > x_124) ? y_148 : x_124);
                    top[14] = ((y_148 > x_124) ? x_124 : y_148);
                    unsigned long long x_125 = top[7];
                    unsigned long long y_149 = top[15];
                    top[7] = ((y_149 > x_125) ? y_149 : x_125);
                    top[15] = ((y_149 > x_125) ? x_125 : y_149);
                    unsigned long long x_126 = top[0];
                    unsigned long long y_150 = top[4];
                    top[0] = ((y_150 > x_126) ? y_150 : x_126);
                    top[4] = ((y_150 > x_126) ? x_126 : y_150);
                    unsigned long long x_127 = top[1];
                    unsigned long long y_151 = top[5];
                    top[1] = ((y_151 > x_127) ? y_151 : x_127);
                    top[5] = ((y_151 > x_127) ? x_127 : y_151);
                    unsigned long long x_128 = top[2];
                    unsigned long long y_152 = top[6];
                    top[2] = ((y_152 > x_128) ? y_152 : x_128);
                    top[6] = ((y_152 > x_128) ? x_128 : y_152);
                    unsigned long long x_129 = top[3];
                    unsigned long long y_153 = top[7];
                    top[3] = ((y_153 > x_129) ? y_153 : x_129);
                    top[7] = ((y_153 > x_129) ? x_129 : y_153);
                    unsigned long long x_130 = top[8];
                    unsigned long long y_154 = top[12];
                    top[8] = ((y_154 > x_130) ? y_154 : x_130);
                    top[12] = ((y_154 > x_130) ? x_130 : y_154);
                    unsigned long long x_131 = top[9];
                    unsigned long long y_155 = top[13];
                    top[9] = ((y_155 > x_131) ? y_155 : x_131);
                    top[13] = ((y_155 > x_131) ? x_131 : y_155);
                    unsigned long long x_132 = top[10];
                    unsigned long long y_156 = top[14];
                    top[10] = ((y_156 > x_132) ? y_156 : x_132);
                    top[14] = ((y_156 > x_132) ? x_132 : y_156);
                    unsigned long long x_133 = top[11];
                    unsigned long long y_157 = top[15];
                    top[11] = ((y_157 > x_133) ? y_157 : x_133);
                    top[15] = ((y_157 > x_133) ? x_133 : y_157);
                    unsigned long long x_134 = top[0];
                    unsigned long long y_158 = top[2];
                    top[0] = ((y_158 > x_134) ? y_158 : x_134);
                    top[2] = ((y_158 > x_134) ? x_134 : y_158);
                    unsigned long long x_135 = top[1];
                    unsigned long long y_159 = top[3];
                    top[1] = ((y_159 > x_135) ? y_159 : x_135);
                    top[3] = ((y_159 > x_135) ? x_135 : y_159);
                    unsigned long long x_136 = top[4];
                    unsigned long long y_160 = top[6];
                    top[4] = ((y_160 > x_136) ? y_160 : x_136);
                    top[6] = ((y_160 > x_136) ? x_136 : y_160);
                    unsigned long long x_137 = top[5];
                    unsigned long long y_161 = top[7];
                    top[5] = ((y_161 > x_137) ? y_161 : x_137);
                    top[7] = ((y_161 > x_137) ? x_137 : y_161);
                    unsigned long long x_138 = top[8];
                    unsigned long long y_162 = top[10];
                    top[8] = ((y_162 > x_138) ? y_162 : x_138);
                    top[10] = ((y_162 > x_138) ? x_138 : y_162);
                    unsigned long long x_139 = top[9];
                    unsigned long long y_163 = top[11];
                    top[9] = ((y_163 > x_139) ? y_163 : x_139);
                    top[11] = ((y_163 > x_139) ? x_139 : y_163);
                    unsigned long long x_140 = top[12];
                    unsigned long long y_164 = top[14];
                    top[12] = ((y_164 > x_140) ? y_164 : x_140);
                    top[14] = ((y_164 > x_140) ? x_140 : y_164);
                    unsigned long long x_141 = top[13];
                    unsigned long long y_165 = top[15];
                    top[13] = ((y_165 > x_141) ? y_165 : x_141);
                    top[15] = ((y_165 > x_141) ? x_141 : y_165);
                    unsigned long long x_142 = top[0];
                    unsigned long long y_166 = top[1];
                    top[0] = ((y_166 > x_142) ? y_166 : x_142);
                    top[1] = ((y_166 > x_142) ? x_142 : y_166);
                    unsigned long long x_143 = top[2];
                    unsigned long long y_167 = top[3];
                    top[2] = ((y_167 > x_143) ? y_167 : x_143);
                    top[3] = ((y_167 > x_143) ? x_143 : y_167);
                    unsigned long long x_144 = top[4];
                    unsigned long long y_168 = top[5];
                    top[4] = ((y_168 > x_144) ? y_168 : x_144);
                    top[5] = ((y_168 > x_144) ? x_144 : y_168);
                    unsigned long long x_145 = top[6];
                    unsigned long long y_169 = top[7];
                    top[6] = ((y_169 > x_145) ? y_169 : x_145);
                    top[7] = ((y_169 > x_145) ? x_145 : y_169);
                    unsigned long long x_146 = top[8];
                    unsigned long long y_170 = top[9];
                    top[8] = ((y_170 > x_146) ? y_170 : x_146);
                    top[9] = ((y_170 > x_146) ? x_146 : y_170);
                    unsigned long long x_147 = top[10];
                    unsigned long long y_171 = top[11];
                    top[10] = ((y_171 > x_147) ? y_171 : x_147);
                    top[11] = ((y_171 > x_147) ? x_147 : y_171);
                    unsigned long long x_148 = top[12];
                    unsigned long long y_172 = top[13];
                    top[12] = ((y_172 > x_148) ? y_172 : x_148);
                    top[13] = ((y_172 > x_148) ? x_148 : y_172);
                    unsigned long long x_149 = top[14];
                    unsigned long long y_173 = top[15];
                    top[14] = ((y_173 > x_149) ? y_173 : x_149);
                    top[15] = ((y_173 > x_149) ? x_149 : y_173);
                    buf[0] = 0;
                    buf[1] = 0;
                    buf[2] = 0;
                    buf[3] = 0;
                    buf[4] = 0;
                    buf[5] = 0;
                    buf[6] = 0;
                    buf[7] = 0;
                }
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w != 0) {
                lists[w * 16 * 32 + lane_0] = top[0];
                lists[(w * 16 + 1) * 32 + lane_0] = top[1];
                lists[(w * 16 + 2) * 32 + lane_0] = top[2];
                lists[(w * 16 + 3) * 32 + lane_0] = top[3];
                lists[(w * 16 + 4) * 32 + lane_0] = top[4];
                lists[(w * 16 + 5) * 32 + lane_0] = top[5];
                lists[(w * 16 + 6) * 32 + lane_0] = top[6];
                lists[(w * 16 + 7) * 32 + lane_0] = top[7];
                lists[(w * 16 + 8) * 32 + lane_0] = top[8];
                lists[(w * 16 + 9) * 32 + lane_0] = top[9];
                lists[(w * 16 + 10) * 32 + lane_0] = top[10];
                lists[(w * 16 + 11) * 32 + lane_0] = top[11];
                lists[(w * 16 + 12) * 32 + lane_0] = top[12];
                lists[(w * 16 + 13) * 32 + lane_0] = top[13];
                lists[(w * 16 + 14) * 32 + lane_0] = top[14];
                lists[(w * 16 + 15) * 32 + lane_0] = top[15];
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w == 0) {
                unsigned long long other[16];
                other[0] = lists[512 + lane_0];
                other[1] = lists[544 + lane_0];
                other[2] = lists[576 + lane_0];
                other[3] = lists[608 + lane_0];
                other[4] = lists[640 + lane_0];
                other[5] = lists[672 + lane_0];
                other[6] = lists[704 + lane_0];
                other[7] = lists[736 + lane_0];
                other[8] = lists[768 + lane_0];
                other[9] = lists[800 + lane_0];
                other[10] = lists[832 + lane_0];
                other[11] = lists[864 + lane_0];
                other[12] = lists[896 + lane_0];
                other[13] = lists[928 + lane_0];
                other[14] = lists[960 + lane_0];
                other[15] = lists[992 + lane_0];
                unsigned long long y_174 = other[15];
                top[0] = ((y_174 > top[0]) ? y_174 : top[0]);
                unsigned long long y_175 = other[14];
                top[1] = ((y_175 > top[1]) ? y_175 : top[1]);
                unsigned long long y_176 = other[13];
                top[2] = ((y_176 > top[2]) ? y_176 : top[2]);
                unsigned long long y_177 = other[12];
                top[3] = ((y_177 > top[3]) ? y_177 : top[3]);
                unsigned long long y_178 = other[11];
                top[4] = ((y_178 > top[4]) ? y_178 : top[4]);
                unsigned long long y_179 = other[10];
                top[5] = ((y_179 > top[5]) ? y_179 : top[5]);
                unsigned long long y_180 = other[9];
                top[6] = ((y_180 > top[6]) ? y_180 : top[6]);
                unsigned long long y_181 = other[8];
                top[7] = ((y_181 > top[7]) ? y_181 : top[7]);
                unsigned long long y_182 = other[7];
                top[8] = ((y_182 > top[8]) ? y_182 : top[8]);
                unsigned long long y_183 = other[6];
                top[9] = ((y_183 > top[9]) ? y_183 : top[9]);
                unsigned long long y_184 = other[5];
                top[10] = ((y_184 > top[10]) ? y_184 : top[10]);
                unsigned long long y_185 = other[4];
                top[11] = ((y_185 > top[11]) ? y_185 : top[11]);
                unsigned long long y_186 = other[3];
                top[12] = ((y_186 > top[12]) ? y_186 : top[12]);
                unsigned long long y_187 = other[2];
                top[13] = ((y_187 > top[13]) ? y_187 : top[13]);
                unsigned long long y_188 = other[1];
                top[14] = ((y_188 > top[14]) ? y_188 : top[14]);
                unsigned long long y_189 = other[0];
                top[15] = ((y_189 > top[15]) ? y_189 : top[15]);
                unsigned long long x_150 = top[0];
                unsigned long long y_190 = top[8];
                top[0] = ((y_190 > x_150) ? y_190 : x_150);
                top[8] = ((y_190 > x_150) ? x_150 : y_190);
                unsigned long long x_150_1 = top[1];
                unsigned long long y_190_1 = top[9];
                top[1] = ((y_190_1 > x_150_1) ? y_190_1 : x_150_1);
                top[9] = ((y_190_1 > x_150_1) ? x_150_1 : y_190_1);
                unsigned long long x_151 = top[2];
                unsigned long long y_191 = top[10];
                top[2] = ((y_191 > x_151) ? y_191 : x_151);
                top[10] = ((y_191 > x_151) ? x_151 : y_191);
                unsigned long long x_152 = top[3];
                unsigned long long y_192 = top[11];
                top[3] = ((y_192 > x_152) ? y_192 : x_152);
                top[11] = ((y_192 > x_152) ? x_152 : y_192);
                unsigned long long x_153 = top[4];
                unsigned long long y_193 = top[12];
                top[4] = ((y_193 > x_153) ? y_193 : x_153);
                top[12] = ((y_193 > x_153) ? x_153 : y_193);
                unsigned long long x_154 = top[5];
                unsigned long long y_194 = top[13];
                top[5] = ((y_194 > x_154) ? y_194 : x_154);
                top[13] = ((y_194 > x_154) ? x_154 : y_194);
                unsigned long long x_155 = top[6];
                unsigned long long y_195 = top[14];
                top[6] = ((y_195 > x_155) ? y_195 : x_155);
                top[14] = ((y_195 > x_155) ? x_155 : y_195);
                unsigned long long x_156 = top[7];
                unsigned long long y_196 = top[15];
                top[7] = ((y_196 > x_156) ? y_196 : x_156);
                top[15] = ((y_196 > x_156) ? x_156 : y_196);
                unsigned long long x_157 = top[0];
                unsigned long long y_197 = top[4];
                top[0] = ((y_197 > x_157) ? y_197 : x_157);
                top[4] = ((y_197 > x_157) ? x_157 : y_197);
                unsigned long long x_158 = top[1];
                unsigned long long y_198 = top[5];
                top[1] = ((y_198 > x_158) ? y_198 : x_158);
                top[5] = ((y_198 > x_158) ? x_158 : y_198);
                unsigned long long x_159 = top[2];
                unsigned long long y_199 = top[6];
                top[2] = ((y_199 > x_159) ? y_199 : x_159);
                top[6] = ((y_199 > x_159) ? x_159 : y_199);
                unsigned long long x_160 = top[3];
                unsigned long long y_200 = top[7];
                top[3] = ((y_200 > x_160) ? y_200 : x_160);
                top[7] = ((y_200 > x_160) ? x_160 : y_200);
                unsigned long long x_161 = top[8];
                unsigned long long y_201 = top[12];
                top[8] = ((y_201 > x_161) ? y_201 : x_161);
                top[12] = ((y_201 > x_161) ? x_161 : y_201);
                unsigned long long x_162 = top[9];
                unsigned long long y_202 = top[13];
                top[9] = ((y_202 > x_162) ? y_202 : x_162);
                top[13] = ((y_202 > x_162) ? x_162 : y_202);
                unsigned long long x_163 = top[10];
                unsigned long long y_203 = top[14];
                top[10] = ((y_203 > x_163) ? y_203 : x_163);
                top[14] = ((y_203 > x_163) ? x_163 : y_203);
                unsigned long long x_164 = top[11];
                unsigned long long y_204 = top[15];
                top[11] = ((y_204 > x_164) ? y_204 : x_164);
                top[15] = ((y_204 > x_164) ? x_164 : y_204);
                unsigned long long x_165 = top[0];
                unsigned long long y_205 = top[2];
                top[0] = ((y_205 > x_165) ? y_205 : x_165);
                top[2] = ((y_205 > x_165) ? x_165 : y_205);
                unsigned long long x_166 = top[1];
                unsigned long long y_206 = top[3];
                top[1] = ((y_206 > x_166) ? y_206 : x_166);
                top[3] = ((y_206 > x_166) ? x_166 : y_206);
                unsigned long long x_167 = top[4];
                unsigned long long y_207 = top[6];
                top[4] = ((y_207 > x_167) ? y_207 : x_167);
                top[6] = ((y_207 > x_167) ? x_167 : y_207);
                unsigned long long x_168 = top[5];
                unsigned long long y_208 = top[7];
                top[5] = ((y_208 > x_168) ? y_208 : x_168);
                top[7] = ((y_208 > x_168) ? x_168 : y_208);
                unsigned long long x_169 = top[8];
                unsigned long long y_209 = top[10];
                top[8] = ((y_209 > x_169) ? y_209 : x_169);
                top[10] = ((y_209 > x_169) ? x_169 : y_209);
                unsigned long long x_170 = top[9];
                unsigned long long y_210 = top[11];
                top[9] = ((y_210 > x_170) ? y_210 : x_170);
                top[11] = ((y_210 > x_170) ? x_170 : y_210);
                unsigned long long x_171 = top[12];
                unsigned long long y_211 = top[14];
                top[12] = ((y_211 > x_171) ? y_211 : x_171);
                top[14] = ((y_211 > x_171) ? x_171 : y_211);
                unsigned long long x_172 = top[13];
                unsigned long long y_212 = top[15];
                top[13] = ((y_212 > x_172) ? y_212 : x_172);
                top[15] = ((y_212 > x_172) ? x_172 : y_212);
                unsigned long long x_173 = top[0];
                unsigned long long y_213 = top[1];
                top[0] = ((y_213 > x_173) ? y_213 : x_173);
                top[1] = ((y_213 > x_173) ? x_173 : y_213);
                unsigned long long x_174 = top[2];
                unsigned long long y_214 = top[3];
                top[2] = ((y_214 > x_174) ? y_214 : x_174);
                top[3] = ((y_214 > x_174) ? x_174 : y_214);
                unsigned long long x_175 = top[4];
                unsigned long long y_215 = top[5];
                top[4] = ((y_215 > x_175) ? y_215 : x_175);
                top[5] = ((y_215 > x_175) ? x_175 : y_215);
                unsigned long long x_176 = top[6];
                unsigned long long y_216 = top[7];
                top[6] = ((y_216 > x_176) ? y_216 : x_176);
                top[7] = ((y_216 > x_176) ? x_176 : y_216);
                unsigned long long x_177 = top[8];
                unsigned long long y_217 = top[9];
                top[8] = ((y_217 > x_177) ? y_217 : x_177);
                top[9] = ((y_217 > x_177) ? x_177 : y_217);
                unsigned long long x_178 = top[10];
                unsigned long long y_218 = top[11];
                top[10] = ((y_218 > x_178) ? y_218 : x_178);
                top[11] = ((y_218 > x_178) ? x_178 : y_218);
                unsigned long long x_179 = top[12];
                unsigned long long y_219 = top[13];
                top[12] = ((y_219 > x_179) ? y_219 : x_179);
                top[13] = ((y_219 > x_179) ? x_179 : y_219);
                unsigned long long x_180 = top[14];
                unsigned long long y_220 = top[15];
                top[14] = ((y_220 > x_180) ? y_220 : x_180);
                top[15] = ((y_220 > x_180) ? x_180 : y_220);
                other[0] = lists[1024 + lane_0];
                other[1] = lists[1056 + lane_0];
                other[2] = lists[1088 + lane_0];
                other[3] = lists[1120 + lane_0];
                other[4] = lists[1152 + lane_0];
                other[5] = lists[1184 + lane_0];
                other[6] = lists[1216 + lane_0];
                other[7] = lists[1248 + lane_0];
                other[8] = lists[1280 + lane_0];
                other[9] = lists[1312 + lane_0];
                other[10] = lists[1344 + lane_0];
                other[11] = lists[1376 + lane_0];
                other[12] = lists[1408 + lane_0];
                other[13] = lists[1440 + lane_0];
                other[14] = lists[1472 + lane_0];
                other[15] = lists[1504 + lane_0];
                unsigned long long y_221 = other[15];
                top[0] = ((y_221 > top[0]) ? y_221 : top[0]);
                unsigned long long y_222 = other[14];
                top[1] = ((y_222 > top[1]) ? y_222 : top[1]);
                unsigned long long y_223 = other[13];
                top[2] = ((y_223 > top[2]) ? y_223 : top[2]);
                unsigned long long y_224 = other[12];
                top[3] = ((y_224 > top[3]) ? y_224 : top[3]);
                unsigned long long y_225 = other[11];
                top[4] = ((y_225 > top[4]) ? y_225 : top[4]);
                unsigned long long y_226 = other[10];
                top[5] = ((y_226 > top[5]) ? y_226 : top[5]);
                unsigned long long y_227 = other[9];
                top[6] = ((y_227 > top[6]) ? y_227 : top[6]);
                unsigned long long y_228 = other[8];
                top[7] = ((y_228 > top[7]) ? y_228 : top[7]);
                unsigned long long y_229 = other[7];
                top[8] = ((y_229 > top[8]) ? y_229 : top[8]);
                unsigned long long y_230 = other[6];
                top[9] = ((y_230 > top[9]) ? y_230 : top[9]);
                unsigned long long y_231 = other[5];
                top[10] = ((y_231 > top[10]) ? y_231 : top[10]);
                unsigned long long y_232 = other[4];
                top[11] = ((y_232 > top[11]) ? y_232 : top[11]);
                unsigned long long y_233 = other[3];
                top[12] = ((y_233 > top[12]) ? y_233 : top[12]);
                unsigned long long y_234 = other[2];
                top[13] = ((y_234 > top[13]) ? y_234 : top[13]);
                unsigned long long y_235 = other[1];
                top[14] = ((y_235 > top[14]) ? y_235 : top[14]);
                unsigned long long y_236 = other[0];
                top[15] = ((y_236 > top[15]) ? y_236 : top[15]);
                unsigned long long x_181 = top[0];
                unsigned long long y_237 = top[8];
                top[0] = ((y_237 > x_181) ? y_237 : x_181);
                top[8] = ((y_237 > x_181) ? x_181 : y_237);
                unsigned long long x_182 = top[1];
                unsigned long long y_238 = top[9];
                top[1] = ((y_238 > x_182) ? y_238 : x_182);
                top[9] = ((y_238 > x_182) ? x_182 : y_238);
                unsigned long long x_183 = top[2];
                unsigned long long y_239 = top[10];
                top[2] = ((y_239 > x_183) ? y_239 : x_183);
                top[10] = ((y_239 > x_183) ? x_183 : y_239);
                unsigned long long x_184 = top[3];
                unsigned long long y_240 = top[11];
                top[3] = ((y_240 > x_184) ? y_240 : x_184);
                top[11] = ((y_240 > x_184) ? x_184 : y_240);
                unsigned long long x_185 = top[4];
                unsigned long long y_241 = top[12];
                top[4] = ((y_241 > x_185) ? y_241 : x_185);
                top[12] = ((y_241 > x_185) ? x_185 : y_241);
                unsigned long long x_186 = top[5];
                unsigned long long y_242 = top[13];
                top[5] = ((y_242 > x_186) ? y_242 : x_186);
                top[13] = ((y_242 > x_186) ? x_186 : y_242);
                unsigned long long x_187 = top[6];
                unsigned long long y_243 = top[14];
                top[6] = ((y_243 > x_187) ? y_243 : x_187);
                top[14] = ((y_243 > x_187) ? x_187 : y_243);
                unsigned long long x_188 = top[7];
                unsigned long long y_244 = top[15];
                top[7] = ((y_244 > x_188) ? y_244 : x_188);
                top[15] = ((y_244 > x_188) ? x_188 : y_244);
                unsigned long long x_189 = top[0];
                unsigned long long y_245 = top[4];
                top[0] = ((y_245 > x_189) ? y_245 : x_189);
                top[4] = ((y_245 > x_189) ? x_189 : y_245);
                unsigned long long x_190 = top[1];
                unsigned long long y_246 = top[5];
                top[1] = ((y_246 > x_190) ? y_246 : x_190);
                top[5] = ((y_246 > x_190) ? x_190 : y_246);
                unsigned long long x_191 = top[2];
                unsigned long long y_247 = top[6];
                top[2] = ((y_247 > x_191) ? y_247 : x_191);
                top[6] = ((y_247 > x_191) ? x_191 : y_247);
                unsigned long long x_192 = top[3];
                unsigned long long y_248 = top[7];
                top[3] = ((y_248 > x_192) ? y_248 : x_192);
                top[7] = ((y_248 > x_192) ? x_192 : y_248);
                unsigned long long x_193 = top[8];
                unsigned long long y_249 = top[12];
                top[8] = ((y_249 > x_193) ? y_249 : x_193);
                top[12] = ((y_249 > x_193) ? x_193 : y_249);
                unsigned long long x_194 = top[9];
                unsigned long long y_250 = top[13];
                top[9] = ((y_250 > x_194) ? y_250 : x_194);
                top[13] = ((y_250 > x_194) ? x_194 : y_250);
                unsigned long long x_195 = top[10];
                unsigned long long y_251 = top[14];
                top[10] = ((y_251 > x_195) ? y_251 : x_195);
                top[14] = ((y_251 > x_195) ? x_195 : y_251);
                unsigned long long x_196 = top[11];
                unsigned long long y_252 = top[15];
                top[11] = ((y_252 > x_196) ? y_252 : x_196);
                top[15] = ((y_252 > x_196) ? x_196 : y_252);
                unsigned long long x_197 = top[0];
                unsigned long long y_253 = top[2];
                top[0] = ((y_253 > x_197) ? y_253 : x_197);
                top[2] = ((y_253 > x_197) ? x_197 : y_253);
                unsigned long long x_198 = top[1];
                unsigned long long y_254 = top[3];
                top[1] = ((y_254 > x_198) ? y_254 : x_198);
                top[3] = ((y_254 > x_198) ? x_198 : y_254);
                unsigned long long x_199 = top[4];
                unsigned long long y_255 = top[6];
                top[4] = ((y_255 > x_199) ? y_255 : x_199);
                top[6] = ((y_255 > x_199) ? x_199 : y_255);
                unsigned long long x_200 = top[5];
                unsigned long long y_256 = top[7];
                top[5] = ((y_256 > x_200) ? y_256 : x_200);
                top[7] = ((y_256 > x_200) ? x_200 : y_256);
                unsigned long long x_201 = top[8];
                unsigned long long y_257 = top[10];
                top[8] = ((y_257 > x_201) ? y_257 : x_201);
                top[10] = ((y_257 > x_201) ? x_201 : y_257);
                unsigned long long x_202 = top[9];
                unsigned long long y_258 = top[11];
                top[9] = ((y_258 > x_202) ? y_258 : x_202);
                top[11] = ((y_258 > x_202) ? x_202 : y_258);
                unsigned long long x_203 = top[12];
                unsigned long long y_259 = top[14];
                top[12] = ((y_259 > x_203) ? y_259 : x_203);
                top[14] = ((y_259 > x_203) ? x_203 : y_259);
                unsigned long long x_204 = top[13];
                unsigned long long y_260 = top[15];
                top[13] = ((y_260 > x_204) ? y_260 : x_204);
                top[15] = ((y_260 > x_204) ? x_204 : y_260);
                unsigned long long x_205 = top[0];
                unsigned long long y_261 = top[1];
                top[0] = ((y_261 > x_205) ? y_261 : x_205);
                top[1] = ((y_261 > x_205) ? x_205 : y_261);
                unsigned long long x_206 = top[2];
                unsigned long long y_262 = top[3];
                top[2] = ((y_262 > x_206) ? y_262 : x_206);
                top[3] = ((y_262 > x_206) ? x_206 : y_262);
                unsigned long long x_207 = top[4];
                unsigned long long y_263 = top[5];
                top[4] = ((y_263 > x_207) ? y_263 : x_207);
                top[5] = ((y_263 > x_207) ? x_207 : y_263);
                unsigned long long x_208 = top[6];
                unsigned long long y_264 = top[7];
                top[6] = ((y_264 > x_208) ? y_264 : x_208);
                top[7] = ((y_264 > x_208) ? x_208 : y_264);
                unsigned long long x_209 = top[8];
                unsigned long long y_265 = top[9];
                top[8] = ((y_265 > x_209) ? y_265 : x_209);
                top[9] = ((y_265 > x_209) ? x_209 : y_265);
                unsigned long long x_210 = top[10];
                unsigned long long y_266 = top[11];
                top[10] = ((y_266 > x_210) ? y_266 : x_210);
                top[11] = ((y_266 > x_210) ? x_210 : y_266);
                unsigned long long x_211 = top[12];
                unsigned long long y_267 = top[13];
                top[12] = ((y_267 > x_211) ? y_267 : x_211);
                top[13] = ((y_267 > x_211) ? x_211 : y_267);
                unsigned long long x_212 = top[14];
                unsigned long long y_268 = top[15];
                top[14] = ((y_268 > x_212) ? y_268 : x_212);
                top[15] = ((y_268 > x_212) ? x_212 : y_268);
                other[0] = lists[1536 + lane_0];
                other[1] = lists[1568 + lane_0];
                other[2] = lists[1600 + lane_0];
                other[3] = lists[1632 + lane_0];
                other[4] = lists[1664 + lane_0];
                other[5] = lists[1696 + lane_0];
                other[6] = lists[1728 + lane_0];
                other[7] = lists[1760 + lane_0];
                other[8] = lists[1792 + lane_0];
                other[9] = lists[1824 + lane_0];
                other[10] = lists[1856 + lane_0];
                other[11] = lists[1888 + lane_0];
                other[12] = lists[1920 + lane_0];
                other[13] = lists[1952 + lane_0];
                other[14] = lists[1984 + lane_0];
                other[15] = lists[2016 + lane_0];
                unsigned long long y_269 = other[15];
                top[0] = ((y_269 > top[0]) ? y_269 : top[0]);
                unsigned long long y_270 = other[14];
                top[1] = ((y_270 > top[1]) ? y_270 : top[1]);
                unsigned long long y_271 = other[13];
                top[2] = ((y_271 > top[2]) ? y_271 : top[2]);
                unsigned long long y_272 = other[12];
                top[3] = ((y_272 > top[3]) ? y_272 : top[3]);
                unsigned long long y_273 = other[11];
                top[4] = ((y_273 > top[4]) ? y_273 : top[4]);
                unsigned long long y_274 = other[10];
                top[5] = ((y_274 > top[5]) ? y_274 : top[5]);
                unsigned long long y_275 = other[9];
                top[6] = ((y_275 > top[6]) ? y_275 : top[6]);
                unsigned long long y_276 = other[8];
                top[7] = ((y_276 > top[7]) ? y_276 : top[7]);
                unsigned long long y_277 = other[7];
                top[8] = ((y_277 > top[8]) ? y_277 : top[8]);
                unsigned long long y_278 = other[6];
                top[9] = ((y_278 > top[9]) ? y_278 : top[9]);
                unsigned long long y_279 = other[5];
                top[10] = ((y_279 > top[10]) ? y_279 : top[10]);
                unsigned long long y_280 = other[4];
                top[11] = ((y_280 > top[11]) ? y_280 : top[11]);
                unsigned long long y_281 = other[3];
                top[12] = ((y_281 > top[12]) ? y_281 : top[12]);
                unsigned long long y_282 = other[2];
                top[13] = ((y_282 > top[13]) ? y_282 : top[13]);
                unsigned long long y_283 = other[1];
                top[14] = ((y_283 > top[14]) ? y_283 : top[14]);
                unsigned long long y_284 = other[0];
                top[15] = ((y_284 > top[15]) ? y_284 : top[15]);
                unsigned long long x_213 = top[0];
                unsigned long long y_285 = top[8];
                top[0] = ((y_285 > x_213) ? y_285 : x_213);
                top[8] = ((y_285 > x_213) ? x_213 : y_285);
                unsigned long long x_214 = top[1];
                unsigned long long y_286 = top[9];
                top[1] = ((y_286 > x_214) ? y_286 : x_214);
                top[9] = ((y_286 > x_214) ? x_214 : y_286);
                unsigned long long x_215 = top[2];
                unsigned long long y_287 = top[10];
                top[2] = ((y_287 > x_215) ? y_287 : x_215);
                top[10] = ((y_287 > x_215) ? x_215 : y_287);
                unsigned long long x_216 = top[3];
                unsigned long long y_288 = top[11];
                top[3] = ((y_288 > x_216) ? y_288 : x_216);
                top[11] = ((y_288 > x_216) ? x_216 : y_288);
                unsigned long long x_217 = top[4];
                unsigned long long y_289 = top[12];
                top[4] = ((y_289 > x_217) ? y_289 : x_217);
                top[12] = ((y_289 > x_217) ? x_217 : y_289);
                unsigned long long x_218 = top[5];
                unsigned long long y_290 = top[13];
                top[5] = ((y_290 > x_218) ? y_290 : x_218);
                top[13] = ((y_290 > x_218) ? x_218 : y_290);
                unsigned long long x_219 = top[6];
                unsigned long long y_291 = top[14];
                top[6] = ((y_291 > x_219) ? y_291 : x_219);
                top[14] = ((y_291 > x_219) ? x_219 : y_291);
                unsigned long long x_220 = top[7];
                unsigned long long y_292 = top[15];
                top[7] = ((y_292 > x_220) ? y_292 : x_220);
                top[15] = ((y_292 > x_220) ? x_220 : y_292);
                unsigned long long x_221 = top[0];
                unsigned long long y_293 = top[4];
                top[0] = ((y_293 > x_221) ? y_293 : x_221);
                top[4] = ((y_293 > x_221) ? x_221 : y_293);
                unsigned long long x_222 = top[1];
                unsigned long long y_294 = top[5];
                top[1] = ((y_294 > x_222) ? y_294 : x_222);
                top[5] = ((y_294 > x_222) ? x_222 : y_294);
                unsigned long long x_223 = top[2];
                unsigned long long y_295 = top[6];
                top[2] = ((y_295 > x_223) ? y_295 : x_223);
                top[6] = ((y_295 > x_223) ? x_223 : y_295);
                unsigned long long x_224 = top[3];
                unsigned long long y_296 = top[7];
                top[3] = ((y_296 > x_224) ? y_296 : x_224);
                top[7] = ((y_296 > x_224) ? x_224 : y_296);
                unsigned long long x_225 = top[8];
                unsigned long long y_297 = top[12];
                top[8] = ((y_297 > x_225) ? y_297 : x_225);
                top[12] = ((y_297 > x_225) ? x_225 : y_297);
                unsigned long long x_226 = top[9];
                unsigned long long y_298 = top[13];
                top[9] = ((y_298 > x_226) ? y_298 : x_226);
                top[13] = ((y_298 > x_226) ? x_226 : y_298);
                unsigned long long x_227 = top[10];
                unsigned long long y_299 = top[14];
                top[10] = ((y_299 > x_227) ? y_299 : x_227);
                top[14] = ((y_299 > x_227) ? x_227 : y_299);
                unsigned long long x_228 = top[11];
                unsigned long long y_300 = top[15];
                top[11] = ((y_300 > x_228) ? y_300 : x_228);
                top[15] = ((y_300 > x_228) ? x_228 : y_300);
                unsigned long long x_229 = top[0];
                unsigned long long y_301 = top[2];
                top[0] = ((y_301 > x_229) ? y_301 : x_229);
                top[2] = ((y_301 > x_229) ? x_229 : y_301);
                unsigned long long x_230 = top[1];
                unsigned long long y_302 = top[3];
                top[1] = ((y_302 > x_230) ? y_302 : x_230);
                top[3] = ((y_302 > x_230) ? x_230 : y_302);
                unsigned long long x_231 = top[4];
                unsigned long long y_303 = top[6];
                top[4] = ((y_303 > x_231) ? y_303 : x_231);
                top[6] = ((y_303 > x_231) ? x_231 : y_303);
                unsigned long long x_232 = top[5];
                unsigned long long y_304 = top[7];
                top[5] = ((y_304 > x_232) ? y_304 : x_232);
                top[7] = ((y_304 > x_232) ? x_232 : y_304);
                unsigned long long x_233 = top[8];
                unsigned long long y_305 = top[10];
                top[8] = ((y_305 > x_233) ? y_305 : x_233);
                top[10] = ((y_305 > x_233) ? x_233 : y_305);
                unsigned long long x_234 = top[9];
                unsigned long long y_306 = top[11];
                top[9] = ((y_306 > x_234) ? y_306 : x_234);
                top[11] = ((y_306 > x_234) ? x_234 : y_306);
                unsigned long long x_235 = top[12];
                unsigned long long y_307 = top[14];
                top[12] = ((y_307 > x_235) ? y_307 : x_235);
                top[14] = ((y_307 > x_235) ? x_235 : y_307);
                unsigned long long x_236 = top[13];
                unsigned long long y_308 = top[15];
                top[13] = ((y_308 > x_236) ? y_308 : x_236);
                top[15] = ((y_308 > x_236) ? x_236 : y_308);
                unsigned long long x_237 = top[0];
                unsigned long long y_309 = top[1];
                top[0] = ((y_309 > x_237) ? y_309 : x_237);
                top[1] = ((y_309 > x_237) ? x_237 : y_309);
                unsigned long long x_238 = top[2];
                unsigned long long y_310 = top[3];
                top[2] = ((y_310 > x_238) ? y_310 : x_238);
                top[3] = ((y_310 > x_238) ? x_238 : y_310);
                unsigned long long x_239 = top[4];
                unsigned long long y_311 = top[5];
                top[4] = ((y_311 > x_239) ? y_311 : x_239);
                top[5] = ((y_311 > x_239) ? x_239 : y_311);
                unsigned long long x_240 = top[6];
                unsigned long long y_312 = top[7];
                top[6] = ((y_312 > x_240) ? y_312 : x_240);
                top[7] = ((y_312 > x_240) ? x_240 : y_312);
                unsigned long long x_241 = top[8];
                unsigned long long y_313 = top[9];
                top[8] = ((y_313 > x_241) ? y_313 : x_241);
                top[9] = ((y_313 > x_241) ? x_241 : y_313);
                unsigned long long x_242 = top[10];
                unsigned long long y_314 = top[11];
                top[10] = ((y_314 > x_242) ? y_314 : x_242);
                top[11] = ((y_314 > x_242) ? x_242 : y_314);
                unsigned long long x_243 = top[12];
                unsigned long long y_315 = top[13];
                top[12] = ((y_315 > x_243) ? y_315 : x_243);
                top[13] = ((y_315 > x_243) ? x_243 : y_315);
                unsigned long long x_244 = top[14];
                unsigned long long y_316 = top[15];
                top[14] = ((y_316 > x_244) ? y_316 : x_244);
                top[15] = ((y_316 > x_244) ? x_244 : y_316);
                if (live != 0) {
                    long long dst = (long long)(t * num_heads + h) * (long long)pool_row_stride + (long long)(split * 16);
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
            int n_tiles_1 = (n_blk_1 + 1) / 2;
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
                    i0 = tile_1 * 2;
                    if (i0 % 32 == 0) {
                        gi = i0 + 32 + lane_0_1;
                        ids_next = 0;
                        if (gi < n_blk_1) {
                            ids_next = block_table[bt_base + split_1 + gi * splits];
                        }
                    }
                    nb = n_blk_1 - i0;
                    if (nb > 2) {
                        nb = 2;
                    }
                    int pg_0 = __shfl_sync(0xFFFFFFFF, ids, i0 % 32);
                    int pg_1 = __shfl_sync(0xFFFFFFFF, ids, i0 % 32 + 1);
                    mbarrier_wait(k_empty_addr + (k_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(k_full_addr + (k_stage) * 8, (unsigned int)nb * 4096);
                        if (nb > 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 8192 + 0), "l"((&k)), "r"(0), "r"(0), "r"(pg_0),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                        if (nb > 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_k_addr + k_stage * 8192 + 4096), "l"((&k)), "r"(0), "r"(0), "r"(pg_1),
                                   "r"(k_full_addr + (k_stage) * 8), "l"(0x12F0000000000000ULL) : "memory");
                        }
                    }
                    if ((i0 + 2) % 32 == 0) {
                        ids = ids_next;
                    }
                    k_stage += 1;
                    if (k_stage == 6) { k_stage = 0; _phase_k_empty ^= 1; }
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
            int n_tiles_2 = (n_blk_2 + 1) / 2;
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
                    int b_lo = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_1) * 512);
                    int a_lo = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(a_lo);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(b_lo);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (slot * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135266320, 1);
                        }
                    }
                    elect_commit2(acc_full_addr + (slot) * 8, k_empty_addr + (k_stage_1) * 8);
                    k_stage_1 += 1;
                    if (k_stage_1 == 6) { k_stage_1 = 0; _phase_k_full ^= 1; }
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
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
