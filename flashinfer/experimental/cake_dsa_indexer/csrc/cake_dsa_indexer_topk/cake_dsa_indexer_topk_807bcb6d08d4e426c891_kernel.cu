/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define NUM_MAIN_STAGES 1
#define SMEM_S_OUT_ID_OFF 1024
#define SMEM_S_OUT_ID_STAGE_BYTES 8192
#define SMEM_S_OUT_ID_STRIDE 8192
#define SMEM_S_OUT_SC_OFF 9216
#define SMEM_S_OUT_SC_STAGE_BYTES 8192
#define SMEM_S_OUT_SC_STRIDE 8192
#define SMEM_S_BITS_OFF 17408
#define SMEM_S_BITS_STAGE_BYTES 16384
#define SMEM_S_BITS_STRIDE 16384
#define SMEM_S_BKT_OFF 33792
#define SMEM_S_BKT_STAGE_BYTES 2048
#define SMEM_S_BKT_STRIDE 2048
#define SMEM_S_XCH_OFF 35840
#define SMEM_S_XCH_STAGE_BYTES 384
#define SMEM_S_XCH_STRIDE 384
#define SMEM_TOTAL 36224
#define THREADS 256
#define RANK_XCH_PAD 16

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsa_indexer_topk_807bcb6d08d4e426c891(int* __restrict__ Indices, float* __restrict__ Scores, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int top_k, int num_segments)
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
    #define rank_row_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    int* s_out_id = reinterpret_cast<int*>(smem_raw + SMEM_S_OUT_ID_OFF);
    const int s_out_id_addr = smem + SMEM_S_OUT_ID_OFF;
    unsigned int* s_out_sc = reinterpret_cast<unsigned int*>(smem_raw + SMEM_S_OUT_SC_OFF);
    const int s_out_sc_addr = smem + SMEM_S_OUT_SC_OFF;
    unsigned int* s_bits = reinterpret_cast<unsigned int*>(smem_raw + SMEM_S_BITS_OFF);
    const int s_bits_addr = smem + SMEM_S_BITS_OFF;
    unsigned int* s_bkt = reinterpret_cast<unsigned int*>(smem_raw + SMEM_S_BKT_OFF);
    const int s_bkt_addr = smem + SMEM_S_BKT_OFF;
    unsigned int* s_xch = reinterpret_cast<unsigned int*>(smem_raw + SMEM_S_XCH_OFF);
    const int s_xch_addr = smem + SMEM_S_XCH_OFF;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // rank_row_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    int row = bid;
    long long row_base = (long long)row * (long long)top_k;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(rank_row_full_addr, top_k * 8);
            cp_async_bulk_gmem2smem(s_out_id_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(Indices) + ((unsigned long long)row_base * (unsigned long long)4)), top_k * 4, rank_row_full_addr);
            cp_async_bulk_gmem2smem(s_out_sc_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(Scores) + ((unsigned long long)row_base * (unsigned long long)4)), top_k * 4, rank_row_full_addr);
        }
    }
    int seg = 0;
    #pragma unroll 1
    for (int s = 1; s < num_segments; s++) {
        if (row >= cu_seqlens_q[s]) {
            seg += 1;
        }
    }
    int lk = cu_seqlens_k[seg + 1] - cu_seqlens_k[seg];
    int n_words = (lk + 31) / 32;
    int n_slabs = (n_words + 256 - 1) / 256;
    int n_bkts = (lk + 255) / 256;
    int n_bslabs = (n_bkts + 256 - 1) / 256;
    unsigned int lk_u = (unsigned int)lk;
    unsigned int zero_u = 0;
    unsigned int one_u = 1;
    unsigned int lanes_below = (one_u << (unsigned int)lane_1) - 1;
    int kids[8];
    unsigned int bits[8];
    mbarrier_wait(rank_row_full_addr, 0);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int pos = i * 256 + tid;
        kids[i] = -1;
        bits[i] = 4286578688;
        unsigned int is_pad = 0;
        if (pos < top_k) {
            int kid = s_out_id[pos];
            unsigned int score_bits = s_out_sc[pos];
            kids[i] = kid;
            bits[i] = score_bits;
            is_pad = 1;
            if (lk_u > (unsigned int)kid) {
                is_pad = 0;
            }
        }
        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, is_pad != 0);
        unsigned int pad_mask = _vote_0;
        if (lane_1 == 0) {
            int _popc_0 = __popc(pad_mask);
            s_xch[RANK_XCH_PAD + i * 8 + warp_1] = (unsigned int)_popc_0;
        }
    }
    #pragma unroll 1
    for (int j = 0; j < n_slabs; j++) {
        s_bits[j * 256 + tid] = zero_u;
    }
    #pragma unroll 1
    for (int j_1 = 0; j_1 < n_bslabs; j_1++) {
        s_bkt[j_1 * 256 + tid] = zero_u;
    }
    __syncthreads();
    if (warp_1 == 0) {
        unsigned int pad_sum = 0;
        unsigned int pad_e = s_xch[RANK_XCH_PAD + lane_1 * 2];
        pad_sum += pad_e;
        unsigned int pad_e_0 = s_xch[RANK_XCH_PAD + lane_1 * 2 + 1];
        pad_sum += pad_e_0;
        uint32_t _warp_scan_sum_u32_0 = pad_sum;
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
        unsigned int pad_incl = _warp_scan_sum_u32_0;
        unsigned int pad_run = pad_incl - pad_sum;
        unsigned int pad_f = s_xch[RANK_XCH_PAD + lane_1 * 2];
        s_xch[RANK_XCH_PAD + lane_1 * 2] = pad_run;
        pad_run += pad_f;
        unsigned int pad_f_1 = s_xch[RANK_XCH_PAD + lane_1 * 2 + 1];
        s_xch[RANK_XCH_PAD + lane_1 * 2 + 1] = pad_run;
        pad_run += pad_f_1;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        unsigned int x_s = (unsigned int)kids[i_1];
        if (x_s < lk_u) {
            unsigned int bit_s = one_u << (x_s & 31);
            atomicAdd(&s_bits[(int)(x_s >> 5)], bit_s);
            atomicAdd(&s_bkt[(int)(x_s >> 8)], one_u);
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int j_2 = 0; j_2 < n_bslabs; j_2++) {
        int b_a = j_2 * 256 + tid;
        unsigned int c_a = s_bkt[b_a];
        uint32_t _warp_scan_sum_u32_1 = c_a;
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
        unsigned int incl_a = _warp_scan_sum_u32_1;
        s_bkt[b_a] = incl_a - c_a;
        if (lane_1 == 31) {
            s_xch[j_2 * 8 + warp_1] = incl_a;
        }
    }
    __syncthreads();
    unsigned int running = 0;
    #pragma unroll 1
    for (int j_3 = 0; j_3 < n_bslabs; j_3++) {
        int b_b = j_3 * 256 + tid;
        unsigned int mine = running;
        unsigned int slab_tot = 0;
        unsigned int tot = s_xch[j_3 * 8];
        mine += ((warp_1 > 0) ? tot : zero_u);
        slab_tot += tot;
        unsigned int tot_0 = s_xch[j_3 * 8 + 1];
        mine += ((warp_1 > 1) ? tot_0 : zero_u);
        slab_tot += tot_0;
        unsigned int tot_1 = s_xch[j_3 * 8 + 2];
        mine += ((warp_1 > 2) ? tot_1 : zero_u);
        slab_tot += tot_1;
        unsigned int tot_2 = s_xch[j_3 * 8 + 3];
        mine += ((warp_1 > 3) ? tot_2 : zero_u);
        slab_tot += tot_2;
        unsigned int tot_3 = s_xch[j_3 * 8 + 4];
        mine += ((warp_1 > 4) ? tot_3 : zero_u);
        slab_tot += tot_3;
        unsigned int tot_4 = s_xch[j_3 * 8 + 5];
        mine += ((warp_1 > 5) ? tot_4 : zero_u);
        slab_tot += tot_4;
        unsigned int tot_5 = s_xch[j_3 * 8 + 6];
        mine += ((warp_1 > 6) ? tot_5 : zero_u);
        slab_tot += tot_5;
        unsigned int tot_6 = s_xch[j_3 * 8 + 7];
        mine += ((warp_1 > 7) ? tot_6 : zero_u);
        slab_tot += tot_6;
        unsigned int lane_b = s_bkt[b_b];
        s_bkt[b_b] = mine + lane_b;
        running += slab_tot;
    }
    unsigned int n_valid = running;
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; i_2++) {
        int pos_o = i_2 * 256 + tid;
        int kid_o = kids[i_2];
        unsigned int x_o = (unsigned int)kid_o;
        unsigned int is_pad_o = 0;
        if (pos_o < top_k) {
            is_pad_o = 1;
            if (x_o < lk_u) {
                is_pad_o = 0;
            }
        }
        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, is_pad_o != 0);
        unsigned int pad_mask_o = _vote_1;
        if (pos_o < top_k) {
            unsigned int dst = 0;
            if (x_o < lk_u) {
                int bkt_o = (int)(x_o >> 8);
                int wbase_o = bkt_o * 8;
                int w_in_o = (int)(x_o >> 5 & 7);
                unsigned int below_o = (one_u << (x_o & 31)) - one_u;
                unsigned int acc_o = s_bkt[bkt_o];
                unsigned int bw_w = s_bits[wbase_o];
                int _popc_1 = __popc(bw_w);
                unsigned int c_w = (unsigned int)_popc_1;
                acc_o += ((w_in_o > 0) ? c_w : zero_u);
                unsigned int bw_w_0 = s_bits[wbase_o + 1];
                int _popc_2 = __popc(bw_w_0);
                unsigned int c_w_1 = (unsigned int)_popc_2;
                acc_o += ((w_in_o > 1) ? c_w_1 : zero_u);
                unsigned int bw_w_2 = s_bits[wbase_o + 2];
                int _popc_3 = __popc(bw_w_2);
                unsigned int c_w_3 = (unsigned int)_popc_3;
                acc_o += ((w_in_o > 2) ? c_w_3 : zero_u);
                unsigned int bw_w_4 = s_bits[wbase_o + 3];
                int _popc_4 = __popc(bw_w_4);
                unsigned int c_w_5 = (unsigned int)_popc_4;
                acc_o += ((w_in_o > 3) ? c_w_5 : zero_u);
                unsigned int bw_w_6 = s_bits[wbase_o + 4];
                int _popc_5 = __popc(bw_w_6);
                unsigned int c_w_7 = (unsigned int)_popc_5;
                acc_o += ((w_in_o > 4) ? c_w_7 : zero_u);
                unsigned int bw_w_8 = s_bits[wbase_o + 5];
                int _popc_6 = __popc(bw_w_8);
                unsigned int c_w_9 = (unsigned int)_popc_6;
                acc_o += ((w_in_o > 5) ? c_w_9 : zero_u);
                unsigned int bw_w_10 = s_bits[wbase_o + 6];
                int _popc_7 = __popc(bw_w_10);
                unsigned int c_w_11 = (unsigned int)_popc_7;
                acc_o += ((w_in_o > 6) ? c_w_11 : zero_u);
                unsigned int bw_o = s_bits[wbase_o + w_in_o];
                unsigned int masked_o = bw_o & below_o;
                int _popc_8 = __popc(masked_o);
                dst = acc_o + (unsigned int)_popc_8;
            } else {
                unsigned int pad_before = s_xch[RANK_XCH_PAD + i_2 * 8 + warp_1];
                int _popc_9 = __popc(pad_mask_o & lanes_below);
                dst = n_valid + pad_before + (unsigned int)_popc_9;
            }
            int dst_i = (int)dst;
            unsigned int bits_o = bits[i_2];
            s_out_id[dst_i] = kid_o;
            s_out_sc[dst_i] = bits_o;
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            {
                void* _cpbulk_dst_0 = reinterpret_cast<void*>(Indices + row_base);
                asm volatile(
                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                    :: "l"(_cpbulk_dst_0), "r"(s_out_id_addr), "r"((uint32_t)(top_k * 4))
                    : "memory");
            }
            {
                void* _cpbulk_dst_1 = reinterpret_cast<void*>(Scores + row_base);
                asm volatile(
                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                    :: "l"(_cpbulk_dst_1), "r"(s_out_sc_addr), "r"((uint32_t)(top_k * 4))
                    : "memory");
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
