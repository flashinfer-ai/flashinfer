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
#define SMEM_S_OUT_ID_OFF 0
#define SMEM_S_OUT_ID_STAGE_BYTES 8192
#define SMEM_S_OUT_ID_STRIDE 8192
#define SMEM_S_OUT_SC_OFF 8192
#define SMEM_S_OUT_SC_STAGE_BYTES 8192
#define SMEM_S_OUT_SC_STRIDE 8192
#define SMEM_S_BITS_OFF 16384
#define SMEM_S_BITS_STAGE_BYTES 2048
#define SMEM_S_BITS_STRIDE 2048
#define SMEM_S_PREF_OFF 18432
#define SMEM_S_PREF_STAGE_BYTES 1024
#define SMEM_S_PREF_STRIDE 1024
#define SMEM_S_XCH_OFF 19456
#define SMEM_S_XCH_STAGE_BYTES 352
#define SMEM_S_XCH_STRIDE 352
#define SMEM_TOTAL 19840
#define THREADS 256
#define RANK_XCH_PAD 16

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsa_indexer_topk_dabd609a7901c33aaf43(int* __restrict__ Indices, float* __restrict__ Scores, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int top_k, int num_segments)
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
    uint16_t* s_pref = reinterpret_cast<uint16_t*>(smem_raw + SMEM_S_PREF_OFF);
    const int s_pref_addr = smem + SMEM_S_PREF_OFF;
    unsigned int* s_xch = reinterpret_cast<unsigned int*>(smem_raw + SMEM_S_XCH_OFF);
    const int s_xch_addr = smem + SMEM_S_XCH_OFF;

    // === Task calls (dependency order) ===
    int lane_1 = tid % 32;
    int warp_1 = tid / 32;
    int row = bid;
    long long row_base = (long long)row * (long long)top_k;
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
    unsigned int lk_u = (unsigned int)lk;
    unsigned int zero_u = 0;
    unsigned int one_u = 1;
    unsigned int lanes_below = (one_u << (unsigned int)lane_1) - 1;
    int kids[8];
    unsigned int bits[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int pos = i * 256 + tid;
        kids[i] = -1;
        bits[i] = 4286578688;
        unsigned int is_pad = 0;
        if (pos < top_k) {
            long long slot = row_base + (long long)pos;
            int kid = Indices[slot];
            float score = Scores[slot];
            unsigned int score_bits = 0;
            score_bits = reinterpret_cast<unsigned int*>(&score)[0];
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
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int j_1 = 0; j_1 < n_slabs; j_1++) {
        int w_a = j_1 * 256 + tid;
        unsigned int bw_a = s_bits[w_a];
        int _popc_1 = __popc(bw_a);
        unsigned int c_a = (unsigned int)_popc_1;
        uint32_t _warp_scan_sum_u32_1 = c_a;
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
        unsigned int incl_a = _warp_scan_sum_u32_1;
        s_pref[w_a] = (uint16_t)(incl_a - c_a);
        if (lane_1 == 31) {
            s_xch[j_1 * 8 + warp_1] = incl_a;
        }
    }
    __syncthreads();
    unsigned int running = 0;
    #pragma unroll 1
    for (int j_2 = 0; j_2 < n_slabs; j_2++) {
        int w_b = j_2 * 256 + tid;
        unsigned int mine = running;
        unsigned int slab_tot = 0;
        unsigned int tot = s_xch[j_2 * 8];
        mine += ((warp_1 > 0) ? tot : zero_u);
        slab_tot += tot;
        unsigned int tot_0 = s_xch[j_2 * 8 + 1];
        mine += ((warp_1 > 1) ? tot_0 : zero_u);
        slab_tot += tot_0;
        unsigned int tot_1 = s_xch[j_2 * 8 + 2];
        mine += ((warp_1 > 2) ? tot_1 : zero_u);
        slab_tot += tot_1;
        unsigned int tot_2 = s_xch[j_2 * 8 + 3];
        mine += ((warp_1 > 3) ? tot_2 : zero_u);
        slab_tot += tot_2;
        unsigned int tot_3 = s_xch[j_2 * 8 + 4];
        mine += ((warp_1 > 4) ? tot_3 : zero_u);
        slab_tot += tot_3;
        unsigned int tot_4 = s_xch[j_2 * 8 + 5];
        mine += ((warp_1 > 5) ? tot_4 : zero_u);
        slab_tot += tot_4;
        unsigned int tot_5 = s_xch[j_2 * 8 + 6];
        mine += ((warp_1 > 6) ? tot_5 : zero_u);
        slab_tot += tot_5;
        unsigned int tot_6 = s_xch[j_2 * 8 + 7];
        mine += ((warp_1 > 7) ? tot_6 : zero_u);
        slab_tot += tot_6;
        unsigned int lane_b = (unsigned int)s_pref[w_b];
        s_pref[w_b] = (uint16_t)(mine + lane_b);
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
                int w_o = (int)(x_o >> 5);
                unsigned int below_o = (one_u << (x_o & 31)) - one_u;
                unsigned int bw_o = s_bits[w_o];
                unsigned int pref_o = (unsigned int)s_pref[w_o];
                unsigned int masked_o = bw_o & below_o;
                int _popc_2 = __popc(masked_o);
                dst = pref_o + (unsigned int)_popc_2;
            } else {
                unsigned int pad_before = s_xch[RANK_XCH_PAD + i_2 * 8 + warp_1];
                int _popc_3 = __popc(pad_mask_o & lanes_below);
                dst = n_valid + pad_before + (unsigned int)_popc_3;
            }
            int dst_i = (int)dst;
            unsigned int bits_o = bits[i_2];
            s_out_id[dst_i] = kid_o;
            s_out_sc[dst_i] = bits_o;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; i_3++) {
        int pos_f = i_3 * 256 + tid;
        if (pos_f < top_k) {
            int kid_f = s_out_id[pos_f];
            unsigned int bits_f = s_out_sc[pos_f];
            long long slot_f = row_base + (long long)pos_f;
            float out_score = 0.0f;
            out_score = reinterpret_cast<float*>(&bits_f)[0];
            Indices[slot_f] = kid_f;
            Scores[slot_f] = out_score;
        }
    }
}

} // extern "C"
