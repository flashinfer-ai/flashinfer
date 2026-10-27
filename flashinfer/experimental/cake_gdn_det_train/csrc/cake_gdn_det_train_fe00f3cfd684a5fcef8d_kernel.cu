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
#define NUM_MAIN_STAGES 1
#define SMEM_S_W_OFF 0
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 40960
#define SMEM_S_KN_OFF 16384
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 40960
#define SMEM_S_U_OFF 32768
#define SMEM_S_U_STAGE_BYTES 8192
#define SMEM_S_U_STRIDE 40960
#define SMEM_S_G_OFF 81920
#define SMEM_S_G_STAGE_BYTES 512
#define SMEM_S_G_STRIDE 512
#define SMEM_S_STAGE_OFF 82432
#define SMEM_S_STAGE_STAGE_BYTES 16384
#define SMEM_S_STAGE_STRIDE 16384
#define SMEM_S_RED_OFF 98816
#define SMEM_S_RED_STAGE_BYTES 32
#define SMEM_S_RED_STRIDE 32
#define SMEM_TOTAL 98944
#define THREADS 128

#include <math_constants.h>

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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_det_train_fe00f3cfd684a5fcef8d(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, float* __restrict__ g_cs, float* __restrict__ h0, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ v_new, float* __restrict__ final_state, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int* __restrict__ seg_flag, float warmup_thresh_log2)
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
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_w_addr = smem + 0;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_kn_addr = smem + 16384;
    __nv_bfloat16* s_u = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_u_addr = smem + 32768;
    float* s_g = reinterpret_cast<float*>(smem_raw + 81920);
    const int s_g_addr = smem + 81920;
    int* s_stage = reinterpret_cast<int*>(smem_raw + 82432);
    const int s_stage_addr = smem + 82432;
    float* s_red = reinterpret_cast<float*>(smem_raw + 98816);
    const int s_red_addr = smem + 98816;

    // === Task calls (declaration order) ===
    int seg = blockIdx.x;
    int hv = blockIdx.y;
    int v0 = blockIdx.z * 64;
    int c_begin = seg_chunk_start[seg];
    int c_end = seg_chunk_start[seg + 1];
    int seq = seg_seq[seg];
    int is_first = 0;
    int is_last = 0;
    if (seq >= 0) {
        if (c_begin == seq_chunk_start[seq]) {
            is_first = 1;
        }
        if (c_end == seq_chunk_start[seq + 1]) {
            is_last = 1;
        }
    }
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int quad = lane_0 % 4;
    int vwarp = warp_1;
    int khalf = 0;
    int kbase = khalf * 128;
    int vloc_a = vwarp * 16 + lane_0 / 4;
    int vrow_a = v0 + vloc_a;
    int vrow_b = vrow_a + 8;
    long long nvh = (long long)num_v_heads;
    long long hv64 = (long long)hv;
    long long tok_stride = nvh * 128;
    int stage_base = s_stage_addr + (unsigned int)(warp_1 * 4096);
    int stage_u32_base = warp_1 * 1024;
    unsigned int vw[16];
    float acc_h[64];
    float acc_v[32];
    unsigned int hpk[32];
    unsigned int vpk[16];
    unsigned int u_frag[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float g_next[1];
    float g_far[1];
    int state_idx = seg;
    long long in_base = ((long long)state_idx * nvh + hv64) * 16384;
    int _max_0 = ((seq) > (0) ? (seq) : (0));
    long long out_base = ((long long)_max_0 * nvh + hv64) * 16384;
    acc_h[0] = 0.0f;
    acc_h[1] = 0.0f;
    acc_h[2] = 0.0f;
    acc_h[3] = 0.0f;
    acc_h[4] = 0.0f;
    acc_h[5] = 0.0f;
    acc_h[6] = 0.0f;
    acc_h[7] = 0.0f;
    acc_h[8] = 0.0f;
    acc_h[9] = 0.0f;
    acc_h[10] = 0.0f;
    acc_h[11] = 0.0f;
    acc_h[12] = 0.0f;
    acc_h[13] = 0.0f;
    acc_h[14] = 0.0f;
    acc_h[15] = 0.0f;
    acc_h[16] = 0.0f;
    acc_h[17] = 0.0f;
    acc_h[18] = 0.0f;
    acc_h[19] = 0.0f;
    acc_h[20] = 0.0f;
    acc_h[21] = 0.0f;
    acc_h[22] = 0.0f;
    acc_h[23] = 0.0f;
    acc_h[24] = 0.0f;
    acc_h[25] = 0.0f;
    acc_h[26] = 0.0f;
    acc_h[27] = 0.0f;
    acc_h[28] = 0.0f;
    acc_h[29] = 0.0f;
    acc_h[30] = 0.0f;
    acc_h[31] = 0.0f;
    acc_h[32] = 0.0f;
    acc_h[33] = 0.0f;
    acc_h[34] = 0.0f;
    acc_h[35] = 0.0f;
    acc_h[36] = 0.0f;
    acc_h[37] = 0.0f;
    acc_h[38] = 0.0f;
    acc_h[39] = 0.0f;
    acc_h[40] = 0.0f;
    acc_h[41] = 0.0f;
    acc_h[42] = 0.0f;
    acc_h[43] = 0.0f;
    acc_h[44] = 0.0f;
    acc_h[45] = 0.0f;
    acc_h[46] = 0.0f;
    acc_h[47] = 0.0f;
    acc_h[48] = 0.0f;
    acc_h[49] = 0.0f;
    acc_h[50] = 0.0f;
    acc_h[51] = 0.0f;
    acc_h[52] = 0.0f;
    acc_h[53] = 0.0f;
    acc_h[54] = 0.0f;
    acc_h[55] = 0.0f;
    acc_h[56] = 0.0f;
    acc_h[57] = 0.0f;
    acc_h[58] = 0.0f;
    acc_h[59] = 0.0f;
    acc_h[60] = 0.0f;
    acc_h[61] = 0.0f;
    acc_h[62] = 0.0f;
    acc_h[63] = 0.0f;
    int ident = 0;
    int active = 1;
    if (v0 >= 128) {
        ident = 1;
    }
    if (c_end <= c_begin) {
        active = 0;
    }
    if (is_last != 0) {
        active = 0;
    }
    if (ident != 0) {
        if (is_first != 0) {
            if (use_initial_state == 0) {
                active = 0;
            }
        }
    }
    int n_seg = c_end - c_begin;
    float tot = 0.0f;
    if (tid_2 < n_seg) {
        int cw = c_end - 1 - tid_2;
        int cw_len = chunk_len[cw];
        if (cw_len > 0) {
            tot = g_cs[(long long)(chunk_start[cw] + cw_len - 1) * nvh + hv64];
        }
    }
    float up = __shfl_up_sync(0xFFFFFFFF, tot, 1, 32);
    if (lane_0 >= 1) {
        tot = tot + up;
    }
    float up_3 = __shfl_up_sync(0xFFFFFFFF, tot, 2, 32);
    if (lane_0 >= 2) {
        tot = tot + up_3;
    }
    float up_4 = __shfl_up_sync(0xFFFFFFFF, tot, 4, 32);
    if (lane_0 >= 4) {
        tot = tot + up_4;
    }
    float up_5 = __shfl_up_sync(0xFFFFFFFF, tot, 8, 32);
    if (lane_0 >= 8) {
        tot = tot + up_5;
    }
    float up_6 = __shfl_up_sync(0xFFFFFFFF, tot, 16, 32);
    if (lane_0 >= 16) {
        tot = tot + up_6;
    }
    if (lane_0 == 31) {
        s_red[warp_1] = tot;
    }
    __syncthreads();
    float cum = tot;
    if (warp_1 > 0) {
        cum = cum + s_red[0];
    }
    if (warp_1 > 1) {
        cum = cum + s_red[1];
    }
    if (warp_1 > 2) {
        cum = cum + s_red[2];
    }
    if (warp_1 > 3) {
        cum = cum + s_red[3];
    }
    int cand = n_seg;
    if (tid_2 < n_seg) {
        if (cum < warmup_thresh_log2) {
            cand = tid_2;
        }
    }
    int other = __shfl_xor_sync(0xFFFFFFFF, cand, 1);
    int _min_1 = ((cand) < (other) ? (cand) : (other));
    cand = _min_1;
    int other_7 = __shfl_xor_sync(0xFFFFFFFF, cand, 2);
    int _min_2 = ((cand) < (other_7) ? (cand) : (other_7));
    cand = _min_2;
    int other_8 = __shfl_xor_sync(0xFFFFFFFF, cand, 4);
    int _min_3 = ((cand) < (other_8) ? (cand) : (other_8));
    cand = _min_3;
    int other_9 = __shfl_xor_sync(0xFFFFFFFF, cand, 8);
    int _min_4 = ((cand) < (other_9) ? (cand) : (other_9));
    cand = _min_4;
    int other_10 = __shfl_xor_sync(0xFFFFFFFF, cand, 16);
    int _min_5 = ((cand) < (other_10) ? (cand) : (other_10));
    cand = _min_5;
    if (lane_0 == 0) {
        s_stage[warp_1] = cand;
    }
    __syncthreads();
    int n_w = s_stage[0];
    int _min_6 = ((n_w) < (s_stage[1]) ? (n_w) : (s_stage[1]));
    n_w = _min_6;
    int _min_7 = ((n_w) < (s_stage[2]) ? (n_w) : (s_stage[2]));
    n_w = _min_7;
    int _min_8 = ((n_w) < (s_stage[3]) ? (n_w) : (s_stage[3]));
    n_w = _min_8;
    n_w = n_w + 1;
    int fallback = 0;
    if (n_w > n_seg) {
        fallback = 1;
    }
    if (fallback == 0) {
        if (ident != 0) {
            active = 0;
        }
        c_begin = c_end - n_w;
    }
    if (v0 == 0) {
        if (tid_2 == 0) {
            if (n_seg > 0) {
                seg_flag[(long long)seg * nvh + hv64] = fallback;
            }
        }
    }
    int ca = v0 - 128 + vloc_a;
    if (ident != 0) {
        #pragma unroll
        for (int nb = 0; nb < 16; nb++) {
            int kcol = kbase + nb * 8 + quad * 2;
            if (kcol == ca) {
                acc_h[nb * 4] = 1.0f;
            }
            if (kcol + 1 == ca) {
                acc_h[nb * 4 + 1] = 1.0f;
            }
            if (kcol == ca + 8) {
                acc_h[nb * 4 + 2] = 1.0f;
            }
            if (kcol + 1 == ca + 8) {
                acc_h[nb * 4 + 3] = 1.0f;
            }
        }
    }
    int num_chunks = c_end - c_begin;
    int c_read = c_begin;
    if (c_end <= c_begin) {
        c_read = 0;
    }
    int seq_tok0 = chunk_start[c_read];
    int c_last = ((c_end - 1) > (0) ? (c_end - 1) : (0));
    int seq_len = chunk_start[c_last] + chunk_len[c_last] - seq_tok0;
    if (active == 0) {
        num_chunks = 0;
        seq_len = 0;
    }
    int stage_off = 0;
    int g_off = 0;
    int _min_0 = ((seq_len) < (64) ? (seq_len) : (64));
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(seq_tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    int dst0 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(w + src_row), "r"((_min_0 > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 1024), "l"(w + (src_row + 8 * row_stride)), "r"((_min_0 > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(w + (src_row + 16 * row_stride)), "r"((_min_0 > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 3072), "l"(w + (src_row + 24 * row_stride)), "r"((_min_0 > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(w + (src_row + 32 * row_stride)), "r"((_min_0 > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 5120), "l"(w + (src_row + 40 * row_stride)), "r"((_min_0 > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(w + (src_row + 48 * row_stride)), "r"((_min_0 > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 7168), "l"(w + (src_row + 56 * row_stride)), "r"((_min_0 > r0 + 56) ? 16 : 0));
    int r0_11 = tid / 16;
    int c0_12 = tid % 16 * 8;
    long long row_stride_13 = (long long)num_heads * 128;
    long long src_row_14 = (long long)(seq_tok0 + r0_11) * row_stride_13 + (long long)hq * 128 + (long long)c0_12;
    int dst0_15 = s_kn_addr + (unsigned int)stage_off + (unsigned int)(c0_12 / 64 * 8192 + (r0_11 * 128 + c0_12 % 64 * 2 ^ (r0_11 * 128 + c0_12 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15), "l"(kn + src_row_14), "r"((_min_0 > r0_11) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 1024), "l"(kn + (src_row_14 + 8 * row_stride_13)), "r"((_min_0 > r0_11 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 2048), "l"(kn + (src_row_14 + 16 * row_stride_13)), "r"((_min_0 > r0_11 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 3072), "l"(kn + (src_row_14 + 24 * row_stride_13)), "r"((_min_0 > r0_11 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 4096), "l"(kn + (src_row_14 + 32 * row_stride_13)), "r"((_min_0 > r0_11 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 5120), "l"(kn + (src_row_14 + 40 * row_stride_13)), "r"((_min_0 > r0_11 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 6144), "l"(kn + (src_row_14 + 48 * row_stride_13)), "r"((_min_0 > r0_11 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 7168), "l"(kn + (src_row_14 + 56 * row_stride_13)), "r"((_min_0 > r0_11 + 56) ? 16 : 0));
    if (ident == 0) {
        int r0_0 = tid / 8;
        int c0_1 = tid % 8 * 8;
        long long row_stride_2 = (long long)num_v_heads * 128;
        long long src_row_3 = (long long)(seq_tok0 + r0_0) * row_stride_2 + (long long)hv * 128 + (long long)v0 + (long long)c0_1;
        int dst0_4 = s_u_addr + (unsigned int)stage_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4), "l"(u + src_row_3), "r"((_min_0 > r0_0) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 2048), "l"(u + (src_row_3 + 16 * row_stride_2)), "r"((_min_0 > r0_0 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 4096), "l"(u + (src_row_3 + 32 * row_stride_2)), "r"((_min_0 > r0_0 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 6144), "l"(u + (src_row_3 + 48 * row_stride_2)), "r"((_min_0 > r0_0 + 48) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    int _min_1_1 = ((seq_len) < (64) ? (seq_len) : (64));
    int tok = tid;
    if (tok < 64) {
        float gate_val = 0.0f;
        if (tok < _min_1_1) {
            gate_val = g_cs[(long long)(seq_tok0 + tok) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tok] = gate_val;
    }
    int _min_2_1 = ((seq_len - 64) < (64) ? (seq_len - 64) : (64));
    int tok_16 = tid;
    g_next[0] = 0.0f;
    if (tok_16 < _min_2_1) {
        g_next[0] = g_cs[(long long)(seq_tok0 + 64 + tok_16) * (long long)num_v_heads + (long long)hv];
    }
    #pragma unroll 1
    for (int it_chunk = 0; it_chunk < num_chunks; it_chunk++) {
        int chunk = c_begin + it_chunk;
        int tok0 = seq_tok0 + it_chunk * 64;
        int n_valid = ((seq_len - it_chunk * 64) < (64) ? (seq_len - it_chunk * 64) : (64));
        int next_off = 40960 - stage_off;
        int next_g_off = 64 - g_off;
        int _min_3_1 = ((seq_len - (it_chunk + 2) * 64) < (64) ? (seq_len - (it_chunk + 2) * 64) : (64));
        int tok_0 = tid;
        g_far[0] = 0.0f;
        if (tok_0 < _min_3_1) {
            g_far[0] = g_cs[(long long)(tok0 + 128 + tok_0) * (long long)num_v_heads + (long long)hv];
        }
        asm volatile("cp.async.wait_group 0;");
        #pragma unroll
        for (int _lp = 0; _lp < 32; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_h[_lp*2 + 0], acc_h[_lp*2+1 + 0]));
            hpk[_lp] = *(uint32_t*)&_bf2;
        }
        __syncthreads();
        if (num_chunks > it_chunk + 1) {
            int next_tok0 = tok0 + 64;
            int next_len = ((seq_len - (it_chunk + 1) * 64) < (64) ? (seq_len - (it_chunk + 1) * 64) : (64));
            int r0_0_1 = tid / 16;
            int c0_1_1 = tid % 16 * 8;
            long long row_stride_2_1 = (long long)num_v_heads * 128;
            long long src_row_3_1 = (long long)(next_tok0 + r0_0_1) * row_stride_2_1 + (long long)hv * 128 + (long long)c0_1_1;
            int dst0_4_1 = s_w_addr + (unsigned int)next_off + (unsigned int)(c0_1_1 / 64 * 8192 + (r0_0_1 * 128 + c0_1_1 % 64 * 2 ^ (r0_0_1 * 128 + c0_1_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1), "l"(w + src_row_3_1), "r"((next_len > r0_0_1) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 1024), "l"(w + (src_row_3_1 + 8 * row_stride_2_1)), "r"((next_len > r0_0_1 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 2048), "l"(w + (src_row_3_1 + 16 * row_stride_2_1)), "r"((next_len > r0_0_1 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 3072), "l"(w + (src_row_3_1 + 24 * row_stride_2_1)), "r"((next_len > r0_0_1 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 4096), "l"(w + (src_row_3_1 + 32 * row_stride_2_1)), "r"((next_len > r0_0_1 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 5120), "l"(w + (src_row_3_1 + 40 * row_stride_2_1)), "r"((next_len > r0_0_1 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 6144), "l"(w + (src_row_3_1 + 48 * row_stride_2_1)), "r"((next_len > r0_0_1 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 7168), "l"(w + (src_row_3_1 + 56 * row_stride_2_1)), "r"((next_len > r0_0_1 + 56) ? 16 : 0));
            int r0_5 = tid / 16;
            int c0_6 = tid % 16 * 8;
            long long row_stride_7 = (long long)num_heads * 128;
            long long src_row_8 = (long long)(next_tok0 + r0_5) * row_stride_7 + (long long)hq * 128 + (long long)c0_6;
            int dst0_9 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9), "l"(kn + src_row_8), "r"((next_len > r0_5) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 1024), "l"(kn + (src_row_8 + 8 * row_stride_7)), "r"((next_len > r0_5 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 2048), "l"(kn + (src_row_8 + 16 * row_stride_7)), "r"((next_len > r0_5 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 3072), "l"(kn + (src_row_8 + 24 * row_stride_7)), "r"((next_len > r0_5 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 4096), "l"(kn + (src_row_8 + 32 * row_stride_7)), "r"((next_len > r0_5 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 5120), "l"(kn + (src_row_8 + 40 * row_stride_7)), "r"((next_len > r0_5 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 6144), "l"(kn + (src_row_8 + 48 * row_stride_7)), "r"((next_len > r0_5 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 7168), "l"(kn + (src_row_8 + 56 * row_stride_7)), "r"((next_len > r0_5 + 56) ? 16 : 0));
            if (ident == 0) {
                int r0_1 = tid / 8;
                int c0_2 = tid % 8 * 8;
                long long row_stride_3 = (long long)num_v_heads * 128;
                long long src_row_4 = (long long)(next_tok0 + r0_1) * row_stride_3 + (long long)hv * 128 + (long long)v0 + (long long)c0_2;
                int dst0_5 = s_u_addr + (unsigned int)next_off + (unsigned int)(c0_2 / 64 * 8192 + (r0_1 * 128 + c0_2 % 64 * 2 ^ (r0_1 * 128 + c0_2 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5), "l"(u + src_row_4), "r"((next_len > r0_1) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5 + 2048), "l"(u + (src_row_4 + 16 * row_stride_3)), "r"((next_len > r0_1 + 16) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5 + 4096), "l"(u + (src_row_4 + 32 * row_stride_3)), "r"((next_len > r0_1 + 32) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5 + 6144), "l"(u + (src_row_4 + 48 * row_stride_3)), "r"((next_len > r0_1 + 48) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        int _max_9 = ((n_valid) > (1) ? (n_valid) : (1));
        float g_last = s_g[g_off + _max_9 - 1];
        float _exp2_10 = approx_exp2(g_last);
        const float2 _scale2_11 = {_exp2_10, _exp2_10};
        #pragma unroll
        for (int _ls = 0; _ls < 32; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_h)[_ls], _scale2_11);
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            #pragma unroll
            for (int nb3 = 0; nb3 < 4; nb3++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                    : "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((kbase + ks * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8)[0]), "=f"((acc_v + nb3 * 8)[1]), "=f"((acc_v + nb3 * 8)[2]), "=f"((acc_v + nb3 * 8)[3])
                    : "r"((hpk + ks * 4)[0]), "r"((hpk + ks * 4)[1]), "r"((hpk + ks * 4)[2]), "r"((hpk + ks * 4)[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8 + 4)[0]), "=f"((acc_v + nb3 * 8 + 4)[1]), "=f"((acc_v + nb3 * 8 + 4)[2]), "=f"((acc_v + nb3 * 8 + 4)[3])
                    : "r"((hpk + ks * 4)[0]), "r"((hpk + ks * 4)[1]), "r"((hpk + ks * 4)[2]), "r"((hpk + ks * 4)[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[3])));
            }
        }
        float vn[4];
        #pragma unroll
        for (int tb = 0; tb < 4; tb++) {
            if (ident == 0) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(u_frag[0]), "=r"(u_frag[1]), "=r"(u_frag[2]), "=r"(u_frag[3])
                    : "r"(s_u_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
            } else {
                u_frag[0] = 0;
                u_frag[1] = 0;
                u_frag[2] = 0;
                u_frag[3] = 0;
            }
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int tok_1 = (tb * 2 + half) * 8 + quad * 2 + e;
                    float decay = 0.0f;
                    if (tok_1 < n_valid) {
                        float _exp2_12 = approx_exp2(g_last - s_g[g_off + tok_1]);
                        decay = _exp2_12;
                    }
                    float u_a = ((e == 0) ? __uint_as_float(u_frag[half * 2] << 16) : __uint_as_float(u_frag[half * 2] & 4294901760u));
                    float u_b = ((e == 0) ? __uint_as_float(u_frag[half * 2 + 1] << 16) : __uint_as_float(u_frag[half * 2 + 1] & 4294901760u));
                    vn[e] = u_a - acc_v[(tb * 2 + half) * 4 + e];
                    vn[2 + e] = u_b - acc_v[(tb * 2 + half) * 4 + 2 + e];
                    acc_v[(tb * 2 + half) * 4 + e] = vn[e] * decay;
                    acc_v[(tb * 2 + half) * 4 + 2 + e] = vn[2 + e] * decay;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vn[_lp*2 + 0], vn[_lp*2+1 + 0]));
                    vw[((tb * 2 + half) * 2) + _lp] = *(uint32_t*)&_bf2;
                }
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_v[_lp*2 + 0], acc_v[_lp*2+1 + 0]));
            vpk[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int ks2 = 0; ks2 < 4; ks2++) {
            #pragma unroll
            for (int nb5 = 0; nb5 < 8; nb5++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                    : "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((kbase + nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_h + nb5 * 8)[0]), "+f"((acc_h + nb5 * 8)[1]), "+f"((acc_h + nb5 * 8)[2]), "+f"((acc_h + nb5 * 8)[3])
                    : "r"((vpk + ks2 * 4)[0]), "r"((vpk + ks2 * 4)[1]), "r"((vpk + ks2 * 4)[2]), "r"((vpk + ks2 * 4)[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_h + nb5 * 8 + 4)[0]), "+f"((acc_h + nb5 * 8 + 4)[1]), "+f"((acc_h + nb5 * 8 + 4)[2]), "+f"((acc_h + nb5 * 8 + 4)[3])
                    : "r"((vpk + ks2 * 4)[0]), "r"((vpk + ks2 * 4)[1]), "r"((vpk + ks2 * 4)[2]), "r"((vpk + ks2 * 4)[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
            }
        }
        if (num_chunks > it_chunk + 1) {
            #pragma unroll
            for (int gi2 = 0; gi2 < 1; gi2++) {
                int gtok2 = gi2 * 128 + tid_2;
                if (gtok2 < 64) {
                    s_g[next_g_off + gtok2] = g_next[gi2];
                }
            }
        }
        #pragma unroll
        for (int gi3 = 0; gi3 < 1; gi3++) {
            g_next[gi3] = g_far[gi3];
        }
        stage_off = next_off;
        g_off = next_g_off;
    }
    asm volatile("cp.async.wait_group 0;");
    if (active != 0) {
        long long pre_base = ((long long)seg * nvh + hv64) * 32768;
        #pragma unroll
        for (int nb6 = 0; nb6 < 16; nb6++) {
            int kcol2 = kbase + nb6 * 8 + quad * 2;
            seg_state[pre_base + (long long)kcol2 * 256 + (long long)vrow_a] = acc_h[nb6 * 4];
            seg_state[pre_base + (long long)(kcol2 + 1) * 256 + (long long)vrow_a] = acc_h[nb6 * 4 + 1];
            seg_state[pre_base + (long long)kcol2 * 256 + (long long)vrow_b] = acc_h[nb6 * 4 + 2];
            seg_state[pre_base + (long long)(kcol2 + 1) * 256 + (long long)vrow_b] = acc_h[nb6 * 4 + 3];
        }
    }
}

} // extern "C"
