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
#define SMEM_S_KN_OFF 0
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 65536
#define SMEM_S_QN_OFF 16384
#define SMEM_S_QN_STAGE_BYTES 16384
#define SMEM_S_QN_STRIDE 65536
#define SMEM_S_W_OFF 32768
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 65536
#define SMEM_S_DO_OFF 49152
#define SMEM_S_DO_STAGE_BYTES 8192
#define SMEM_S_DO_STRIDE 65536
#define SMEM_S_DVL_OFF 57344
#define SMEM_S_DVL_STAGE_BYTES 8192
#define SMEM_S_DVL_STRIDE 65536
#define SMEM_S_G_OFF 131072
#define SMEM_S_G_STAGE_BYTES 512
#define SMEM_S_G_STRIDE 512
#define SMEM_S_GAIN_OFF 131584
#define SMEM_S_GAIN_STAGE_BYTES 512
#define SMEM_S_GAIN_STRIDE 512
#define SMEM_S_STAGE_OFF 132096
#define SMEM_S_STAGE_STAGE_BYTES 16384
#define SMEM_S_STAGE_STRIDE 16384
#define SMEM_S_RED_OFF 148480
#define SMEM_S_RED_STAGE_BYTES 32
#define SMEM_S_RED_STRIDE 32
#define SMEM_TOTAL 148608
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
kernel_cake_gdn_det_train_9f459af70f5bb9dcf7ba(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv_local, float* __restrict__ g_cs, float* __restrict__ dht, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, float* __restrict__ dh0, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_final_state_grad, int store_initial_state_grad, float scale, int* __restrict__ seg_flag, float warmup_thresh_log2)
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
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kn_addr = smem + 0;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_qn_addr = smem + 16384;
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_w_addr = smem + 32768;
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_do_addr = smem + 49152;
    __nv_bfloat16* s_dvl = reinterpret_cast<__nv_bfloat16*>(smem_raw + 57344);
    const int s_dvl_addr = smem + 57344;
    float* s_g = reinterpret_cast<float*>(smem_raw + 131072);
    const int s_g_addr = smem + 131072;
    float* s_gain = reinterpret_cast<float*>(smem_raw + 131584);
    const int s_gain_addr = smem + 131584;
    int* s_stage = reinterpret_cast<int*>(smem_raw + 132096);
    const int s_stage_addr = smem + 132096;
    float* s_red = reinterpret_cast<float*>(smem_raw + 148480);
    const int s_red_addr = smem + 148480;

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
    float acc_dh[64];
    float acc_v[32];
    unsigned int dhpk[32];
    unsigned int ndvpk[16];
    unsigned int a_frag_mn[4];
    unsigned int dvl_frag[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float g_next[1];
    float g_far[1];
    long long in_base = ((long long)seg * nvh + hv64) * 16384;
    int _max_0 = ((seq) > (0) ? (seq) : (0));
    long long out_base = ((long long)_max_0 * nvh + hv64) * 16384;
    acc_dh[0] = 0.0f;
    acc_dh[1] = 0.0f;
    acc_dh[2] = 0.0f;
    acc_dh[3] = 0.0f;
    acc_dh[4] = 0.0f;
    acc_dh[5] = 0.0f;
    acc_dh[6] = 0.0f;
    acc_dh[7] = 0.0f;
    acc_dh[8] = 0.0f;
    acc_dh[9] = 0.0f;
    acc_dh[10] = 0.0f;
    acc_dh[11] = 0.0f;
    acc_dh[12] = 0.0f;
    acc_dh[13] = 0.0f;
    acc_dh[14] = 0.0f;
    acc_dh[15] = 0.0f;
    acc_dh[16] = 0.0f;
    acc_dh[17] = 0.0f;
    acc_dh[18] = 0.0f;
    acc_dh[19] = 0.0f;
    acc_dh[20] = 0.0f;
    acc_dh[21] = 0.0f;
    acc_dh[22] = 0.0f;
    acc_dh[23] = 0.0f;
    acc_dh[24] = 0.0f;
    acc_dh[25] = 0.0f;
    acc_dh[26] = 0.0f;
    acc_dh[27] = 0.0f;
    acc_dh[28] = 0.0f;
    acc_dh[29] = 0.0f;
    acc_dh[30] = 0.0f;
    acc_dh[31] = 0.0f;
    acc_dh[32] = 0.0f;
    acc_dh[33] = 0.0f;
    acc_dh[34] = 0.0f;
    acc_dh[35] = 0.0f;
    acc_dh[36] = 0.0f;
    acc_dh[37] = 0.0f;
    acc_dh[38] = 0.0f;
    acc_dh[39] = 0.0f;
    acc_dh[40] = 0.0f;
    acc_dh[41] = 0.0f;
    acc_dh[42] = 0.0f;
    acc_dh[43] = 0.0f;
    acc_dh[44] = 0.0f;
    acc_dh[45] = 0.0f;
    acc_dh[46] = 0.0f;
    acc_dh[47] = 0.0f;
    acc_dh[48] = 0.0f;
    acc_dh[49] = 0.0f;
    acc_dh[50] = 0.0f;
    acc_dh[51] = 0.0f;
    acc_dh[52] = 0.0f;
    acc_dh[53] = 0.0f;
    acc_dh[54] = 0.0f;
    acc_dh[55] = 0.0f;
    acc_dh[56] = 0.0f;
    acc_dh[57] = 0.0f;
    acc_dh[58] = 0.0f;
    acc_dh[59] = 0.0f;
    acc_dh[60] = 0.0f;
    acc_dh[61] = 0.0f;
    acc_dh[62] = 0.0f;
    acc_dh[63] = 0.0f;
    int ident = 0;
    int active = 1;
    if (v0 >= 128) {
        ident = 1;
    }
    if (c_end <= c_begin) {
        active = 0;
    }
    if (is_first != 0) {
        active = 0;
    }
    if (ident != 0) {
        if (is_last != 0) {
            if (use_final_state_grad == 0) {
                active = 0;
            }
        }
    }
    int n_seg = c_end - c_begin;
    float tot = 0.0f;
    if (tid_2 < n_seg) {
        int cw = c_begin + tid_2;
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
        c_end = c_begin + n_w;
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
                acc_dh[nb * 4] = 1.0f;
            }
            if (kcol + 1 == ca) {
                acc_dh[nb * 4 + 1] = 1.0f;
            }
            if (kcol == ca + 8) {
                acc_dh[nb * 4 + 2] = 1.0f;
            }
            if (kcol + 1 == ca + 8) {
                acc_dh[nb * 4 + 3] = 1.0f;
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
    int last_tok0 = chunk_start[c_last];
    int last_len = chunk_len[c_last];
    if (active == 0) {
        num_chunks = 0;
        last_len = 0;
    }
    int stage_off = 0;
    int g_off = 0;
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_heads * 128;
    long long src_row = (long long)(last_tok0 + r0) * row_stride + (long long)hq * 128 + (long long)c0;
    int dst0 = s_kn_addr + (unsigned int)stage_off + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(kn + src_row), "r"((last_len > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 1024), "l"(kn + (src_row + 8 * row_stride)), "r"((last_len > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(kn + (src_row + 16 * row_stride)), "r"((last_len > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 3072), "l"(kn + (src_row + 24 * row_stride)), "r"((last_len > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(kn + (src_row + 32 * row_stride)), "r"((last_len > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 5120), "l"(kn + (src_row + 40 * row_stride)), "r"((last_len > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(kn + (src_row + 48 * row_stride)), "r"((last_len > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 7168), "l"(kn + (src_row + 56 * row_stride)), "r"((last_len > r0 + 56) ? 16 : 0));
    int r0_11 = tid / 16;
    int c0_12 = tid % 16 * 8;
    long long row_stride_13 = (long long)num_heads * 128;
    long long src_row_14 = (long long)(last_tok0 + r0_11) * row_stride_13 + (long long)hq * 128 + (long long)c0_12;
    int dst0_15 = s_qn_addr + (unsigned int)stage_off + (unsigned int)(c0_12 / 64 * 8192 + (r0_11 * 128 + c0_12 % 64 * 2 ^ (r0_11 * 128 + c0_12 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15), "l"(qn + src_row_14), "r"((last_len > r0_11) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 1024), "l"(qn + (src_row_14 + 8 * row_stride_13)), "r"((last_len > r0_11 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 2048), "l"(qn + (src_row_14 + 16 * row_stride_13)), "r"((last_len > r0_11 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 3072), "l"(qn + (src_row_14 + 24 * row_stride_13)), "r"((last_len > r0_11 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 4096), "l"(qn + (src_row_14 + 32 * row_stride_13)), "r"((last_len > r0_11 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 5120), "l"(qn + (src_row_14 + 40 * row_stride_13)), "r"((last_len > r0_11 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 6144), "l"(qn + (src_row_14 + 48 * row_stride_13)), "r"((last_len > r0_11 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_15 + 7168), "l"(qn + (src_row_14 + 56 * row_stride_13)), "r"((last_len > r0_11 + 56) ? 16 : 0));
    int r0_16 = tid / 16;
    int c0_17 = tid % 16 * 8;
    long long row_stride_18 = (long long)num_v_heads * 128;
    long long src_row_19 = (long long)(last_tok0 + r0_16) * row_stride_18 + (long long)hv * 128 + (long long)c0_17;
    int dst0_20 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0_17 / 64 * 8192 + (r0_16 * 128 + c0_17 % 64 * 2 ^ (r0_16 * 128 + c0_17 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20), "l"(w + src_row_19), "r"((last_len > r0_16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 1024), "l"(w + (src_row_19 + 8 * row_stride_18)), "r"((last_len > r0_16 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 2048), "l"(w + (src_row_19 + 16 * row_stride_18)), "r"((last_len > r0_16 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 3072), "l"(w + (src_row_19 + 24 * row_stride_18)), "r"((last_len > r0_16 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 4096), "l"(w + (src_row_19 + 32 * row_stride_18)), "r"((last_len > r0_16 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 5120), "l"(w + (src_row_19 + 40 * row_stride_18)), "r"((last_len > r0_16 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 6144), "l"(w + (src_row_19 + 48 * row_stride_18)), "r"((last_len > r0_16 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_20 + 7168), "l"(w + (src_row_19 + 56 * row_stride_18)), "r"((last_len > r0_16 + 56) ? 16 : 0));
    if (ident == 0) {
        int r0_0 = tid / 8;
        int c0_1 = tid % 8 * 8;
        long long row_stride_2 = (long long)num_v_heads * 128;
        long long src_row_3 = (long long)(last_tok0 + r0_0) * row_stride_2 + (long long)hv * 128 + (long long)v0 + (long long)c0_1;
        int dst0_4 = s_do_addr + (unsigned int)stage_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4), "l"(do_ + src_row_3), "r"((last_len > r0_0) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 2048), "l"(do_ + (src_row_3 + 16 * row_stride_2)), "r"((last_len > r0_0 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 4096), "l"(do_ + (src_row_3 + 32 * row_stride_2)), "r"((last_len > r0_0 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 6144), "l"(do_ + (src_row_3 + 48 * row_stride_2)), "r"((last_len > r0_0 + 48) ? 16 : 0));
        int r0_5 = tid / 8;
        int c0_6 = tid % 8 * 8;
        long long row_stride_7 = (long long)num_v_heads * 128;
        long long src_row_8 = (long long)(last_tok0 + r0_5) * row_stride_7 + (long long)hv * 128 + (long long)v0 + (long long)c0_6;
        int dst0_9 = s_dvl_addr + (unsigned int)stage_off + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9), "l"(dv_local + src_row_8), "r"((last_len > r0_5) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 2048), "l"(dv_local + (src_row_8 + 16 * row_stride_7)), "r"((last_len > r0_5 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 4096), "l"(dv_local + (src_row_8 + 32 * row_stride_7)), "r"((last_len > r0_5 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 6144), "l"(dv_local + (src_row_8 + 48 * row_stride_7)), "r"((last_len > r0_5 + 48) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    int tok = tid;
    if (tok < 64) {
        float gate_val = 0.0f;
        if (tok < last_len) {
            gate_val = g_cs[(long long)(last_tok0 + tok) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tok] = gate_val;
        float _exp2_9 = approx_exp2(gate_val);
        s_gain[tok] = _exp2_9 * scale;
    }
    int _min_0 = ((num_chunks - 1) < (1) ? (num_chunks - 1) : (1));
    int tok_21 = tid;
    g_next[0] = 0.0f;
    if (tok_21 < _min_0 * 64) {
        g_next[0] = g_cs[(long long)(last_tok0 - 64 + tok_21) * (long long)num_v_heads + (long long)hv];
    }
    #pragma unroll 1
    for (int rev = 0; rev < num_chunks; rev++) {
        int chunk = c_end - 1 - rev;
        int tok0 = last_tok0 - rev * 64;
        int n_valid = 64;
        if (rev == 0) {
            n_valid = last_len;
        }
        int next_off = 65536 - stage_off;
        int next_g_off = 64 - g_off;
        int _max_10 = ((num_chunks - rev - 2) > (0) ? (num_chunks - rev - 2) : (0));
        int _min_1_1 = ((_max_10) < (1) ? (_max_10) : (1));
        int tok_0 = tid;
        g_far[0] = 0.0f;
        if (tok_0 < _min_1_1 * 64) {
            g_far[0] = g_cs[(long long)(tok0 - 128 + tok_0) * (long long)num_v_heads + (long long)hv];
        }
        if (num_chunks > rev + 1) {
            int next_tok0 = tok0 - 64;
            int next_len = 64;
            int r0_0_1 = tid / 16;
            int c0_1_1 = tid % 16 * 8;
            long long row_stride_2_1 = (long long)num_heads * 128;
            long long src_row_3_1 = (long long)(next_tok0 + r0_0_1) * row_stride_2_1 + (long long)hq * 128 + (long long)c0_1_1;
            int dst0_4_1 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_1_1 / 64 * 8192 + (r0_0_1 * 128 + c0_1_1 % 64 * 2 ^ (r0_0_1 * 128 + c0_1_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1), "l"(kn + src_row_3_1), "r"((next_len > r0_0_1) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 1024), "l"(kn + (src_row_3_1 + 8 * row_stride_2_1)), "r"((next_len > r0_0_1 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 2048), "l"(kn + (src_row_3_1 + 16 * row_stride_2_1)), "r"((next_len > r0_0_1 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 3072), "l"(kn + (src_row_3_1 + 24 * row_stride_2_1)), "r"((next_len > r0_0_1 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 4096), "l"(kn + (src_row_3_1 + 32 * row_stride_2_1)), "r"((next_len > r0_0_1 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 5120), "l"(kn + (src_row_3_1 + 40 * row_stride_2_1)), "r"((next_len > r0_0_1 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 6144), "l"(kn + (src_row_3_1 + 48 * row_stride_2_1)), "r"((next_len > r0_0_1 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4_1 + 7168), "l"(kn + (src_row_3_1 + 56 * row_stride_2_1)), "r"((next_len > r0_0_1 + 56) ? 16 : 0));
            int r0_5_1 = tid / 16;
            int c0_6_1 = tid % 16 * 8;
            long long row_stride_7_1 = (long long)num_heads * 128;
            long long src_row_8_1 = (long long)(next_tok0 + r0_5_1) * row_stride_7_1 + (long long)hq * 128 + (long long)c0_6_1;
            int dst0_9_1 = s_qn_addr + (unsigned int)next_off + (unsigned int)(c0_6_1 / 64 * 8192 + (r0_5_1 * 128 + c0_6_1 % 64 * 2 ^ (r0_5_1 * 128 + c0_6_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1), "l"(qn + src_row_8_1), "r"((next_len > r0_5_1) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 1024), "l"(qn + (src_row_8_1 + 8 * row_stride_7_1)), "r"((next_len > r0_5_1 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 2048), "l"(qn + (src_row_8_1 + 16 * row_stride_7_1)), "r"((next_len > r0_5_1 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 3072), "l"(qn + (src_row_8_1 + 24 * row_stride_7_1)), "r"((next_len > r0_5_1 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 4096), "l"(qn + (src_row_8_1 + 32 * row_stride_7_1)), "r"((next_len > r0_5_1 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 5120), "l"(qn + (src_row_8_1 + 40 * row_stride_7_1)), "r"((next_len > r0_5_1 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 6144), "l"(qn + (src_row_8_1 + 48 * row_stride_7_1)), "r"((next_len > r0_5_1 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 7168), "l"(qn + (src_row_8_1 + 56 * row_stride_7_1)), "r"((next_len > r0_5_1 + 56) ? 16 : 0));
            int r0_10 = tid / 16;
            int c0_11 = tid % 16 * 8;
            long long row_stride_12 = (long long)num_v_heads * 128;
            long long src_row_13 = (long long)(next_tok0 + r0_10) * row_stride_12 + (long long)hv * 128 + (long long)c0_11;
            int dst0_14 = s_w_addr + (unsigned int)next_off + (unsigned int)(c0_11 / 64 * 8192 + (r0_10 * 128 + c0_11 % 64 * 2 ^ (r0_10 * 128 + c0_11 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14), "l"(w + src_row_13), "r"((next_len > r0_10) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 1024), "l"(w + (src_row_13 + 8 * row_stride_12)), "r"((next_len > r0_10 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 2048), "l"(w + (src_row_13 + 16 * row_stride_12)), "r"((next_len > r0_10 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 3072), "l"(w + (src_row_13 + 24 * row_stride_12)), "r"((next_len > r0_10 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 4096), "l"(w + (src_row_13 + 32 * row_stride_12)), "r"((next_len > r0_10 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 5120), "l"(w + (src_row_13 + 40 * row_stride_12)), "r"((next_len > r0_10 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 6144), "l"(w + (src_row_13 + 48 * row_stride_12)), "r"((next_len > r0_10 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 7168), "l"(w + (src_row_13 + 56 * row_stride_12)), "r"((next_len > r0_10 + 56) ? 16 : 0));
            if (ident == 0) {
                int r0_1 = tid / 8;
                int c0_2 = tid % 8 * 8;
                long long row_stride_3 = (long long)num_v_heads * 128;
                long long src_row_4 = (long long)(next_tok0 + r0_1) * row_stride_3 + (long long)hv * 128 + (long long)v0 + (long long)c0_2;
                int dst0_5 = s_do_addr + (unsigned int)next_off + (unsigned int)(c0_2 / 64 * 8192 + (r0_1 * 128 + c0_2 % 64 * 2 ^ (r0_1 * 128 + c0_2 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5), "l"(do_ + src_row_4), "r"((next_len > r0_1) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5 + 2048), "l"(do_ + (src_row_4 + 16 * row_stride_3)), "r"((next_len > r0_1 + 16) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5 + 4096), "l"(do_ + (src_row_4 + 32 * row_stride_3)), "r"((next_len > r0_1 + 32) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_5 + 6144), "l"(do_ + (src_row_4 + 48 * row_stride_3)), "r"((next_len > r0_1 + 48) ? 16 : 0));
                int r0_6 = tid / 8;
                int c0_7 = tid % 8 * 8;
                long long row_stride_8 = (long long)num_v_heads * 128;
                long long src_row_9 = (long long)(next_tok0 + r0_6) * row_stride_8 + (long long)hv * 128 + (long long)v0 + (long long)c0_7;
                int dst0_10 = s_dvl_addr + (unsigned int)next_off + (unsigned int)(c0_7 / 64 * 8192 + (r0_6 * 128 + c0_7 % 64 * 2 ^ (r0_6 * 128 + c0_7 % 64 * 2 >> 7 & 7) << 4));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_10), "l"(dv_local + src_row_9), "r"((next_len > r0_6) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_10 + 2048), "l"(dv_local + (src_row_9 + 16 * row_stride_8)), "r"((next_len > r0_6 + 16) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_10 + 4096), "l"(dv_local + (src_row_9 + 32 * row_stride_8)), "r"((next_len > r0_6 + 32) ? 16 : 0));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(dst0_10 + 6144), "l"(dv_local + (src_row_9 + 48 * row_stride_8)), "r"((next_len > r0_6 + 48) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        #pragma unroll
        for (int _lp = 0; _lp < 32; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_dh[_lp*2 + 0], acc_dh[_lp*2+1 + 0]));
            dhpk[_lp] = *(uint32_t*)&_bf2;
        }
        __syncthreads();
        int _max_11 = ((n_valid) > (1) ? (n_valid) : (1));
        float g_last = s_g[g_off + _max_11 - 1];
        float _exp2_12 = approx_exp2(g_last);
        const float2 _scale2_13 = {_exp2_12, _exp2_12};
        #pragma unroll
        for (int _ls = 0; _ls < 32; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_dh)[_ls], _scale2_13);
        #pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            #pragma unroll
            for (int nb3 = 0; nb3 < 4; nb3++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                    : "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((kbase + ks * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8)[0]), "=f"((acc_v + nb3 * 8)[1]), "=f"((acc_v + nb3 * 8)[2]), "=f"((acc_v + nb3 * 8)[3])
                    : "r"((dhpk + ks * 4)[0]), "r"((dhpk + ks * 4)[1]), "r"((dhpk + ks * 4)[2]), "r"((dhpk + ks * 4)[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8 + 4)[0]), "=f"((acc_v + nb3 * 8 + 4)[1]), "=f"((acc_v + nb3 * 8 + 4)[2]), "=f"((acc_v + nb3 * 8 + 4)[3])
                    : "r"((dhpk + ks * 4)[0]), "r"((dhpk + ks * 4)[1]), "r"((dhpk + ks * 4)[2]), "r"((dhpk + ks * 4)[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[3])));
            }
        }
        float dvn[4];
        #pragma unroll
        for (int tb = 0; tb < 4; tb++) {
            if (ident == 0) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(dvl_frag[0]), "=r"(dvl_frag[1]), "=r"(dvl_frag[2]), "=r"(dvl_frag[3])
                    : "r"(s_dvl_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
            } else {
                dvl_frag[0] = 0;
                dvl_frag[1] = 0;
                dvl_frag[2] = 0;
                dvl_frag[3] = 0;
            }
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int tok_1 = (tb * 2 + half) * 8 + quad * 2 + e;
                    float decay = 0.0f;
                    if (tok_1 < n_valid) {
                        float _exp2_14 = approx_exp2(g_last - s_g[g_off + tok_1]);
                        decay = _exp2_14;
                    }
                    float dl_a = ((e == 0) ? __uint_as_float(dvl_frag[half * 2] << 16) : __uint_as_float(dvl_frag[half * 2] & 4294901760u));
                    float dl_b = ((e == 0) ? __uint_as_float(dvl_frag[half * 2 + 1] << 16) : __uint_as_float(dvl_frag[half * 2 + 1] & 4294901760u));
                    dvn[e] = __fmaf_rn(acc_v[(tb * 2 + half) * 4 + e], decay, dl_a);
                    dvn[2 + e] = __fmaf_rn(acc_v[(tb * 2 + half) * 4 + 2 + e], decay, dl_b);
                    acc_v[(tb * 2 + half) * 4 + e] = -dvn[e];
                    acc_v[(tb * 2 + half) * 4 + 2 + e] = -dvn[2 + e];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dvn[_lp*2 + 0], dvn[_lp*2+1 + 0]));
                    vw[((tb * 2 + half) * 2) + _lp] = *(uint32_t*)&_bf2;
                }
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_v[_lp*2 + 0], acc_v[_lp*2+1 + 0]));
            ndvpk[_lp] = *(uint32_t*)&_bf2;
        }
        if (ident != 0) {
            a_frag_mn[0] = 0;
            a_frag_mn[1] = 0;
            a_frag_mn[2] = 0;
            a_frag_mn[3] = 0;
        }
        #pragma unroll
        for (int ks2 = 0; ks2 < 4; ks2++) {
            if (ident == 0) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
                    : "r"(s_do_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((ks2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((ks2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                float gain_lo = s_gain[g_off + ks2 * 16 + quad * 2];
                float gain_hi = s_gain[g_off + ks2 * 16 + quad * 2 + 1];
                float gain_lo8 = s_gain[g_off + ks2 * 16 + 8 + quad * 2];
                float gain_hi8 = s_gain[g_off + ks2 * 16 + 8 + quad * 2 + 1];
                uint32_t _bf16x2_scale_pair_15;
                {
                    uint32_t _bf16x2_pair_16 = a_frag_mn[0];
                    float _bf16x2_lo_16;
                    float _bf16x2_hi_16;
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_16) : "h"((uint16_t)(_bf16x2_pair_16 & 0xFFFFu)));
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_16) : "h"((uint16_t)(_bf16x2_pair_16 >> 16)));
                    _bf16x2_lo_16 *= gain_lo;
                    _bf16x2_hi_16 *= gain_hi;
                    uint32_t _bf16x2_out_16;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_16) : "f"(_bf16x2_hi_16), "f"(_bf16x2_lo_16));
                    _bf16x2_scale_pair_15 = _bf16x2_out_16;
                }
                a_frag_mn[0] = _bf16x2_scale_pair_15;
                uint32_t _bf16x2_scale_pair_17;
                {
                    uint32_t _bf16x2_pair_18 = a_frag_mn[1];
                    float _bf16x2_lo_18;
                    float _bf16x2_hi_18;
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_18) : "h"((uint16_t)(_bf16x2_pair_18 & 0xFFFFu)));
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_18) : "h"((uint16_t)(_bf16x2_pair_18 >> 16)));
                    _bf16x2_lo_18 *= gain_lo;
                    _bf16x2_hi_18 *= gain_hi;
                    uint32_t _bf16x2_out_18;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_18) : "f"(_bf16x2_hi_18), "f"(_bf16x2_lo_18));
                    _bf16x2_scale_pair_17 = _bf16x2_out_18;
                }
                a_frag_mn[1] = _bf16x2_scale_pair_17;
                uint32_t _bf16x2_scale_pair_19;
                {
                    uint32_t _bf16x2_pair_20 = a_frag_mn[2];
                    float _bf16x2_lo_20;
                    float _bf16x2_hi_20;
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_20) : "h"((uint16_t)(_bf16x2_pair_20 & 0xFFFFu)));
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_20) : "h"((uint16_t)(_bf16x2_pair_20 >> 16)));
                    _bf16x2_lo_20 *= gain_lo8;
                    _bf16x2_hi_20 *= gain_hi8;
                    uint32_t _bf16x2_out_20;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_20) : "f"(_bf16x2_hi_20), "f"(_bf16x2_lo_20));
                    _bf16x2_scale_pair_19 = _bf16x2_out_20;
                }
                a_frag_mn[2] = _bf16x2_scale_pair_19;
                uint32_t _bf16x2_scale_pair_21;
                {
                    uint32_t _bf16x2_pair_22 = a_frag_mn[3];
                    float _bf16x2_lo_22;
                    float _bf16x2_hi_22;
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_lo_22) : "h"((uint16_t)(_bf16x2_pair_22 & 0xFFFFu)));
                    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(_bf16x2_hi_22) : "h"((uint16_t)(_bf16x2_pair_22 >> 16)));
                    _bf16x2_lo_22 *= gain_lo8;
                    _bf16x2_hi_22 *= gain_hi8;
                    uint32_t _bf16x2_out_22;
                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_bf16x2_out_22) : "f"(_bf16x2_hi_22), "f"(_bf16x2_lo_22));
                    _bf16x2_scale_pair_21 = _bf16x2_out_22;
                }
                a_frag_mn[3] = _bf16x2_scale_pair_21;
            }
            #pragma unroll
            for (int nb5 = 0; nb5 < 8; nb5++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                    : "r"(s_qn_addr + (unsigned int)stage_off + (unsigned int)((kbase + nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dh + nb5 * 8)[0]), "+f"((acc_dh + nb5 * 8)[1]), "+f"((acc_dh + nb5 * 8)[2]), "+f"((acc_dh + nb5 * 8)[3])
                    : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dh + nb5 * 8 + 4)[0]), "+f"((acc_dh + nb5 * 8 + 4)[1]), "+f"((acc_dh + nb5 * 8 + 4)[2]), "+f"((acc_dh + nb5 * 8 + 4)[3])
                    : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                    : "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((kbase + nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dh + nb5 * 8)[0]), "+f"((acc_dh + nb5 * 8)[1]), "+f"((acc_dh + nb5 * 8)[2]), "+f"((acc_dh + nb5 * 8)[3])
                    : "r"((ndvpk + ks2 * 4)[0]), "r"((ndvpk + ks2 * 4)[1]), "r"((ndvpk + ks2 * 4)[2]), "r"((ndvpk + ks2 * 4)[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dh + nb5 * 8 + 4)[0]), "+f"((acc_dh + nb5 * 8 + 4)[1]), "+f"((acc_dh + nb5 * 8 + 4)[2]), "+f"((acc_dh + nb5 * 8 + 4)[3])
                    : "r"((ndvpk + ks2 * 4)[0]), "r"((ndvpk + ks2 * 4)[1]), "r"((ndvpk + ks2 * 4)[2]), "r"((ndvpk + ks2 * 4)[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
            }
        }
        if (num_chunks > rev + 1) {
            #pragma unroll
            for (int gi2 = 0; gi2 < 1; gi2++) {
                int gtok2 = gi2 * 128 + tid_2;
                if (gtok2 < 64) {
                    s_g[next_g_off + gtok2] = g_next[gi2];
                    float _exp2_23 = approx_exp2(g_next[gi2]);
                    s_gain[next_g_off + gtok2] = _exp2_23 * scale;
                }
            }
        }
        #pragma unroll
        for (int gi3 = 0; gi3 < 1; gi3++) {
            g_next[gi3] = g_far[gi3];
        }
        __syncthreads();
        stage_off = next_off;
        g_off = next_g_off;
    }
    asm volatile("cp.async.wait_group 0;");
    if (active != 0) {
        long long pre_base = ((long long)seg * nvh + hv64) * 32768;
        #pragma unroll
        for (int nb6 = 0; nb6 < 16; nb6++) {
            int kcol2 = kbase + nb6 * 8 + quad * 2;
            seg_state[pre_base + (long long)kcol2 * 256 + (long long)vrow_a] = acc_dh[nb6 * 4];
            seg_state[pre_base + (long long)(kcol2 + 1) * 256 + (long long)vrow_a] = acc_dh[nb6 * 4 + 1];
            seg_state[pre_base + (long long)kcol2 * 256 + (long long)vrow_b] = acc_dh[nb6 * 4 + 2];
            seg_state[pre_base + (long long)(kcol2 + 1) * 256 + (long long)vrow_b] = acc_dh[nb6 * 4 + 3];
        }
    }
}

} // extern "C"
