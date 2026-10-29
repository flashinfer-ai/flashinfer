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
#define SMEM_S_RED_STAGE_BYTES 16
#define SMEM_S_RED_STRIDE 16
#define SMEM_S_X32_OFF 0
#define SMEM_S_X32_STAGE_BYTES 32768
#define SMEM_S_X32_STRIDE 32768
#define SMEM_S_XHI_OFF 32768
#define SMEM_S_XHI_STAGE_BYTES 16384
#define SMEM_S_XHI_STRIDE 16384
#define SMEM_S_XLO_OFF 49152
#define SMEM_S_XLO_STAGE_BYTES 16384
#define SMEM_S_XLO_STRIDE 16384
#define SMEM_S_MHI_OFF 65536
#define SMEM_S_MHI_STAGE_BYTES 8192
#define SMEM_S_MHI_STRIDE 8192
#define SMEM_S_MLO_OFF 73728
#define SMEM_S_MLO_STAGE_BYTES 8192
#define SMEM_S_MLO_STRIDE 8192
#define SMEM_S_M32_OFF 82432
#define SMEM_S_M32_STAGE_BYTES 16384
#define SMEM_S_M32_STRIDE 16384
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
kernel_cake_gdn_det_train_de8e119d987f2d740f62(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, float* __restrict__ g_cs, float* __restrict__ h0, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ v_new, float* __restrict__ final_state, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int* __restrict__ seg_flag)
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
    float* s_x32 = reinterpret_cast<float*>(smem_raw + 0);
    const int s_x32_addr = smem + 0;
    __nv_bfloat16* s_xhi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_xhi_addr = smem + 32768;
    __nv_bfloat16* s_xlo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_xlo_addr = smem + 49152;
    __nv_bfloat16* s_mhi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_mhi_addr = smem + 65536;
    __nv_bfloat16* s_mlo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 73728);
    const int s_mlo_addr = smem + 73728;
    float* s_m32 = reinterpret_cast<float*>(smem_raw + 82432);
    const int s_m32_addr = smem + 82432;

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
    int _max_0 = ((seq) > (0) ? (seq) : (0));
    state_idx = _max_0;
    long long in_base = ((long long)state_idx * nvh + hv64) * 16384;
    int _max_1 = ((seq) > (0) ? (seq) : (0));
    long long out_base = ((long long)_max_1 * nvh + hv64) * 16384;
    int ident = 0;
    int active = 1;
    int early_c_read = c_begin;
    if (c_end <= c_begin) {
        early_c_read = 0;
    }
    int early_tok0 = chunk_start[early_c_read];
    int early_c_last = ((c_end - 1) > (0) ? (c_end - 1) : (0));
    int early_len = chunk_start[early_c_last] + chunk_len[early_c_last] - early_tok0;
    int early_off = 0;
    int _min_0 = ((early_len) < (64) ? (early_len) : (64));
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(early_tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    int dst0 = s_w_addr + (unsigned int)early_off + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
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
    int r0_3 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_5 = (long long)num_heads * 128;
    long long src_row_6 = (long long)(early_tok0 + r0_3) * row_stride_5 + (long long)hq * 128 + (long long)c0_4;
    int dst0_7 = s_kn_addr + (unsigned int)early_off + (unsigned int)(c0_4 / 64 * 8192 + (r0_3 * 128 + c0_4 % 64 * 2 ^ (r0_3 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7), "l"(kn + src_row_6), "r"((_min_0 > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 1024), "l"(kn + (src_row_6 + 8 * row_stride_5)), "r"((_min_0 > r0_3 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 2048), "l"(kn + (src_row_6 + 16 * row_stride_5)), "r"((_min_0 > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 3072), "l"(kn + (src_row_6 + 24 * row_stride_5)), "r"((_min_0 > r0_3 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 4096), "l"(kn + (src_row_6 + 32 * row_stride_5)), "r"((_min_0 > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 5120), "l"(kn + (src_row_6 + 40 * row_stride_5)), "r"((_min_0 > r0_3 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 6144), "l"(kn + (src_row_6 + 48 * row_stride_5)), "r"((_min_0 > r0_3 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 7168), "l"(kn + (src_row_6 + 56 * row_stride_5)), "r"((_min_0 > r0_3 + 56) ? 16 : 0));
    int r0_8 = tid / 8;
    int c0_9 = tid % 8 * 8;
    long long row_stride_10 = (long long)num_v_heads * 128;
    long long src_row_11 = (long long)(early_tok0 + r0_8) * row_stride_10 + (long long)hv * 128 + (long long)v0 + (long long)c0_9;
    int dst0_12 = s_u_addr + (unsigned int)early_off + (unsigned int)(c0_9 / 64 * 8192 + (r0_8 * 128 + c0_9 % 64 * 2 ^ (r0_8 * 128 + c0_9 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12), "l"(u + src_row_11), "r"((_min_0 > r0_8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 2048), "l"(u + (src_row_11 + 16 * row_stride_10)), "r"((_min_0 > r0_8 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 4096), "l"(u + (src_row_11 + 32 * row_stride_10)), "r"((_min_0 > r0_8 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 6144), "l"(u + (src_row_11 + 48 * row_stride_10)), "r"((_min_0 > r0_8 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int chain_ran = 0;
    if (is_first != 0) {
        if (use_initial_state != 0) {
            #pragma unroll
            for (int nb = 0; nb < 16; nb++) {
                int kcol = kbase + nb * 8 + quad * 2;
                acc_h[nb * 4] = h0[in_base + (long long)kcol * 128 + (long long)vrow_a];
                acc_h[nb * 4 + 1] = h0[in_base + (long long)(kcol + 1) * 128 + (long long)vrow_a];
                acc_h[nb * 4 + 2] = h0[in_base + (long long)kcol * 128 + (long long)vrow_b];
                acc_h[nb * 4 + 3] = h0[in_base + (long long)(kcol + 1) * 128 + (long long)vrow_b];
            }
        } else {
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
        }
    }
    if (is_first == 0) {
        if (seq < 0) {
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
        } else {
            long long nvh_0 = (long long)num_v_heads;
            long long hv64_1 = (long long)hv;
            int lane_2 = lane;
            int warp_3 = warp;
            int quad_4 = lane_2 % 4;
            int row_q = lane_2 / 4;
            int vloc_b = vloc_a + 8;
            int vrow_b_5 = vrow_a + 8;
            int replayed = 0;
            long long pre_base_p0 = ((long long)(seg - 1) * nvh_0 + hv64_1) * 32768;
            int kcol0 = kbase + quad_4 * 2;
            acc_h[0] = seg_state[pre_base_p0 + (long long)kcol0 * 256 + (long long)vrow_a];
            acc_h[1] = seg_state[pre_base_p0 + (long long)(kcol0 + 1) * 256 + (long long)vrow_a];
            acc_h[2] = seg_state[pre_base_p0 + (long long)kcol0 * 256 + (long long)vrow_b_5];
            acc_h[3] = seg_state[pre_base_p0 + (long long)(kcol0 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_6 = kbase + 8 + quad_4 * 2;
            acc_h[4] = seg_state[pre_base_p0 + (long long)kcol0_6 * 256 + (long long)vrow_a];
            acc_h[5] = seg_state[pre_base_p0 + (long long)(kcol0_6 + 1) * 256 + (long long)vrow_a];
            acc_h[6] = seg_state[pre_base_p0 + (long long)kcol0_6 * 256 + (long long)vrow_b_5];
            acc_h[7] = seg_state[pre_base_p0 + (long long)(kcol0_6 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_7 = kbase + 16 + quad_4 * 2;
            acc_h[8] = seg_state[pre_base_p0 + (long long)kcol0_7 * 256 + (long long)vrow_a];
            acc_h[9] = seg_state[pre_base_p0 + (long long)(kcol0_7 + 1) * 256 + (long long)vrow_a];
            acc_h[10] = seg_state[pre_base_p0 + (long long)kcol0_7 * 256 + (long long)vrow_b_5];
            acc_h[11] = seg_state[pre_base_p0 + (long long)(kcol0_7 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_8 = kbase + 24 + quad_4 * 2;
            acc_h[12] = seg_state[pre_base_p0 + (long long)kcol0_8 * 256 + (long long)vrow_a];
            acc_h[13] = seg_state[pre_base_p0 + (long long)(kcol0_8 + 1) * 256 + (long long)vrow_a];
            acc_h[14] = seg_state[pre_base_p0 + (long long)kcol0_8 * 256 + (long long)vrow_b_5];
            acc_h[15] = seg_state[pre_base_p0 + (long long)(kcol0_8 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_9 = kbase + 32 + quad_4 * 2;
            acc_h[16] = seg_state[pre_base_p0 + (long long)kcol0_9 * 256 + (long long)vrow_a];
            acc_h[17] = seg_state[pre_base_p0 + (long long)(kcol0_9 + 1) * 256 + (long long)vrow_a];
            acc_h[18] = seg_state[pre_base_p0 + (long long)kcol0_9 * 256 + (long long)vrow_b_5];
            acc_h[19] = seg_state[pre_base_p0 + (long long)(kcol0_9 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_10 = kbase + 40 + quad_4 * 2;
            acc_h[20] = seg_state[pre_base_p0 + (long long)kcol0_10 * 256 + (long long)vrow_a];
            acc_h[21] = seg_state[pre_base_p0 + (long long)(kcol0_10 + 1) * 256 + (long long)vrow_a];
            acc_h[22] = seg_state[pre_base_p0 + (long long)kcol0_10 * 256 + (long long)vrow_b_5];
            acc_h[23] = seg_state[pre_base_p0 + (long long)(kcol0_10 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_11 = kbase + 48 + quad_4 * 2;
            acc_h[24] = seg_state[pre_base_p0 + (long long)kcol0_11 * 256 + (long long)vrow_a];
            acc_h[25] = seg_state[pre_base_p0 + (long long)(kcol0_11 + 1) * 256 + (long long)vrow_a];
            acc_h[26] = seg_state[pre_base_p0 + (long long)kcol0_11 * 256 + (long long)vrow_b_5];
            acc_h[27] = seg_state[pre_base_p0 + (long long)(kcol0_11 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_12 = kbase + 56 + quad_4 * 2;
            acc_h[28] = seg_state[pre_base_p0 + (long long)kcol0_12 * 256 + (long long)vrow_a];
            acc_h[29] = seg_state[pre_base_p0 + (long long)(kcol0_12 + 1) * 256 + (long long)vrow_a];
            acc_h[30] = seg_state[pre_base_p0 + (long long)kcol0_12 * 256 + (long long)vrow_b_5];
            acc_h[31] = seg_state[pre_base_p0 + (long long)(kcol0_12 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_13 = kbase + 64 + quad_4 * 2;
            acc_h[32] = seg_state[pre_base_p0 + (long long)kcol0_13 * 256 + (long long)vrow_a];
            acc_h[33] = seg_state[pre_base_p0 + (long long)(kcol0_13 + 1) * 256 + (long long)vrow_a];
            acc_h[34] = seg_state[pre_base_p0 + (long long)kcol0_13 * 256 + (long long)vrow_b_5];
            acc_h[35] = seg_state[pre_base_p0 + (long long)(kcol0_13 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_14 = kbase + 72 + quad_4 * 2;
            acc_h[36] = seg_state[pre_base_p0 + (long long)kcol0_14 * 256 + (long long)vrow_a];
            acc_h[37] = seg_state[pre_base_p0 + (long long)(kcol0_14 + 1) * 256 + (long long)vrow_a];
            acc_h[38] = seg_state[pre_base_p0 + (long long)kcol0_14 * 256 + (long long)vrow_b_5];
            acc_h[39] = seg_state[pre_base_p0 + (long long)(kcol0_14 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_15 = kbase + 80 + quad_4 * 2;
            acc_h[40] = seg_state[pre_base_p0 + (long long)kcol0_15 * 256 + (long long)vrow_a];
            acc_h[41] = seg_state[pre_base_p0 + (long long)(kcol0_15 + 1) * 256 + (long long)vrow_a];
            acc_h[42] = seg_state[pre_base_p0 + (long long)kcol0_15 * 256 + (long long)vrow_b_5];
            acc_h[43] = seg_state[pre_base_p0 + (long long)(kcol0_15 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_16 = kbase + 88 + quad_4 * 2;
            acc_h[44] = seg_state[pre_base_p0 + (long long)kcol0_16 * 256 + (long long)vrow_a];
            acc_h[45] = seg_state[pre_base_p0 + (long long)(kcol0_16 + 1) * 256 + (long long)vrow_a];
            acc_h[46] = seg_state[pre_base_p0 + (long long)kcol0_16 * 256 + (long long)vrow_b_5];
            acc_h[47] = seg_state[pre_base_p0 + (long long)(kcol0_16 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_17 = kbase + 96 + quad_4 * 2;
            acc_h[48] = seg_state[pre_base_p0 + (long long)kcol0_17 * 256 + (long long)vrow_a];
            acc_h[49] = seg_state[pre_base_p0 + (long long)(kcol0_17 + 1) * 256 + (long long)vrow_a];
            acc_h[50] = seg_state[pre_base_p0 + (long long)kcol0_17 * 256 + (long long)vrow_b_5];
            acc_h[51] = seg_state[pre_base_p0 + (long long)(kcol0_17 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_18 = kbase + 104 + quad_4 * 2;
            acc_h[52] = seg_state[pre_base_p0 + (long long)kcol0_18 * 256 + (long long)vrow_a];
            acc_h[53] = seg_state[pre_base_p0 + (long long)(kcol0_18 + 1) * 256 + (long long)vrow_a];
            acc_h[54] = seg_state[pre_base_p0 + (long long)kcol0_18 * 256 + (long long)vrow_b_5];
            acc_h[55] = seg_state[pre_base_p0 + (long long)(kcol0_18 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_19 = kbase + 112 + quad_4 * 2;
            acc_h[56] = seg_state[pre_base_p0 + (long long)kcol0_19 * 256 + (long long)vrow_a];
            acc_h[57] = seg_state[pre_base_p0 + (long long)(kcol0_19 + 1) * 256 + (long long)vrow_a];
            acc_h[58] = seg_state[pre_base_p0 + (long long)kcol0_19 * 256 + (long long)vrow_b_5];
            acc_h[59] = seg_state[pre_base_p0 + (long long)(kcol0_19 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_20 = kbase + 120 + quad_4 * 2;
            acc_h[60] = seg_state[pre_base_p0 + (long long)kcol0_20 * 256 + (long long)vrow_a];
            acc_h[61] = seg_state[pre_base_p0 + (long long)(kcol0_20 + 1) * 256 + (long long)vrow_a];
            acc_h[62] = seg_state[pre_base_p0 + (long long)kcol0_20 * 256 + (long long)vrow_b_5];
            acc_h[63] = seg_state[pre_base_p0 + (long long)(kcol0_20 + 1) * 256 + (long long)vrow_b_5];
            int p = seg - 1;
            int p_first = 1;
            if (p > 0) {
                if (seg_seq[p - 1] == seq) {
                    p_first = 0;
                }
            }
            int has_m = seg_flag[(long long)p * nvh_0 + hv64_1];
            if (p_first != 0) {
                if (use_initial_state == 0) {
                    has_m = 0;
                }
            }
            int stop = 0;
            #pragma unroll 1
            for (int it = 0; it < seg; it++) {
                if (stop == 0) {
                    if (has_m == 0) {
                        stop = 1;
                    } else if (p_first != 0) {
                        stop = 1;
                    } else {
                        p = p - 1;
                        p_first = 1;
                        if (p > 0) {
                            if (seg_seq[p - 1] == seq) {
                                p_first = 0;
                            }
                        }
                        has_m = seg_flag[(long long)p * nvh_0 + hv64_1];
                        if (p_first != 0) {
                            if (use_initial_state == 0) {
                                has_m = 0;
                            }
                        }
                    }
                }
            }
            int q0 = p + 1;
            if (has_m != 0) {
                q0 = p;
            }
            int n_steps = seg - q0;
            long long pre_base_p = ((long long)p * nvh_0 + hv64_1) * 32768;
            if (n_steps > 0) {
                replayed = 1;
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                float acc[16];
                unsigned int a_hi[4];
                unsigned int a_lo[4];
                unsigned int b_hi[4];
                unsigned int b_lo[4];
                int m0 = warp_3 / 2 * 16;
                int nb0 = warp_3 % 2 * 2;
                if (has_m != 0) {
                    long long init_base = ((long long)seq * nvh_0 + hv64_1) * 16384;
                    int idx = tid * 4;
                    int row = idx / 64;
                    int col = idx % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row * 128 + (long long)col)));
                    int idx_0 = (128 + tid) * 4;
                    int row_1 = idx_0 / 64;
                    int col_2 = idx_0 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_0 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_1 * 128 + (long long)col_2)));
                    int idx_3 = (256 + tid) * 4;
                    int row_4 = idx_3 / 64;
                    int col_5 = idx_3 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_3 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_4 * 128 + (long long)col_5)));
                    int idx_6 = (384 + tid) * 4;
                    int row_7 = idx_6 / 64;
                    int col_8 = idx_6 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_6 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_7 * 128 + (long long)col_8)));
                    int idx_9 = (512 + tid) * 4;
                    int row_10 = idx_9 / 64;
                    int col_11 = idx_9 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_9 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_10 * 128 + (long long)col_11)));
                    int idx_12 = (640 + tid) * 4;
                    int row_13 = idx_12 / 64;
                    int col_14 = idx_12 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_12 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_13 * 128 + (long long)col_14)));
                    int idx_15 = (768 + tid) * 4;
                    int row_16 = idx_15 / 64;
                    int col_17 = idx_15 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_15 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_16 * 128 + (long long)col_17)));
                    int idx_18 = (896 + tid) * 4;
                    int row_19 = idx_18 / 64;
                    int col_20 = idx_18 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_18 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_19 * 128 + (long long)col_20)));
                    int idx_21 = (1024 + tid) * 4;
                    int row_22 = idx_21 / 64;
                    int col_23 = idx_21 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_21 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_22 * 128 + (long long)col_23)));
                    int idx_24 = (1152 + tid) * 4;
                    int row_25 = idx_24 / 64;
                    int col_26 = idx_24 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_24 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_25 * 128 + (long long)col_26)));
                    int idx_27 = (1280 + tid) * 4;
                    int row_28 = idx_27 / 64;
                    int col_29 = idx_27 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_27 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_28 * 128 + (long long)col_29)));
                    int idx_30 = (1408 + tid) * 4;
                    int row_31 = idx_30 / 64;
                    int col_32 = idx_30 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_30 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_31 * 128 + (long long)col_32)));
                    int idx_33 = (1536 + tid) * 4;
                    int row_34 = idx_33 / 64;
                    int col_35 = idx_33 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_33 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_34 * 128 + (long long)col_35)));
                    int idx_36 = (1664 + tid) * 4;
                    int row_37 = idx_36 / 64;
                    int col_38 = idx_36 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_36 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_37 * 128 + (long long)col_38)));
                    int idx_39 = (1792 + tid) * 4;
                    int row_40 = idx_39 / 64;
                    int col_41 = idx_39 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_39 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_40 * 128 + (long long)col_41)));
                    int idx_42 = (1920 + tid) * 4;
                    int row_43 = idx_42 / 64;
                    int col_44 = idx_42 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_42 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_43 * 128 + (long long)col_44)));
                } else {
                    int idx_1 = tid * 4;
                    int row_2 = idx_1 / 64;
                    int col_1 = idx_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2 * 256 + (long long)col_1)));
                    int idx_0_1 = (128 + tid) * 4;
                    int row_1_1 = idx_0_1 / 64;
                    int col_2_1 = idx_0_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_0_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_1_1 * 256 + (long long)col_2_1)));
                    int idx_3_1 = (256 + tid) * 4;
                    int row_4_1 = idx_3_1 / 64;
                    int col_5_1 = idx_3_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_3_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_4_1 * 256 + (long long)col_5_1)));
                    int idx_6_1 = (384 + tid) * 4;
                    int row_7_1 = idx_6_1 / 64;
                    int col_8_1 = idx_6_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_6_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_7_1 * 256 + (long long)col_8_1)));
                    int idx_9_1 = (512 + tid) * 4;
                    int row_10_1 = idx_9_1 / 64;
                    int col_11_1 = idx_9_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_9_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_10_1 * 256 + (long long)col_11_1)));
                    int idx_12_1 = (640 + tid) * 4;
                    int row_13_1 = idx_12_1 / 64;
                    int col_14_1 = idx_12_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_12_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_13_1 * 256 + (long long)col_14_1)));
                    int idx_15_1 = (768 + tid) * 4;
                    int row_16_1 = idx_15_1 / 64;
                    int col_17_1 = idx_15_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_15_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_16_1 * 256 + (long long)col_17_1)));
                    int idx_18_1 = (896 + tid) * 4;
                    int row_19_1 = idx_18_1 / 64;
                    int col_20_1 = idx_18_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_18_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_19_1 * 256 + (long long)col_20_1)));
                    int idx_21_1 = (1024 + tid) * 4;
                    int row_22_1 = idx_21_1 / 64;
                    int col_23_1 = idx_21_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_21_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_22_1 * 256 + (long long)col_23_1)));
                    int idx_24_1 = (1152 + tid) * 4;
                    int row_25_1 = idx_24_1 / 64;
                    int col_26_1 = idx_24_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_24_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_25_1 * 256 + (long long)col_26_1)));
                    int idx_27_1 = (1280 + tid) * 4;
                    int row_28_1 = idx_27_1 / 64;
                    int col_29_1 = idx_27_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_27_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_28_1 * 256 + (long long)col_29_1)));
                    int idx_30_1 = (1408 + tid) * 4;
                    int row_31_1 = idx_30_1 / 64;
                    int col_32_1 = idx_30_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_30_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_31_1 * 256 + (long long)col_32_1)));
                    int idx_33_1 = (1536 + tid) * 4;
                    int row_34_1 = idx_33_1 / 64;
                    int col_35_1 = idx_33_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_33_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_34_1 * 256 + (long long)col_35_1)));
                    int idx_36_1 = (1664 + tid) * 4;
                    int row_37_1 = idx_36_1 / 64;
                    int col_38_1 = idx_36_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_36_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_37_1 * 256 + (long long)col_38_1)));
                    int idx_39_1 = (1792 + tid) * 4;
                    int row_40_1 = idx_39_1 / 64;
                    int col_41_1 = idx_39_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_39_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_40_1 * 256 + (long long)col_41_1)));
                    int idx_42_1 = (1920 + tid) * 4;
                    int row_43_1 = idx_42_1 / 64;
                    int col_44_1 = idx_42_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_42_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_43_1 * 256 + (long long)col_44_1)));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                float vals[4];
                unsigned int hi[2];
                unsigned int lo[2];
                float res[4];
                #pragma unroll 4
                for (int i = 0; i < 16; i++) {
                    int idx_2 = (i * 128 + tid) * 4;
                    int row_3 = idx_2 / 64;
                    int col_3 = idx_2 % 64;
                    vals[0] = s_x32[idx_2];
                    vals[1] = s_x32[idx_2 + 1];
                    vals[2] = s_x32[idx_2 + 2];
                    vals[3] = s_x32[idx_2 + 3];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        hi[_lp] = *(uint32_t*)&_bf2;
                    }
                    res[0] = vals[0] - __uint_as_float(hi[0] << 16);
                    res[1] = vals[1] - __uint_as_float(hi[0] & 4294901760u);
                    res[2] = vals[2] - __uint_as_float(hi[1] << 16);
                    res[3] = vals[3] - __uint_as_float(hi[1] & 4294901760u);
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 0], res[_lp*2+1 + 0]));
                        lo[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xhi_addr + (unsigned int)(col_3 / 64 * 16384 + (row_3 * 128 + col_3 % 64 * 2 ^ (row_3 * 128 + col_3 % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xlo_addr + (unsigned int)(col_3 / 64 * 16384 + (row_3 * 128 + col_3 % 64 * 2 ^ (row_3 * 128 + col_3 % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
                }
                __syncthreads();
                #pragma unroll 1
                for (int st = 0; st < n_steps; st++) {
                    int q = q0 + st;
                    long long pre_base_q = ((long long)q * nvh_0 + hv64_1) * 32768;
                    int idx_4 = tid * 4;
                    int row_5 = idx_4 / 64;
                    int col_4 = idx_4 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_4 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_5 * 256 + (long long)col_4)));
                    int idx_0_2 = (128 + tid) * 4;
                    int row_1_2 = idx_0_2 / 64;
                    int col_2_2 = idx_0_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_0_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_1_2 * 256 + (long long)col_2_2)));
                    int idx_3_2 = (256 + tid) * 4;
                    int row_4_2 = idx_3_2 / 64;
                    int col_5_2 = idx_3_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_3_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_4_2 * 256 + (long long)col_5_2)));
                    int idx_6_2 = (384 + tid) * 4;
                    int row_7_2 = idx_6_2 / 64;
                    int col_8_2 = idx_6_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_6_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_7_2 * 256 + (long long)col_8_2)));
                    int idx_9_2 = (512 + tid) * 4;
                    int row_10_2 = idx_9_2 / 64;
                    int col_11_2 = idx_9_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_9_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_10_2 * 256 + (long long)col_11_2)));
                    int idx_12_2 = (640 + tid) * 4;
                    int row_13_2 = idx_12_2 / 64;
                    int col_14_2 = idx_12_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_12_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_13_2 * 256 + (long long)col_14_2)));
                    int idx_15_2 = (768 + tid) * 4;
                    int row_16_2 = idx_15_2 / 64;
                    int col_17_2 = idx_15_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_15_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_16_2 * 256 + (long long)col_17_2)));
                    int idx_18_2 = (896 + tid) * 4;
                    int row_19_2 = idx_18_2 / 64;
                    int col_20_2 = idx_18_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_18_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_19_2 * 256 + (long long)col_20_2)));
                    int idx_21_2 = (1024 + tid) * 4;
                    int row_22_2 = idx_21_2 / 64;
                    int col_23_2 = idx_21_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_21_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_22_2 * 256 + (long long)col_23_2)));
                    int idx_24_2 = (1152 + tid) * 4;
                    int row_25_2 = idx_24_2 / 64;
                    int col_26_2 = idx_24_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_24_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_25_2 * 256 + (long long)col_26_2)));
                    int idx_27_2 = (1280 + tid) * 4;
                    int row_28_2 = idx_27_2 / 64;
                    int col_29_2 = idx_27_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_27_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_28_2 * 256 + (long long)col_29_2)));
                    int idx_30_2 = (1408 + tid) * 4;
                    int row_31_2 = idx_30_2 / 64;
                    int col_32_2 = idx_30_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_30_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_31_2 * 256 + (long long)col_32_2)));
                    int idx_33_2 = (1536 + tid) * 4;
                    int row_34_2 = idx_33_2 / 64;
                    int col_35_2 = idx_33_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_33_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_34_2 * 256 + (long long)col_35_2)));
                    int idx_36_2 = (1664 + tid) * 4;
                    int row_37_2 = idx_36_2 / 64;
                    int col_38_2 = idx_36_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_36_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_37_2 * 256 + (long long)col_38_2)));
                    int idx_39_2 = (1792 + tid) * 4;
                    int row_40_2 = idx_39_2 / 64;
                    int col_41_2 = idx_39_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_39_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_40_2 * 256 + (long long)col_41_2)));
                    int idx_42_2 = (1920 + tid) * 4;
                    int row_43_2 = idx_42_2 / 64;
                    int col_44_2 = idx_42_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_42_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_43_2 * 256 + (long long)col_44_2)));
                    int idx_45 = tid * 4;
                    int row_46 = idx_45 / 128;
                    int col_47 = idx_45 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_45 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_46 * 256 + (long long)col_47)));
                    int idx_48 = (128 + tid) * 4;
                    int row_49 = idx_48 / 128;
                    int col_50 = idx_48 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_48 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_49 * 256 + (long long)col_50)));
                    int idx_51 = (256 + tid) * 4;
                    int row_52 = idx_51 / 128;
                    int col_53 = idx_51 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_51 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_52 * 256 + (long long)col_53)));
                    int idx_54 = (384 + tid) * 4;
                    int row_55 = idx_54 / 128;
                    int col_56 = idx_54 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_54 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_55 * 256 + (long long)col_56)));
                    int idx_57 = (512 + tid) * 4;
                    int row_58 = idx_57 / 128;
                    int col_59 = idx_57 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_57 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_58 * 256 + (long long)col_59)));
                    int idx_60 = (640 + tid) * 4;
                    int row_61 = idx_60 / 128;
                    int col_62 = idx_60 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_60 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_61 * 256 + (long long)col_62)));
                    int idx_63 = (768 + tid) * 4;
                    int row_64 = idx_63 / 128;
                    int col_65 = idx_63 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_63 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_64 * 256 + (long long)col_65)));
                    int idx_66 = (896 + tid) * 4;
                    int row_67 = idx_66 / 128;
                    int col_68 = idx_66 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_66 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_67 * 256 + (long long)col_68)));
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_69[4];
                    unsigned int hi_70[2];
                    unsigned int lo_71[2];
                    float res_72[4];
                    #pragma unroll 4
                    for (int i_1 = 0; i_1 < 8; i_1++) {
                        int idx_1_1 = (i_1 * 128 + tid) * 4;
                        int row_2_1 = idx_1_1 / 128;
                        int col_3_1 = idx_1_1 % 128;
                        vals_69[0] = s_m32[idx_1_1];
                        vals_69[1] = s_m32[idx_1_1 + 1];
                        vals_69[2] = s_m32[idx_1_1 + 2];
                        vals_69[3] = s_m32[idx_1_1 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_69[_lp*2 + 0], vals_69[_lp*2+1 + 0]));
                            hi_70[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_72[0] = vals_69[0] - __uint_as_float(hi_70[0] << 16);
                        res_72[1] = vals_69[1] - __uint_as_float(hi_70[0] & 4294901760u);
                        res_72[2] = vals_69[2] - __uint_as_float(hi_70[1] << 16);
                        res_72[3] = vals_69[3] - __uint_as_float(hi_70[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_72[_lp*2 + 0], res_72[_lp*2+1 + 0]));
                            lo_71[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_3_1 / 64 * 4096 + (row_2_1 * 128 + col_3_1 % 64 * 2 ^ (row_2_1 * 128 + col_3_1 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_70[0]), "r"(hi_70[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_3_1 / 64 * 4096 + (row_2_1 * 128 + col_3_1 % 64 * 2 ^ (row_2_1 * 128 + col_3_1 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_71[0]), "r"(lo_71[1]) : "memory");
                    }
                    __syncthreads();
                    int idx_73 = tid * 4;
                    int row_74 = idx_73 / 128;
                    int col_75 = idx_73 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_73 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_74 * 256 + (long long)col_75)));
                    int idx_76 = (128 + tid) * 4;
                    int row_77 = idx_76 / 128;
                    int col_78 = idx_76 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_76 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_77 * 256 + (long long)col_78)));
                    int idx_79 = (256 + tid) * 4;
                    int row_80 = idx_79 / 128;
                    int col_81 = idx_79 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_79 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_80 * 256 + (long long)col_81)));
                    int idx_82 = (384 + tid) * 4;
                    int row_83 = idx_82 / 128;
                    int col_84 = idx_82 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_82 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_83 * 256 + (long long)col_84)));
                    int idx_85 = (512 + tid) * 4;
                    int row_86 = idx_85 / 128;
                    int col_87 = idx_85 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_85 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_86 * 256 + (long long)col_87)));
                    int idx_88 = (640 + tid) * 4;
                    int row_89 = idx_88 / 128;
                    int col_90 = idx_88 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_88 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_89 * 256 + (long long)col_90)));
                    int idx_91 = (768 + tid) * 4;
                    int row_92 = idx_91 / 128;
                    int col_93 = idx_91 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_91 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_92 * 256 + (long long)col_93)));
                    int idx_94 = (896 + tid) * 4;
                    int row_95 = idx_94 / 128;
                    int col_96 = idx_94 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_94 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_95 * 256 + (long long)col_96)));
                    asm volatile("cp.async.commit_group;");
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    int krow = m0 + row_q;
                    int colv = nb0 * 16 + quad_4 * 2;
                    int ia = krow * 64 + colv;
                    int ib = (krow + 8) * 64 + colv;
                    s_x32[ia] = acc[0] + s_x32[ia];
                    s_x32[ia + 1] = acc[1] + s_x32[ia + 1];
                    s_x32[ib] = acc[2] + s_x32[ib];
                    s_x32[ib + 1] = acc[3] + s_x32[ib + 1];
                    int colv_97 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_98 = krow * 64 + colv_97;
                    int ib_99 = (krow + 8) * 64 + colv_97;
                    s_x32[ia_98] = acc[4] + s_x32[ia_98];
                    s_x32[ia_98 + 1] = acc[5] + s_x32[ia_98 + 1];
                    s_x32[ib_99] = acc[6] + s_x32[ib_99];
                    s_x32[ib_99 + 1] = acc[7] + s_x32[ib_99 + 1];
                    int colv_100 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_101 = krow * 64 + colv_100;
                    int ib_102 = (krow + 8) * 64 + colv_100;
                    s_x32[ia_101] = acc[8] + s_x32[ia_101];
                    s_x32[ia_101 + 1] = acc[9] + s_x32[ia_101 + 1];
                    s_x32[ib_102] = acc[10] + s_x32[ib_102];
                    s_x32[ib_102 + 1] = acc[11] + s_x32[ib_102 + 1];
                    int colv_103 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_104 = krow * 64 + colv_103;
                    int ib_105 = (krow + 8) * 64 + colv_103;
                    s_x32[ia_104] = acc[12] + s_x32[ia_104];
                    s_x32[ia_104 + 1] = acc[13] + s_x32[ia_104 + 1];
                    s_x32[ib_105] = acc[14] + s_x32[ib_105];
                    s_x32[ib_105 + 1] = acc[15] + s_x32[ib_105 + 1];
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_106[4];
                    unsigned int hi_107[2];
                    unsigned int lo_108[2];
                    float res_109[4];
                    #pragma unroll 4
                    for (int i_2 = 0; i_2 < 8; i_2++) {
                        int idx_1_2 = (i_2 * 128 + tid) * 4;
                        int row_2_2 = idx_1_2 / 128;
                        int col_3_2 = idx_1_2 % 128;
                        vals_106[0] = s_m32[idx_1_2];
                        vals_106[1] = s_m32[idx_1_2 + 1];
                        vals_106[2] = s_m32[idx_1_2 + 2];
                        vals_106[3] = s_m32[idx_1_2 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_106[_lp*2 + 0], vals_106[_lp*2+1 + 0]));
                            hi_107[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_109[0] = vals_106[0] - __uint_as_float(hi_107[0] << 16);
                        res_109[1] = vals_106[1] - __uint_as_float(hi_107[0] & 4294901760u);
                        res_109[2] = vals_106[2] - __uint_as_float(hi_107[1] << 16);
                        res_109[3] = vals_106[3] - __uint_as_float(hi_107[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_109[_lp*2 + 0], res_109[_lp*2+1 + 0]));
                            lo_108[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_3_2 / 64 * 4096 + (row_2_2 * 128 + col_3_2 % 64 * 2 ^ (row_2_2 * 128 + col_3_2 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_107[0]), "r"(hi_107[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_3_2 / 64 * 4096 + (row_2_2 * 128 + col_3_2 % 64 * 2 ^ (row_2_2 * 128 + col_3_2 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_108[0]), "r"(lo_108[1]) : "memory");
                    }
                    __syncthreads();
                    int idx_110 = tid * 4;
                    int row_111 = idx_110 / 128;
                    int col_112 = idx_110 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_110 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_111 * 256 + (long long)col_112)));
                    int idx_113 = (128 + tid) * 4;
                    int row_114 = idx_113 / 128;
                    int col_115 = idx_113 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_113 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_114 * 256 + (long long)col_115)));
                    int idx_116 = (256 + tid) * 4;
                    int row_117 = idx_116 / 128;
                    int col_118 = idx_116 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_116 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_117 * 256 + (long long)col_118)));
                    int idx_119 = (384 + tid) * 4;
                    int row_120 = idx_119 / 128;
                    int col_121 = idx_119 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_119 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_120 * 256 + (long long)col_121)));
                    int idx_122 = (512 + tid) * 4;
                    int row_123 = idx_122 / 128;
                    int col_124 = idx_122 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_122 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_123 * 256 + (long long)col_124)));
                    int idx_125 = (640 + tid) * 4;
                    int row_126 = idx_125 / 128;
                    int col_127 = idx_125 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_125 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_126 * 256 + (long long)col_127)));
                    int idx_128 = (768 + tid) * 4;
                    int row_129 = idx_128 / 128;
                    int col_130 = idx_128 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_128 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_129 * 256 + (long long)col_130)));
                    int idx_131 = (896 + tid) * 4;
                    int row_132 = idx_131 / 128;
                    int col_133 = idx_131 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_131 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_132 * 256 + (long long)col_133)));
                    asm volatile("cp.async.commit_group;");
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    int krow_134 = 32 + m0 + row_q;
                    int colv_135 = nb0 * 16 + quad_4 * 2;
                    int ia_136 = krow_134 * 64 + colv_135;
                    int ib_137 = (krow_134 + 8) * 64 + colv_135;
                    s_x32[ia_136] = acc[0] + s_x32[ia_136];
                    s_x32[ia_136 + 1] = acc[1] + s_x32[ia_136 + 1];
                    s_x32[ib_137] = acc[2] + s_x32[ib_137];
                    s_x32[ib_137 + 1] = acc[3] + s_x32[ib_137 + 1];
                    int colv_138 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_139 = krow_134 * 64 + colv_138;
                    int ib_140 = (krow_134 + 8) * 64 + colv_138;
                    s_x32[ia_139] = acc[4] + s_x32[ia_139];
                    s_x32[ia_139 + 1] = acc[5] + s_x32[ia_139 + 1];
                    s_x32[ib_140] = acc[6] + s_x32[ib_140];
                    s_x32[ib_140 + 1] = acc[7] + s_x32[ib_140 + 1];
                    int colv_141 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_142 = krow_134 * 64 + colv_141;
                    int ib_143 = (krow_134 + 8) * 64 + colv_141;
                    s_x32[ia_142] = acc[8] + s_x32[ia_142];
                    s_x32[ia_142 + 1] = acc[9] + s_x32[ia_142 + 1];
                    s_x32[ib_143] = acc[10] + s_x32[ib_143];
                    s_x32[ib_143 + 1] = acc[11] + s_x32[ib_143 + 1];
                    int colv_144 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_145 = krow_134 * 64 + colv_144;
                    int ib_146 = (krow_134 + 8) * 64 + colv_144;
                    s_x32[ia_145] = acc[12] + s_x32[ia_145];
                    s_x32[ia_145 + 1] = acc[13] + s_x32[ia_145 + 1];
                    s_x32[ib_146] = acc[14] + s_x32[ib_146];
                    s_x32[ib_146 + 1] = acc[15] + s_x32[ib_146 + 1];
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_147[4];
                    unsigned int hi_148[2];
                    unsigned int lo_149[2];
                    float res_150[4];
                    #pragma unroll 4
                    for (int i_3 = 0; i_3 < 8; i_3++) {
                        int idx_1_3 = (i_3 * 128 + tid) * 4;
                        int row_2_3 = idx_1_3 / 128;
                        int col_3_3 = idx_1_3 % 128;
                        vals_147[0] = s_m32[idx_1_3];
                        vals_147[1] = s_m32[idx_1_3 + 1];
                        vals_147[2] = s_m32[idx_1_3 + 2];
                        vals_147[3] = s_m32[idx_1_3 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_147[_lp*2 + 0], vals_147[_lp*2+1 + 0]));
                            hi_148[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_150[0] = vals_147[0] - __uint_as_float(hi_148[0] << 16);
                        res_150[1] = vals_147[1] - __uint_as_float(hi_148[0] & 4294901760u);
                        res_150[2] = vals_147[2] - __uint_as_float(hi_148[1] << 16);
                        res_150[3] = vals_147[3] - __uint_as_float(hi_148[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_150[_lp*2 + 0], res_150[_lp*2+1 + 0]));
                            lo_149[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_3_3 / 64 * 4096 + (row_2_3 * 128 + col_3_3 % 64 * 2 ^ (row_2_3 * 128 + col_3_3 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_148[0]), "r"(hi_148[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_3_3 / 64 * 4096 + (row_2_3 * 128 + col_3_3 % 64 * 2 ^ (row_2_3 * 128 + col_3_3 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_149[0]), "r"(lo_149[1]) : "memory");
                    }
                    __syncthreads();
                    int idx_151 = tid * 4;
                    int row_152 = idx_151 / 128;
                    int col_153 = idx_151 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_151 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_152 * 256 + (long long)col_153)));
                    int idx_154 = (128 + tid) * 4;
                    int row_155 = idx_154 / 128;
                    int col_156 = idx_154 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_154 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_155 * 256 + (long long)col_156)));
                    int idx_157 = (256 + tid) * 4;
                    int row_158 = idx_157 / 128;
                    int col_159 = idx_157 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_157 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_158 * 256 + (long long)col_159)));
                    int idx_160 = (384 + tid) * 4;
                    int row_161 = idx_160 / 128;
                    int col_162 = idx_160 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_160 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_161 * 256 + (long long)col_162)));
                    int idx_163 = (512 + tid) * 4;
                    int row_164 = idx_163 / 128;
                    int col_165 = idx_163 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_163 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_164 * 256 + (long long)col_165)));
                    int idx_166 = (640 + tid) * 4;
                    int row_167 = idx_166 / 128;
                    int col_168 = idx_166 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_166 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_167 * 256 + (long long)col_168)));
                    int idx_169 = (768 + tid) * 4;
                    int row_170 = idx_169 / 128;
                    int col_171 = idx_169 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_169 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_170 * 256 + (long long)col_171)));
                    int idx_172 = (896 + tid) * 4;
                    int row_173 = idx_172 / 128;
                    int col_174 = idx_172 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_172 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_173 * 256 + (long long)col_174)));
                    asm volatile("cp.async.commit_group;");
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    int krow_175 = 64 + m0 + row_q;
                    int colv_176 = nb0 * 16 + quad_4 * 2;
                    int ia_177 = krow_175 * 64 + colv_176;
                    int ib_178 = (krow_175 + 8) * 64 + colv_176;
                    s_x32[ia_177] = acc[0] + s_x32[ia_177];
                    s_x32[ia_177 + 1] = acc[1] + s_x32[ia_177 + 1];
                    s_x32[ib_178] = acc[2] + s_x32[ib_178];
                    s_x32[ib_178 + 1] = acc[3] + s_x32[ib_178 + 1];
                    int colv_179 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_180 = krow_175 * 64 + colv_179;
                    int ib_181 = (krow_175 + 8) * 64 + colv_179;
                    s_x32[ia_180] = acc[4] + s_x32[ia_180];
                    s_x32[ia_180 + 1] = acc[5] + s_x32[ia_180 + 1];
                    s_x32[ib_181] = acc[6] + s_x32[ib_181];
                    s_x32[ib_181 + 1] = acc[7] + s_x32[ib_181 + 1];
                    int colv_182 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_183 = krow_175 * 64 + colv_182;
                    int ib_184 = (krow_175 + 8) * 64 + colv_182;
                    s_x32[ia_183] = acc[8] + s_x32[ia_183];
                    s_x32[ia_183 + 1] = acc[9] + s_x32[ia_183 + 1];
                    s_x32[ib_184] = acc[10] + s_x32[ib_184];
                    s_x32[ib_184 + 1] = acc[11] + s_x32[ib_184 + 1];
                    int colv_185 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_186 = krow_175 * 64 + colv_185;
                    int ib_187 = (krow_175 + 8) * 64 + colv_185;
                    s_x32[ia_186] = acc[12] + s_x32[ia_186];
                    s_x32[ia_186 + 1] = acc[13] + s_x32[ia_186 + 1];
                    s_x32[ib_187] = acc[14] + s_x32[ib_187];
                    s_x32[ib_187 + 1] = acc[15] + s_x32[ib_187 + 1];
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_188[4];
                    unsigned int hi_189[2];
                    unsigned int lo_190[2];
                    float res_191[4];
                    #pragma unroll 4
                    for (int i_4 = 0; i_4 < 8; i_4++) {
                        int idx_1_4 = (i_4 * 128 + tid) * 4;
                        int row_2_4 = idx_1_4 / 128;
                        int col_3_4 = idx_1_4 % 128;
                        vals_188[0] = s_m32[idx_1_4];
                        vals_188[1] = s_m32[idx_1_4 + 1];
                        vals_188[2] = s_m32[idx_1_4 + 2];
                        vals_188[3] = s_m32[idx_1_4 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_188[_lp*2 + 0], vals_188[_lp*2+1 + 0]));
                            hi_189[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_191[0] = vals_188[0] - __uint_as_float(hi_189[0] << 16);
                        res_191[1] = vals_188[1] - __uint_as_float(hi_189[0] & 4294901760u);
                        res_191[2] = vals_188[2] - __uint_as_float(hi_189[1] << 16);
                        res_191[3] = vals_188[3] - __uint_as_float(hi_189[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_191[_lp*2 + 0], res_191[_lp*2+1 + 0]));
                            lo_190[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_3_4 / 64 * 4096 + (row_2_4 * 128 + col_3_4 % 64 * 2 ^ (row_2_4 * 128 + col_3_4 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_189[0]), "r"(hi_189[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_3_4 / 64 * 4096 + (row_2_4 * 128 + col_3_4 % 64 * 2 ^ (row_2_4 * 128 + col_3_4 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_190[0]), "r"(lo_190[1]) : "memory");
                    }
                    __syncthreads();
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    acc[4] = 0.0f;
                    acc[5] = 0.0f;
                    acc[6] = 0.0f;
                    acc[7] = 0.0f;
                    acc[8] = 0.0f;
                    acc[9] = 0.0f;
                    acc[10] = 0.0f;
                    acc[11] = 0.0f;
                    acc[12] = 0.0f;
                    acc[13] = 0.0f;
                    acc[14] = 0.0f;
                    acc[15] = 0.0f;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)(lane / 16 * 8 / 64 * 4096 + ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 ^ ((m0 + lane % 16) * 128 + lane / 16 * 8 % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ (lane % 16 * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((16 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((32 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (32 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((32 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((48 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (48 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((48 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((64 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (64 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((64 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((80 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (80 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((80 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((96 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (96 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((96 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                        : "r"(s_mhi_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                        : "r"(s_mlo_addr + (unsigned int)((112 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (112 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)((nb0 * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + (nb0 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                        : "r"(s_xhi_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                        : "r"(s_xlo_addr + (unsigned int)(((nb0 + 1) * 16 + lane / 16 * 8) / 64 * 16384 + ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 ^ ((112 + lane % 16) * 128 + ((nb0 + 1) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                    int krow_192 = 96 + m0 + row_q;
                    int colv_193 = nb0 * 16 + quad_4 * 2;
                    int ia_194 = krow_192 * 64 + colv_193;
                    int ib_195 = (krow_192 + 8) * 64 + colv_193;
                    s_x32[ia_194] = acc[0] + s_x32[ia_194];
                    s_x32[ia_194 + 1] = acc[1] + s_x32[ia_194 + 1];
                    s_x32[ib_195] = acc[2] + s_x32[ib_195];
                    s_x32[ib_195 + 1] = acc[3] + s_x32[ib_195 + 1];
                    int colv_196 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_197 = krow_192 * 64 + colv_196;
                    int ib_198 = (krow_192 + 8) * 64 + colv_196;
                    s_x32[ia_197] = acc[4] + s_x32[ia_197];
                    s_x32[ia_197 + 1] = acc[5] + s_x32[ia_197 + 1];
                    s_x32[ib_198] = acc[6] + s_x32[ib_198];
                    s_x32[ib_198 + 1] = acc[7] + s_x32[ib_198 + 1];
                    int colv_199 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_200 = krow_192 * 64 + colv_199;
                    int ib_201 = (krow_192 + 8) * 64 + colv_199;
                    s_x32[ia_200] = acc[8] + s_x32[ia_200];
                    s_x32[ia_200 + 1] = acc[9] + s_x32[ia_200 + 1];
                    s_x32[ib_201] = acc[10] + s_x32[ib_201];
                    s_x32[ib_201 + 1] = acc[11] + s_x32[ib_201 + 1];
                    int colv_202 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_203 = krow_192 * 64 + colv_202;
                    int ib_204 = (krow_192 + 8) * 64 + colv_202;
                    s_x32[ia_203] = acc[12] + s_x32[ia_203];
                    s_x32[ia_203 + 1] = acc[13] + s_x32[ia_203 + 1];
                    s_x32[ib_204] = acc[14] + s_x32[ib_204];
                    s_x32[ib_204 + 1] = acc[15] + s_x32[ib_204 + 1];
                    __syncthreads();
                    if (n_steps > st + 1) {
                        float vals_0[4];
                        unsigned int hi_1[2];
                        unsigned int lo_2[2];
                        float res_3[4];
                        #pragma unroll 4
                        for (int i_5 = 0; i_5 < 16; i_5++) {
                            int idx_1_5 = (i_5 * 128 + tid) * 4;
                            int row_2_5 = idx_1_5 / 64;
                            int col_3_5 = idx_1_5 % 64;
                            vals_0[0] = s_x32[idx_1_5];
                            vals_0[1] = s_x32[idx_1_5 + 1];
                            vals_0[2] = s_x32[idx_1_5 + 2];
                            vals_0[3] = s_x32[idx_1_5 + 3];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                                hi_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            res_3[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
                            res_3[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
                            res_3[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
                            res_3[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
                                lo_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xhi_addr + (unsigned int)(col_3_5 / 64 * 16384 + (row_2_5 * 128 + col_3_5 % 64 * 2 ^ (row_2_5 * 128 + col_3_5 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xlo_addr + (unsigned int)(col_3_5 / 64 * 16384 + (row_2_5 * 128 + col_3_5 % 64 * 2 ^ (row_2_5 * 128 + col_3_5 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
                        }
                        __syncthreads();
                    }
                }
                int kcol5 = kbase + quad_4 * 2;
                acc_h[0] = s_x32[kcol5 * 64 + vloc_a];
                acc_h[1] = s_x32[(kcol5 + 1) * 64 + vloc_a];
                acc_h[2] = s_x32[kcol5 * 64 + vloc_b];
                acc_h[3] = s_x32[(kcol5 + 1) * 64 + vloc_b];
                int kcol5_0 = kbase + 8 + quad_4 * 2;
                acc_h[4] = s_x32[kcol5_0 * 64 + vloc_a];
                acc_h[5] = s_x32[(kcol5_0 + 1) * 64 + vloc_a];
                acc_h[6] = s_x32[kcol5_0 * 64 + vloc_b];
                acc_h[7] = s_x32[(kcol5_0 + 1) * 64 + vloc_b];
                int kcol5_1 = kbase + 16 + quad_4 * 2;
                acc_h[8] = s_x32[kcol5_1 * 64 + vloc_a];
                acc_h[9] = s_x32[(kcol5_1 + 1) * 64 + vloc_a];
                acc_h[10] = s_x32[kcol5_1 * 64 + vloc_b];
                acc_h[11] = s_x32[(kcol5_1 + 1) * 64 + vloc_b];
                int kcol5_2 = kbase + 24 + quad_4 * 2;
                acc_h[12] = s_x32[kcol5_2 * 64 + vloc_a];
                acc_h[13] = s_x32[(kcol5_2 + 1) * 64 + vloc_a];
                acc_h[14] = s_x32[kcol5_2 * 64 + vloc_b];
                acc_h[15] = s_x32[(kcol5_2 + 1) * 64 + vloc_b];
                int kcol5_3 = kbase + 32 + quad_4 * 2;
                acc_h[16] = s_x32[kcol5_3 * 64 + vloc_a];
                acc_h[17] = s_x32[(kcol5_3 + 1) * 64 + vloc_a];
                acc_h[18] = s_x32[kcol5_3 * 64 + vloc_b];
                acc_h[19] = s_x32[(kcol5_3 + 1) * 64 + vloc_b];
                int kcol5_4 = kbase + 40 + quad_4 * 2;
                acc_h[20] = s_x32[kcol5_4 * 64 + vloc_a];
                acc_h[21] = s_x32[(kcol5_4 + 1) * 64 + vloc_a];
                acc_h[22] = s_x32[kcol5_4 * 64 + vloc_b];
                acc_h[23] = s_x32[(kcol5_4 + 1) * 64 + vloc_b];
                int kcol5_5 = kbase + 48 + quad_4 * 2;
                acc_h[24] = s_x32[kcol5_5 * 64 + vloc_a];
                acc_h[25] = s_x32[(kcol5_5 + 1) * 64 + vloc_a];
                acc_h[26] = s_x32[kcol5_5 * 64 + vloc_b];
                acc_h[27] = s_x32[(kcol5_5 + 1) * 64 + vloc_b];
                int kcol5_6 = kbase + 56 + quad_4 * 2;
                acc_h[28] = s_x32[kcol5_6 * 64 + vloc_a];
                acc_h[29] = s_x32[(kcol5_6 + 1) * 64 + vloc_a];
                acc_h[30] = s_x32[kcol5_6 * 64 + vloc_b];
                acc_h[31] = s_x32[(kcol5_6 + 1) * 64 + vloc_b];
                int kcol5_7 = kbase + 64 + quad_4 * 2;
                acc_h[32] = s_x32[kcol5_7 * 64 + vloc_a];
                acc_h[33] = s_x32[(kcol5_7 + 1) * 64 + vloc_a];
                acc_h[34] = s_x32[kcol5_7 * 64 + vloc_b];
                acc_h[35] = s_x32[(kcol5_7 + 1) * 64 + vloc_b];
                int kcol5_8 = kbase + 72 + quad_4 * 2;
                acc_h[36] = s_x32[kcol5_8 * 64 + vloc_a];
                acc_h[37] = s_x32[(kcol5_8 + 1) * 64 + vloc_a];
                acc_h[38] = s_x32[kcol5_8 * 64 + vloc_b];
                acc_h[39] = s_x32[(kcol5_8 + 1) * 64 + vloc_b];
                int kcol5_9 = kbase + 80 + quad_4 * 2;
                acc_h[40] = s_x32[kcol5_9 * 64 + vloc_a];
                acc_h[41] = s_x32[(kcol5_9 + 1) * 64 + vloc_a];
                acc_h[42] = s_x32[kcol5_9 * 64 + vloc_b];
                acc_h[43] = s_x32[(kcol5_9 + 1) * 64 + vloc_b];
                int kcol5_10 = kbase + 88 + quad_4 * 2;
                acc_h[44] = s_x32[kcol5_10 * 64 + vloc_a];
                acc_h[45] = s_x32[(kcol5_10 + 1) * 64 + vloc_a];
                acc_h[46] = s_x32[kcol5_10 * 64 + vloc_b];
                acc_h[47] = s_x32[(kcol5_10 + 1) * 64 + vloc_b];
                int kcol5_11 = kbase + 96 + quad_4 * 2;
                acc_h[48] = s_x32[kcol5_11 * 64 + vloc_a];
                acc_h[49] = s_x32[(kcol5_11 + 1) * 64 + vloc_a];
                acc_h[50] = s_x32[kcol5_11 * 64 + vloc_b];
                acc_h[51] = s_x32[(kcol5_11 + 1) * 64 + vloc_b];
                int kcol5_12 = kbase + 104 + quad_4 * 2;
                acc_h[52] = s_x32[kcol5_12 * 64 + vloc_a];
                acc_h[53] = s_x32[(kcol5_12 + 1) * 64 + vloc_a];
                acc_h[54] = s_x32[kcol5_12 * 64 + vloc_b];
                acc_h[55] = s_x32[(kcol5_12 + 1) * 64 + vloc_b];
                int kcol5_13 = kbase + 112 + quad_4 * 2;
                acc_h[56] = s_x32[kcol5_13 * 64 + vloc_a];
                acc_h[57] = s_x32[(kcol5_13 + 1) * 64 + vloc_a];
                acc_h[58] = s_x32[kcol5_13 * 64 + vloc_b];
                acc_h[59] = s_x32[(kcol5_13 + 1) * 64 + vloc_b];
                int kcol5_14 = kbase + 120 + quad_4 * 2;
                acc_h[60] = s_x32[kcol5_14 * 64 + vloc_a];
                acc_h[61] = s_x32[(kcol5_14 + 1) * 64 + vloc_a];
                acc_h[62] = s_x32[kcol5_14 * 64 + vloc_b];
                acc_h[63] = s_x32[(kcol5_14 + 1) * 64 + vloc_b];
                __syncthreads();
            }
            chain_ran = replayed;
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
    int stage_off = 0;
    int g_off = 0;
    if (chain_ran != 0) {
        int _min_1 = ((seq_len) < (64) ? (seq_len) : (64));
        int r0_0 = tid / 16;
        int c0_1 = tid % 16 * 8;
        long long row_stride_2 = (long long)num_v_heads * 128;
        long long src_row_3 = (long long)(seq_tok0 + r0_0) * row_stride_2 + (long long)hv * 128 + (long long)c0_1;
        int dst0_4 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4), "l"(w + src_row_3), "r"((_min_1 > r0_0) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 1024), "l"(w + (src_row_3 + 8 * row_stride_2)), "r"((_min_1 > r0_0 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 2048), "l"(w + (src_row_3 + 16 * row_stride_2)), "r"((_min_1 > r0_0 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 3072), "l"(w + (src_row_3 + 24 * row_stride_2)), "r"((_min_1 > r0_0 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 4096), "l"(w + (src_row_3 + 32 * row_stride_2)), "r"((_min_1 > r0_0 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 5120), "l"(w + (src_row_3 + 40 * row_stride_2)), "r"((_min_1 > r0_0 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 6144), "l"(w + (src_row_3 + 48 * row_stride_2)), "r"((_min_1 > r0_0 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 7168), "l"(w + (src_row_3 + 56 * row_stride_2)), "r"((_min_1 > r0_0 + 56) ? 16 : 0));
        int r0_5 = tid / 16;
        int c0_6 = tid % 16 * 8;
        long long row_stride_7 = (long long)num_heads * 128;
        long long src_row_8 = (long long)(seq_tok0 + r0_5) * row_stride_7 + (long long)hq * 128 + (long long)c0_6;
        int dst0_9 = s_kn_addr + (unsigned int)stage_off + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9), "l"(kn + src_row_8), "r"((_min_1 > r0_5) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 1024), "l"(kn + (src_row_8 + 8 * row_stride_7)), "r"((_min_1 > r0_5 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 2048), "l"(kn + (src_row_8 + 16 * row_stride_7)), "r"((_min_1 > r0_5 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 3072), "l"(kn + (src_row_8 + 24 * row_stride_7)), "r"((_min_1 > r0_5 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 4096), "l"(kn + (src_row_8 + 32 * row_stride_7)), "r"((_min_1 > r0_5 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 5120), "l"(kn + (src_row_8 + 40 * row_stride_7)), "r"((_min_1 > r0_5 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 6144), "l"(kn + (src_row_8 + 48 * row_stride_7)), "r"((_min_1 > r0_5 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 7168), "l"(kn + (src_row_8 + 56 * row_stride_7)), "r"((_min_1 > r0_5 + 56) ? 16 : 0));
        int r0_10 = tid / 8;
        int c0_11 = tid % 8 * 8;
        long long row_stride_12 = (long long)num_v_heads * 128;
        long long src_row_13 = (long long)(seq_tok0 + r0_10) * row_stride_12 + (long long)hv * 128 + (long long)v0 + (long long)c0_11;
        int dst0_14 = s_u_addr + (unsigned int)stage_off + (unsigned int)(c0_11 / 64 * 8192 + (r0_10 * 128 + c0_11 % 64 * 2 ^ (r0_10 * 128 + c0_11 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_14), "l"(u + src_row_13), "r"((_min_1 > r0_10) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_14 + 2048), "l"(u + (src_row_13 + 16 * row_stride_12)), "r"((_min_1 > r0_10 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_14 + 4096), "l"(u + (src_row_13 + 32 * row_stride_12)), "r"((_min_1 > r0_10 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_14 + 6144), "l"(u + (src_row_13 + 48 * row_stride_12)), "r"((_min_1 > r0_10 + 48) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
    }
    int _min_2 = ((seq_len) < (64) ? (seq_len) : (64));
    int tok = tid;
    if (tok < 64) {
        float gate_val = 0.0f;
        if (tok < _min_2) {
            gate_val = g_cs[(long long)(seq_tok0 + tok) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tok] = gate_val;
    }
    int _min_3 = ((seq_len - 64) < (64) ? (seq_len - 64) : (64));
    int tok_13 = tid;
    g_next[0] = 0.0f;
    if (tok_13 < _min_3) {
        g_next[0] = g_cs[(long long)(seq_tok0 + 64 + tok_13) * (long long)num_v_heads + (long long)hv];
    }
    #pragma unroll 1
    for (int it_chunk = 0; it_chunk < num_chunks; it_chunk++) {
        int chunk = c_begin + it_chunk;
        int tok0 = seq_tok0 + it_chunk * 64;
        int n_valid = ((seq_len - it_chunk * 64) < (64) ? (seq_len - it_chunk * 64) : (64));
        int next_off = 40960 - stage_off;
        int next_g_off = 64 - g_off;
        int _min_4 = ((seq_len - (it_chunk + 2) * 64) < (64) ? (seq_len - (it_chunk + 2) * 64) : (64));
        int tok_0 = tid;
        g_far[0] = 0.0f;
        if (tok_0 < _min_4) {
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
            int r0_5_1 = tid / 16;
            int c0_6_1 = tid % 16 * 8;
            long long row_stride_7_1 = (long long)num_heads * 128;
            long long src_row_8_1 = (long long)(next_tok0 + r0_5_1) * row_stride_7_1 + (long long)hq * 128 + (long long)c0_6_1;
            int dst0_9_1 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_6_1 / 64 * 8192 + (r0_5_1 * 128 + c0_6_1 % 64 * 2 ^ (r0_5_1 * 128 + c0_6_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1), "l"(kn + src_row_8_1), "r"((next_len > r0_5_1) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 1024), "l"(kn + (src_row_8_1 + 8 * row_stride_7_1)), "r"((next_len > r0_5_1 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 2048), "l"(kn + (src_row_8_1 + 16 * row_stride_7_1)), "r"((next_len > r0_5_1 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 3072), "l"(kn + (src_row_8_1 + 24 * row_stride_7_1)), "r"((next_len > r0_5_1 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 4096), "l"(kn + (src_row_8_1 + 32 * row_stride_7_1)), "r"((next_len > r0_5_1 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 5120), "l"(kn + (src_row_8_1 + 40 * row_stride_7_1)), "r"((next_len > r0_5_1 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 6144), "l"(kn + (src_row_8_1 + 48 * row_stride_7_1)), "r"((next_len > r0_5_1 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9_1 + 7168), "l"(kn + (src_row_8_1 + 56 * row_stride_7_1)), "r"((next_len > r0_5_1 + 56) ? 16 : 0));
            int r0_10_1 = tid / 8;
            int c0_11_1 = tid % 8 * 8;
            long long row_stride_12_1 = (long long)num_v_heads * 128;
            long long src_row_13_1 = (long long)(next_tok0 + r0_10_1) * row_stride_12_1 + (long long)hv * 128 + (long long)v0 + (long long)c0_11_1;
            int dst0_14_1 = s_u_addr + (unsigned int)next_off + (unsigned int)(c0_11_1 / 64 * 8192 + (r0_10_1 * 128 + c0_11_1 % 64 * 2 ^ (r0_10_1 * 128 + c0_11_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14_1), "l"(u + src_row_13_1), "r"((next_len > r0_10_1) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14_1 + 2048), "l"(u + (src_row_13_1 + 16 * row_stride_12_1)), "r"((next_len > r0_10_1 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14_1 + 4096), "l"(u + (src_row_13_1 + 32 * row_stride_12_1)), "r"((next_len > r0_10_1 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14_1 + 6144), "l"(u + (src_row_13_1 + 48 * row_stride_12_1)), "r"((next_len > r0_10_1 + 48) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
        }
        int _max_2 = ((n_valid) > (1) ? (n_valid) : (1));
        float g_last = s_g[g_off + _max_2 - 1];
        float _exp2_3 = approx_exp2(g_last);
        const float2 _scale2_4 = {_exp2_3, _exp2_3};
        #pragma unroll
        for (int _ls = 0; _ls < 32; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_h)[_ls], _scale2_4);
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
        long long tape_base = ((long long)chunk * nvh + hv64) * 16384 + (long long)(v0 + vwarp * 16) * 128 + (long long)kbase;
        int m_idx = lane_0 / 8;
        int r_idx = lane_0 % 8;
        int st_row = m_idx % 2 * 8 + r_idx;
        int st_nb = m_idx / 2;
        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&hpk[0])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[1])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[2])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[3]))
            : "memory");
        int st_nb_1 = 2 + m_idx / 2;
        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_1 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&hpk[4])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[5])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[6])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[7]))
            : "memory");
        int st_nb_2 = 4 + m_idx / 2;
        uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_2 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&hpk[8])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[9])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[10])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[11]))
            : "memory");
        int st_nb_3 = 6 + m_idx / 2;
        uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_3 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&hpk[12])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[13])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[14])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[15]))
            : "memory");
        int st_nb_4 = 8 + m_idx / 2;
        uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_4 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&hpk[16])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[17])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[18])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[19]))
            : "memory");
        int st_nb_5 = 10 + m_idx / 2;
        uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_5 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&hpk[20])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[21])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[22])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[23]))
            : "memory");
        int st_nb_6 = 12 + m_idx / 2;
        uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_6 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&hpk[24])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[25])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[26])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[27]))
            : "memory");
        int st_nb_7 = 14 + m_idx / 2;
        uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_7 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&hpk[28])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[29])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[30])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[31]))
            : "memory");
        __syncwarp();
        int ld_row_lo = lane_0 / 16;
        int ld_ch = lane_0 % 16;
        int ld_row = ld_row_lo;
        int words[4];
        {
            const int4* _ivptr_13 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row * 64 + ((ld_ch ^ ld_row % 8) << 2));
            int4 _ivld_13;
            _ivld_13 = *_ivptr_13;
            words[0 + 0] = _ivld_13.x;
            words[0 + 1] = _ivld_13.y;
            words[0 + 2] = _ivld_13.z;
            words[0 + 3] = _ivld_13.w;
        }
        {
            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_8 = 2 + ld_row_lo;
        int words_9[4];
        {
            const int4* _ivptr_14 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_8 * 64 + ((ld_ch ^ ld_row_8 % 8) << 2));
            int4 _ivld_14;
            _ivld_14 = *_ivptr_14;
            words_9[0 + 0] = _ivld_14.x;
            words_9[0 + 1] = _ivld_14.y;
            words_9[0 + 2] = _ivld_14.z;
            words_9[0 + 3] = _ivld_14.w;
        }
        {
            int4 _iv4 = make_int4(words_9[0 + 0], words_9[0 + 1], words_9[0 + 2], words_9[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_8 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_10 = 4 + ld_row_lo;
        int words_11[4];
        {
            const int4* _ivptr_15 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_10 * 64 + ((ld_ch ^ ld_row_10 % 8) << 2));
            int4 _ivld_15;
            _ivld_15 = *_ivptr_15;
            words_11[0 + 0] = _ivld_15.x;
            words_11[0 + 1] = _ivld_15.y;
            words_11[0 + 2] = _ivld_15.z;
            words_11[0 + 3] = _ivld_15.w;
        }
        {
            int4 _iv4 = make_int4(words_11[0 + 0], words_11[0 + 1], words_11[0 + 2], words_11[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_10 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_12 = 6 + ld_row_lo;
        int words_13[4];
        {
            const int4* _ivptr_16 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_12 * 64 + ((ld_ch ^ ld_row_12 % 8) << 2));
            int4 _ivld_16;
            _ivld_16 = *_ivptr_16;
            words_13[0 + 0] = _ivld_16.x;
            words_13[0 + 1] = _ivld_16.y;
            words_13[0 + 2] = _ivld_16.z;
            words_13[0 + 3] = _ivld_16.w;
        }
        {
            int4 _iv4 = make_int4(words_13[0 + 0], words_13[0 + 1], words_13[0 + 2], words_13[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_12 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_14 = 8 + ld_row_lo;
        int words_15[4];
        {
            const int4* _ivptr_17 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_14 * 64 + ((ld_ch ^ ld_row_14 % 8) << 2));
            int4 _ivld_17;
            _ivld_17 = *_ivptr_17;
            words_15[0 + 0] = _ivld_17.x;
            words_15[0 + 1] = _ivld_17.y;
            words_15[0 + 2] = _ivld_17.z;
            words_15[0 + 3] = _ivld_17.w;
        }
        {
            int4 _iv4 = make_int4(words_15[0 + 0], words_15[0 + 1], words_15[0 + 2], words_15[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_14 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_16 = 10 + ld_row_lo;
        int words_17[4];
        {
            const int4* _ivptr_18 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_16 * 64 + ((ld_ch ^ ld_row_16 % 8) << 2));
            int4 _ivld_18;
            _ivld_18 = *_ivptr_18;
            words_17[0 + 0] = _ivld_18.x;
            words_17[0 + 1] = _ivld_18.y;
            words_17[0 + 2] = _ivld_18.z;
            words_17[0 + 3] = _ivld_18.w;
        }
        {
            int4 _iv4 = make_int4(words_17[0 + 0], words_17[0 + 1], words_17[0 + 2], words_17[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_16 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_18 = 12 + ld_row_lo;
        int words_19[4];
        {
            const int4* _ivptr_19 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_18 * 64 + ((ld_ch ^ ld_row_18 % 8) << 2));
            int4 _ivld_19;
            _ivld_19 = *_ivptr_19;
            words_19[0 + 0] = _ivld_19.x;
            words_19[0 + 1] = _ivld_19.y;
            words_19[0 + 2] = _ivld_19.z;
            words_19[0 + 3] = _ivld_19.w;
        }
        {
            int4 _iv4 = make_int4(words_19[0 + 0], words_19[0 + 1], words_19[0 + 2], words_19[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_18 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_20 = 14 + ld_row_lo;
        int words_21[4];
        {
            const int4* _ivptr_20 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_20 * 64 + ((ld_ch ^ ld_row_20 % 8) << 2));
            int4 _ivld_20;
            _ivld_20 = *_ivptr_20;
            words_21[0 + 0] = _ivld_20.x;
            words_21[0 + 1] = _ivld_20.y;
            words_21[0 + 2] = _ivld_20.z;
            words_21[0 + 3] = _ivld_20.w;
        }
        {
            int4 _iv4 = make_int4(words_21[0 + 0], words_21[0 + 1], words_21[0 + 2], words_21[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_20 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        __syncwarp();
        float vn[4];
        #pragma unroll
        for (int tb = 0; tb < 4; tb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(u_frag[0]), "=r"(u_frag[1]), "=r"(u_frag[2]), "=r"(u_frag[3])
                : "r"(s_u_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int tok_1 = (tb * 2 + half) * 8 + quad * 2 + e;
                    float decay = 0.0f;
                    if (tok_1 < n_valid) {
                        float _exp2_21 = approx_exp2(g_last - s_g[g_off + tok_1]);
                        decay = _exp2_21;
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
        if (khalf == 0) {
            long long row_base = ((long long)tok0 * nvh + hv64) * 128 + (long long)v0 + (long long)(vwarp * 16);
            int m_idx_0 = lane_0 / 8;
            int r_idx_1 = lane_0 % 8;
            int st_half = m_idx_0 % 2;
            int st_tok = m_idx_0 / 2 * 8 + r_idx_1;
            uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(stage_base + st_tok * 32 + ((st_half ^ st_tok >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&vw[0])), "r"(*reinterpret_cast<const uint32_t*>(&vw[1])), "r"(*reinterpret_cast<const uint32_t*>(&vw[2])), "r"(*reinterpret_cast<const uint32_t*>(&vw[3]))
                : "memory");
            int st_tok_2 = (2 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(stage_base + st_tok_2 * 32 + ((st_half ^ st_tok_2 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&vw[4])), "r"(*reinterpret_cast<const uint32_t*>(&vw[5])), "r"(*reinterpret_cast<const uint32_t*>(&vw[6])), "r"(*reinterpret_cast<const uint32_t*>(&vw[7]))
                : "memory");
            int st_tok_3 = (4 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(stage_base + st_tok_3 * 32 + ((st_half ^ st_tok_3 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&vw[8])), "r"(*reinterpret_cast<const uint32_t*>(&vw[9])), "r"(*reinterpret_cast<const uint32_t*>(&vw[10])), "r"(*reinterpret_cast<const uint32_t*>(&vw[11]))
                : "memory");
            int st_tok_4 = (6 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(stage_base + st_tok_4 * 32 + ((st_half ^ st_tok_4 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&vw[12])), "r"(*reinterpret_cast<const uint32_t*>(&vw[13])), "r"(*reinterpret_cast<const uint32_t*>(&vw[14])), "r"(*reinterpret_cast<const uint32_t*>(&vw[15]))
                : "memory");
            __syncwarp();
            int ld_half = lane_0 % 2;
            int ld_tok = lane_0 / 2;
            int words_5[4];
            {
                const int4* _ivptr_26 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok * 8 + ((ld_half ^ ld_tok >> 2 & 1) << 2));
                int4 _ivld_26;
                _ivld_26 = *_ivptr_26;
                words_5[0 + 0] = _ivld_26.x;
                words_5[0 + 1] = _ivld_26.y;
                words_5[0 + 2] = _ivld_26.z;
                words_5[0 + 3] = _ivld_26.w;
            }
            if (ld_tok < n_valid) {
                {
                    int4 _iv4 = make_int4(words_5[0 + 0], words_5[0 + 1], words_5[0 + 2], words_5[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_6 = 16 + lane_0 / 2;
            int words_7[4];
            {
                const int4* _ivptr_27 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_6 * 8 + ((ld_half ^ ld_tok_6 >> 2 & 1) << 2));
                int4 _ivld_27;
                _ivld_27 = *_ivptr_27;
                words_7[0 + 0] = _ivld_27.x;
                words_7[0 + 1] = _ivld_27.y;
                words_7[0 + 2] = _ivld_27.z;
                words_7[0 + 3] = _ivld_27.w;
            }
            if (ld_tok_6 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_7[0 + 0], words_7[0 + 1], words_7[0 + 2], words_7[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_6 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_8 = 32 + lane_0 / 2;
            int words_10[4];
            {
                const int4* _ivptr_28 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_8 * 8 + ((ld_half ^ ld_tok_8 >> 2 & 1) << 2));
                int4 _ivld_28;
                _ivld_28 = *_ivptr_28;
                words_10[0 + 0] = _ivld_28.x;
                words_10[0 + 1] = _ivld_28.y;
                words_10[0 + 2] = _ivld_28.z;
                words_10[0 + 3] = _ivld_28.w;
            }
            if (ld_tok_8 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_10[0 + 0], words_10[0 + 1], words_10[0 + 2], words_10[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_8 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_11 = 48 + lane_0 / 2;
            int words_12[4];
            {
                const int4* _ivptr_29 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_11 * 8 + ((ld_half ^ ld_tok_11 >> 2 & 1) << 2));
                int4 _ivld_29;
                _ivld_29 = *_ivptr_29;
                words_12[0 + 0] = _ivld_29.x;
                words_12[0 + 1] = _ivld_29.y;
                words_12[0 + 2] = _ivld_29.z;
                words_12[0 + 3] = _ivld_29.w;
            }
            if (ld_tok_11 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_12[0 + 0], words_12[0 + 1], words_12[0 + 2], words_12[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_11 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            __syncwarp();
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
    if (store_final_state != 0) {
        if (is_last != 0) {
            #pragma unroll
            for (int nb6 = 0; nb6 < 16; nb6++) {
                int kcol2 = kbase + nb6 * 8 + quad * 2;
                final_state[out_base + (long long)kcol2 * 128 + (long long)vrow_a] = acc_h[nb6 * 4];
                final_state[out_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_a] = acc_h[nb6 * 4 + 1];
                final_state[out_base + (long long)kcol2 * 128 + (long long)vrow_b] = acc_h[nb6 * 4 + 2];
                final_state[out_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_b] = acc_h[nb6 * 4 + 3];
            }
        }
    }
}

} // extern "C"
