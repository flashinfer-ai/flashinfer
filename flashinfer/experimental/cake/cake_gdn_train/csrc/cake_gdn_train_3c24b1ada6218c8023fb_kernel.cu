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
kernel_cake_gdn_train_3c24b1ada6218c8023fb(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, float* __restrict__ g_cs, float* __restrict__ h0, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ v_new, float* __restrict__ final_state, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int* __restrict__ seg_flag)
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
    int r0_0 = tid / 16;
    int c0_0 = tid % 16 * 8;
    long long row_stride_0 = (long long)num_heads * 128;
    long long src_row_0 = (long long)(early_tok0 + r0_0) * row_stride_0 + (long long)hq * 128 + (long long)c0_0;
    int dst0_0 = s_kn_addr + (unsigned int)early_off + (unsigned int)(c0_0 / 64 * 8192 + (r0_0 * 128 + c0_0 % 64 * 2 ^ (r0_0 * 128 + c0_0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0), "l"(kn + src_row_0), "r"((_min_0 > r0_0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 1024), "l"(kn + (src_row_0 + 8 * row_stride_0)), "r"((_min_0 > r0_0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 2048), "l"(kn + (src_row_0 + 16 * row_stride_0)), "r"((_min_0 > r0_0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 3072), "l"(kn + (src_row_0 + 24 * row_stride_0)), "r"((_min_0 > r0_0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 4096), "l"(kn + (src_row_0 + 32 * row_stride_0)), "r"((_min_0 > r0_0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 5120), "l"(kn + (src_row_0 + 40 * row_stride_0)), "r"((_min_0 > r0_0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 6144), "l"(kn + (src_row_0 + 48 * row_stride_0)), "r"((_min_0 > r0_0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_0 + 7168), "l"(kn + (src_row_0 + 56 * row_stride_0)), "r"((_min_0 > r0_0 + 56) ? 16 : 0));
    int r0_1 = tid / 8;
    int c0_1 = tid % 8 * 8;
    long long row_stride_1 = (long long)num_v_heads * 128;
    long long src_row_1 = (long long)(early_tok0 + r0_1) * row_stride_1 + (long long)hv * 128 + (long long)v0 + (long long)c0_1;
    int dst0_1 = s_u_addr + (unsigned int)early_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_1 * 128 + c0_1 % 64 * 2 ^ (r0_1 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1), "l"(u + src_row_1), "r"((_min_0 > r0_1) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1 + 2048), "l"(u + (src_row_1 + 16 * row_stride_1)), "r"((_min_0 > r0_1 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1 + 4096), "l"(u + (src_row_1 + 32 * row_stride_1)), "r"((_min_0 > r0_1 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_1 + 6144), "l"(u + (src_row_1 + 48 * row_stride_1)), "r"((_min_0 > r0_1 + 48) ? 16 : 0));
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
            int kcol0_1_0 = kbase + quad_4 * 2;
            acc_h[0] = seg_state[pre_base_p0 + (long long)kcol0_1_0 * 256 + (long long)vrow_a];
            acc_h[1] = seg_state[pre_base_p0 + (long long)(kcol0_1_0 + 1) * 256 + (long long)vrow_a];
            acc_h[2] = seg_state[pre_base_p0 + (long long)kcol0_1_0 * 256 + (long long)vrow_b_5];
            acc_h[3] = seg_state[pre_base_p0 + (long long)(kcol0_1_0 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_1 = kbase + 8 + quad_4 * 2;
            acc_h[4] = seg_state[pre_base_p0 + (long long)kcol0_1_1 * 256 + (long long)vrow_a];
            acc_h[5] = seg_state[pre_base_p0 + (long long)(kcol0_1_1 + 1) * 256 + (long long)vrow_a];
            acc_h[6] = seg_state[pre_base_p0 + (long long)kcol0_1_1 * 256 + (long long)vrow_b_5];
            acc_h[7] = seg_state[pre_base_p0 + (long long)(kcol0_1_1 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_2 = kbase + 16 + quad_4 * 2;
            acc_h[8] = seg_state[pre_base_p0 + (long long)kcol0_1_2 * 256 + (long long)vrow_a];
            acc_h[9] = seg_state[pre_base_p0 + (long long)(kcol0_1_2 + 1) * 256 + (long long)vrow_a];
            acc_h[10] = seg_state[pre_base_p0 + (long long)kcol0_1_2 * 256 + (long long)vrow_b_5];
            acc_h[11] = seg_state[pre_base_p0 + (long long)(kcol0_1_2 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_3 = kbase + 24 + quad_4 * 2;
            acc_h[12] = seg_state[pre_base_p0 + (long long)kcol0_1_3 * 256 + (long long)vrow_a];
            acc_h[13] = seg_state[pre_base_p0 + (long long)(kcol0_1_3 + 1) * 256 + (long long)vrow_a];
            acc_h[14] = seg_state[pre_base_p0 + (long long)kcol0_1_3 * 256 + (long long)vrow_b_5];
            acc_h[15] = seg_state[pre_base_p0 + (long long)(kcol0_1_3 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_4 = kbase + 32 + quad_4 * 2;
            acc_h[16] = seg_state[pre_base_p0 + (long long)kcol0_1_4 * 256 + (long long)vrow_a];
            acc_h[17] = seg_state[pre_base_p0 + (long long)(kcol0_1_4 + 1) * 256 + (long long)vrow_a];
            acc_h[18] = seg_state[pre_base_p0 + (long long)kcol0_1_4 * 256 + (long long)vrow_b_5];
            acc_h[19] = seg_state[pre_base_p0 + (long long)(kcol0_1_4 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_5 = kbase + 40 + quad_4 * 2;
            acc_h[20] = seg_state[pre_base_p0 + (long long)kcol0_1_5 * 256 + (long long)vrow_a];
            acc_h[21] = seg_state[pre_base_p0 + (long long)(kcol0_1_5 + 1) * 256 + (long long)vrow_a];
            acc_h[22] = seg_state[pre_base_p0 + (long long)kcol0_1_5 * 256 + (long long)vrow_b_5];
            acc_h[23] = seg_state[pre_base_p0 + (long long)(kcol0_1_5 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_6 = kbase + 48 + quad_4 * 2;
            acc_h[24] = seg_state[pre_base_p0 + (long long)kcol0_1_6 * 256 + (long long)vrow_a];
            acc_h[25] = seg_state[pre_base_p0 + (long long)(kcol0_1_6 + 1) * 256 + (long long)vrow_a];
            acc_h[26] = seg_state[pre_base_p0 + (long long)kcol0_1_6 * 256 + (long long)vrow_b_5];
            acc_h[27] = seg_state[pre_base_p0 + (long long)(kcol0_1_6 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_7 = kbase + 56 + quad_4 * 2;
            acc_h[28] = seg_state[pre_base_p0 + (long long)kcol0_1_7 * 256 + (long long)vrow_a];
            acc_h[29] = seg_state[pre_base_p0 + (long long)(kcol0_1_7 + 1) * 256 + (long long)vrow_a];
            acc_h[30] = seg_state[pre_base_p0 + (long long)kcol0_1_7 * 256 + (long long)vrow_b_5];
            acc_h[31] = seg_state[pre_base_p0 + (long long)(kcol0_1_7 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_8 = kbase + 64 + quad_4 * 2;
            acc_h[32] = seg_state[pre_base_p0 + (long long)kcol0_1_8 * 256 + (long long)vrow_a];
            acc_h[33] = seg_state[pre_base_p0 + (long long)(kcol0_1_8 + 1) * 256 + (long long)vrow_a];
            acc_h[34] = seg_state[pre_base_p0 + (long long)kcol0_1_8 * 256 + (long long)vrow_b_5];
            acc_h[35] = seg_state[pre_base_p0 + (long long)(kcol0_1_8 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_9 = kbase + 72 + quad_4 * 2;
            acc_h[36] = seg_state[pre_base_p0 + (long long)kcol0_1_9 * 256 + (long long)vrow_a];
            acc_h[37] = seg_state[pre_base_p0 + (long long)(kcol0_1_9 + 1) * 256 + (long long)vrow_a];
            acc_h[38] = seg_state[pre_base_p0 + (long long)kcol0_1_9 * 256 + (long long)vrow_b_5];
            acc_h[39] = seg_state[pre_base_p0 + (long long)(kcol0_1_9 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_10 = kbase + 80 + quad_4 * 2;
            acc_h[40] = seg_state[pre_base_p0 + (long long)kcol0_1_10 * 256 + (long long)vrow_a];
            acc_h[41] = seg_state[pre_base_p0 + (long long)(kcol0_1_10 + 1) * 256 + (long long)vrow_a];
            acc_h[42] = seg_state[pre_base_p0 + (long long)kcol0_1_10 * 256 + (long long)vrow_b_5];
            acc_h[43] = seg_state[pre_base_p0 + (long long)(kcol0_1_10 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_11 = kbase + 88 + quad_4 * 2;
            acc_h[44] = seg_state[pre_base_p0 + (long long)kcol0_1_11 * 256 + (long long)vrow_a];
            acc_h[45] = seg_state[pre_base_p0 + (long long)(kcol0_1_11 + 1) * 256 + (long long)vrow_a];
            acc_h[46] = seg_state[pre_base_p0 + (long long)kcol0_1_11 * 256 + (long long)vrow_b_5];
            acc_h[47] = seg_state[pre_base_p0 + (long long)(kcol0_1_11 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_12 = kbase + 96 + quad_4 * 2;
            acc_h[48] = seg_state[pre_base_p0 + (long long)kcol0_1_12 * 256 + (long long)vrow_a];
            acc_h[49] = seg_state[pre_base_p0 + (long long)(kcol0_1_12 + 1) * 256 + (long long)vrow_a];
            acc_h[50] = seg_state[pre_base_p0 + (long long)kcol0_1_12 * 256 + (long long)vrow_b_5];
            acc_h[51] = seg_state[pre_base_p0 + (long long)(kcol0_1_12 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_13 = kbase + 104 + quad_4 * 2;
            acc_h[52] = seg_state[pre_base_p0 + (long long)kcol0_1_13 * 256 + (long long)vrow_a];
            acc_h[53] = seg_state[pre_base_p0 + (long long)(kcol0_1_13 + 1) * 256 + (long long)vrow_a];
            acc_h[54] = seg_state[pre_base_p0 + (long long)kcol0_1_13 * 256 + (long long)vrow_b_5];
            acc_h[55] = seg_state[pre_base_p0 + (long long)(kcol0_1_13 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_14 = kbase + 112 + quad_4 * 2;
            acc_h[56] = seg_state[pre_base_p0 + (long long)kcol0_1_14 * 256 + (long long)vrow_a];
            acc_h[57] = seg_state[pre_base_p0 + (long long)(kcol0_1_14 + 1) * 256 + (long long)vrow_a];
            acc_h[58] = seg_state[pre_base_p0 + (long long)kcol0_1_14 * 256 + (long long)vrow_b_5];
            acc_h[59] = seg_state[pre_base_p0 + (long long)(kcol0_1_14 + 1) * 256 + (long long)vrow_b_5];
            int kcol0_1_15 = kbase + 120 + quad_4 * 2;
            acc_h[60] = seg_state[pre_base_p0 + (long long)kcol0_1_15 * 256 + (long long)vrow_a];
            acc_h[61] = seg_state[pre_base_p0 + (long long)(kcol0_1_15 + 1) * 256 + (long long)vrow_a];
            acc_h[62] = seg_state[pre_base_p0 + (long long)kcol0_1_15 * 256 + (long long)vrow_b_5];
            acc_h[63] = seg_state[pre_base_p0 + (long long)(kcol0_1_15 + 1) * 256 + (long long)vrow_b_5];
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
                    int idx_1_0 = tid * 4;
                    int row_2_0 = idx_1_0 / 64;
                    int col_1_0 = idx_1_0 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_0 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_0 * 128 + (long long)col_1_0)));
                    int idx_1_1 = (128 + tid) * 4;
                    int row_2_1 = idx_1_1 / 64;
                    int col_1_1 = idx_1_1 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_1 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_1 * 128 + (long long)col_1_1)));
                    int idx_1_2 = (256 + tid) * 4;
                    int row_2_2 = idx_1_2 / 64;
                    int col_1_2 = idx_1_2 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_2 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_2 * 128 + (long long)col_1_2)));
                    int idx_1_3 = (384 + tid) * 4;
                    int row_2_3 = idx_1_3 / 64;
                    int col_1_3 = idx_1_3 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_3 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_3 * 128 + (long long)col_1_3)));
                    int idx_1_4 = (512 + tid) * 4;
                    int row_2_4 = idx_1_4 / 64;
                    int col_1_4 = idx_1_4 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_4 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_4 * 128 + (long long)col_1_4)));
                    int idx_1_5 = (640 + tid) * 4;
                    int row_2_5 = idx_1_5 / 64;
                    int col_1_5 = idx_1_5 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_5 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_5 * 128 + (long long)col_1_5)));
                    int idx_1_6 = (768 + tid) * 4;
                    int row_2_6 = idx_1_6 / 64;
                    int col_1_6 = idx_1_6 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_6 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_6 * 128 + (long long)col_1_6)));
                    int idx_1_7 = (896 + tid) * 4;
                    int row_2_7 = idx_1_7 / 64;
                    int col_1_7 = idx_1_7 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_7 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_7 * 128 + (long long)col_1_7)));
                    int idx_1_8 = (1024 + tid) * 4;
                    int row_2_8 = idx_1_8 / 64;
                    int col_1_8 = idx_1_8 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_8 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_8 * 128 + (long long)col_1_8)));
                    int idx_1_9 = (1152 + tid) * 4;
                    int row_2_9 = idx_1_9 / 64;
                    int col_1_9 = idx_1_9 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_9 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_9 * 128 + (long long)col_1_9)));
                    int idx_1_10 = (1280 + tid) * 4;
                    int row_2_10 = idx_1_10 / 64;
                    int col_1_10 = idx_1_10 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_10 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_10 * 128 + (long long)col_1_10)));
                    int idx_1_11 = (1408 + tid) * 4;
                    int row_2_11 = idx_1_11 / 64;
                    int col_1_11 = idx_1_11 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_11 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_11 * 128 + (long long)col_1_11)));
                    int idx_1_12 = (1536 + tid) * 4;
                    int row_2_12 = idx_1_12 / 64;
                    int col_1_12 = idx_1_12 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_12 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_12 * 128 + (long long)col_1_12)));
                    int idx_1_13 = (1664 + tid) * 4;
                    int row_2_13 = idx_1_13 / 64;
                    int col_1_13 = idx_1_13 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_13 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_13 * 128 + (long long)col_1_13)));
                    int idx_1_14 = (1792 + tid) * 4;
                    int row_2_14 = idx_1_14 / 64;
                    int col_1_14 = idx_1_14 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_14 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_14 * 128 + (long long)col_1_14)));
                    int idx_1_15 = (1920 + tid) * 4;
                    int row_2_15 = idx_1_15 / 64;
                    int col_1_15 = idx_1_15 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_15 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_2_15 * 128 + (long long)col_1_15)));
                } else {
                    int idx_1_16 = tid * 4;
                    int row_2_16 = idx_1_16 / 64;
                    int col_1_16 = idx_1_16 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_16 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_16 * 256 + (long long)col_1_16)));
                    int idx_1_17 = (128 + tid) * 4;
                    int row_2_17 = idx_1_17 / 64;
                    int col_1_17 = idx_1_17 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_17 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_17 * 256 + (long long)col_1_17)));
                    int idx_1_18 = (256 + tid) * 4;
                    int row_2_18 = idx_1_18 / 64;
                    int col_1_18 = idx_1_18 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_18 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_18 * 256 + (long long)col_1_18)));
                    int idx_1_19 = (384 + tid) * 4;
                    int row_2_19 = idx_1_19 / 64;
                    int col_1_19 = idx_1_19 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_19 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_19 * 256 + (long long)col_1_19)));
                    int idx_1_20 = (512 + tid) * 4;
                    int row_2_20 = idx_1_20 / 64;
                    int col_1_20 = idx_1_20 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_20 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_20 * 256 + (long long)col_1_20)));
                    int idx_1_21 = (640 + tid) * 4;
                    int row_2_21 = idx_1_21 / 64;
                    int col_1_21 = idx_1_21 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_21 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_21 * 256 + (long long)col_1_21)));
                    int idx_1_22 = (768 + tid) * 4;
                    int row_2_22 = idx_1_22 / 64;
                    int col_1_22 = idx_1_22 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_22 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_22 * 256 + (long long)col_1_22)));
                    int idx_1_23 = (896 + tid) * 4;
                    int row_2_23 = idx_1_23 / 64;
                    int col_1_23 = idx_1_23 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_23 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_23 * 256 + (long long)col_1_23)));
                    int idx_1_24 = (1024 + tid) * 4;
                    int row_2_24 = idx_1_24 / 64;
                    int col_1_24 = idx_1_24 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_24 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_24 * 256 + (long long)col_1_24)));
                    int idx_1_25 = (1152 + tid) * 4;
                    int row_2_25 = idx_1_25 / 64;
                    int col_1_25 = idx_1_25 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_25 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_25 * 256 + (long long)col_1_25)));
                    int idx_1_26 = (1280 + tid) * 4;
                    int row_2_26 = idx_1_26 / 64;
                    int col_1_26 = idx_1_26 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_26 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_26 * 256 + (long long)col_1_26)));
                    int idx_1_27 = (1408 + tid) * 4;
                    int row_2_27 = idx_1_27 / 64;
                    int col_1_27 = idx_1_27 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_27 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_27 * 256 + (long long)col_1_27)));
                    int idx_1_28 = (1536 + tid) * 4;
                    int row_2_28 = idx_1_28 / 64;
                    int col_1_28 = idx_1_28 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_28 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_28 * 256 + (long long)col_1_28)));
                    int idx_1_29 = (1664 + tid) * 4;
                    int row_2_29 = idx_1_29 / 64;
                    int col_1_29 = idx_1_29 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_29 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_29 * 256 + (long long)col_1_29)));
                    int idx_1_30 = (1792 + tid) * 4;
                    int row_2_30 = idx_1_30 / 64;
                    int col_1_30 = idx_1_30 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_30 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_30 * 256 + (long long)col_1_30)));
                    int idx_1_31 = (1920 + tid) * 4;
                    int row_2_31 = idx_1_31 / 64;
                    int col_1_31 = idx_1_31 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_31 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2_31 * 256 + (long long)col_1_31)));
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
                    int idx = (i * 128 + tid) * 4;
                    int row = idx / 64;
                    int col = idx % 64;
                    vals[0] = s_x32[idx];
                    vals[1] = s_x32[idx + 1];
                    vals[2] = s_x32[idx + 2];
                    vals[3] = s_x32[idx + 3];
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
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xhi_addr + (unsigned int)(col / 64 * 16384 + (row * 128 + col % 64 * 2 ^ (row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xlo_addr + (unsigned int)(col / 64 * 16384 + (row * 128 + col % 64 * 2 ^ (row * 128 + col % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
                }
                __syncthreads();
                #pragma unroll 1
                for (int st = 0; st < n_steps; st++) {
                    int q = q0 + st;
                    long long pre_base_q = ((long long)q * nvh_0 + hv64_1) * 32768;
                    int idx_1_32 = tid * 4;
                    int row_2_32 = idx_1_32 / 64;
                    int col_1_32 = idx_1_32 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_32 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_32 * 256 + (long long)col_1_32)));
                    int idx_1_33 = (128 + tid) * 4;
                    int row_2_33 = idx_1_33 / 64;
                    int col_1_33 = idx_1_33 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_33 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_33 * 256 + (long long)col_1_33)));
                    int idx_1_34 = (256 + tid) * 4;
                    int row_2_34 = idx_1_34 / 64;
                    int col_1_34 = idx_1_34 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_34 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_34 * 256 + (long long)col_1_34)));
                    int idx_1_35 = (384 + tid) * 4;
                    int row_2_35 = idx_1_35 / 64;
                    int col_1_35 = idx_1_35 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_35 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_35 * 256 + (long long)col_1_35)));
                    int idx_1_36 = (512 + tid) * 4;
                    int row_2_36 = idx_1_36 / 64;
                    int col_1_36 = idx_1_36 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_36 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_36 * 256 + (long long)col_1_36)));
                    int idx_1_37 = (640 + tid) * 4;
                    int row_2_37 = idx_1_37 / 64;
                    int col_1_37 = idx_1_37 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_37 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_37 * 256 + (long long)col_1_37)));
                    int idx_1_38 = (768 + tid) * 4;
                    int row_2_38 = idx_1_38 / 64;
                    int col_1_38 = idx_1_38 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_38 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_38 * 256 + (long long)col_1_38)));
                    int idx_1_39 = (896 + tid) * 4;
                    int row_2_39 = idx_1_39 / 64;
                    int col_1_39 = idx_1_39 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_39 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_39 * 256 + (long long)col_1_39)));
                    int idx_1_40 = (1024 + tid) * 4;
                    int row_2_40 = idx_1_40 / 64;
                    int col_1_40 = idx_1_40 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_40 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_40 * 256 + (long long)col_1_40)));
                    int idx_1_41 = (1152 + tid) * 4;
                    int row_2_41 = idx_1_41 / 64;
                    int col_1_41 = idx_1_41 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_41 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_41 * 256 + (long long)col_1_41)));
                    int idx_1_42 = (1280 + tid) * 4;
                    int row_2_42 = idx_1_42 / 64;
                    int col_1_42 = idx_1_42 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_42 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_42 * 256 + (long long)col_1_42)));
                    int idx_1_43 = (1408 + tid) * 4;
                    int row_2_43 = idx_1_43 / 64;
                    int col_1_43 = idx_1_43 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_43 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_43 * 256 + (long long)col_1_43)));
                    int idx_1_44 = (1536 + tid) * 4;
                    int row_2_44 = idx_1_44 / 64;
                    int col_1_44 = idx_1_44 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_44 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_44 * 256 + (long long)col_1_44)));
                    int idx_1_45 = (1664 + tid) * 4;
                    int row_2_45 = idx_1_45 / 64;
                    int col_1_45 = idx_1_45 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_45 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_45 * 256 + (long long)col_1_45)));
                    int idx_1_46 = (1792 + tid) * 4;
                    int row_2_46 = idx_1_46 / 64;
                    int col_1_46 = idx_1_46 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_46 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_46 * 256 + (long long)col_1_46)));
                    int idx_1_47 = (1920 + tid) * 4;
                    int row_2_47 = idx_1_47 / 64;
                    int col_1_47 = idx_1_47 % 64;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1_47 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_2_47 * 256 + (long long)col_1_47)));
                    int idx_0 = tid * 4;
                    int row_0 = idx_0 / 128;
                    int col_0 = idx_0 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_0 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_0 * 256 + (long long)col_0)));
                    int idx_1 = (128 + tid) * 4;
                    int row_1 = idx_1 / 128;
                    int col_1 = idx_1 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_1 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_1 * 256 + (long long)col_1)));
                    int idx_2 = (256 + tid) * 4;
                    int row_2 = idx_2 / 128;
                    int col_2 = idx_2 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_2 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_2 * 256 + (long long)col_2)));
                    int idx_3 = (384 + tid) * 4;
                    int row_3 = idx_3 / 128;
                    int col_3 = idx_3 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_3 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_3 * 256 + (long long)col_3)));
                    int idx_4 = (512 + tid) * 4;
                    int row_4 = idx_4 / 128;
                    int col_4 = idx_4 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_4 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_4 * 256 + (long long)col_4)));
                    int idx_5 = (640 + tid) * 4;
                    int row_5 = idx_5 / 128;
                    int col_5 = idx_5 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_5 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_5 * 256 + (long long)col_5)));
                    int idx_6 = (768 + tid) * 4;
                    int row_6 = idx_6 / 128;
                    int col_6 = idx_6 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_6 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_6 * 256 + (long long)col_6)));
                    int idx_7 = (896 + tid) * 4;
                    int row_7 = idx_7 / 128;
                    int col_7 = idx_7 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_7 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_7 * 256 + (long long)col_7)));
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_0[4];
                    unsigned int hi_0[2];
                    unsigned int lo_0[2];
                    float res_0[4];
                    #pragma unroll 4
                    for (int i_1 = 0; i_1 < 8; i_1++) {
                        int idx_8 = (i_1 * 128 + tid) * 4;
                        int row_8 = idx_8 / 128;
                        int col_8 = idx_8 % 128;
                        vals_0[0] = s_m32[idx_8];
                        vals_0[1] = s_m32[idx_8 + 1];
                        vals_0[2] = s_m32[idx_8 + 2];
                        vals_0[3] = s_m32[idx_8 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                            hi_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_0[0] = vals_0[0] - __uint_as_float(hi_0[0] << 16);
                        res_0[1] = vals_0[1] - __uint_as_float(hi_0[0] & 4294901760u);
                        res_0[2] = vals_0[2] - __uint_as_float(hi_0[1] << 16);
                        res_0[3] = vals_0[3] - __uint_as_float(hi_0[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_0[_lp*2 + 0], res_0[_lp*2+1 + 0]));
                            lo_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_8 / 64 * 4096 + (row_8 * 128 + col_8 % 64 * 2 ^ (row_8 * 128 + col_8 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_0[0]), "r"(hi_0[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_8 / 64 * 4096 + (row_8 * 128 + col_8 % 64 * 2 ^ (row_8 * 128 + col_8 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_0[0]), "r"(lo_0[1]) : "memory");
                    }
                    __syncthreads();
                    int idx_8_1 = tid * 4;
                    int row_8_1 = idx_8_1 / 128;
                    int col_8_1 = idx_8_1 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_8_1 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_8_1 * 256 + (long long)col_8_1)));
                    int idx_9 = (128 + tid) * 4;
                    int row_9 = idx_9 / 128;
                    int col_9 = idx_9 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_9 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_9 * 256 + (long long)col_9)));
                    int idx_10 = (256 + tid) * 4;
                    int row_10 = idx_10 / 128;
                    int col_10 = idx_10 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_10 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_10 * 256 + (long long)col_10)));
                    int idx_11 = (384 + tid) * 4;
                    int row_11 = idx_11 / 128;
                    int col_11 = idx_11 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_11 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_11 * 256 + (long long)col_11)));
                    int idx_12 = (512 + tid) * 4;
                    int row_12 = idx_12 / 128;
                    int col_12 = idx_12 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_12 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_12 * 256 + (long long)col_12)));
                    int idx_13 = (640 + tid) * 4;
                    int row_13 = idx_13 / 128;
                    int col_13 = idx_13 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_13 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_13 * 256 + (long long)col_13)));
                    int idx_14 = (768 + tid) * 4;
                    int row_14 = idx_14 / 128;
                    int col_14 = idx_14 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_14 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_14 * 256 + (long long)col_14)));
                    int idx_15 = (896 + tid) * 4;
                    int row_15 = idx_15 / 128;
                    int col_15 = idx_15 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_15 * 4)), "l"(seg_state + (pre_base_q + 8320 + (long long)row_15 * 256 + (long long)col_15)));
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
                    int colv_1_0 = nb0 * 16 + quad_4 * 2;
                    int ia_1_0 = krow * 64 + colv_1_0;
                    int ib_1_0 = (krow + 8) * 64 + colv_1_0;
                    s_x32[ia_1_0] = acc[0] + s_x32[ia_1_0];
                    s_x32[ia_1_0 + 1] = acc[1] + s_x32[ia_1_0 + 1];
                    s_x32[ib_1_0] = acc[2] + s_x32[ib_1_0];
                    s_x32[ib_1_0 + 1] = acc[3] + s_x32[ib_1_0 + 1];
                    int colv_1_1 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_1_1 = krow * 64 + colv_1_1;
                    int ib_1_1 = (krow + 8) * 64 + colv_1_1;
                    s_x32[ia_1_1] = acc[4] + s_x32[ia_1_1];
                    s_x32[ia_1_1 + 1] = acc[5] + s_x32[ia_1_1 + 1];
                    s_x32[ib_1_1] = acc[6] + s_x32[ib_1_1];
                    s_x32[ib_1_1 + 1] = acc[7] + s_x32[ib_1_1 + 1];
                    int colv_1_2 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_1_2 = krow * 64 + colv_1_2;
                    int ib_1_2 = (krow + 8) * 64 + colv_1_2;
                    s_x32[ia_1_2] = acc[8] + s_x32[ia_1_2];
                    s_x32[ia_1_2 + 1] = acc[9] + s_x32[ia_1_2 + 1];
                    s_x32[ib_1_2] = acc[10] + s_x32[ib_1_2];
                    s_x32[ib_1_2 + 1] = acc[11] + s_x32[ib_1_2 + 1];
                    int colv_1_3 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_1_3 = krow * 64 + colv_1_3;
                    int ib_1_3 = (krow + 8) * 64 + colv_1_3;
                    s_x32[ia_1_3] = acc[12] + s_x32[ia_1_3];
                    s_x32[ia_1_3 + 1] = acc[13] + s_x32[ia_1_3 + 1];
                    s_x32[ib_1_3] = acc[14] + s_x32[ib_1_3];
                    s_x32[ib_1_3 + 1] = acc[15] + s_x32[ib_1_3 + 1];
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_1[4];
                    unsigned int hi_1[2];
                    unsigned int lo_1[2];
                    float res_1[4];
                    #pragma unroll 4
                    for (int i_2 = 0; i_2 < 8; i_2++) {
                        int idx_16 = (i_2 * 128 + tid) * 4;
                        int row_16 = idx_16 / 128;
                        int col_16 = idx_16 % 128;
                        vals_1[0] = s_m32[idx_16];
                        vals_1[1] = s_m32[idx_16 + 1];
                        vals_1[2] = s_m32[idx_16 + 2];
                        vals_1[3] = s_m32[idx_16 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_1[_lp*2 + 0], vals_1[_lp*2+1 + 0]));
                            hi_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_1[0] = vals_1[0] - __uint_as_float(hi_1[0] << 16);
                        res_1[1] = vals_1[1] - __uint_as_float(hi_1[0] & 4294901760u);
                        res_1[2] = vals_1[2] - __uint_as_float(hi_1[1] << 16);
                        res_1[3] = vals_1[3] - __uint_as_float(hi_1[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_1[_lp*2 + 0], res_1[_lp*2+1 + 0]));
                            lo_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_16 / 64 * 4096 + (row_16 * 128 + col_16 % 64 * 2 ^ (row_16 * 128 + col_16 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_16 / 64 * 4096 + (row_16 * 128 + col_16 % 64 * 2 ^ (row_16 * 128 + col_16 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_1[0]), "r"(lo_1[1]) : "memory");
                    }
                    __syncthreads();
                    int idx_16_1 = tid * 4;
                    int row_16_1 = idx_16_1 / 128;
                    int col_16_1 = idx_16_1 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_16_1 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_16_1 * 256 + (long long)col_16_1)));
                    int idx_17 = (128 + tid) * 4;
                    int row_17 = idx_17 / 128;
                    int col_17 = idx_17 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_17 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_17 * 256 + (long long)col_17)));
                    int idx_18 = (256 + tid) * 4;
                    int row_18 = idx_18 / 128;
                    int col_18 = idx_18 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_18 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_18 * 256 + (long long)col_18)));
                    int idx_19 = (384 + tid) * 4;
                    int row_19 = idx_19 / 128;
                    int col_19 = idx_19 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_19 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_19 * 256 + (long long)col_19)));
                    int idx_20 = (512 + tid) * 4;
                    int row_20 = idx_20 / 128;
                    int col_20 = idx_20 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_20 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_20 * 256 + (long long)col_20)));
                    int idx_21 = (640 + tid) * 4;
                    int row_21 = idx_21 / 128;
                    int col_21 = idx_21 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_21 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_21 * 256 + (long long)col_21)));
                    int idx_22 = (768 + tid) * 4;
                    int row_22 = idx_22 / 128;
                    int col_22 = idx_22 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_22 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_22 * 256 + (long long)col_22)));
                    int idx_23 = (896 + tid) * 4;
                    int row_23 = idx_23 / 128;
                    int col_23 = idx_23 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_23 * 4)), "l"(seg_state + (pre_base_q + 16512 + (long long)row_23 * 256 + (long long)col_23)));
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
                    int colv_1_4 = nb0 * 16 + quad_4 * 2;
                    int ia_1_4 = krow_134 * 64 + colv_1_4;
                    int ib_1_4 = (krow_134 + 8) * 64 + colv_1_4;
                    s_x32[ia_1_4] = acc[0] + s_x32[ia_1_4];
                    s_x32[ia_1_4 + 1] = acc[1] + s_x32[ia_1_4 + 1];
                    s_x32[ib_1_4] = acc[2] + s_x32[ib_1_4];
                    s_x32[ib_1_4 + 1] = acc[3] + s_x32[ib_1_4 + 1];
                    int colv_1_5 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_1_5 = krow_134 * 64 + colv_1_5;
                    int ib_1_5 = (krow_134 + 8) * 64 + colv_1_5;
                    s_x32[ia_1_5] = acc[4] + s_x32[ia_1_5];
                    s_x32[ia_1_5 + 1] = acc[5] + s_x32[ia_1_5 + 1];
                    s_x32[ib_1_5] = acc[6] + s_x32[ib_1_5];
                    s_x32[ib_1_5 + 1] = acc[7] + s_x32[ib_1_5 + 1];
                    int colv_1_6 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_1_6 = krow_134 * 64 + colv_1_6;
                    int ib_1_6 = (krow_134 + 8) * 64 + colv_1_6;
                    s_x32[ia_1_6] = acc[8] + s_x32[ia_1_6];
                    s_x32[ia_1_6 + 1] = acc[9] + s_x32[ia_1_6 + 1];
                    s_x32[ib_1_6] = acc[10] + s_x32[ib_1_6];
                    s_x32[ib_1_6 + 1] = acc[11] + s_x32[ib_1_6 + 1];
                    int colv_1_7 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_1_7 = krow_134 * 64 + colv_1_7;
                    int ib_1_7 = (krow_134 + 8) * 64 + colv_1_7;
                    s_x32[ia_1_7] = acc[12] + s_x32[ia_1_7];
                    s_x32[ia_1_7 + 1] = acc[13] + s_x32[ia_1_7 + 1];
                    s_x32[ib_1_7] = acc[14] + s_x32[ib_1_7];
                    s_x32[ib_1_7 + 1] = acc[15] + s_x32[ib_1_7 + 1];
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_2[4];
                    unsigned int hi_2[2];
                    unsigned int lo_2[2];
                    float res_2[4];
                    #pragma unroll 4
                    for (int i_3 = 0; i_3 < 8; i_3++) {
                        int idx_24 = (i_3 * 128 + tid) * 4;
                        int row_24 = idx_24 / 128;
                        int col_24 = idx_24 % 128;
                        vals_2[0] = s_m32[idx_24];
                        vals_2[1] = s_m32[idx_24 + 1];
                        vals_2[2] = s_m32[idx_24 + 2];
                        vals_2[3] = s_m32[idx_24 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_2[_lp*2 + 0], vals_2[_lp*2+1 + 0]));
                            hi_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_2[0] = vals_2[0] - __uint_as_float(hi_2[0] << 16);
                        res_2[1] = vals_2[1] - __uint_as_float(hi_2[0] & 4294901760u);
                        res_2[2] = vals_2[2] - __uint_as_float(hi_2[1] << 16);
                        res_2[3] = vals_2[3] - __uint_as_float(hi_2[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_2[_lp*2 + 0], res_2[_lp*2+1 + 0]));
                            lo_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_24 / 64 * 4096 + (row_24 * 128 + col_24 % 64 * 2 ^ (row_24 * 128 + col_24 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_2[0]), "r"(hi_2[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_24 / 64 * 4096 + (row_24 * 128 + col_24 % 64 * 2 ^ (row_24 * 128 + col_24 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
                    }
                    __syncthreads();
                    int idx_24_1 = tid * 4;
                    int row_24_1 = idx_24_1 / 128;
                    int col_24_1 = idx_24_1 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_24_1 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_24_1 * 256 + (long long)col_24_1)));
                    int idx_25 = (128 + tid) * 4;
                    int row_25 = idx_25 / 128;
                    int col_25 = idx_25 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_25 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_25 * 256 + (long long)col_25)));
                    int idx_26 = (256 + tid) * 4;
                    int row_26 = idx_26 / 128;
                    int col_26 = idx_26 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_26 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_26 * 256 + (long long)col_26)));
                    int idx_27 = (384 + tid) * 4;
                    int row_27 = idx_27 / 128;
                    int col_27 = idx_27 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_27 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_27 * 256 + (long long)col_27)));
                    int idx_28 = (512 + tid) * 4;
                    int row_28 = idx_28 / 128;
                    int col_28 = idx_28 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_28 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_28 * 256 + (long long)col_28)));
                    int idx_29 = (640 + tid) * 4;
                    int row_29 = idx_29 / 128;
                    int col_29 = idx_29 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_29 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_29 * 256 + (long long)col_29)));
                    int idx_30 = (768 + tid) * 4;
                    int row_30 = idx_30 / 128;
                    int col_30 = idx_30 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_30 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_30 * 256 + (long long)col_30)));
                    int idx_31 = (896 + tid) * 4;
                    int row_31 = idx_31 / 128;
                    int col_31 = idx_31 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_31 * 4)), "l"(seg_state + (pre_base_q + 24704 + (long long)row_31 * 256 + (long long)col_31)));
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
                    int colv_1_8 = nb0 * 16 + quad_4 * 2;
                    int ia_1_8 = krow_175 * 64 + colv_1_8;
                    int ib_1_8 = (krow_175 + 8) * 64 + colv_1_8;
                    s_x32[ia_1_8] = acc[0] + s_x32[ia_1_8];
                    s_x32[ia_1_8 + 1] = acc[1] + s_x32[ia_1_8 + 1];
                    s_x32[ib_1_8] = acc[2] + s_x32[ib_1_8];
                    s_x32[ib_1_8 + 1] = acc[3] + s_x32[ib_1_8 + 1];
                    int colv_1_9 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_1_9 = krow_175 * 64 + colv_1_9;
                    int ib_1_9 = (krow_175 + 8) * 64 + colv_1_9;
                    s_x32[ia_1_9] = acc[4] + s_x32[ia_1_9];
                    s_x32[ia_1_9 + 1] = acc[5] + s_x32[ia_1_9 + 1];
                    s_x32[ib_1_9] = acc[6] + s_x32[ib_1_9];
                    s_x32[ib_1_9 + 1] = acc[7] + s_x32[ib_1_9 + 1];
                    int colv_1_10 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_1_10 = krow_175 * 64 + colv_1_10;
                    int ib_1_10 = (krow_175 + 8) * 64 + colv_1_10;
                    s_x32[ia_1_10] = acc[8] + s_x32[ia_1_10];
                    s_x32[ia_1_10 + 1] = acc[9] + s_x32[ia_1_10 + 1];
                    s_x32[ib_1_10] = acc[10] + s_x32[ib_1_10];
                    s_x32[ib_1_10 + 1] = acc[11] + s_x32[ib_1_10 + 1];
                    int colv_1_11 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_1_11 = krow_175 * 64 + colv_1_11;
                    int ib_1_11 = (krow_175 + 8) * 64 + colv_1_11;
                    s_x32[ia_1_11] = acc[12] + s_x32[ia_1_11];
                    s_x32[ia_1_11 + 1] = acc[13] + s_x32[ia_1_11 + 1];
                    s_x32[ib_1_11] = acc[14] + s_x32[ib_1_11];
                    s_x32[ib_1_11 + 1] = acc[15] + s_x32[ib_1_11 + 1];
                    asm volatile("cp.async.wait_group 0;");
                    __syncthreads();
                    float vals_3[4];
                    unsigned int hi_3[2];
                    unsigned int lo_3[2];
                    float res_3[4];
                    #pragma unroll 4
                    for (int i_4 = 0; i_4 < 8; i_4++) {
                        int idx_32 = (i_4 * 128 + tid) * 4;
                        int row_32 = idx_32 / 128;
                        int col_32 = idx_32 % 128;
                        vals_3[0] = s_m32[idx_32];
                        vals_3[1] = s_m32[idx_32 + 1];
                        vals_3[2] = s_m32[idx_32 + 2];
                        vals_3[3] = s_m32[idx_32 + 3];
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_3[_lp*2 + 0], vals_3[_lp*2+1 + 0]));
                            hi_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        res_3[0] = vals_3[0] - __uint_as_float(hi_3[0] << 16);
                        res_3[1] = vals_3[1] - __uint_as_float(hi_3[0] & 4294901760u);
                        res_3[2] = vals_3[2] - __uint_as_float(hi_3[1] << 16);
                        res_3[3] = vals_3[3] - __uint_as_float(hi_3[1] & 4294901760u);
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
                            lo_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_32 / 64 * 4096 + (row_32 * 128 + col_32 % 64 * 2 ^ (row_32 * 128 + col_32 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_3[0]), "r"(hi_3[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_32 / 64 * 4096 + (row_32 * 128 + col_32 % 64 * 2 ^ (row_32 * 128 + col_32 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_3[0]), "r"(lo_3[1]) : "memory");
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
                    int colv_1_12 = nb0 * 16 + quad_4 * 2;
                    int ia_1_12 = krow_192 * 64 + colv_1_12;
                    int ib_1_12 = (krow_192 + 8) * 64 + colv_1_12;
                    s_x32[ia_1_12] = acc[0] + s_x32[ia_1_12];
                    s_x32[ia_1_12 + 1] = acc[1] + s_x32[ia_1_12 + 1];
                    s_x32[ib_1_12] = acc[2] + s_x32[ib_1_12];
                    s_x32[ib_1_12 + 1] = acc[3] + s_x32[ib_1_12 + 1];
                    int colv_1_13 = nb0 * 16 + 8 + quad_4 * 2;
                    int ia_1_13 = krow_192 * 64 + colv_1_13;
                    int ib_1_13 = (krow_192 + 8) * 64 + colv_1_13;
                    s_x32[ia_1_13] = acc[4] + s_x32[ia_1_13];
                    s_x32[ia_1_13 + 1] = acc[5] + s_x32[ia_1_13 + 1];
                    s_x32[ib_1_13] = acc[6] + s_x32[ib_1_13];
                    s_x32[ib_1_13 + 1] = acc[7] + s_x32[ib_1_13 + 1];
                    int colv_1_14 = (nb0 + 1) * 16 + quad_4 * 2;
                    int ia_1_14 = krow_192 * 64 + colv_1_14;
                    int ib_1_14 = (krow_192 + 8) * 64 + colv_1_14;
                    s_x32[ia_1_14] = acc[8] + s_x32[ia_1_14];
                    s_x32[ia_1_14 + 1] = acc[9] + s_x32[ia_1_14 + 1];
                    s_x32[ib_1_14] = acc[10] + s_x32[ib_1_14];
                    s_x32[ib_1_14 + 1] = acc[11] + s_x32[ib_1_14 + 1];
                    int colv_1_15 = (nb0 + 1) * 16 + 8 + quad_4 * 2;
                    int ia_1_15 = krow_192 * 64 + colv_1_15;
                    int ib_1_15 = (krow_192 + 8) * 64 + colv_1_15;
                    s_x32[ia_1_15] = acc[12] + s_x32[ia_1_15];
                    s_x32[ia_1_15 + 1] = acc[13] + s_x32[ia_1_15 + 1];
                    s_x32[ib_1_15] = acc[14] + s_x32[ib_1_15];
                    s_x32[ib_1_15 + 1] = acc[15] + s_x32[ib_1_15 + 1];
                    __syncthreads();
                    if (n_steps > st + 1) {
                        float vals_4[4];
                        unsigned int hi_4[2];
                        unsigned int lo_4[2];
                        float res_4[4];
                        #pragma unroll 4
                        for (int i_5 = 0; i_5 < 16; i_5++) {
                            int idx_33 = (i_5 * 128 + tid) * 4;
                            int row_33 = idx_33 / 64;
                            int col_33 = idx_33 % 64;
                            vals_4[0] = s_x32[idx_33];
                            vals_4[1] = s_x32[idx_33 + 1];
                            vals_4[2] = s_x32[idx_33 + 2];
                            vals_4[3] = s_x32[idx_33 + 3];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_4[_lp*2 + 0], vals_4[_lp*2+1 + 0]));
                                hi_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            res_4[0] = vals_4[0] - __uint_as_float(hi_4[0] << 16);
                            res_4[1] = vals_4[1] - __uint_as_float(hi_4[0] & 4294901760u);
                            res_4[2] = vals_4[2] - __uint_as_float(hi_4[1] << 16);
                            res_4[3] = vals_4[3] - __uint_as_float(hi_4[1] & 4294901760u);
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_4[_lp*2 + 0], res_4[_lp*2+1 + 0]));
                                lo_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xhi_addr + (unsigned int)(col_33 / 64 * 16384 + (row_33 * 128 + col_33 % 64 * 2 ^ (row_33 * 128 + col_33 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_4[0]), "r"(hi_4[1]) : "memory");
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xlo_addr + (unsigned int)(col_33 / 64 * 16384 + (row_33 * 128 + col_33 % 64 * 2 ^ (row_33 * 128 + col_33 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_4[0]), "r"(lo_4[1]) : "memory");
                        }
                        __syncthreads();
                    }
                }
                int kcol5_15_0 = kbase + quad_4 * 2;
                acc_h[0] = s_x32[kcol5_15_0 * 64 + vloc_a];
                acc_h[1] = s_x32[(kcol5_15_0 + 1) * 64 + vloc_a];
                acc_h[2] = s_x32[kcol5_15_0 * 64 + vloc_b];
                acc_h[3] = s_x32[(kcol5_15_0 + 1) * 64 + vloc_b];
                int kcol5_15_1 = kbase + 8 + quad_4 * 2;
                acc_h[4] = s_x32[kcol5_15_1 * 64 + vloc_a];
                acc_h[5] = s_x32[(kcol5_15_1 + 1) * 64 + vloc_a];
                acc_h[6] = s_x32[kcol5_15_1 * 64 + vloc_b];
                acc_h[7] = s_x32[(kcol5_15_1 + 1) * 64 + vloc_b];
                int kcol5_15_2 = kbase + 16 + quad_4 * 2;
                acc_h[8] = s_x32[kcol5_15_2 * 64 + vloc_a];
                acc_h[9] = s_x32[(kcol5_15_2 + 1) * 64 + vloc_a];
                acc_h[10] = s_x32[kcol5_15_2 * 64 + vloc_b];
                acc_h[11] = s_x32[(kcol5_15_2 + 1) * 64 + vloc_b];
                int kcol5_15_3 = kbase + 24 + quad_4 * 2;
                acc_h[12] = s_x32[kcol5_15_3 * 64 + vloc_a];
                acc_h[13] = s_x32[(kcol5_15_3 + 1) * 64 + vloc_a];
                acc_h[14] = s_x32[kcol5_15_3 * 64 + vloc_b];
                acc_h[15] = s_x32[(kcol5_15_3 + 1) * 64 + vloc_b];
                int kcol5_15_4 = kbase + 32 + quad_4 * 2;
                acc_h[16] = s_x32[kcol5_15_4 * 64 + vloc_a];
                acc_h[17] = s_x32[(kcol5_15_4 + 1) * 64 + vloc_a];
                acc_h[18] = s_x32[kcol5_15_4 * 64 + vloc_b];
                acc_h[19] = s_x32[(kcol5_15_4 + 1) * 64 + vloc_b];
                int kcol5_15_5 = kbase + 40 + quad_4 * 2;
                acc_h[20] = s_x32[kcol5_15_5 * 64 + vloc_a];
                acc_h[21] = s_x32[(kcol5_15_5 + 1) * 64 + vloc_a];
                acc_h[22] = s_x32[kcol5_15_5 * 64 + vloc_b];
                acc_h[23] = s_x32[(kcol5_15_5 + 1) * 64 + vloc_b];
                int kcol5_15_6 = kbase + 48 + quad_4 * 2;
                acc_h[24] = s_x32[kcol5_15_6 * 64 + vloc_a];
                acc_h[25] = s_x32[(kcol5_15_6 + 1) * 64 + vloc_a];
                acc_h[26] = s_x32[kcol5_15_6 * 64 + vloc_b];
                acc_h[27] = s_x32[(kcol5_15_6 + 1) * 64 + vloc_b];
                int kcol5_15_7 = kbase + 56 + quad_4 * 2;
                acc_h[28] = s_x32[kcol5_15_7 * 64 + vloc_a];
                acc_h[29] = s_x32[(kcol5_15_7 + 1) * 64 + vloc_a];
                acc_h[30] = s_x32[kcol5_15_7 * 64 + vloc_b];
                acc_h[31] = s_x32[(kcol5_15_7 + 1) * 64 + vloc_b];
                int kcol5_15_8 = kbase + 64 + quad_4 * 2;
                acc_h[32] = s_x32[kcol5_15_8 * 64 + vloc_a];
                acc_h[33] = s_x32[(kcol5_15_8 + 1) * 64 + vloc_a];
                acc_h[34] = s_x32[kcol5_15_8 * 64 + vloc_b];
                acc_h[35] = s_x32[(kcol5_15_8 + 1) * 64 + vloc_b];
                int kcol5_15_9 = kbase + 72 + quad_4 * 2;
                acc_h[36] = s_x32[kcol5_15_9 * 64 + vloc_a];
                acc_h[37] = s_x32[(kcol5_15_9 + 1) * 64 + vloc_a];
                acc_h[38] = s_x32[kcol5_15_9 * 64 + vloc_b];
                acc_h[39] = s_x32[(kcol5_15_9 + 1) * 64 + vloc_b];
                int kcol5_15_10 = kbase + 80 + quad_4 * 2;
                acc_h[40] = s_x32[kcol5_15_10 * 64 + vloc_a];
                acc_h[41] = s_x32[(kcol5_15_10 + 1) * 64 + vloc_a];
                acc_h[42] = s_x32[kcol5_15_10 * 64 + vloc_b];
                acc_h[43] = s_x32[(kcol5_15_10 + 1) * 64 + vloc_b];
                int kcol5_15_11 = kbase + 88 + quad_4 * 2;
                acc_h[44] = s_x32[kcol5_15_11 * 64 + vloc_a];
                acc_h[45] = s_x32[(kcol5_15_11 + 1) * 64 + vloc_a];
                acc_h[46] = s_x32[kcol5_15_11 * 64 + vloc_b];
                acc_h[47] = s_x32[(kcol5_15_11 + 1) * 64 + vloc_b];
                int kcol5_15_12 = kbase + 96 + quad_4 * 2;
                acc_h[48] = s_x32[kcol5_15_12 * 64 + vloc_a];
                acc_h[49] = s_x32[(kcol5_15_12 + 1) * 64 + vloc_a];
                acc_h[50] = s_x32[kcol5_15_12 * 64 + vloc_b];
                acc_h[51] = s_x32[(kcol5_15_12 + 1) * 64 + vloc_b];
                int kcol5_15_13 = kbase + 104 + quad_4 * 2;
                acc_h[52] = s_x32[kcol5_15_13 * 64 + vloc_a];
                acc_h[53] = s_x32[(kcol5_15_13 + 1) * 64 + vloc_a];
                acc_h[54] = s_x32[kcol5_15_13 * 64 + vloc_b];
                acc_h[55] = s_x32[(kcol5_15_13 + 1) * 64 + vloc_b];
                int kcol5_15_14 = kbase + 112 + quad_4 * 2;
                acc_h[56] = s_x32[kcol5_15_14 * 64 + vloc_a];
                acc_h[57] = s_x32[(kcol5_15_14 + 1) * 64 + vloc_a];
                acc_h[58] = s_x32[kcol5_15_14 * 64 + vloc_b];
                acc_h[59] = s_x32[(kcol5_15_14 + 1) * 64 + vloc_b];
                int kcol5_15_15 = kbase + 120 + quad_4 * 2;
                acc_h[60] = s_x32[kcol5_15_15 * 64 + vloc_a];
                acc_h[61] = s_x32[(kcol5_15_15 + 1) * 64 + vloc_a];
                acc_h[62] = s_x32[kcol5_15_15 * 64 + vloc_b];
                acc_h[63] = s_x32[(kcol5_15_15 + 1) * 64 + vloc_b];
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
        int r0_2 = tid / 16;
        int c0_2 = tid % 16 * 8;
        long long row_stride_2 = (long long)num_v_heads * 128;
        long long src_row_2 = (long long)(seq_tok0 + r0_2) * row_stride_2 + (long long)hv * 128 + (long long)c0_2;
        int dst0_2 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0_2 / 64 * 8192 + (r0_2 * 128 + c0_2 % 64 * 2 ^ (r0_2 * 128 + c0_2 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2), "l"(w + src_row_2), "r"((_min_1 > r0_2) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 1024), "l"(w + (src_row_2 + 8 * row_stride_2)), "r"((_min_1 > r0_2 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 2048), "l"(w + (src_row_2 + 16 * row_stride_2)), "r"((_min_1 > r0_2 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 3072), "l"(w + (src_row_2 + 24 * row_stride_2)), "r"((_min_1 > r0_2 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 4096), "l"(w + (src_row_2 + 32 * row_stride_2)), "r"((_min_1 > r0_2 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 5120), "l"(w + (src_row_2 + 40 * row_stride_2)), "r"((_min_1 > r0_2 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 6144), "l"(w + (src_row_2 + 48 * row_stride_2)), "r"((_min_1 > r0_2 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_2 + 7168), "l"(w + (src_row_2 + 56 * row_stride_2)), "r"((_min_1 > r0_2 + 56) ? 16 : 0));
        int r0_3 = tid / 16;
        int c0_3 = tid % 16 * 8;
        long long row_stride_3 = (long long)num_heads * 128;
        long long src_row_3 = (long long)(seq_tok0 + r0_3) * row_stride_3 + (long long)hq * 128 + (long long)c0_3;
        int dst0_3 = s_kn_addr + (unsigned int)stage_off + (unsigned int)(c0_3 / 64 * 8192 + (r0_3 * 128 + c0_3 % 64 * 2 ^ (r0_3 * 128 + c0_3 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3), "l"(kn + src_row_3), "r"((_min_1 > r0_3) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 1024), "l"(kn + (src_row_3 + 8 * row_stride_3)), "r"((_min_1 > r0_3 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 2048), "l"(kn + (src_row_3 + 16 * row_stride_3)), "r"((_min_1 > r0_3 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 3072), "l"(kn + (src_row_3 + 24 * row_stride_3)), "r"((_min_1 > r0_3 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 4096), "l"(kn + (src_row_3 + 32 * row_stride_3)), "r"((_min_1 > r0_3 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 5120), "l"(kn + (src_row_3 + 40 * row_stride_3)), "r"((_min_1 > r0_3 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 6144), "l"(kn + (src_row_3 + 48 * row_stride_3)), "r"((_min_1 > r0_3 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_3 + 7168), "l"(kn + (src_row_3 + 56 * row_stride_3)), "r"((_min_1 > r0_3 + 56) ? 16 : 0));
        int r0_4 = tid / 8;
        int c0_4 = tid % 8 * 8;
        long long row_stride_4 = (long long)num_v_heads * 128;
        long long src_row_4 = (long long)(seq_tok0 + r0_4) * row_stride_4 + (long long)hv * 128 + (long long)v0 + (long long)c0_4;
        int dst0_4 = s_u_addr + (unsigned int)stage_off + (unsigned int)(c0_4 / 64 * 8192 + (r0_4 * 128 + c0_4 % 64 * 2 ^ (r0_4 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4), "l"(u + src_row_4), "r"((_min_1 > r0_4) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 2048), "l"(u + (src_row_4 + 16 * row_stride_4)), "r"((_min_1 > r0_4 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 4096), "l"(u + (src_row_4 + 32 * row_stride_4)), "r"((_min_1 > r0_4 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 6144), "l"(u + (src_row_4 + 48 * row_stride_4)), "r"((_min_1 > r0_4 + 48) ? 16 : 0));
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
    int tok_0 = tid;
    g_next[0] = 0.0f;
    if (tok_0 < _min_3) {
        g_next[0] = g_cs[(long long)(seq_tok0 + 64 + tok_0) * (long long)num_v_heads + (long long)hv];
    }
    #pragma unroll 1
    for (int it_chunk = 0; it_chunk < num_chunks; it_chunk++) {
        int chunk = c_begin + it_chunk;
        int tok0 = seq_tok0 + it_chunk * 64;
        int n_valid = ((seq_len - it_chunk * 64) < (64) ? (seq_len - it_chunk * 64) : (64));
        int next_off = 40960 - stage_off;
        int next_g_off = 64 - g_off;
        int _min_4 = ((seq_len - (it_chunk + 2) * 64) < (64) ? (seq_len - (it_chunk + 2) * 64) : (64));
        int tok_1 = tid;
        g_far[0] = 0.0f;
        if (tok_1 < _min_4) {
            g_far[0] = g_cs[(long long)(tok0 + 128 + tok_1) * (long long)num_v_heads + (long long)hv];
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
            int r0_5 = tid / 16;
            int c0_5 = tid % 16 * 8;
            long long row_stride_5 = (long long)num_v_heads * 128;
            long long src_row_5 = (long long)(next_tok0 + r0_5) * row_stride_5 + (long long)hv * 128 + (long long)c0_5;
            int dst0_5 = s_w_addr + (unsigned int)next_off + (unsigned int)(c0_5 / 64 * 8192 + (r0_5 * 128 + c0_5 % 64 * 2 ^ (r0_5 * 128 + c0_5 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5), "l"(w + src_row_5), "r"((next_len > r0_5) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 1024), "l"(w + (src_row_5 + 8 * row_stride_5)), "r"((next_len > r0_5 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 2048), "l"(w + (src_row_5 + 16 * row_stride_5)), "r"((next_len > r0_5 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 3072), "l"(w + (src_row_5 + 24 * row_stride_5)), "r"((next_len > r0_5 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 4096), "l"(w + (src_row_5 + 32 * row_stride_5)), "r"((next_len > r0_5 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 5120), "l"(w + (src_row_5 + 40 * row_stride_5)), "r"((next_len > r0_5 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 6144), "l"(w + (src_row_5 + 48 * row_stride_5)), "r"((next_len > r0_5 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 7168), "l"(w + (src_row_5 + 56 * row_stride_5)), "r"((next_len > r0_5 + 56) ? 16 : 0));
            int r0_6 = tid / 16;
            int c0_6 = tid % 16 * 8;
            long long row_stride_6 = (long long)num_heads * 128;
            long long src_row_6 = (long long)(next_tok0 + r0_6) * row_stride_6 + (long long)hq * 128 + (long long)c0_6;
            int dst0_6 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_6 / 64 * 8192 + (r0_6 * 128 + c0_6 % 64 * 2 ^ (r0_6 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6), "l"(kn + src_row_6), "r"((next_len > r0_6) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 1024), "l"(kn + (src_row_6 + 8 * row_stride_6)), "r"((next_len > r0_6 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 2048), "l"(kn + (src_row_6 + 16 * row_stride_6)), "r"((next_len > r0_6 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 3072), "l"(kn + (src_row_6 + 24 * row_stride_6)), "r"((next_len > r0_6 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 4096), "l"(kn + (src_row_6 + 32 * row_stride_6)), "r"((next_len > r0_6 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 5120), "l"(kn + (src_row_6 + 40 * row_stride_6)), "r"((next_len > r0_6 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 6144), "l"(kn + (src_row_6 + 48 * row_stride_6)), "r"((next_len > r0_6 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_6 + 7168), "l"(kn + (src_row_6 + 56 * row_stride_6)), "r"((next_len > r0_6 + 56) ? 16 : 0));
            int r0_7 = tid / 8;
            int c0_7 = tid % 8 * 8;
            long long row_stride_7 = (long long)num_v_heads * 128;
            long long src_row_7 = (long long)(next_tok0 + r0_7) * row_stride_7 + (long long)hv * 128 + (long long)v0 + (long long)c0_7;
            int dst0_7 = s_u_addr + (unsigned int)next_off + (unsigned int)(c0_7 / 64 * 8192 + (r0_7 * 128 + c0_7 % 64 * 2 ^ (r0_7 * 128 + c0_7 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_7), "l"(u + src_row_7), "r"((next_len > r0_7) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_7 + 2048), "l"(u + (src_row_7 + 16 * row_stride_7)), "r"((next_len > r0_7 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_7 + 4096), "l"(u + (src_row_7 + 32 * row_stride_7)), "r"((next_len > r0_7 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_7 + 6144), "l"(u + (src_row_7 + 48 * row_stride_7)), "r"((next_len > r0_7 + 48) ? 16 : 0));
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
        int st_nb_8_0 = m_idx / 2;
        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_0 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&hpk[0])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[1])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[2])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[3]))
            : "memory");
        int st_nb_8_1 = 2 + m_idx / 2;
        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_1 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&hpk[4])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[5])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[6])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[7]))
            : "memory");
        int st_nb_8_2 = 4 + m_idx / 2;
        uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_2 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&hpk[8])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[9])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[10])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[11]))
            : "memory");
        int st_nb_8_3 = 6 + m_idx / 2;
        uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_3 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&hpk[12])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[13])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[14])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[15]))
            : "memory");
        int st_nb_8_4 = 8 + m_idx / 2;
        uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_4 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&hpk[16])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[17])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[18])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[19]))
            : "memory");
        int st_nb_8_5 = 10 + m_idx / 2;
        uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_5 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&hpk[20])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[21])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[22])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[23]))
            : "memory");
        int st_nb_8_6 = 12 + m_idx / 2;
        uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_6 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&hpk[24])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[25])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[26])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[27]))
            : "memory");
        int st_nb_8_7 = 14 + m_idx / 2;
        uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_8_7 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&hpk[28])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[29])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[30])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[31]))
            : "memory");
        __syncwarp();
        int ld_row_lo = lane_0 / 16;
        int ld_ch = lane_0 % 16;
        int ld_row_1_0 = ld_row_lo;
        int words_1_0[4];
        {
            const int4* _ivptr_13 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_0 * 64 + ((ld_ch ^ ld_row_1_0 % 8) << 2));
            int4 _ivld_13;
            _ivld_13 = *_ivptr_13;
            words_1_0[0 + 0] = _ivld_13.x;
            words_1_0[0 + 1] = _ivld_13.y;
            words_1_0[0 + 2] = _ivld_13.z;
            words_1_0[0 + 3] = _ivld_13.w;
        }
        {
            int4 _iv4 = make_int4(words_1_0[0 + 0], words_1_0[0 + 1], words_1_0[0 + 2], words_1_0[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_0 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_1 = 2 + ld_row_lo;
        int words_1_1[4];
        {
            const int4* _ivptr_14 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_1 * 64 + ((ld_ch ^ ld_row_1_1 % 8) << 2));
            int4 _ivld_14;
            _ivld_14 = *_ivptr_14;
            words_1_1[0 + 0] = _ivld_14.x;
            words_1_1[0 + 1] = _ivld_14.y;
            words_1_1[0 + 2] = _ivld_14.z;
            words_1_1[0 + 3] = _ivld_14.w;
        }
        {
            int4 _iv4 = make_int4(words_1_1[0 + 0], words_1_1[0 + 1], words_1_1[0 + 2], words_1_1[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_1 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_2 = 4 + ld_row_lo;
        int words_1_2[4];
        {
            const int4* _ivptr_15 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_2 * 64 + ((ld_ch ^ ld_row_1_2 % 8) << 2));
            int4 _ivld_15;
            _ivld_15 = *_ivptr_15;
            words_1_2[0 + 0] = _ivld_15.x;
            words_1_2[0 + 1] = _ivld_15.y;
            words_1_2[0 + 2] = _ivld_15.z;
            words_1_2[0 + 3] = _ivld_15.w;
        }
        {
            int4 _iv4 = make_int4(words_1_2[0 + 0], words_1_2[0 + 1], words_1_2[0 + 2], words_1_2[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_2 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_3 = 6 + ld_row_lo;
        int words_1_3[4];
        {
            const int4* _ivptr_16 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_3 * 64 + ((ld_ch ^ ld_row_1_3 % 8) << 2));
            int4 _ivld_16;
            _ivld_16 = *_ivptr_16;
            words_1_3[0 + 0] = _ivld_16.x;
            words_1_3[0 + 1] = _ivld_16.y;
            words_1_3[0 + 2] = _ivld_16.z;
            words_1_3[0 + 3] = _ivld_16.w;
        }
        {
            int4 _iv4 = make_int4(words_1_3[0 + 0], words_1_3[0 + 1], words_1_3[0 + 2], words_1_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_3 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_4 = 8 + ld_row_lo;
        int words_1_4[4];
        {
            const int4* _ivptr_17 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_4 * 64 + ((ld_ch ^ ld_row_1_4 % 8) << 2));
            int4 _ivld_17;
            _ivld_17 = *_ivptr_17;
            words_1_4[0 + 0] = _ivld_17.x;
            words_1_4[0 + 1] = _ivld_17.y;
            words_1_4[0 + 2] = _ivld_17.z;
            words_1_4[0 + 3] = _ivld_17.w;
        }
        {
            int4 _iv4 = make_int4(words_1_4[0 + 0], words_1_4[0 + 1], words_1_4[0 + 2], words_1_4[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_4 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_5 = 10 + ld_row_lo;
        int words_1_5[4];
        {
            const int4* _ivptr_18 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_5 * 64 + ((ld_ch ^ ld_row_1_5 % 8) << 2));
            int4 _ivld_18;
            _ivld_18 = *_ivptr_18;
            words_1_5[0 + 0] = _ivld_18.x;
            words_1_5[0 + 1] = _ivld_18.y;
            words_1_5[0 + 2] = _ivld_18.z;
            words_1_5[0 + 3] = _ivld_18.w;
        }
        {
            int4 _iv4 = make_int4(words_1_5[0 + 0], words_1_5[0 + 1], words_1_5[0 + 2], words_1_5[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_5 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_6 = 12 + ld_row_lo;
        int words_1_6[4];
        {
            const int4* _ivptr_19 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_6 * 64 + ((ld_ch ^ ld_row_1_6 % 8) << 2));
            int4 _ivld_19;
            _ivld_19 = *_ivptr_19;
            words_1_6[0 + 0] = _ivld_19.x;
            words_1_6[0 + 1] = _ivld_19.y;
            words_1_6[0 + 2] = _ivld_19.z;
            words_1_6[0 + 3] = _ivld_19.w;
        }
        {
            int4 _iv4 = make_int4(words_1_6[0 + 0], words_1_6[0 + 1], words_1_6[0 + 2], words_1_6[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_6 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_1_7 = 14 + ld_row_lo;
        int words_1_7[4];
        {
            const int4* _ivptr_20 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_1_7 * 64 + ((ld_ch ^ ld_row_1_7 % 8) << 2));
            int4 _ivld_20;
            _ivld_20 = *_ivptr_20;
            words_1_7[0 + 0] = _ivld_20.x;
            words_1_7[0 + 1] = _ivld_20.y;
            words_1_7[0 + 2] = _ivld_20.z;
            words_1_7[0 + 3] = _ivld_20.w;
        }
        {
            int4 _iv4 = make_int4(words_1_7[0 + 0], words_1_7[0 + 1], words_1_7[0 + 2], words_1_7[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_1_7 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
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
                    int tok_1_0 = (tb * 2 + half) * 8 + quad * 2 + e;
                    float decay = 0.0f;
                    if (tok_1_0 < n_valid) {
                        float _exp2_21 = approx_exp2(g_last - s_g[g_off + tok_1_0]);
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
            int st_tok_1_0 = m_idx_0 / 2 * 8 + r_idx_1;
            uint32_t _stmatrix_addr_22 = static_cast<uint32_t>(stage_base + st_tok_1_0 * 32 + ((st_half ^ st_tok_1_0 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_22), "r"(*reinterpret_cast<const uint32_t*>(&vw[0])), "r"(*reinterpret_cast<const uint32_t*>(&vw[1])), "r"(*reinterpret_cast<const uint32_t*>(&vw[2])), "r"(*reinterpret_cast<const uint32_t*>(&vw[3]))
                : "memory");
            int st_tok_1_1 = (2 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(stage_base + st_tok_1_1 * 32 + ((st_half ^ st_tok_1_1 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&vw[4])), "r"(*reinterpret_cast<const uint32_t*>(&vw[5])), "r"(*reinterpret_cast<const uint32_t*>(&vw[6])), "r"(*reinterpret_cast<const uint32_t*>(&vw[7]))
                : "memory");
            int st_tok_1_2 = (4 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(stage_base + st_tok_1_2 * 32 + ((st_half ^ st_tok_1_2 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&vw[8])), "r"(*reinterpret_cast<const uint32_t*>(&vw[9])), "r"(*reinterpret_cast<const uint32_t*>(&vw[10])), "r"(*reinterpret_cast<const uint32_t*>(&vw[11]))
                : "memory");
            int st_tok_1_3 = (6 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(stage_base + st_tok_1_3 * 32 + ((st_half ^ st_tok_1_3 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&vw[12])), "r"(*reinterpret_cast<const uint32_t*>(&vw[13])), "r"(*reinterpret_cast<const uint32_t*>(&vw[14])), "r"(*reinterpret_cast<const uint32_t*>(&vw[15]))
                : "memory");
            __syncwarp();
            int ld_half = lane_0 % 2;
            int ld_tok_1_0 = lane_0 / 2;
            int words_0[4];
            {
                const int4* _ivptr_26 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_1_0 * 8 + ((ld_half ^ ld_tok_1_0 >> 2 & 1) << 2));
                int4 _ivld_26;
                _ivld_26 = *_ivptr_26;
                words_0[0 + 0] = _ivld_26.x;
                words_0[0 + 1] = _ivld_26.y;
                words_0[0 + 2] = _ivld_26.z;
                words_0[0 + 3] = _ivld_26.w;
            }
            if (ld_tok_1_0 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_0[0 + 0], words_0[0 + 1], words_0[0 + 2], words_0[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_1_0 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_1_1 = 16 + lane_0 / 2;
            int words_1[4];
            {
                const int4* _ivptr_27 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_1_1 * 8 + ((ld_half ^ ld_tok_1_1 >> 2 & 1) << 2));
                int4 _ivld_27;
                _ivld_27 = *_ivptr_27;
                words_1[0 + 0] = _ivld_27.x;
                words_1[0 + 1] = _ivld_27.y;
                words_1[0 + 2] = _ivld_27.z;
                words_1[0 + 3] = _ivld_27.w;
            }
            if (ld_tok_1_1 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_1[0 + 0], words_1[0 + 1], words_1[0 + 2], words_1[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_1_1 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_1_2 = 32 + lane_0 / 2;
            int words_2[4];
            {
                const int4* _ivptr_28 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_1_2 * 8 + ((ld_half ^ ld_tok_1_2 >> 2 & 1) << 2));
                int4 _ivld_28;
                _ivld_28 = *_ivptr_28;
                words_2[0 + 0] = _ivld_28.x;
                words_2[0 + 1] = _ivld_28.y;
                words_2[0 + 2] = _ivld_28.z;
                words_2[0 + 3] = _ivld_28.w;
            }
            if (ld_tok_1_2 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_2[0 + 0], words_2[0 + 1], words_2[0 + 2], words_2[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_1_2 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_1_3 = 48 + lane_0 / 2;
            int words_3[4];
            {
                const int4* _ivptr_29 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_1_3 * 8 + ((ld_half ^ ld_tok_1_3 >> 2 & 1) << 2));
                int4 _ivld_29;
                _ivld_29 = *_ivptr_29;
                words_3[0 + 0] = _ivld_29.x;
                words_3[0 + 1] = _ivld_29.y;
                words_3[0 + 2] = _ivld_29.z;
                words_3[0 + 3] = _ivld_29.w;
            }
            if (ld_tok_1_3 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_1_3 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
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
