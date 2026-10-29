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
#define SMEM_S_STAGE_STAGE_BYTES 8192
#define SMEM_S_STAGE_STRIDE 8192
#define SMEM_S_RED_OFF 140288
#define SMEM_S_RED_STAGE_BYTES 16384
#define SMEM_S_RED_STRIDE 16384
#define SMEM_TOTAL 156672
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
kernel_cake_gdn_det_train_f69f9431ee7ecdc5d3f7(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv_local, float* __restrict__ g_cs, float* __restrict__ dht, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, float* __restrict__ dh0, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_final_state_grad, int store_initial_state_grad, float scale)
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
    float* s_red = reinterpret_cast<float*>(smem_raw + 140288);
    const int s_red_addr = smem + 140288;

    // === Task calls (declaration order) ===
    int seg = blockIdx.x;
    int hv = blockIdx.y;
    int v0 = blockIdx.z * 32;
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
    int vwarp = warp_1 / 2;
    int khalf = warp_1 % 2;
    int kbase = khalf * 64;
    int vloc_a = vwarp * 16 + lane_0 / 4;
    int vrow_a = v0 + vloc_a;
    int vrow_b = vrow_a + 8;
    long long tok_stride = (long long)num_v_heads * 128;
    int stage_base = s_stage_addr + (unsigned int)(warp_1 * 2048);
    int stage_u32_base = warp_1 * 512;
    unsigned int vw[16];
    float acc_dh[32];
    float acc_v[32];
    unsigned int dhpk[16];
    unsigned int ndvpk[16];
    unsigned int a_frag_mn[4];
    unsigned int dvl_frag[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float g_next[1];
    float g_far[1];
    long long in_base = ((long long)seg * (long long)num_v_heads + (long long)hv) * 16384;
    int _max_0 = ((seq) > (0) ? (seq) : (0));
    long long out_base = ((long long)_max_0 * (long long)num_v_heads + (long long)hv) * 16384;
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
    if (use_final_state_grad != 0) {
        #pragma unroll
        for (int nb = 0; nb < 8; nb++) {
            int kcol = kbase + nb * 8 + quad * 2;
            acc_dh[nb * 4] = dht[in_base + (long long)kcol * 128 + (long long)vrow_a];
            acc_dh[nb * 4 + 1] = dht[in_base + (long long)(kcol + 1) * 128 + (long long)vrow_a];
            acc_dh[nb * 4 + 2] = dht[in_base + (long long)kcol * 128 + (long long)vrow_b];
            acc_dh[nb * 4 + 3] = dht[in_base + (long long)(kcol + 1) * 128 + (long long)vrow_b];
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
    int stage_off = 0;
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
    int r0_3 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_5 = (long long)num_heads * 128;
    long long src_row_6 = (long long)(last_tok0 + r0_3) * row_stride_5 + (long long)hq * 128 + (long long)c0_4;
    int dst0_7 = s_qn_addr + (unsigned int)stage_off + (unsigned int)(c0_4 / 64 * 8192 + (r0_3 * 128 + c0_4 % 64 * 2 ^ (r0_3 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7), "l"(qn + src_row_6), "r"((last_len > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 1024), "l"(qn + (src_row_6 + 8 * row_stride_5)), "r"((last_len > r0_3 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 2048), "l"(qn + (src_row_6 + 16 * row_stride_5)), "r"((last_len > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 3072), "l"(qn + (src_row_6 + 24 * row_stride_5)), "r"((last_len > r0_3 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 4096), "l"(qn + (src_row_6 + 32 * row_stride_5)), "r"((last_len > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 5120), "l"(qn + (src_row_6 + 40 * row_stride_5)), "r"((last_len > r0_3 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 6144), "l"(qn + (src_row_6 + 48 * row_stride_5)), "r"((last_len > r0_3 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 7168), "l"(qn + (src_row_6 + 56 * row_stride_5)), "r"((last_len > r0_3 + 56) ? 16 : 0));
    int r0_8 = tid / 16;
    int c0_9 = tid % 16 * 8;
    long long row_stride_10 = (long long)num_v_heads * 128;
    long long src_row_11 = (long long)(last_tok0 + r0_8) * row_stride_10 + (long long)hv * 128 + (long long)c0_9;
    int dst0_12 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0_9 / 64 * 8192 + (r0_8 * 128 + c0_9 % 64 * 2 ^ (r0_8 * 128 + c0_9 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12), "l"(w + src_row_11), "r"((last_len > r0_8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 1024), "l"(w + (src_row_11 + 8 * row_stride_10)), "r"((last_len > r0_8 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 2048), "l"(w + (src_row_11 + 16 * row_stride_10)), "r"((last_len > r0_8 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 3072), "l"(w + (src_row_11 + 24 * row_stride_10)), "r"((last_len > r0_8 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 4096), "l"(w + (src_row_11 + 32 * row_stride_10)), "r"((last_len > r0_8 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 5120), "l"(w + (src_row_11 + 40 * row_stride_10)), "r"((last_len > r0_8 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 6144), "l"(w + (src_row_11 + 48 * row_stride_10)), "r"((last_len > r0_8 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 7168), "l"(w + (src_row_11 + 56 * row_stride_10)), "r"((last_len > r0_8 + 56) ? 16 : 0));
    int r0_13 = tid / 4;
    int c0_14 = tid % 4 * 8;
    long long row_stride_15 = (long long)num_v_heads * 128;
    long long src_row_16 = (long long)(last_tok0 + r0_13) * row_stride_15 + (long long)hv * 128 + (long long)v0 + (long long)c0_14;
    int dst0_17 = s_do_addr + (unsigned int)stage_off + (unsigned int)(c0_14 / 64 * 8192 + (r0_13 * 128 + c0_14 % 64 * 2 ^ (r0_13 * 128 + c0_14 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17), "l"(do_ + src_row_16), "r"((last_len > r0_13) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 4096), "l"(do_ + (src_row_16 + 32 * row_stride_15)), "r"((last_len > r0_13 + 32) ? 16 : 0));
    int r0_18 = tid / 4;
    int c0_19 = tid % 4 * 8;
    long long row_stride_20 = (long long)num_v_heads * 128;
    long long src_row_21 = (long long)(last_tok0 + r0_18) * row_stride_20 + (long long)hv * 128 + (long long)v0 + (long long)c0_19;
    int dst0_22 = s_dvl_addr + (unsigned int)stage_off + (unsigned int)(c0_19 / 64 * 8192 + (r0_18 * 128 + c0_19 % 64 * 2 ^ (r0_18 * 128 + c0_19 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22), "l"(dv_local + src_row_21), "r"((last_len > r0_18) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 4096), "l"(dv_local + (src_row_21 + 32 * row_stride_20)), "r"((last_len > r0_18 + 32) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int tok = tid;
    if (tok < 64) {
        float gate_val = 0.0f;
        if (tok < last_len) {
            gate_val = g_cs[(long long)(last_tok0 + tok) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tok] = gate_val;
        float _exp2_1 = approx_exp2(gate_val);
        s_gain[tok] = _exp2_1 * scale;
    }
    int _min_0 = ((num_chunks - 1) < (1) ? (num_chunks - 1) : (1));
    int tok_23 = tid;
    g_next[0] = 0.0f;
    if (tok_23 < _min_0 * 64) {
        g_next[0] = g_cs[(long long)(last_tok0 - 64 + tok_23) * (long long)num_v_heads + (long long)hv];
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
        int g_off = rev % 2 * 64;
        int next_g_off = 64 - g_off;
        int _max_2 = ((num_chunks - rev - 2) > (0) ? (num_chunks - rev - 2) : (0));
        int _min_1 = ((_max_2) < (1) ? (_max_2) : (1));
        int tok_0 = tid;
        g_far[0] = 0.0f;
        if (tok_0 < _min_1 * 64) {
            g_far[0] = g_cs[(long long)(tok0 - 128 + tok_0) * (long long)num_v_heads + (long long)hv];
        }
        if (num_chunks > rev + 1) {
            int next_tok0 = tok0 - 64;
            int next_len = 64;
            int r0_0 = tid / 16;
            int c0_1 = tid % 16 * 8;
            long long row_stride_2 = (long long)num_heads * 128;
            long long src_row_3 = (long long)(next_tok0 + r0_0) * row_stride_2 + (long long)hq * 128 + (long long)c0_1;
            int dst0_4 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4), "l"(kn + src_row_3), "r"((next_len > r0_0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 1024), "l"(kn + (src_row_3 + 8 * row_stride_2)), "r"((next_len > r0_0 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 2048), "l"(kn + (src_row_3 + 16 * row_stride_2)), "r"((next_len > r0_0 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 3072), "l"(kn + (src_row_3 + 24 * row_stride_2)), "r"((next_len > r0_0 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 4096), "l"(kn + (src_row_3 + 32 * row_stride_2)), "r"((next_len > r0_0 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 5120), "l"(kn + (src_row_3 + 40 * row_stride_2)), "r"((next_len > r0_0 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 6144), "l"(kn + (src_row_3 + 48 * row_stride_2)), "r"((next_len > r0_0 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 7168), "l"(kn + (src_row_3 + 56 * row_stride_2)), "r"((next_len > r0_0 + 56) ? 16 : 0));
            int r0_5 = tid / 16;
            int c0_6 = tid % 16 * 8;
            long long row_stride_7 = (long long)num_heads * 128;
            long long src_row_8 = (long long)(next_tok0 + r0_5) * row_stride_7 + (long long)hq * 128 + (long long)c0_6;
            int dst0_9 = s_qn_addr + (unsigned int)next_off + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9), "l"(qn + src_row_8), "r"((next_len > r0_5) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 1024), "l"(qn + (src_row_8 + 8 * row_stride_7)), "r"((next_len > r0_5 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 2048), "l"(qn + (src_row_8 + 16 * row_stride_7)), "r"((next_len > r0_5 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 3072), "l"(qn + (src_row_8 + 24 * row_stride_7)), "r"((next_len > r0_5 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 4096), "l"(qn + (src_row_8 + 32 * row_stride_7)), "r"((next_len > r0_5 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 5120), "l"(qn + (src_row_8 + 40 * row_stride_7)), "r"((next_len > r0_5 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 6144), "l"(qn + (src_row_8 + 48 * row_stride_7)), "r"((next_len > r0_5 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 7168), "l"(qn + (src_row_8 + 56 * row_stride_7)), "r"((next_len > r0_5 + 56) ? 16 : 0));
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
            int r0_15 = tid / 4;
            int c0_16 = tid % 4 * 8;
            long long row_stride_17 = (long long)num_v_heads * 128;
            long long src_row_18 = (long long)(next_tok0 + r0_15) * row_stride_17 + (long long)hv * 128 + (long long)v0 + (long long)c0_16;
            int dst0_19 = s_do_addr + (unsigned int)next_off + (unsigned int)(c0_16 / 64 * 8192 + (r0_15 * 128 + c0_16 % 64 * 2 ^ (r0_15 * 128 + c0_16 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_19), "l"(do_ + src_row_18), "r"((next_len > r0_15) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_19 + 4096), "l"(do_ + (src_row_18 + 32 * row_stride_17)), "r"((next_len > r0_15 + 32) ? 16 : 0));
            int r0_20 = tid / 4;
            int c0_21 = tid % 4 * 8;
            long long row_stride_22 = (long long)num_v_heads * 128;
            long long src_row_23 = (long long)(next_tok0 + r0_20) * row_stride_22 + (long long)hv * 128 + (long long)v0 + (long long)c0_21;
            int dst0_24 = s_dvl_addr + (unsigned int)next_off + (unsigned int)(c0_21 / 64 * 8192 + (r0_20 * 128 + c0_21 % 64 * 2 ^ (r0_20 * 128 + c0_21 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_24), "l"(dv_local + src_row_23), "r"((next_len > r0_20) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_24 + 4096), "l"(dv_local + (src_row_23 + 32 * row_stride_22)), "r"((next_len > r0_20 + 32) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_dh[_lp*2 + 0], acc_dh[_lp*2+1 + 0]));
            dhpk[_lp] = *(uint32_t*)&_bf2;
        }
        __syncthreads();
        int _max_3 = ((n_valid) > (1) ? (n_valid) : (1));
        float g_last = s_g[g_off + _max_3 - 1];
        float _exp2_4 = approx_exp2(g_last);
        const float2 _scale2_5 = {_exp2_4, _exp2_4};
        #pragma unroll
        for (int _ls = 0; _ls < 16; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_dh)[_ls], _scale2_5);
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
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
        long long tape_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384 + (long long)(v0 + vwarp * 16) * 128 + (long long)kbase;
        int m_idx = lane_0 / 8;
        int r_idx = lane_0 % 8;
        int st_row = m_idx % 2 * 8 + r_idx;
        int st_nb = m_idx / 2;
        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[0])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[1])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[2])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[3]))
            : "memory");
        int st_nb_1 = 2 + m_idx / 2;
        uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb_1 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[4])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[5])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[6])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[7]))
            : "memory");
        int st_nb_2 = 4 + m_idx / 2;
        uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb_2 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[8])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[9])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[10])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[11]))
            : "memory");
        int st_nb_3 = 6 + m_idx / 2;
        uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb_3 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[12])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[13])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[14])), "r"(*reinterpret_cast<const uint32_t*>(&dhpk[15]))
            : "memory");
        __syncwarp();
        int ld_row_lo = lane_0 / 8;
        int ld_ch = lane_0 % 8;
        int ld_row = ld_row_lo;
        int words[4];
        {
            const int4* _ivptr_10 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row * 32 + ((ld_ch ^ ld_row % 8) << 2));
            int4 _ivld_10;
            _ivld_10 = *_ivptr_10;
            words[0 + 0] = _ivld_10.x;
            words[0 + 1] = _ivld_10.y;
            words[0 + 2] = _ivld_10.z;
            words[0 + 3] = _ivld_10.w;
        }
        {
            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(dh_out) + (tape_base + (long long)(ld_row * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_4 = 4 + ld_row_lo;
        int words_5[4];
        {
            const int4* _ivptr_11 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_4 * 32 + ((ld_ch ^ ld_row_4 % 8) << 2));
            int4 _ivld_11;
            _ivld_11 = *_ivptr_11;
            words_5[0 + 0] = _ivld_11.x;
            words_5[0 + 1] = _ivld_11.y;
            words_5[0 + 2] = _ivld_11.z;
            words_5[0 + 3] = _ivld_11.w;
        }
        {
            int4 _iv4 = make_int4(words_5[0 + 0], words_5[0 + 1], words_5[0 + 2], words_5[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(dh_out) + (tape_base + (long long)(ld_row_4 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_6 = 8 + ld_row_lo;
        int words_7[4];
        {
            const int4* _ivptr_12 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_6 * 32 + ((ld_ch ^ ld_row_6 % 8) << 2));
            int4 _ivld_12;
            _ivld_12 = *_ivptr_12;
            words_7[0 + 0] = _ivld_12.x;
            words_7[0 + 1] = _ivld_12.y;
            words_7[0 + 2] = _ivld_12.z;
            words_7[0 + 3] = _ivld_12.w;
        }
        {
            int4 _iv4 = make_int4(words_7[0 + 0], words_7[0 + 1], words_7[0 + 2], words_7[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(dh_out) + (tape_base + (long long)(ld_row_6 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_8 = 12 + ld_row_lo;
        int words_9[4];
        {
            const int4* _ivptr_13 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_8 * 32 + ((ld_ch ^ ld_row_8 % 8) << 2));
            int4 _ivld_13;
            _ivld_13 = *_ivptr_13;
            words_9[0 + 0] = _ivld_13.x;
            words_9[0 + 1] = _ivld_13.y;
            words_9[0 + 2] = _ivld_13.z;
            words_9[0 + 3] = _ivld_13.w;
        }
        {
            int4 _iv4 = make_int4(words_9[0 + 0], words_9[0 + 1], words_9[0 + 2], words_9[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(dh_out) + (tape_base + (long long)(ld_row_8 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        __syncwarp();
        int own_byte = warp_1 * 4096 + lane_0 * 16;
        int partner_byte = (warp_1 ^ 1) * 4096 + lane_0 * 16;
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)own_byte), "f"(acc_v[0]), "f"(acc_v[1]), "f"(acc_v[2]), "f"(acc_v[3]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 512)), "f"(acc_v[4]), "f"(acc_v[5]), "f"(acc_v[6]), "f"(acc_v[7]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 1024)), "f"(acc_v[8]), "f"(acc_v[9]), "f"(acc_v[10]), "f"(acc_v[11]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 1536)), "f"(acc_v[12]), "f"(acc_v[13]), "f"(acc_v[14]), "f"(acc_v[15]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 2048)), "f"(acc_v[16]), "f"(acc_v[17]), "f"(acc_v[18]), "f"(acc_v[19]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 2560)), "f"(acc_v[20]), "f"(acc_v[21]), "f"(acc_v[22]), "f"(acc_v[23]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 3072)), "f"(acc_v[24]), "f"(acc_v[25]), "f"(acc_v[26]), "f"(acc_v[27]));
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(s_red_addr + (unsigned int)(own_byte + 3584)), "f"(acc_v[28]), "f"(acc_v[29]), "f"(acc_v[30]), "f"(acc_v[31]));
        __syncthreads();
        float part[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part[0])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)partner_byte));
        acc_v[0] = acc_v[0] + part[0];
        acc_v[1] = acc_v[1] + part[1];
        acc_v[2] = acc_v[2] + part[2];
        acc_v[3] = acc_v[3] + part[3];
        float part_10[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_10[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 512)));
        acc_v[4] = acc_v[4] + part_10[0];
        acc_v[5] = acc_v[5] + part_10[1];
        acc_v[6] = acc_v[6] + part_10[2];
        acc_v[7] = acc_v[7] + part_10[3];
        float part_11[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_11[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 1024)));
        acc_v[8] = acc_v[8] + part_11[0];
        acc_v[9] = acc_v[9] + part_11[1];
        acc_v[10] = acc_v[10] + part_11[2];
        acc_v[11] = acc_v[11] + part_11[3];
        float part_12[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_12[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 1536)));
        acc_v[12] = acc_v[12] + part_12[0];
        acc_v[13] = acc_v[13] + part_12[1];
        acc_v[14] = acc_v[14] + part_12[2];
        acc_v[15] = acc_v[15] + part_12[3];
        float part_13[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_13[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 2048)));
        acc_v[16] = acc_v[16] + part_13[0];
        acc_v[17] = acc_v[17] + part_13[1];
        acc_v[18] = acc_v[18] + part_13[2];
        acc_v[19] = acc_v[19] + part_13[3];
        float part_14[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_14[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_14[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_14[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 2560)));
        acc_v[20] = acc_v[20] + part_14[0];
        acc_v[21] = acc_v[21] + part_14[1];
        acc_v[22] = acc_v[22] + part_14[2];
        acc_v[23] = acc_v[23] + part_14[3];
        float part_15[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_15[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_15[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_15[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 3072)));
        acc_v[24] = acc_v[24] + part_15[0];
        acc_v[25] = acc_v[25] + part_15[1];
        acc_v[26] = acc_v[26] + part_15[2];
        acc_v[27] = acc_v[27] + part_15[3];
        float part_16[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&part_16[0])), "=r"(*reinterpret_cast<uint32_t*>(&part_16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&part_16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&part_16[(0) + 3]))
            : "r"(s_red_addr + (unsigned int)(partner_byte + 3584)));
        acc_v[28] = acc_v[28] + part_16[0];
        acc_v[29] = acc_v[29] + part_16[1];
        acc_v[30] = acc_v[30] + part_16[2];
        acc_v[31] = acc_v[31] + part_16[3];
        float dvn[4];
        #pragma unroll
        for (int tb = 0; tb < 4; tb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(dvl_frag[0]), "=r"(dvl_frag[1]), "=r"(dvl_frag[2]), "=r"(dvl_frag[3])
                : "r"(s_dvl_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
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
        #pragma unroll
        for (int ks2 = 0; ks2 < 4; ks2++) {
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
            #pragma unroll
            for (int nb5 = 0; nb5 < 4; nb5++) {
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
        if (khalf == 0) {
            long long row_base = ((long long)tok0 * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)(vwarp * 16);
            int m_idx_0 = lane_0 / 8;
            int r_idx_1 = lane_0 % 8;
            int st_half = m_idx_0 % 2;
            int st_tok = m_idx_0 / 2 * 8 + r_idx_1;
            uint32_t _stmatrix_addr_23 = static_cast<uint32_t>(stage_base + st_tok * 32 + ((st_half ^ st_tok >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_23), "r"(*reinterpret_cast<const uint32_t*>(&vw[0])), "r"(*reinterpret_cast<const uint32_t*>(&vw[1])), "r"(*reinterpret_cast<const uint32_t*>(&vw[2])), "r"(*reinterpret_cast<const uint32_t*>(&vw[3]))
                : "memory");
            int st_tok_2 = (2 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_24 = static_cast<uint32_t>(stage_base + st_tok_2 * 32 + ((st_half ^ st_tok_2 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_24), "r"(*reinterpret_cast<const uint32_t*>(&vw[4])), "r"(*reinterpret_cast<const uint32_t*>(&vw[5])), "r"(*reinterpret_cast<const uint32_t*>(&vw[6])), "r"(*reinterpret_cast<const uint32_t*>(&vw[7]))
                : "memory");
            int st_tok_3 = (4 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_25 = static_cast<uint32_t>(stage_base + st_tok_3 * 32 + ((st_half ^ st_tok_3 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_25), "r"(*reinterpret_cast<const uint32_t*>(&vw[8])), "r"(*reinterpret_cast<const uint32_t*>(&vw[9])), "r"(*reinterpret_cast<const uint32_t*>(&vw[10])), "r"(*reinterpret_cast<const uint32_t*>(&vw[11]))
                : "memory");
            int st_tok_4 = (6 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_26 = static_cast<uint32_t>(stage_base + st_tok_4 * 32 + ((st_half ^ st_tok_4 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_26), "r"(*reinterpret_cast<const uint32_t*>(&vw[12])), "r"(*reinterpret_cast<const uint32_t*>(&vw[13])), "r"(*reinterpret_cast<const uint32_t*>(&vw[14])), "r"(*reinterpret_cast<const uint32_t*>(&vw[15]))
                : "memory");
            __syncwarp();
            int ld_half = lane_0 % 2;
            int ld_tok = lane_0 / 2;
            int words_6[4];
            {
                const int4* _ivptr_27 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok * 8 + ((ld_half ^ ld_tok >> 2 & 1) << 2));
                int4 _ivld_27;
                _ivld_27 = *_ivptr_27;
                words_6[0 + 0] = _ivld_27.x;
                words_6[0 + 1] = _ivld_27.y;
                words_6[0 + 2] = _ivld_27.z;
                words_6[0 + 3] = _ivld_27.w;
            }
            if (ld_tok < n_valid) {
                {
                    int4 _iv4 = make_int4(words_6[0 + 0], words_6[0 + 1], words_6[0 + 2], words_6[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(dv2) + (row_base + (long long)ld_tok * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_7 = 16 + lane_0 / 2;
            int words_8[4];
            {
                const int4* _ivptr_28 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_7 * 8 + ((ld_half ^ ld_tok_7 >> 2 & 1) << 2));
                int4 _ivld_28;
                _ivld_28 = *_ivptr_28;
                words_8[0 + 0] = _ivld_28.x;
                words_8[0 + 1] = _ivld_28.y;
                words_8[0 + 2] = _ivld_28.z;
                words_8[0 + 3] = _ivld_28.w;
            }
            if (ld_tok_7 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_8[0 + 0], words_8[0 + 1], words_8[0 + 2], words_8[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(dv2) + (row_base + (long long)ld_tok_7 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_9 = 32 + lane_0 / 2;
            int words_10[4];
            {
                const int4* _ivptr_29 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_9 * 8 + ((ld_half ^ ld_tok_9 >> 2 & 1) << 2));
                int4 _ivld_29;
                _ivld_29 = *_ivptr_29;
                words_10[0 + 0] = _ivld_29.x;
                words_10[0 + 1] = _ivld_29.y;
                words_10[0 + 2] = _ivld_29.z;
                words_10[0 + 3] = _ivld_29.w;
            }
            if (ld_tok_9 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_10[0 + 0], words_10[0 + 1], words_10[0 + 2], words_10[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(dv2) + (row_base + (long long)ld_tok_9 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_11 = 48 + lane_0 / 2;
            int words_12[4];
            {
                const int4* _ivptr_30 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_11 * 8 + ((ld_half ^ ld_tok_11 >> 2 & 1) << 2));
                int4 _ivld_30;
                _ivld_30 = *_ivptr_30;
                words_12[0 + 0] = _ivld_30.x;
                words_12[0 + 1] = _ivld_30.y;
                words_12[0 + 2] = _ivld_30.z;
                words_12[0 + 3] = _ivld_30.w;
            }
            if (ld_tok_11 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_12[0 + 0], words_12[0 + 1], words_12[0 + 2], words_12[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(dv2) + (row_base + (long long)ld_tok_11 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            __syncwarp();
        }
        if (num_chunks > rev + 1) {
            #pragma unroll
            for (int gi2 = 0; gi2 < 1; gi2++) {
                int gtok2 = gi2 * 128 + tid_2;
                if (gtok2 < 64) {
                    s_g[next_g_off + gtok2] = g_next[gi2];
                    float _exp2_31 = approx_exp2(g_next[gi2]);
                    s_gain[next_g_off + gtok2] = _exp2_31 * scale;
                }
            }
        }
        #pragma unroll
        for (int gi3 = 0; gi3 < 1; gi3++) {
            g_next[gi3] = g_far[gi3];
        }
        __syncthreads();
        stage_off = next_off;
    }
    asm volatile("cp.async.wait_group 0;");
    if (store_initial_state_grad != 0) {
        if (is_first != 0) {
            #pragma unroll
            for (int nb6 = 0; nb6 < 8; nb6++) {
                int kcol2 = kbase + nb6 * 8 + quad * 2;
                dh0[out_base + (long long)kcol2 * 128 + (long long)vrow_a] = acc_dh[nb6 * 4];
                dh0[out_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_a] = acc_dh[nb6 * 4 + 1];
                dh0[out_base + (long long)kcol2 * 128 + (long long)vrow_b] = acc_dh[nb6 * 4 + 2];
                dh0[out_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_b] = acc_dh[nb6 * 4 + 3];
            }
        }
    }
}

} // extern "C"
