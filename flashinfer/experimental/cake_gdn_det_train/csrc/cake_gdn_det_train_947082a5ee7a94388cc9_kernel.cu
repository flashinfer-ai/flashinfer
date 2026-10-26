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
#define SMEM_S_KN_STAGE_BYTES 17408
#define SMEM_S_KN_STRIDE 17408
#define SMEM_S_V_OFF 34816
#define SMEM_S_V_STAGE_BYTES 17408
#define SMEM_S_V_STRIDE 17408
#define SMEM_S_A_OFF 17408
#define SMEM_S_A_STAGE_BYTES 9216
#define SMEM_S_A_STRIDE 9216
#define SMEM_S_LT_OFF 17408
#define SMEM_S_LT_STAGE_BYTES 16384
#define SMEM_S_LT_STRIDE 16384
#define SMEM_S_X_OFF 34816
#define SMEM_S_X_STAGE_BYTES 16384
#define SMEM_S_X_STRIDE 16384
#define SMEM_S_T_OFF 51200
#define SMEM_S_T_STAGE_BYTES 3840
#define SMEM_S_T_STRIDE 3840
#define SMEM_S_G_OFF 55040
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 55296
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_S_STAGE_OFF 26624
#define SMEM_S_STAGE_STAGE_BYTES 8192
#define SMEM_S_STAGE_STRIDE 8192
#define SMEM_TOTAL 55552
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_947082a5ee7a94388cc9(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ A_out, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int recompute)
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
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 34816);
    const int s_v_addr = smem + 34816;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int s_a_addr = smem + 17408;
    float* s_lt = reinterpret_cast<float*>(smem_raw + 17408);
    const int s_lt_addr = smem + 17408;
    float* s_x = reinterpret_cast<float*>(smem_raw + 34816);
    const int s_x_addr = smem + 34816;
    float* s_t = reinterpret_cast<float*>(smem_raw + 51200);
    const int s_t_addr = smem + 51200;
    float* s_g = reinterpret_cast<float*>(smem_raw + 55040);
    const int s_g_addr = smem + 55040;
    float* s_b = reinterpret_cast<float*>(smem_raw + 55296);
    const int s_b_addr = smem + 55296;
    int* s_stage = reinterpret_cast<int*>(smem_raw + 26624);
    const int s_stage_addr = smem + 26624;

    // === Task calls (declaration order) ===
    int chunk = blockIdx.x;
    int hv = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int frag_r = lane_0 / 4;
    int col_q = lane_0 % 4 * 2;
    int m0 = warp_1 / 2 * 32;
    int nn0 = warp_1 % 2 * 64;
    unsigned int stage_base = s_stage_addr + (unsigned int)(warp_1 * 2048);
    int r0 = tid / 16;
    int c0 = (tid % 8 + 8 - tid / 16 + tid / 8 % 2 * (tid % 8 + (tid % 8 + tid / 16 & 8) - (tid % 8 + 8 - tid / 16))) * 8;
    long long row_stride = (long long)num_heads * 128;
    long long src_row = (long long)(tok0 + r0) * row_stride + (long long)hq * 128 + (long long)c0;
    unsigned int dst0 = s_kn_addr + (unsigned int)(r0 * 272 + c0 * 2);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(kn + src_row), "r"((n_valid > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2176), "l"(kn + (src_row + 8 * row_stride)), "r"((n_valid > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4352), "l"(kn + (src_row + 16 * row_stride)), "r"((n_valid > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6528), "l"(kn + (src_row + 24 * row_stride)), "r"((n_valid > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 8704), "l"(kn + (src_row + 32 * row_stride)), "r"((n_valid > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 10880), "l"(kn + (src_row + 40 * row_stride)), "r"((n_valid > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 13056), "l"(kn + (src_row + 48 * row_stride)), "r"((n_valid > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 15232), "l"(kn + (src_row + 56 * row_stride)), "r"((n_valid > r0 + 56) ? 16 : 0));
    if (recompute != 0) {
        int r0_0 = tid / 8;
        int c0_1 = tid % 8 * 8;
        long long row_stride_2 = (long long)num_v_heads * 64;
        long long src_row_3 = (long long)(tok0 + r0_0) * row_stride_2 + (long long)hv * 64 + (long long)c0_1;
        unsigned int dst0_4 = s_a_addr + (unsigned int)(r0_0 * 144 + c0_1 * 2);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4), "l"(A_out + src_row_3), "r"((n_valid > r0_0) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 2304), "l"(A_out + (src_row_3 + 16 * row_stride_2)), "r"((n_valid > r0_0 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 4608), "l"(A_out + (src_row_3 + 32 * row_stride_2)), "r"((n_valid > r0_0 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_4 + 6912), "l"(A_out + (src_row_3 + 48 * row_stride_2)), "r"((n_valid > r0_0 + 48) ? 16 : 0));
        int r0_5 = tid / 16;
        int c0_6 = (tid % 8 + 8 - tid / 16 + tid / 8 % 2 * (tid % 8 + (tid % 8 + tid / 16 & 8) - (tid % 8 + 8 - tid / 16))) * 8;
        long long row_stride_7 = (long long)num_v_heads * 128;
        long long src_row_8 = (long long)(tok0 + r0_5) * row_stride_7 + (long long)hv * 128 + (long long)c0_6;
        unsigned int dst0_9 = s_v_addr + (unsigned int)(r0_5 * 272 + c0_6 * 2);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9), "l"(v + src_row_8), "r"((n_valid > r0_5) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 2176), "l"(v + (src_row_8 + 8 * row_stride_7)), "r"((n_valid > r0_5 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 4352), "l"(v + (src_row_8 + 16 * row_stride_7)), "r"((n_valid > r0_5 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 6528), "l"(v + (src_row_8 + 24 * row_stride_7)), "r"((n_valid > r0_5 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 8704), "l"(v + (src_row_8 + 32 * row_stride_7)), "r"((n_valid > r0_5 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 10880), "l"(v + (src_row_8 + 40 * row_stride_7)), "r"((n_valid > r0_5 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 13056), "l"(v + (src_row_8 + 48 * row_stride_7)), "r"((n_valid > r0_5 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_9 + 15232), "l"(v + (src_row_8 + 56 * row_stride_7)), "r"((n_valid > r0_5 + 56) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        float beta_val = 0.0f;
        if (tid_2 < n_valid) {
            long long gate_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
            gate_val = g_cs[gate_index];
            beta_val = beta[gate_index];
        }
        s_g[tid_2] = gate_val;
        s_b[tid_2] = beta_val;
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int a_frag0[4];
    unsigned int a_frag1[4];
    unsigned int b_frag_k0[4];
    unsigned int b_frag_k1[4];
    unsigned int b_frag_k2[4];
    unsigned int b_frag_mn[4];
    float acc[32];
    unsigned int words[4];
    float vals8[8];
    unsigned int pk4[4];
    if (recompute == 0) {
        if (warp_1 == 0) {
            unsigned int la_base = s_kn_addr + (unsigned int)(lane_0 % 16 * 272 + lane_0 / 16 * 8 * 2);
            unsigned int lb_base = s_kn_addr + (unsigned int)((8 * (lane_0 / 16) + lane_0 % 8) * 272 + lane_0 % 16 / 8 * 8 * 2);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 32)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 64)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 96)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 128)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 160)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 192)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                : "r"(la_base + 4352 + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base + 4352 + 224)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            float g_row[4];
            float b_row[4];
            int row_r = frag_r;
            g_row[0] = s_g[row_r];
            b_row[0] = s_b[row_r];
            int row_r_0 = 8 + frag_r;
            g_row[1] = s_g[row_r_0];
            b_row[1] = s_b[row_r_0];
            int row_r_1 = 16 + frag_r;
            g_row[2] = s_g[row_r_1];
            b_row[2] = s_b[row_r_1];
            int row_r_2 = 24 + frag_r;
            g_row[3] = s_g[row_r_2];
            b_row[3] = s_b[row_r_2];
            int col = col_q;
            float g_col = s_g[col];
            int row = frag_r;
            float val = 0.0f;
            if (col < row) {
                if (row < n_valid) {
                    float _exp2_0 = approx_exp2(g_row[0] - g_col);
                    val = b_row[0] * _exp2_0 * acc[0];
                }
            }
            s_lt[col * 64 + (row ^ (col >> 1 & 3) << 3)] = val;
            int row_3 = 8 + frag_r;
            float val_4 = 0.0f;
            if (col < row_3) {
                if (row_3 < n_valid) {
                    float _exp2_1 = approx_exp2(g_row[1] - g_col);
                    val_4 = b_row[1] * _exp2_1 * acc[2];
                }
            }
            s_lt[col * 64 + (row_3 ^ (col >> 1 & 3) << 3)] = val_4;
            int col_5 = col_q + 1;
            float g_col_6 = s_g[col_5];
            int row_7 = frag_r;
            float val_8 = 0.0f;
            if (col_5 < row_7) {
                if (row_7 < n_valid) {
                    float _exp2_2 = approx_exp2(g_row[0] - g_col_6);
                    val_8 = b_row[0] * _exp2_2 * acc[1];
                }
            }
            s_lt[col_5 * 64 + (row_7 ^ (col_5 >> 1 & 3) << 3)] = val_8;
            int row_9 = 8 + frag_r;
            float val_10 = 0.0f;
            if (col_5 < row_9) {
                if (row_9 < n_valid) {
                    float _exp2_3 = approx_exp2(g_row[1] - g_col_6);
                    val_10 = b_row[1] * _exp2_3 * acc[3];
                }
            }
            s_lt[col_5 * 64 + (row_9 ^ (col_5 >> 1 & 3) << 3)] = val_10;
            int col_11 = 8 + col_q;
            float g_col_12 = s_g[col_11];
            int row_13 = frag_r;
            float val_14 = 0.0f;
            if (col_11 < row_13) {
                if (row_13 < n_valid) {
                    float _exp2_4 = approx_exp2(g_row[0] - g_col_12);
                    val_14 = b_row[0] * _exp2_4 * acc[4];
                }
            }
            s_lt[col_11 * 64 + (row_13 ^ (col_11 >> 1 & 3) << 3)] = val_14;
            int row_15 = 8 + frag_r;
            float val_16 = 0.0f;
            if (col_11 < row_15) {
                if (row_15 < n_valid) {
                    float _exp2_5 = approx_exp2(g_row[1] - g_col_12);
                    val_16 = b_row[1] * _exp2_5 * acc[6];
                }
            }
            s_lt[col_11 * 64 + (row_15 ^ (col_11 >> 1 & 3) << 3)] = val_16;
            int col_17 = 8 + col_q + 1;
            float g_col_18 = s_g[col_17];
            int row_19 = frag_r;
            float val_20 = 0.0f;
            if (col_17 < row_19) {
                if (row_19 < n_valid) {
                    float _exp2_6 = approx_exp2(g_row[0] - g_col_18);
                    val_20 = b_row[0] * _exp2_6 * acc[5];
                }
            }
            s_lt[col_17 * 64 + (row_19 ^ (col_17 >> 1 & 3) << 3)] = val_20;
            int row_21 = 8 + frag_r;
            float val_22 = 0.0f;
            if (col_17 < row_21) {
                if (row_21 < n_valid) {
                    float _exp2_7 = approx_exp2(g_row[1] - g_col_18);
                    val_22 = b_row[1] * _exp2_7 * acc[7];
                }
            }
            s_lt[col_17 * 64 + (row_21 ^ (col_17 >> 1 & 3) << 3)] = val_22;
            int col_23 = col_q;
            float g_col_24 = s_g[col_23];
            int row_25 = 16 + frag_r;
            float val_26 = 0.0f;
            if (col_23 < row_25) {
                if (row_25 < n_valid) {
                    float _exp2_8 = approx_exp2(g_row[2] - g_col_24);
                    val_26 = b_row[2] * _exp2_8 * acc[8];
                }
            }
            s_lt[col_23 * 64 + (row_25 ^ (col_23 >> 1 & 3) << 3)] = val_26;
            int row_27 = 24 + frag_r;
            float val_28 = 0.0f;
            if (col_23 < row_27) {
                if (row_27 < n_valid) {
                    float _exp2_9 = approx_exp2(g_row[3] - g_col_24);
                    val_28 = b_row[3] * _exp2_9 * acc[10];
                }
            }
            s_lt[col_23 * 64 + (row_27 ^ (col_23 >> 1 & 3) << 3)] = val_28;
            int col_29 = col_q + 1;
            float g_col_30 = s_g[col_29];
            int row_31 = 16 + frag_r;
            float val_32 = 0.0f;
            if (col_29 < row_31) {
                if (row_31 < n_valid) {
                    float _exp2_10 = approx_exp2(g_row[2] - g_col_30);
                    val_32 = b_row[2] * _exp2_10 * acc[9];
                }
            }
            s_lt[col_29 * 64 + (row_31 ^ (col_29 >> 1 & 3) << 3)] = val_32;
            int row_33 = 24 + frag_r;
            float val_34 = 0.0f;
            if (col_29 < row_33) {
                if (row_33 < n_valid) {
                    float _exp2_11 = approx_exp2(g_row[3] - g_col_30);
                    val_34 = b_row[3] * _exp2_11 * acc[11];
                }
            }
            s_lt[col_29 * 64 + (row_33 ^ (col_29 >> 1 & 3) << 3)] = val_34;
            int col_35 = 8 + col_q;
            float g_col_36 = s_g[col_35];
            int row_37 = 16 + frag_r;
            float val_38 = 0.0f;
            if (col_35 < row_37) {
                if (row_37 < n_valid) {
                    float _exp2_12 = approx_exp2(g_row[2] - g_col_36);
                    val_38 = b_row[2] * _exp2_12 * acc[12];
                }
            }
            s_lt[col_35 * 64 + (row_37 ^ (col_35 >> 1 & 3) << 3)] = val_38;
            int row_39 = 24 + frag_r;
            float val_40 = 0.0f;
            if (col_35 < row_39) {
                if (row_39 < n_valid) {
                    float _exp2_13 = approx_exp2(g_row[3] - g_col_36);
                    val_40 = b_row[3] * _exp2_13 * acc[14];
                }
            }
            s_lt[col_35 * 64 + (row_39 ^ (col_35 >> 1 & 3) << 3)] = val_40;
            int col_41 = 8 + col_q + 1;
            float g_col_42 = s_g[col_41];
            int row_43 = 16 + frag_r;
            float val_44 = 0.0f;
            if (col_41 < row_43) {
                if (row_43 < n_valid) {
                    float _exp2_14 = approx_exp2(g_row[2] - g_col_42);
                    val_44 = b_row[2] * _exp2_14 * acc[13];
                }
            }
            s_lt[col_41 * 64 + (row_43 ^ (col_41 >> 1 & 3) << 3)] = val_44;
            int row_45 = 24 + frag_r;
            float val_46 = 0.0f;
            if (col_41 < row_45) {
                if (row_45 < n_valid) {
                    float _exp2_15 = approx_exp2(g_row[3] - g_col_42);
                    val_46 = b_row[3] * _exp2_15 * acc[15];
                }
            }
            s_lt[col_41 * 64 + (row_45 ^ (col_41 >> 1 & 3) << 3)] = val_46;
            int col_47 = 16 + col_q;
            float g_col_48 = s_g[col_47];
            int row_49 = 16 + frag_r;
            float val_50 = 0.0f;
            if (col_47 < row_49) {
                if (row_49 < n_valid) {
                    float _exp2_16 = approx_exp2(g_row[2] - g_col_48);
                    val_50 = b_row[2] * _exp2_16 * acc[16];
                }
            }
            s_lt[col_47 * 64 + (row_49 ^ (col_47 >> 1 & 3) << 3)] = val_50;
            int row_51 = 24 + frag_r;
            float val_52 = 0.0f;
            if (col_47 < row_51) {
                if (row_51 < n_valid) {
                    float _exp2_17 = approx_exp2(g_row[3] - g_col_48);
                    val_52 = b_row[3] * _exp2_17 * acc[18];
                }
            }
            s_lt[col_47 * 64 + (row_51 ^ (col_47 >> 1 & 3) << 3)] = val_52;
            int col_53 = 16 + col_q + 1;
            float g_col_54 = s_g[col_53];
            int row_55 = 16 + frag_r;
            float val_56 = 0.0f;
            if (col_53 < row_55) {
                if (row_55 < n_valid) {
                    float _exp2_18 = approx_exp2(g_row[2] - g_col_54);
                    val_56 = b_row[2] * _exp2_18 * acc[17];
                }
            }
            s_lt[col_53 * 64 + (row_55 ^ (col_53 >> 1 & 3) << 3)] = val_56;
            int row_57 = 24 + frag_r;
            float val_58 = 0.0f;
            if (col_53 < row_57) {
                if (row_57 < n_valid) {
                    float _exp2_19 = approx_exp2(g_row[3] - g_col_54);
                    val_58 = b_row[3] * _exp2_19 * acc[19];
                }
            }
            s_lt[col_53 * 64 + (row_57 ^ (col_53 >> 1 & 3) << 3)] = val_58;
            int col_59 = 24 + col_q;
            float g_col_60 = s_g[col_59];
            int row_61 = 16 + frag_r;
            float val_62 = 0.0f;
            if (col_59 < row_61) {
                if (row_61 < n_valid) {
                    float _exp2_20 = approx_exp2(g_row[2] - g_col_60);
                    val_62 = b_row[2] * _exp2_20 * acc[20];
                }
            }
            s_lt[col_59 * 64 + (row_61 ^ (col_59 >> 1 & 3) << 3)] = val_62;
            int row_63 = 24 + frag_r;
            float val_64 = 0.0f;
            if (col_59 < row_63) {
                if (row_63 < n_valid) {
                    float _exp2_21 = approx_exp2(g_row[3] - g_col_60);
                    val_64 = b_row[3] * _exp2_21 * acc[22];
                }
            }
            s_lt[col_59 * 64 + (row_63 ^ (col_59 >> 1 & 3) << 3)] = val_64;
            int col_65 = 24 + col_q + 1;
            float g_col_66 = s_g[col_65];
            int row_67 = 16 + frag_r;
            float val_68 = 0.0f;
            if (col_65 < row_67) {
                if (row_67 < n_valid) {
                    float _exp2_22 = approx_exp2(g_row[2] - g_col_66);
                    val_68 = b_row[2] * _exp2_22 * acc[21];
                }
            }
            s_lt[col_65 * 64 + (row_67 ^ (col_65 >> 1 & 3) << 3)] = val_68;
            int row_69 = 24 + frag_r;
            float val_70 = 0.0f;
            if (col_65 < row_69) {
                if (row_69 < n_valid) {
                    float _exp2_23 = approx_exp2(g_row[3] - g_col_66);
                    val_70 = b_row[3] * _exp2_23 * acc[23];
                }
            }
            s_lt[col_65 * 64 + (row_69 ^ (col_65 >> 1 & 3) << 3)] = val_70;
        }
        if (warp_1 == 2) {
            unsigned int la_base_1 = s_kn_addr + (unsigned int)(lane_0 % 16 * 272 + lane_0 / 16 * 8 * 2);
            unsigned int lb_base_1 = s_kn_addr + (unsigned int)((8 * (lane_0 / 16) + lane_0 % 8) * 272 + lane_0 % 16 / 8 * 8 * 2);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 32)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 64)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 96)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 128)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 160)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 192)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_1 + 8704 + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_1 + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_1 + 4352 + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k2[0]), "=r"(b_frag_k2[1]), "=r"(b_frag_k2[2]), "=r"(b_frag_k2[3])
                : "r"(lb_base_1 + 8704 + 224)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[0]), "r"(b_frag_k2[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k2[2]), "r"(b_frag_k2[(2) + 1]));
            float g_row_1[4];
            float b_row_1[4];
            int row_r_3 = 32 + frag_r;
            g_row_1[0] = s_g[row_r_3];
            b_row_1[0] = s_b[row_r_3];
            int row_r_0_1 = 40 + frag_r;
            g_row_1[1] = s_g[row_r_0_1];
            b_row_1[1] = s_b[row_r_0_1];
            int col_1 = col_q;
            float g_col_1 = s_g[col_1];
            int row_1 = 32 + frag_r;
            float val_1 = 0.0f;
            if (col_1 < row_1) {
                if (row_1 < n_valid) {
                    float _exp2_24 = approx_exp2(g_row_1[0] - g_col_1);
                    val_1 = b_row_1[0] * _exp2_24 * acc[0];
                }
            }
            s_lt[col_1 * 64 + (row_1 ^ (col_1 >> 1 & 3) << 3)] = val_1;
            int row_1_1 = 40 + frag_r;
            float val_2 = 0.0f;
            if (col_1 < row_1_1) {
                if (row_1_1 < n_valid) {
                    float _exp2_25 = approx_exp2(g_row_1[1] - g_col_1);
                    val_2 = b_row_1[1] * _exp2_25 * acc[2];
                }
            }
            s_lt[col_1 * 64 + (row_1_1 ^ (col_1 >> 1 & 3) << 3)] = val_2;
            int col_3 = col_q + 1;
            float g_col_4 = s_g[col_3];
            int row_5 = 32 + frag_r;
            float val_6 = 0.0f;
            if (col_3 < row_5) {
                if (row_5 < n_valid) {
                    float _exp2_26 = approx_exp2(g_row_1[0] - g_col_4);
                    val_6 = b_row_1[0] * _exp2_26 * acc[1];
                }
            }
            s_lt[col_3 * 64 + (row_5 ^ (col_3 >> 1 & 3) << 3)] = val_6;
            int row_7_1 = 40 + frag_r;
            float val_8_1 = 0.0f;
            if (col_3 < row_7_1) {
                if (row_7_1 < n_valid) {
                    float _exp2_27 = approx_exp2(g_row_1[1] - g_col_4);
                    val_8_1 = b_row_1[1] * _exp2_27 * acc[3];
                }
            }
            s_lt[col_3 * 64 + (row_7_1 ^ (col_3 >> 1 & 3) << 3)] = val_8_1;
            int col_9 = 8 + col_q;
            float g_col_10 = s_g[col_9];
            int row_11 = 32 + frag_r;
            float val_12 = 0.0f;
            if (col_9 < row_11) {
                if (row_11 < n_valid) {
                    float _exp2_28 = approx_exp2(g_row_1[0] - g_col_10);
                    val_12 = b_row_1[0] * _exp2_28 * acc[4];
                }
            }
            s_lt[col_9 * 64 + (row_11 ^ (col_9 >> 1 & 3) << 3)] = val_12;
            int row_13_1 = 40 + frag_r;
            float val_14_1 = 0.0f;
            if (col_9 < row_13_1) {
                if (row_13_1 < n_valid) {
                    float _exp2_29 = approx_exp2(g_row_1[1] - g_col_10);
                    val_14_1 = b_row_1[1] * _exp2_29 * acc[6];
                }
            }
            s_lt[col_9 * 64 + (row_13_1 ^ (col_9 >> 1 & 3) << 3)] = val_14_1;
            int col_15 = 8 + col_q + 1;
            float g_col_16 = s_g[col_15];
            int row_17 = 32 + frag_r;
            float val_18 = 0.0f;
            if (col_15 < row_17) {
                if (row_17 < n_valid) {
                    float _exp2_30 = approx_exp2(g_row_1[0] - g_col_16);
                    val_18 = b_row_1[0] * _exp2_30 * acc[5];
                }
            }
            s_lt[col_15 * 64 + (row_17 ^ (col_15 >> 1 & 3) << 3)] = val_18;
            int row_19_1 = 40 + frag_r;
            float val_20_1 = 0.0f;
            if (col_15 < row_19_1) {
                if (row_19_1 < n_valid) {
                    float _exp2_31 = approx_exp2(g_row_1[1] - g_col_16);
                    val_20_1 = b_row_1[1] * _exp2_31 * acc[7];
                }
            }
            s_lt[col_15 * 64 + (row_19_1 ^ (col_15 >> 1 & 3) << 3)] = val_20_1;
            int col_21 = 16 + col_q;
            float g_col_22 = s_g[col_21];
            int row_23 = 32 + frag_r;
            float val_24 = 0.0f;
            if (col_21 < row_23) {
                if (row_23 < n_valid) {
                    float _exp2_32 = approx_exp2(g_row_1[0] - g_col_22);
                    val_24 = b_row_1[0] * _exp2_32 * acc[8];
                }
            }
            s_lt[col_21 * 64 + (row_23 ^ (col_21 >> 1 & 3) << 3)] = val_24;
            int row_25_1 = 40 + frag_r;
            float val_26_1 = 0.0f;
            if (col_21 < row_25_1) {
                if (row_25_1 < n_valid) {
                    float _exp2_33 = approx_exp2(g_row_1[1] - g_col_22);
                    val_26_1 = b_row_1[1] * _exp2_33 * acc[10];
                }
            }
            s_lt[col_21 * 64 + (row_25_1 ^ (col_21 >> 1 & 3) << 3)] = val_26_1;
            int col_27 = 16 + col_q + 1;
            float g_col_28 = s_g[col_27];
            int row_29 = 32 + frag_r;
            float val_30 = 0.0f;
            if (col_27 < row_29) {
                if (row_29 < n_valid) {
                    float _exp2_34 = approx_exp2(g_row_1[0] - g_col_28);
                    val_30 = b_row_1[0] * _exp2_34 * acc[9];
                }
            }
            s_lt[col_27 * 64 + (row_29 ^ (col_27 >> 1 & 3) << 3)] = val_30;
            int row_31_1 = 40 + frag_r;
            float val_32_1 = 0.0f;
            if (col_27 < row_31_1) {
                if (row_31_1 < n_valid) {
                    float _exp2_35 = approx_exp2(g_row_1[1] - g_col_28);
                    val_32_1 = b_row_1[1] * _exp2_35 * acc[11];
                }
            }
            s_lt[col_27 * 64 + (row_31_1 ^ (col_27 >> 1 & 3) << 3)] = val_32_1;
            int col_33 = 24 + col_q;
            float g_col_34 = s_g[col_33];
            int row_35 = 32 + frag_r;
            float val_36 = 0.0f;
            if (col_33 < row_35) {
                if (row_35 < n_valid) {
                    float _exp2_36 = approx_exp2(g_row_1[0] - g_col_34);
                    val_36 = b_row_1[0] * _exp2_36 * acc[12];
                }
            }
            s_lt[col_33 * 64 + (row_35 ^ (col_33 >> 1 & 3) << 3)] = val_36;
            int row_37_1 = 40 + frag_r;
            float val_38_1 = 0.0f;
            if (col_33 < row_37_1) {
                if (row_37_1 < n_valid) {
                    float _exp2_37 = approx_exp2(g_row_1[1] - g_col_34);
                    val_38_1 = b_row_1[1] * _exp2_37 * acc[14];
                }
            }
            s_lt[col_33 * 64 + (row_37_1 ^ (col_33 >> 1 & 3) << 3)] = val_38_1;
            int col_39 = 24 + col_q + 1;
            float g_col_40 = s_g[col_39];
            int row_41 = 32 + frag_r;
            float val_42 = 0.0f;
            if (col_39 < row_41) {
                if (row_41 < n_valid) {
                    float _exp2_38 = approx_exp2(g_row_1[0] - g_col_40);
                    val_42 = b_row_1[0] * _exp2_38 * acc[13];
                }
            }
            s_lt[col_39 * 64 + (row_41 ^ (col_39 >> 1 & 3) << 3)] = val_42;
            int row_43_1 = 40 + frag_r;
            float val_44_1 = 0.0f;
            if (col_39 < row_43_1) {
                if (row_43_1 < n_valid) {
                    float _exp2_39 = approx_exp2(g_row_1[1] - g_col_40);
                    val_44_1 = b_row_1[1] * _exp2_39 * acc[15];
                }
            }
            s_lt[col_39 * 64 + (row_43_1 ^ (col_39 >> 1 & 3) << 3)] = val_44_1;
            int col_45 = 32 + col_q;
            float g_col_46 = s_g[col_45];
            int row_47 = 32 + frag_r;
            float val_48 = 0.0f;
            if (col_45 < row_47) {
                if (row_47 < n_valid) {
                    float _exp2_40 = approx_exp2(g_row_1[0] - g_col_46);
                    val_48 = b_row_1[0] * _exp2_40 * acc[16];
                }
            }
            s_lt[col_45 * 64 + (row_47 ^ (col_45 >> 1 & 3) << 3)] = val_48;
            int row_49_1 = 40 + frag_r;
            float val_50_1 = 0.0f;
            if (col_45 < row_49_1) {
                if (row_49_1 < n_valid) {
                    float _exp2_41 = approx_exp2(g_row_1[1] - g_col_46);
                    val_50_1 = b_row_1[1] * _exp2_41 * acc[18];
                }
            }
            s_lt[col_45 * 64 + (row_49_1 ^ (col_45 >> 1 & 3) << 3)] = val_50_1;
            int col_51 = 32 + col_q + 1;
            float g_col_52 = s_g[col_51];
            int row_53 = 32 + frag_r;
            float val_54 = 0.0f;
            if (col_51 < row_53) {
                if (row_53 < n_valid) {
                    float _exp2_42 = approx_exp2(g_row_1[0] - g_col_52);
                    val_54 = b_row_1[0] * _exp2_42 * acc[17];
                }
            }
            s_lt[col_51 * 64 + (row_53 ^ (col_51 >> 1 & 3) << 3)] = val_54;
            int row_55_1 = 40 + frag_r;
            float val_56_1 = 0.0f;
            if (col_51 < row_55_1) {
                if (row_55_1 < n_valid) {
                    float _exp2_43 = approx_exp2(g_row_1[1] - g_col_52);
                    val_56_1 = b_row_1[1] * _exp2_43 * acc[19];
                }
            }
            s_lt[col_51 * 64 + (row_55_1 ^ (col_51 >> 1 & 3) << 3)] = val_56_1;
            int col_57 = 40 + col_q;
            float g_col_58 = s_g[col_57];
            int row_59 = 32 + frag_r;
            float val_60 = 0.0f;
            if (col_57 < row_59) {
                if (row_59 < n_valid) {
                    float _exp2_44 = approx_exp2(g_row_1[0] - g_col_58);
                    val_60 = b_row_1[0] * _exp2_44 * acc[20];
                }
            }
            s_lt[col_57 * 64 + (row_59 ^ (col_57 >> 1 & 3) << 3)] = val_60;
            int row_61_1 = 40 + frag_r;
            float val_62_1 = 0.0f;
            if (col_57 < row_61_1) {
                if (row_61_1 < n_valid) {
                    float _exp2_45 = approx_exp2(g_row_1[1] - g_col_58);
                    val_62_1 = b_row_1[1] * _exp2_45 * acc[22];
                }
            }
            s_lt[col_57 * 64 + (row_61_1 ^ (col_57 >> 1 & 3) << 3)] = val_62_1;
            int col_63 = 40 + col_q + 1;
            float g_col_64 = s_g[col_63];
            int row_65 = 32 + frag_r;
            float val_66 = 0.0f;
            if (col_63 < row_65) {
                if (row_65 < n_valid) {
                    float _exp2_46 = approx_exp2(g_row_1[0] - g_col_64);
                    val_66 = b_row_1[0] * _exp2_46 * acc[21];
                }
            }
            s_lt[col_63 * 64 + (row_65 ^ (col_63 >> 1 & 3) << 3)] = val_66;
            int row_67_1 = 40 + frag_r;
            float val_68_1 = 0.0f;
            if (col_63 < row_67_1) {
                if (row_67_1 < n_valid) {
                    float _exp2_47 = approx_exp2(g_row_1[1] - g_col_64);
                    val_68_1 = b_row_1[1] * _exp2_47 * acc[23];
                }
            }
            s_lt[col_63 * 64 + (row_67_1 ^ (col_63 >> 1 & 3) << 3)] = val_68_1;
        }
        if (warp_1 % 2 == 1) {
            int l_col_base = warp_1 / 2 * 32;
            unsigned int la_base_2 = s_kn_addr + (unsigned int)(lane_0 % 16 * 272 + lane_0 / 16 * 8 * 2);
            unsigned int lb_base_2 = s_kn_addr + (unsigned int)((l_col_base + 8 * (lane_0 / 16) + lane_0 % 8) * 272 + lane_0 % 16 / 8 * 8 * 2);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 32)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 32)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 64)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 64)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 96)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 96)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 128)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 128)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 160)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 160)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 192)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 192)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                : "r"(la_base_2 + 13056 + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                : "r"(lb_base_2 + 224)
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                : "r"(lb_base_2 + 4352 + 224)
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]));
            float g_row_2[4];
            float b_row_2[4];
            int row_r_4 = 48 + frag_r;
            g_row_2[0] = s_g[row_r_4];
            b_row_2[0] = s_b[row_r_4];
            int row_r_0_2 = 56 + frag_r;
            g_row_2[1] = s_g[row_r_0_2];
            b_row_2[1] = s_b[row_r_0_2];
            int col_2 = l_col_base + col_q;
            float g_col_2 = s_g[col_2];
            int row_2 = 48 + frag_r;
            float val_3 = 0.0f;
            if (col_2 < row_2) {
                if (row_2 < n_valid) {
                    float _exp2_48 = approx_exp2(g_row_2[0] - g_col_2);
                    val_3 = b_row_2[0] * _exp2_48 * acc[0];
                }
            }
            s_lt[col_2 * 64 + (row_2 ^ (col_2 >> 1 & 3) << 3)] = val_3;
            int row_1_2 = 56 + frag_r;
            float val_2_1 = 0.0f;
            if (col_2 < row_1_2) {
                if (row_1_2 < n_valid) {
                    float _exp2_49 = approx_exp2(g_row_2[1] - g_col_2);
                    val_2_1 = b_row_2[1] * _exp2_49 * acc[2];
                }
            }
            s_lt[col_2 * 64 + (row_1_2 ^ (col_2 >> 1 & 3) << 3)] = val_2_1;
            int col_3_1 = l_col_base + col_q + 1;
            float g_col_4_1 = s_g[col_3_1];
            int row_5_1 = 48 + frag_r;
            float val_6_1 = 0.0f;
            if (col_3_1 < row_5_1) {
                if (row_5_1 < n_valid) {
                    float _exp2_50 = approx_exp2(g_row_2[0] - g_col_4_1);
                    val_6_1 = b_row_2[0] * _exp2_50 * acc[1];
                }
            }
            s_lt[col_3_1 * 64 + (row_5_1 ^ (col_3_1 >> 1 & 3) << 3)] = val_6_1;
            int row_7_2 = 56 + frag_r;
            float val_8_2 = 0.0f;
            if (col_3_1 < row_7_2) {
                if (row_7_2 < n_valid) {
                    float _exp2_51 = approx_exp2(g_row_2[1] - g_col_4_1);
                    val_8_2 = b_row_2[1] * _exp2_51 * acc[3];
                }
            }
            s_lt[col_3_1 * 64 + (row_7_2 ^ (col_3_1 >> 1 & 3) << 3)] = val_8_2;
            int col_9_1 = l_col_base + 8 + col_q;
            float g_col_10_1 = s_g[col_9_1];
            int row_11_1 = 48 + frag_r;
            float val_12_1 = 0.0f;
            if (col_9_1 < row_11_1) {
                if (row_11_1 < n_valid) {
                    float _exp2_52 = approx_exp2(g_row_2[0] - g_col_10_1);
                    val_12_1 = b_row_2[0] * _exp2_52 * acc[4];
                }
            }
            s_lt[col_9_1 * 64 + (row_11_1 ^ (col_9_1 >> 1 & 3) << 3)] = val_12_1;
            int row_13_2 = 56 + frag_r;
            float val_14_2 = 0.0f;
            if (col_9_1 < row_13_2) {
                if (row_13_2 < n_valid) {
                    float _exp2_53 = approx_exp2(g_row_2[1] - g_col_10_1);
                    val_14_2 = b_row_2[1] * _exp2_53 * acc[6];
                }
            }
            s_lt[col_9_1 * 64 + (row_13_2 ^ (col_9_1 >> 1 & 3) << 3)] = val_14_2;
            int col_15_1 = l_col_base + 8 + col_q + 1;
            float g_col_16_1 = s_g[col_15_1];
            int row_17_1 = 48 + frag_r;
            float val_18_1 = 0.0f;
            if (col_15_1 < row_17_1) {
                if (row_17_1 < n_valid) {
                    float _exp2_54 = approx_exp2(g_row_2[0] - g_col_16_1);
                    val_18_1 = b_row_2[0] * _exp2_54 * acc[5];
                }
            }
            s_lt[col_15_1 * 64 + (row_17_1 ^ (col_15_1 >> 1 & 3) << 3)] = val_18_1;
            int row_19_2 = 56 + frag_r;
            float val_20_2 = 0.0f;
            if (col_15_1 < row_19_2) {
                if (row_19_2 < n_valid) {
                    float _exp2_55 = approx_exp2(g_row_2[1] - g_col_16_1);
                    val_20_2 = b_row_2[1] * _exp2_55 * acc[7];
                }
            }
            s_lt[col_15_1 * 64 + (row_19_2 ^ (col_15_1 >> 1 & 3) << 3)] = val_20_2;
            int col_21_1 = l_col_base + 16 + col_q;
            float g_col_22_1 = s_g[col_21_1];
            int row_23_1 = 48 + frag_r;
            float val_24_1 = 0.0f;
            if (col_21_1 < row_23_1) {
                if (row_23_1 < n_valid) {
                    float _exp2_56 = approx_exp2(g_row_2[0] - g_col_22_1);
                    val_24_1 = b_row_2[0] * _exp2_56 * acc[8];
                }
            }
            s_lt[col_21_1 * 64 + (row_23_1 ^ (col_21_1 >> 1 & 3) << 3)] = val_24_1;
            int row_25_2 = 56 + frag_r;
            float val_26_2 = 0.0f;
            if (col_21_1 < row_25_2) {
                if (row_25_2 < n_valid) {
                    float _exp2_57 = approx_exp2(g_row_2[1] - g_col_22_1);
                    val_26_2 = b_row_2[1] * _exp2_57 * acc[10];
                }
            }
            s_lt[col_21_1 * 64 + (row_25_2 ^ (col_21_1 >> 1 & 3) << 3)] = val_26_2;
            int col_27_1 = l_col_base + 16 + col_q + 1;
            float g_col_28_1 = s_g[col_27_1];
            int row_29_1 = 48 + frag_r;
            float val_30_1 = 0.0f;
            if (col_27_1 < row_29_1) {
                if (row_29_1 < n_valid) {
                    float _exp2_58 = approx_exp2(g_row_2[0] - g_col_28_1);
                    val_30_1 = b_row_2[0] * _exp2_58 * acc[9];
                }
            }
            s_lt[col_27_1 * 64 + (row_29_1 ^ (col_27_1 >> 1 & 3) << 3)] = val_30_1;
            int row_31_2 = 56 + frag_r;
            float val_32_2 = 0.0f;
            if (col_27_1 < row_31_2) {
                if (row_31_2 < n_valid) {
                    float _exp2_59 = approx_exp2(g_row_2[1] - g_col_28_1);
                    val_32_2 = b_row_2[1] * _exp2_59 * acc[11];
                }
            }
            s_lt[col_27_1 * 64 + (row_31_2 ^ (col_27_1 >> 1 & 3) << 3)] = val_32_2;
            int col_33_1 = l_col_base + 16 + 8 + col_q;
            float g_col_34_1 = s_g[col_33_1];
            int row_35_1 = 48 + frag_r;
            float val_36_1 = 0.0f;
            if (col_33_1 < row_35_1) {
                if (row_35_1 < n_valid) {
                    float _exp2_60 = approx_exp2(g_row_2[0] - g_col_34_1);
                    val_36_1 = b_row_2[0] * _exp2_60 * acc[12];
                }
            }
            s_lt[col_33_1 * 64 + (row_35_1 ^ (col_33_1 >> 1 & 3) << 3)] = val_36_1;
            int row_37_2 = 56 + frag_r;
            float val_38_2 = 0.0f;
            if (col_33_1 < row_37_2) {
                if (row_37_2 < n_valid) {
                    float _exp2_61 = approx_exp2(g_row_2[1] - g_col_34_1);
                    val_38_2 = b_row_2[1] * _exp2_61 * acc[14];
                }
            }
            s_lt[col_33_1 * 64 + (row_37_2 ^ (col_33_1 >> 1 & 3) << 3)] = val_38_2;
            int col_39_1 = l_col_base + 16 + 8 + col_q + 1;
            float g_col_40_1 = s_g[col_39_1];
            int row_41_1 = 48 + frag_r;
            float val_42_1 = 0.0f;
            if (col_39_1 < row_41_1) {
                if (row_41_1 < n_valid) {
                    float _exp2_62 = approx_exp2(g_row_2[0] - g_col_40_1);
                    val_42_1 = b_row_2[0] * _exp2_62 * acc[13];
                }
            }
            s_lt[col_39_1 * 64 + (row_41_1 ^ (col_39_1 >> 1 & 3) << 3)] = val_42_1;
            int row_43_2 = 56 + frag_r;
            float val_44_2 = 0.0f;
            if (col_39_1 < row_43_2) {
                if (row_43_2 < n_valid) {
                    float _exp2_63 = approx_exp2(g_row_2[1] - g_col_40_1);
                    val_44_2 = b_row_2[1] * _exp2_63 * acc[15];
                }
            }
            s_lt[col_39_1 * 64 + (row_43_2 ^ (col_39_1 >> 1 & 3) << 3)] = val_44_2;
        }
        __syncthreads();
        float zero4[4];
        zero4[0] = 0.0f;
        zero4[1] = 0.0f;
        zero4[2] = 0.0f;
        zero4[3] = 0.0f;
        int z_half = tid_2 / 64;
        int zr = tid_2 % 64 / 4;
        int zc = tid_2 % 4 * 4;
        int zbi = 0;
        int zbj = 1 + z_half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)(((zbi * 16 + zr) * 64 + (zbj * 16 + zc ^ (zbi * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(zero4[0])), "r"(__float_as_uint(zero4[1])), "r"(__float_as_uint(zero4[2])), "r"(__float_as_uint(zero4[3])));
        int zbi_0 = z_half;
        int zbj_1 = 3 + -1 * z_half;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)(((zbi_0 * 16 + zr) * 64 + (zbj_1 * 16 + zc ^ (zbi_0 * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(zero4[0])), "r"(__float_as_uint(zero4[1])), "r"(__float_as_uint(zero4[2])), "r"(__float_as_uint(zero4[3])));
        int zbi_2 = 1 + z_half;
        int zbj_3 = 3;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)(((zbi_2 * 16 + zr) * 64 + (zbj_3 * 16 + zc ^ (zbi_2 * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(zero4[0])), "r"(__float_as_uint(zero4[1])), "r"(__float_as_uint(zero4[2])), "r"(__float_as_uint(zero4[3])));
        if (tid_2 < 64) {
            int dblk = tid_2 / 16;
            int dcol = tid_2 % 16;
            int drow0 = dblk * 16;
            int lt_d[2];
            lt_d[0] = drow0 * 65;
            lt_d[1] = drow0 * 64 + (drow0 ^ 16);
            int x_d[2];
            x_d[0] = drow0 * 65 + dcol;
            x_d[1] = drow0 * 64 + (drow0 ^ 16) + dcol;
            float xcol[16];
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                float xv = 0.0f;
                if (di == dcol) {
                    xv = 1.0f;
                }
                #pragma unroll
                for (int dj = 0; dj < di; dj++) {
                    xv = __fmaf_rn(-s_lt[lt_d[(dj >> 1 & 3) << 3 >> 4 & 1] + (dj * 64 + (di ^ (dj >> 1 & 3) << 3 & 8))], xcol[dj], xv);
                }
                xcol[di] = xv;
            }
            #pragma unroll
            for (int di2 = 0; di2 < 16; di2++) {
                s_x[x_d[di2 >> 2 & 1] + di2 * 64] = xcol[di2];
            }
        } else {
            int s_row0 = (tid_2 - 64) / 16;
            int s_col = (tid_2 - 64) % 16 * 8;
            int base[1];
            base[0] = s_row0 * 272 + s_col * 2;
            float _exp2_0_1 = approx_exp2(s_g[s_row0]);
            int o = base[0];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0] * _exp2_0_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0] * _exp2_0_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0] * _exp2_0_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0] * _exp2_0_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0] * _exp2_0_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0] * _exp2_0_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0] * _exp2_0_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0] * _exp2_0_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_1_1 = approx_exp2(s_g[s_row0 + 4]);
            int o_0 = base[0] + 1088;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_0));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 4] * _exp2_1_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 4] * _exp2_1_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_0), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_2_1 = approx_exp2(s_g[s_row0 + 8]);
            int o_1 = base[0] + 2176;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_1));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 8] * _exp2_2_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 8] * _exp2_2_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_3_1 = approx_exp2(s_g[s_row0 + 12]);
            int o_2 = base[0] + 3264;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_2));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 12] * _exp2_3_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 12] * _exp2_3_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_2), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_4_1 = approx_exp2(s_g[s_row0 + 16]);
            int o_3 = base[0] + 4352;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_3));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 16] * _exp2_4_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 16] * _exp2_4_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_3), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_5_1 = approx_exp2(s_g[s_row0 + 20]);
            int o_4 = base[0] + 5440;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_4));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 20] * _exp2_5_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 20] * _exp2_5_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_4), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_6_1 = approx_exp2(s_g[s_row0 + 24]);
            int o_5 = base[0] + 6528;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_5));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 24] * _exp2_6_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 24] * _exp2_6_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_5), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_7_1 = approx_exp2(s_g[s_row0 + 28]);
            int o_6 = base[0] + 7616;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_6));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 28] * _exp2_7_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 28] * _exp2_7_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_6), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_8_1 = approx_exp2(s_g[s_row0 + 32]);
            int o_7 = base[0] + 8704;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_7));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 32] * _exp2_8_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 32] * _exp2_8_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_7), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_9_1 = approx_exp2(s_g[s_row0 + 36]);
            int o_8 = base[0] + 9792;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_8));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 36] * _exp2_9_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 36] * _exp2_9_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_8), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_10_1 = approx_exp2(s_g[s_row0 + 40]);
            int o_9 = base[0] + 10880;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_9));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 40] * _exp2_10_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 40] * _exp2_10_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_9), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_11_1 = approx_exp2(s_g[s_row0 + 44]);
            int o_10 = base[0] + 11968;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_10));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 44] * _exp2_11_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 44] * _exp2_11_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_10), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_12_1 = approx_exp2(s_g[s_row0 + 48]);
            int o_11 = base[0] + 13056;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_11));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 48] * _exp2_12_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 48] * _exp2_12_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_11), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_13_1 = approx_exp2(s_g[s_row0 + 52]);
            int o_12 = base[0] + 14144;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_12));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 52] * _exp2_13_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 52] * _exp2_13_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_12), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_14_1 = approx_exp2(s_g[s_row0 + 56]);
            int o_13 = base[0] + 15232;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_13));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 56] * _exp2_14_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 56] * _exp2_14_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_13), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
            float _exp2_15_1 = approx_exp2(s_g[s_row0 + 60]);
            int o_14 = base[0] + 16320;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                : "r"(s_kn_addr + (unsigned int)o_14));
            vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0 + 60] * _exp2_15_1);
            vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0 + 60] * _exp2_15_1);
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
                pk4[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_14), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        }
        __syncthreads();
        float acc8[8];
        float lv[2];
        float xv4[4];
        float xi8[8];
        float tv16[16];
        int tr0 = lane_0 / 8 / 2 * 8 + lane_0 / 8 % 2 * 2 + lane_0 % 8 / 4 * 4;
        int tc0 = lane_0 % 4 * 4;
        int r3 = warp_1 * 4 + lane_0 / 8;
        int c3 = lane_0 % 8 * 2;
        if (warp_1 < 3) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 1) * 16 + tr0 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tr0 * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (tr0 + 1) * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        if (warp_1 < 3) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            acc8[0] = -acc8[0];
            acc8[1] = -acc8[1];
            acc8[2] = -acc8[2];
            acc8[3] = -acc8[3];
            acc8[4] = -acc8[4];
            acc8[5] = -acc8[5];
            acc8[6] = -acc8[6];
            acc8[7] = -acc8[7];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + tr0 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + tr0 + 1) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        if (warp_1 < 2) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 1 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 1 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 2 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 2 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 3 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 3 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 4 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 4 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 5 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 5 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 6 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 6 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 7 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 7 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 8 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 8 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 9 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 9 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 10 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 10 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 11 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 11 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 12 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 12 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 13 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 13 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 14 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 14 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 2) * 16 + tr0 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + tc0 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1]))
                : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + ((warp_1 + 2) * 16 + tr0 ^ ((warp_1 + 1) * 16 + 15 >> 1 & 3) << 3)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 1) * 16 + 15 >> 2 & 1) << 4)) * 4)));
            acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
            acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
            acc8[2] = __fmaf_rn(lv[0], xv4[2], acc8[2]);
            acc8[3] = __fmaf_rn(lv[0], xv4[3], acc8[3]);
            acc8[4] = __fmaf_rn(lv[1], xv4[0], acc8[4]);
            acc8[5] = __fmaf_rn(lv[1], xv4[1], acc8[5]);
            acc8[6] = __fmaf_rn(lv[1], xv4[2], acc8[6]);
            acc8[7] = __fmaf_rn(lv[1], xv4[3], acc8[7]);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tr0 * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (tr0 + 1) * 20 + tc0) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        if (warp_1 < 2) {
            acc8[0] = 0.0f;
            acc8[1] = 0.0f;
            acc8[2] = 0.0f;
            acc8[3] = 0.0f;
            acc8[4] = 0.0f;
            acc8[5] = 0.0f;
            acc8[6] = 0.0f;
            acc8[7] = 0.0f;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&xi8[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(4) + 3]))
                : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + tc0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + tc0) * 4)));
            acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[0], tv16[2], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[0], tv16[3], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[4], tv16[0], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[4], tv16[1], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[4], tv16[2], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[4], tv16[3], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[1], tv16[4], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[1], tv16[5], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[1], tv16[6], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[1], tv16[7], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[5], tv16[4], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[5], tv16[5], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[5], tv16[6], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[5], tv16[7], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[2], tv16[8], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[2], tv16[9], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[2], tv16[10], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[2], tv16[11], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[6], tv16[8], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[6], tv16[9], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[6], tv16[10], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[6], tv16[11], acc8[7]);
            acc8[0] = __fmaf_rn(xi8[3], tv16[12], acc8[0]);
            acc8[1] = __fmaf_rn(xi8[3], tv16[13], acc8[1]);
            acc8[2] = __fmaf_rn(xi8[3], tv16[14], acc8[2]);
            acc8[3] = __fmaf_rn(xi8[3], tv16[15], acc8[3]);
            acc8[4] = __fmaf_rn(xi8[7], tv16[12], acc8[4]);
            acc8[5] = __fmaf_rn(xi8[7], tv16[13], acc8[5]);
            acc8[6] = __fmaf_rn(xi8[7], tv16[14], acc8[6]);
            acc8[7] = __fmaf_rn(xi8[7], tv16[15], acc8[7]);
            acc8[0] = -acc8[0];
            acc8[1] = -acc8[1];
            acc8[2] = -acc8[2];
            acc8[3] = -acc8[3];
            acc8[4] = -acc8[4];
            acc8[5] = -acc8[5];
            acc8[6] = -acc8[6];
            acc8[7] = -acc8[7];
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 2) * 16 + tr0 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])), "r"(__float_as_uint(acc8[2])), "r"(__float_as_uint(acc8[3])));
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + tr0 + 1) * 64 + (warp_1 * 16 + tc0 ^ ((warp_1 + 2) * 16 + tr0 + 1 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[4])), "r"(__float_as_uint(acc8[5])), "r"(__float_as_uint(acc8[6])), "r"(__float_as_uint(acc8[7])));
        }
        __syncthreads();
        acc8[0] = 0.0f;
        acc8[1] = 0.0f;
        acc8[2] = 0.0f;
        acc8[3] = 0.0f;
        acc8[4] = 0.0f;
        acc8[5] = 0.0f;
        acc8[6] = 0.0f;
        acc8[7] = 0.0f;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((48 + r3 ^ 0) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((c3 ^ 0) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1024 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1024 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2048 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2048 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((64 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((64 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1088 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1088 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2112 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2112 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((128 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((128 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1152 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1152 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2176 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2176 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((192 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((192 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1216 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1216 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2240 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2240 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((256 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((256 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1280 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1280 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2304 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2304 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((320 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((320 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1344 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1344 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2368 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2368 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((384 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((384 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1408 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1408 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2432 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2432 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((448 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((448 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1472 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1472 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2496 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2496 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((512 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((512 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1536 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1536 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2560 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2560 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((576 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((576 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1600 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1600 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2624 + (48 + r3 ^ 0)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2624 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((640 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((640 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1664 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1664 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2688 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2688 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((704 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((704 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1728 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1728 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2752 + (48 + r3 ^ 8)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2752 + (c3 ^ 0)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((768 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((768 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1792 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1792 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2816 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2816 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((832 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((832 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1856 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1856 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2880 + (48 + r3 ^ 16)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2880 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((896 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((896 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1920 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1920 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((2944 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((2944 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((960 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((960 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((1984 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((1984 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])) : "r"(s_lt_addr + (unsigned int)((3008 + (48 + r3 ^ 24)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1]))
            : "r"(s_x_addr + (unsigned int)((3008 + (c3 ^ 16)) * 4)));
        acc8[0] = __fmaf_rn(lv[0], xv4[0], acc8[0]);
        acc8[1] = __fmaf_rn(lv[0], xv4[1], acc8[1]);
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_t_addr + (unsigned int)((r3 * 20 + c3) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])));
        __syncthreads();
        acc8[0] = 0.0f;
        acc8[1] = 0.0f;
        acc8[2] = 0.0f;
        acc8[3] = 0.0f;
        acc8[4] = 0.0f;
        acc8[5] = 0.0f;
        acc8[6] = 0.0f;
        acc8[7] = 0.0f;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (48 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)(c3 * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((20 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((40 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((60 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (52 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)((80 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((100 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((120 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((140 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (56 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)((160 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((180 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((200 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((220 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&xi8[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi8[(0) + 3]))
            : "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (60 ^ (48 + r3 >> 2 & 1) << 4)) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1]))
            : "r"(s_t_addr + (unsigned int)((240 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(2) + 1]))
            : "r"(s_t_addr + (unsigned int)((260 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1]))
            : "r"(s_t_addr + (unsigned int)((280 + c3) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tv16[6])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(6) + 1]))
            : "r"(s_t_addr + (unsigned int)((300 + c3) * 4)));
        acc8[0] = __fmaf_rn(xi8[0], tv16[0], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[0], tv16[1], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[1], tv16[2], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[1], tv16[3], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[2], tv16[4], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[2], tv16[5], acc8[1]);
        acc8[0] = __fmaf_rn(xi8[3], tv16[6], acc8[0]);
        acc8[1] = __fmaf_rn(xi8[3], tv16[7], acc8[1]);
        acc8[0] = -acc8[0];
        acc8[1] = -acc8[1];
        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_x_addr + (unsigned int)(((48 + r3) * 64 + (c3 ^ (48 + r3 >> 2 & 1) << 4)) * 4)), "r"(__float_as_uint(acc8[0])), "r"(__float_as_uint(acc8[1])));
        __syncthreads();
        float quad[4];
        unsigned int pk2[2];
        int p_row = tid_2 / 16;
        int p_col = tid_2 % 16 * 4;
        unsigned int x_pub = s_x_addr + (unsigned int)((p_row * 64 + (p_col ^ (p_row >> 2 & 1) << 4)) * 4);
        int a_pub = p_row * 144 + p_col * 2;
        long long a_idx0 = ((long long)(tok0 + p_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)p_col;
        long long a_rows8 = (long long)num_v_heads * 512;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                : "r"(x_pub + (unsigned int)(t * 8 * 64 * 4)));
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 0], quad[_lp*2+1 + 0]));
                pk2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_a_addr + (unsigned int)(a_pub + t * 8 * 144)), "r"(pk2[0]), "r"(pk2[1]));
            if (n_valid > p_row + t * 8) {
                unsigned int _packed_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 0], quad[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 2], quad[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(A_out + (a_idx0 + a_rows8 * (long long)t))[0] = reinterpret_cast<int2*>(_packed_0)[0];
            }
        }
        __syncthreads();
        int r0_4 = tid / 16;
        int c0_5 = (tid % 8 + 8 - tid / 16 + tid / 8 % 2 * (tid % 8 + (tid % 8 + tid / 16 & 8) - (tid % 8 + 8 - tid / 16))) * 8;
        long long row_stride_6 = (long long)num_v_heads * 128;
        long long src_row_7 = (long long)(tok0 + r0_4) * row_stride_6 + (long long)hv * 128 + (long long)c0_5;
        unsigned int dst0_8 = s_v_addr + (unsigned int)(r0_4 * 272 + c0_5 * 2);
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8), "l"(v + src_row_7), "r"((n_valid > r0_4) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 2176), "l"(v + (src_row_7 + 8 * row_stride_6)), "r"((n_valid > r0_4 + 8) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 4352), "l"(v + (src_row_7 + 16 * row_stride_6)), "r"((n_valid > r0_4 + 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 6528), "l"(v + (src_row_7 + 24 * row_stride_6)), "r"((n_valid > r0_4 + 24) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 8704), "l"(v + (src_row_7 + 32 * row_stride_6)), "r"((n_valid > r0_4 + 32) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 10880), "l"(v + (src_row_7 + 40 * row_stride_6)), "r"((n_valid > r0_4 + 40) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 13056), "l"(v + (src_row_7 + 48 * row_stride_6)), "r"((n_valid > r0_4 + 48) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(dst0_8 + 15232), "l"(v + (src_row_7 + 56 * row_stride_6)), "r"((n_valid > r0_4 + 56) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
    }
    if (recompute != 0) {
        int s_row0_1 = tid_2 / 16;
        int s_col_1 = tid_2 % 16 * 8;
        int base_1[1];
        base_1[0] = s_row0_1 * 272 + s_col_1 * 2;
        float _exp2_16_1 = approx_exp2(s_g[s_row0_1]);
        int o_15 = base_1[0];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_15));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1] * _exp2_16_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1] * _exp2_16_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_15), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_17_1 = approx_exp2(s_g[s_row0_1 + 8]);
        int o_0_1 = base_1[0] + 2176;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_0_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 8] * _exp2_17_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_0_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_18_1 = approx_exp2(s_g[s_row0_1 + 16]);
        int o_1_1 = base_1[0] + 4352;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_1_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 16] * _exp2_18_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_1_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_19_1 = approx_exp2(s_g[s_row0_1 + 24]);
        int o_2_1 = base_1[0] + 6528;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_2_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 24] * _exp2_19_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_2_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_20_1 = approx_exp2(s_g[s_row0_1 + 32]);
        int o_3_1 = base_1[0] + 8704;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_3_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 32] * _exp2_20_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_3_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_21_1 = approx_exp2(s_g[s_row0_1 + 40]);
        int o_4_1 = base_1[0] + 10880;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_4_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 40] * _exp2_21_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_4_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_22_1 = approx_exp2(s_g[s_row0_1 + 48]);
        int o_5_1 = base_1[0] + 13056;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_5_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 48] * _exp2_22_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_5_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        float _exp2_23_1 = approx_exp2(s_g[s_row0_1 + 56]);
        int o_6_1 = base_1[0] + 15232;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(s_kn_addr + (unsigned int)o_6_1));
        vals8[0] = __uint_as_float(words[0] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[2] = __uint_as_float(words[1] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[4] = __uint_as_float(words[2] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[6] = __uint_as_float(words[3] << 16) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[s_row0_1 + 56] * _exp2_23_1);
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
            pk4[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)o_6_1), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
        __syncthreads();
    }
    int m_idx = lane_0 / 8;
    int r_idx = lane_0 % 8;
    int st_row = m_idx % 2 * 8 + r_idx;
    int st_sub = m_idx / 2;
    int ld_row = lane_0 / 4;
    int ld_ch = lane_0 % 4;
    int words_3[4];
    int frag_row = lane_0 % 16;
    int frag_hi = lane_0 / 16 * 8;
    unsigned int a_base = s_a_addr + (unsigned int)((m0 + frag_row) * 144 + frag_hi * 2);
    unsigned int b_base = s_kn_addr + (unsigned int)(frag_row * 272 + (nn0 + frag_hi) * 2);
    int cbase = nn0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[24]), "=f"(acc[(24) + 1]), "=f"(acc[(24) + 2]), "=f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[28]), "=f"(acc[(28) + 1]), "=f"(acc[(28) + 2]), "=f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304 + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 4352)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 4352 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304 + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 8704)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 8704 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304 + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 13056)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 13056 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s = st_row;
    uint32_t _stmatrix_addr_64 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s * 64 + ((st_sub ^ row_s >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_64), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_4 = st_row;
    uint32_t _stmatrix_addr_65 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_4 * 64 + ((2 + st_sub ^ row_s_4 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_65), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_5 = 16 + st_row;
    uint32_t _stmatrix_addr_66 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_5 * 64 + ((st_sub ^ row_s_5 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_66), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_6 = 16 + st_row;
    uint32_t _stmatrix_addr_67 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_6 * 64 + ((2 + st_sub ^ row_s_6 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_67), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_4 = ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_4 * 64 + ((ld_ch ^ row_4 >> 1 & 3) << 4))));
    int orow = m0 + row_4;
    if (orow < n_valid) {
        long long out = ((long long)(tok0 + orow) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out / 2) = _iv4;
        }
    }
    int row_7_3 = 8 + ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_7_3 * 64 + ((ld_ch ^ row_7_3 >> 1 & 3) << 4))));
    int orow_8 = m0 + row_7_3;
    if (orow_8 < n_valid) {
        long long out_1 = ((long long)(tok0 + orow_8) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_1 / 2) = _iv4;
        }
    }
    int row_9_1 = 16 + ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_9_1 * 64 + ((ld_ch ^ row_9_1 >> 1 & 3) << 4))));
    int orow_10 = m0 + row_9_1;
    if (orow_10 < n_valid) {
        long long out_2 = ((long long)(tok0 + orow_10) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_2 / 2) = _iv4;
        }
    }
    int row_11_2 = 24 + ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_11_2 * 64 + ((ld_ch ^ row_11_2 >> 1 & 3) << 4))));
    int orow_12 = m0 + row_11_2;
    if (orow_12 < n_valid) {
        long long out_3 = ((long long)(tok0 + orow_12) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_3 / 2) = _iv4;
        }
    }
    __syncwarp();
    int cbase_13 = nn0 + 32;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[24]), "=f"(acc[(24) + 1]), "=f"(acc[(24) + 2]), "=f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[28]), "=f"(acc[(28) + 1]), "=f"(acc[(28) + 2]), "=f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304 + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 4352 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 4352 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304 + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 8704 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 8704 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base + 2304 + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 13056 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base + 13056 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_14 = st_row;
    uint32_t _stmatrix_addr_68 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_14 * 64 + ((st_sub ^ row_s_14 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_68), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_15 = st_row;
    uint32_t _stmatrix_addr_69 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_15 * 64 + ((2 + st_sub ^ row_s_15 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_69), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_16 = 16 + st_row;
    uint32_t _stmatrix_addr_70 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_16 * 64 + ((st_sub ^ row_s_16 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_70), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_17 = 16 + st_row;
    uint32_t _stmatrix_addr_71 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_17 * 64 + ((2 + st_sub ^ row_s_17 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_71), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_18 = ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_18 * 64 + ((ld_ch ^ row_18 >> 1 & 3) << 4))));
    int orow_19 = m0 + row_18;
    if (orow_19 < n_valid) {
        long long out_4 = ((long long)(tok0 + orow_19) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_13 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_4 / 2) = _iv4;
        }
    }
    int row_20 = 8 + ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_20 * 64 + ((ld_ch ^ row_20 >> 1 & 3) << 4))));
    int orow_21 = m0 + row_20;
    if (orow_21 < n_valid) {
        long long out_5 = ((long long)(tok0 + orow_21) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_13 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_5 / 2) = _iv4;
        }
    }
    int row_22 = 16 + ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_22 * 64 + ((ld_ch ^ row_22 >> 1 & 3) << 4))));
    int orow_23 = m0 + row_22;
    if (orow_23 < n_valid) {
        long long out_6 = ((long long)(tok0 + orow_23) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_13 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_6 / 2) = _iv4;
        }
    }
    int row_24 = 24 + ld_row;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_24 * 64 + ((ld_ch ^ row_24 >> 1 & 3) << 4))));
    int orow_25 = m0 + row_24;
    if (orow_25 < n_valid) {
        long long out_7 = ((long long)(tok0 + orow_25) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_13 + (long long)(ld_ch * 8);
        {
            int4 _iv4 = make_int4(words_3[0 + 0], words_3[0 + 1], words_3[0 + 2], words_3[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(w_out) + out_7 / 2) = _iv4;
        }
    }
    __syncwarp();
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int s_row0_2 = tid_2 / 16;
    int s_col_2 = tid_2 % 16 * 8;
    int base_2[1];
    base_2[0] = s_row0_2 * 272 + s_col_2 * 2;
    int o_16 = base_2[0];
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_16));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_16), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_26 = base_2[0] + 2176;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_26));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 8];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 8];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 8];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 8];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 8];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 8];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 8];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 8];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_26), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_27 = base_2[0] + 4352;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_27));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_27), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_28 = base_2[0] + 6528;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_28));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 24];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 24];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 24];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 24];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 24];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 24];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 24];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 24];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_28), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_29 = base_2[0] + 8704;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_29));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 32];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 32];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 32];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 32];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 32];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 32];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 32];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 32];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_29), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_30 = base_2[0] + 10880;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_30));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 40];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 40];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 40];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 40];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 40];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 40];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 40];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 40];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_30), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_31 = base_2[0] + 13056;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_31));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 48];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 48];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 48];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 48];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 48];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 48];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 48];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 48];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_31), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    int o_32 = base_2[0] + 15232;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)o_32));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[s_row0_2 + 56];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[s_row0_2 + 56];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[s_row0_2 + 56];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[s_row0_2 + 56];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[s_row0_2 + 56];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[s_row0_2 + 56];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[s_row0_2 + 56];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[s_row0_2 + 56];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)o_32), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]));
    __syncthreads();
    int m_idx_33 = lane_0 / 8;
    int r_idx_34 = lane_0 % 8;
    int st_row_35 = m_idx_33 % 2 * 8 + r_idx_34;
    int st_sub_36 = m_idx_33 / 2;
    int ld_row_37 = lane_0 / 4;
    int ld_ch_38 = lane_0 % 4;
    int words_39[4];
    int frag_row_40 = lane_0 % 16;
    int frag_hi_41 = lane_0 / 16 * 8;
    unsigned int a_base_42 = s_a_addr + (unsigned int)((m0 + frag_row_40) * 144 + frag_hi_41 * 2);
    unsigned int b_base_43 = s_v_addr + (unsigned int)(frag_row_40 * 272 + (nn0 + frag_hi_41) * 2);
    int cbase_44 = nn0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[24]), "=f"(acc[(24) + 1]), "=f"(acc[(24) + 2]), "=f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[28]), "=f"(acc[(28) + 1]), "=f"(acc[(28) + 2]), "=f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42 + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304 + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 4352)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 4352 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42 + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304 + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 8704)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 8704 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42 + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304 + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 13056)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 13056 + 32)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_45 = st_row_35;
    uint32_t _stmatrix_addr_72 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_45 * 64 + ((st_sub_36 ^ row_s_45 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_72), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_46 = st_row_35;
    uint32_t _stmatrix_addr_73 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_46 * 64 + ((2 + st_sub_36 ^ row_s_46 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_73), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_47 = 16 + st_row_35;
    uint32_t _stmatrix_addr_74 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_47 * 64 + ((st_sub_36 ^ row_s_47 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_74), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_48 = 16 + st_row_35;
    uint32_t _stmatrix_addr_75 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_48 * 64 + ((2 + st_sub_36 ^ row_s_48 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_75), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_49_2 = ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_49_2 * 64 + ((ld_ch_38 ^ row_49_2 >> 1 & 3) << 4))));
    int orow_50 = m0 + row_49_2;
    if (orow_50 < n_valid) {
        long long out_8 = ((long long)(tok0 + orow_50) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_44 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_8 / 2) = _iv4;
        }
    }
    int row_51_1 = 8 + ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_51_1 * 64 + ((ld_ch_38 ^ row_51_1 >> 1 & 3) << 4))));
    int orow_52 = m0 + row_51_1;
    if (orow_52 < n_valid) {
        long long out_9 = ((long long)(tok0 + orow_52) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_44 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_9 / 2) = _iv4;
        }
    }
    int row_53_1 = 16 + ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_53_1 * 64 + ((ld_ch_38 ^ row_53_1 >> 1 & 3) << 4))));
    int orow_54 = m0 + row_53_1;
    if (orow_54 < n_valid) {
        long long out_10 = ((long long)(tok0 + orow_54) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_44 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_10 / 2) = _iv4;
        }
    }
    int row_55_2 = 24 + ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_55_2 * 64 + ((ld_ch_38 ^ row_55_2 >> 1 & 3) << 4))));
    int orow_56 = m0 + row_55_2;
    if (orow_56 < n_valid) {
        long long out_11 = ((long long)(tok0 + orow_56) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_44 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_11 / 2) = _iv4;
        }
    }
    __syncwarp();
    int cbase_57 = nn0 + 32;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[24]), "=f"(acc[(24) + 1]), "=f"(acc[(24) + 2]), "=f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(acc[28]), "=f"(acc[(28) + 1]), "=f"(acc[(28) + 2]), "=f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42 + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304 + 32)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 4352 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 4352 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42 + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304 + 64)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 8704 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 8704 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
        : "r"(a_base_42 + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
        : "r"(a_base_42 + 2304 + 96)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 13056 + 64)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[16]), "+f"(acc[(16) + 1]), "+f"(acc[(16) + 2]), "+f"(acc[(16) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[20]), "+f"(acc[(20) + 1]), "+f"(acc[(20) + 2]), "+f"(acc[(20) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
        : "r"(b_base_43 + 13056 + 96)
        : "memory");
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[8]), "+f"(acc[(8) + 1]), "+f"(acc[(8) + 2]), "+f"(acc[(8) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[12]), "+f"(acc[(12) + 1]), "+f"(acc[(12) + 2]), "+f"(acc[(12) + 3])
        : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[24]), "+f"(acc[(24) + 1]), "+f"(acc[(24) + 2]), "+f"(acc[(24) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(acc[28]), "+f"(acc[(28) + 1]), "+f"(acc[(28) + 2]), "+f"(acc[(28) + 3])
        : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_58 = st_row_35;
    uint32_t _stmatrix_addr_76 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_58 * 64 + ((st_sub_36 ^ row_s_58 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_76), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 8], acc[_lp*2+1 + 8]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_59 = st_row_35;
    uint32_t _stmatrix_addr_77 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_59 * 64 + ((2 + st_sub_36 ^ row_s_59 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_77), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 16], acc[_lp*2+1 + 16]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_60 = 16 + st_row_35;
    uint32_t _stmatrix_addr_78 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_60 * 64 + ((st_sub_36 ^ row_s_60 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_78), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 24], acc[_lp*2+1 + 24]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    int row_s_61 = 16 + st_row_35;
    uint32_t _stmatrix_addr_79 = static_cast<uint32_t>(stage_base + (unsigned int)(row_s_61 * 64 + ((2 + st_sub_36 ^ row_s_61 >> 1 & 3) << 4)));
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(_stmatrix_addr_79), "r"(*reinterpret_cast<const uint32_t*>(&pk4[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk4[3]))
        : "memory");
    __syncwarp();
    int row_62 = ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_62 * 64 + ((ld_ch_38 ^ row_62 >> 1 & 3) << 4))));
    int orow_63 = m0 + row_62;
    if (orow_63 < n_valid) {
        long long out_12 = ((long long)(tok0 + orow_63) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_57 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_12 / 2) = _iv4;
        }
    }
    int row_64 = 8 + ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_64 * 64 + ((ld_ch_38 ^ row_64 >> 1 & 3) << 4))));
    int orow_65 = m0 + row_64;
    if (orow_65 < n_valid) {
        long long out_13 = ((long long)(tok0 + orow_65) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_57 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_13 / 2) = _iv4;
        }
    }
    int row_66 = 16 + ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_66 * 64 + ((ld_ch_38 ^ row_66 >> 1 & 3) << 4))));
    int orow_67 = m0 + row_66;
    if (orow_67 < n_valid) {
        long long out_14 = ((long long)(tok0 + orow_67) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_57 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_14 / 2) = _iv4;
        }
    }
    int row_68 = 24 + ld_row_37;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
        : "r"(stage_base + (unsigned int)(row_68 * 64 + ((ld_ch_38 ^ row_68 >> 1 & 3) << 4))));
    int orow_69 = m0 + row_68;
    if (orow_69 < n_valid) {
        long long out_15 = ((long long)(tok0 + orow_69) * (long long)num_v_heads + (long long)hv) * 128 + (long long)cbase_57 + (long long)(ld_ch_38 * 8);
        {
            int4 _iv4 = make_int4(words_39[0 + 0], words_39[0 + 1], words_39[0 + 2], words_39[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(u_out) + out_15 / 2) = _iv4;
        }
    }
    __syncwarp();
}

} // extern "C"
