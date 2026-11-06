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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) __cluster_dims__(8,1,1) void
kernel_cake_dense_projection_gemm_2f07a521e0f7ebd21021(float* __restrict__ X, float* __restrict__ W, float* __restrict__ out, int T, int N, int K, int ldx, int ldw, int ldo)
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
    const unsigned int clusters_x = gridDim.x / 8;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 8;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Prologue
    float* smem_1 = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int rank = blockIdx.x % 8;
    int e0 = blockIdx.x / 8 * 16;
    int t0 = blockIdx.y * 64;
    int seg = K / 8;
    int k0 = rank * seg;
    int chunks = seg / 128;
    int j = lane % 16;
    int ge = tid / 16 % 2;
    int gt = tid / 16 / 2;
    float acc[64];
    float comp[1];
    #pragma unroll
    for (int o = 0; o < 64; o++) {
        acc[o] = 0.0f;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (0 < chunks) {
        unsigned int fill = smem_addr + 0;
        int col0 = k0;
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int wv_ = i * 256 + tid;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(fill + (unsigned int)((wv_ / 32 * 144 + wv_ % 32 * 4) * 4)), "l"(W + ((long long)(e0 + wv_ / 32) * (long long)ldw + (long long)(col0 + wv_ % 32 * 4))));
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
            int xv_ = i_1 * 256 + tid;
            int tok = t0 + xv_ / 32;
            int _min_0 = ((tok) < (T - 1) ? (tok) : (T - 1));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(fill + (unsigned int)(((16 + xv_ / 32) * 144 + xv_ % 32 * 4) * 4)), "l"(X + ((long long)_min_0 * (long long)ldx + (long long)(col0 + xv_ % 32 * 4))), "r"((tok < T) ? 16 : 0));
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int c = 0; c < chunks; c++) {
        if (c + 2 - 1 < chunks) {
            unsigned int fill_1 = smem_addr + (unsigned int)((c + 2 - 1) % 2 * 46080);
            int col0_1 = k0 + (c + 2 - 1) * 128;
            #pragma unroll
            for (int i_2 = 0; i_2 < 2; i_2++) {
                int wv__1 = i_2 * 256 + tid;
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                    :: "r"(fill_1 + (unsigned int)((wv__1 / 32 * 144 + wv__1 % 32 * 4) * 4)), "l"(W + ((long long)(e0 + wv__1 / 32) * (long long)ldw + (long long)(col0_1 + wv__1 % 32 * 4))));
            }
            #pragma unroll
            for (int i_3 = 0; i_3 < 8; i_3++) {
                int xv__1 = i_3 * 256 + tid;
                int tok_1 = t0 + xv__1 / 32;
                int _min_1 = ((tok_1) < (T - 1) ? (tok_1) : (T - 1));
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(fill_1 + (unsigned int)(((16 + xv__1 / 32) * 144 + xv__1 % 32 * 4) * 4)), "l"(X + ((long long)_min_1 * (long long)ldx + (long long)(col0_1 + xv__1 % 32 * 4))), "r"((tok_1 < T) ? 16 : 0));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        __syncthreads();
        unsigned int cur = smem_addr + (unsigned int)(c % 2 * 46080);
        float wv[8];
        float xv[8];
        float yv[1];
        #pragma unroll
        for (int g4 = 0; g4 < 8; g4++) {
            int col_0 = j + 16 * g4;
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])) : "r"(cur + (unsigned int)((ge * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[1])) : "r"(cur + (unsigned int)(((ge + 2) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[2])) : "r"(cur + (unsigned int)(((ge + 4) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[3])) : "r"(cur + (unsigned int)(((ge + 6) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[4])) : "r"(cur + (unsigned int)(((ge + 8) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[5])) : "r"(cur + (unsigned int)(((ge + 10) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[6])) : "r"(cur + (unsigned int)(((ge + 12) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&wv[7])) : "r"(cur + (unsigned int)(((ge + 14) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[0])) : "r"(cur + (unsigned int)(((16 + gt) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[1])) : "r"(cur + (unsigned int)(((16 + gt + 8) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[2])) : "r"(cur + (unsigned int)(((16 + gt + 16) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[3])) : "r"(cur + (unsigned int)(((16 + gt + 24) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[4])) : "r"(cur + (unsigned int)(((16 + gt + 32) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[5])) : "r"(cur + (unsigned int)(((16 + gt + 40) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[6])) : "r"(cur + (unsigned int)(((16 + gt + 48) * 144 + col_0) * 4)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&xv[7])) : "r"(cur + (unsigned int)(((16 + gt + 56) * 144 + col_0) * 4)));
            acc[0] = __fmaf_rn(xv[0], wv[0], acc[0]);
            acc[1] = __fmaf_rn(xv[1], wv[0], acc[1]);
            acc[2] = __fmaf_rn(xv[2], wv[0], acc[2]);
            acc[3] = __fmaf_rn(xv[3], wv[0], acc[3]);
            acc[4] = __fmaf_rn(xv[4], wv[0], acc[4]);
            acc[5] = __fmaf_rn(xv[5], wv[0], acc[5]);
            acc[6] = __fmaf_rn(xv[6], wv[0], acc[6]);
            acc[7] = __fmaf_rn(xv[7], wv[0], acc[7]);
            acc[8] = __fmaf_rn(xv[0], wv[1], acc[8]);
            acc[9] = __fmaf_rn(xv[1], wv[1], acc[9]);
            acc[10] = __fmaf_rn(xv[2], wv[1], acc[10]);
            acc[11] = __fmaf_rn(xv[3], wv[1], acc[11]);
            acc[12] = __fmaf_rn(xv[4], wv[1], acc[12]);
            acc[13] = __fmaf_rn(xv[5], wv[1], acc[13]);
            acc[14] = __fmaf_rn(xv[6], wv[1], acc[14]);
            acc[15] = __fmaf_rn(xv[7], wv[1], acc[15]);
            acc[16] = __fmaf_rn(xv[0], wv[2], acc[16]);
            acc[17] = __fmaf_rn(xv[1], wv[2], acc[17]);
            acc[18] = __fmaf_rn(xv[2], wv[2], acc[18]);
            acc[19] = __fmaf_rn(xv[3], wv[2], acc[19]);
            acc[20] = __fmaf_rn(xv[4], wv[2], acc[20]);
            acc[21] = __fmaf_rn(xv[5], wv[2], acc[21]);
            acc[22] = __fmaf_rn(xv[6], wv[2], acc[22]);
            acc[23] = __fmaf_rn(xv[7], wv[2], acc[23]);
            acc[24] = __fmaf_rn(xv[0], wv[3], acc[24]);
            acc[25] = __fmaf_rn(xv[1], wv[3], acc[25]);
            acc[26] = __fmaf_rn(xv[2], wv[3], acc[26]);
            acc[27] = __fmaf_rn(xv[3], wv[3], acc[27]);
            acc[28] = __fmaf_rn(xv[4], wv[3], acc[28]);
            acc[29] = __fmaf_rn(xv[5], wv[3], acc[29]);
            acc[30] = __fmaf_rn(xv[6], wv[3], acc[30]);
            acc[31] = __fmaf_rn(xv[7], wv[3], acc[31]);
            acc[32] = __fmaf_rn(xv[0], wv[4], acc[32]);
            acc[33] = __fmaf_rn(xv[1], wv[4], acc[33]);
            acc[34] = __fmaf_rn(xv[2], wv[4], acc[34]);
            acc[35] = __fmaf_rn(xv[3], wv[4], acc[35]);
            acc[36] = __fmaf_rn(xv[4], wv[4], acc[36]);
            acc[37] = __fmaf_rn(xv[5], wv[4], acc[37]);
            acc[38] = __fmaf_rn(xv[6], wv[4], acc[38]);
            acc[39] = __fmaf_rn(xv[7], wv[4], acc[39]);
            acc[40] = __fmaf_rn(xv[0], wv[5], acc[40]);
            acc[41] = __fmaf_rn(xv[1], wv[5], acc[41]);
            acc[42] = __fmaf_rn(xv[2], wv[5], acc[42]);
            acc[43] = __fmaf_rn(xv[3], wv[5], acc[43]);
            acc[44] = __fmaf_rn(xv[4], wv[5], acc[44]);
            acc[45] = __fmaf_rn(xv[5], wv[5], acc[45]);
            acc[46] = __fmaf_rn(xv[6], wv[5], acc[46]);
            acc[47] = __fmaf_rn(xv[7], wv[5], acc[47]);
            acc[48] = __fmaf_rn(xv[0], wv[6], acc[48]);
            acc[49] = __fmaf_rn(xv[1], wv[6], acc[49]);
            acc[50] = __fmaf_rn(xv[2], wv[6], acc[50]);
            acc[51] = __fmaf_rn(xv[3], wv[6], acc[51]);
            acc[52] = __fmaf_rn(xv[4], wv[6], acc[52]);
            acc[53] = __fmaf_rn(xv[5], wv[6], acc[53]);
            acc[54] = __fmaf_rn(xv[6], wv[6], acc[54]);
            acc[55] = __fmaf_rn(xv[7], wv[6], acc[55]);
            acc[56] = __fmaf_rn(xv[0], wv[7], acc[56]);
            acc[57] = __fmaf_rn(xv[1], wv[7], acc[57]);
            acc[58] = __fmaf_rn(xv[2], wv[7], acc[58]);
            acc[59] = __fmaf_rn(xv[3], wv[7], acc[59]);
            acc[60] = __fmaf_rn(xv[4], wv[7], acc[60]);
            acc[61] = __fmaf_rn(xv[5], wv[7], acc[61]);
            acc[62] = __fmaf_rn(xv[6], wv[7], acc[62]);
            acc[63] = __fmaf_rn(xv[7], wv[7], acc[63]);
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;");
    #pragma unroll
    for (int o_0 = 0; o_0 < 64; o_0++) {
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, acc[o_0], 1);
        acc[o_0] = acc[o_0] + _shfl_xor_sync_2;
    }
    #pragma unroll
    for (int o_1 = 0; o_1 < 64; o_1++) {
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, acc[o_1], 2);
        acc[o_1] = acc[o_1] + _shfl_xor_sync_3;
    }
    #pragma unroll
    for (int o_2 = 0; o_2 < 64; o_2++) {
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, acc[o_2], 4);
        acc[o_2] = acc[o_2] + _shfl_xor_sync_4;
    }
    #pragma unroll
    for (int o_3 = 0; o_3 < 64; o_3++) {
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, acc[o_3], 8);
        acc[o_3] = acc[o_3] + _shfl_xor_sync_5;
    }
    if (j == 0) {
        int red_o_0 = ge * 64 + gt;
        smem_1[red_o_0] = acc[0];
    }
    if (j == 1) {
        int red_o_1 = ge * 64 + gt + 8;
        smem_1[red_o_1] = acc[1];
    }
    if (j == 2) {
        int red_o_2 = ge * 64 + gt + 16;
        smem_1[red_o_2] = acc[2];
    }
    if (j == 3) {
        int red_o_3 = ge * 64 + gt + 24;
        smem_1[red_o_3] = acc[3];
    }
    if (j == 4) {
        int red_o_4 = ge * 64 + gt + 32;
        smem_1[red_o_4] = acc[4];
    }
    if (j == 5) {
        int red_o_5 = ge * 64 + gt + 40;
        smem_1[red_o_5] = acc[5];
    }
    if (j == 6) {
        int red_o_6 = ge * 64 + gt + 48;
        smem_1[red_o_6] = acc[6];
    }
    if (j == 7) {
        int red_o_7 = ge * 64 + gt + 56;
        smem_1[red_o_7] = acc[7];
    }
    if (j == 8) {
        int red_o_8 = (ge + 2) * 64 + gt;
        smem_1[red_o_8] = acc[8];
    }
    if (j == 9) {
        int red_o_9 = (ge + 2) * 64 + gt + 8;
        smem_1[red_o_9] = acc[9];
    }
    if (j == 10) {
        int red_o_10 = (ge + 2) * 64 + gt + 16;
        smem_1[red_o_10] = acc[10];
    }
    if (j == 11) {
        int red_o_11 = (ge + 2) * 64 + gt + 24;
        smem_1[red_o_11] = acc[11];
    }
    if (j == 12) {
        int red_o_12 = (ge + 2) * 64 + gt + 32;
        smem_1[red_o_12] = acc[12];
    }
    if (j == 13) {
        int red_o_13 = (ge + 2) * 64 + gt + 40;
        smem_1[red_o_13] = acc[13];
    }
    if (j == 14) {
        int red_o_14 = (ge + 2) * 64 + gt + 48;
        smem_1[red_o_14] = acc[14];
    }
    if (j == 15) {
        int red_o_15 = (ge + 2) * 64 + gt + 56;
        smem_1[red_o_15] = acc[15];
    }
    if (j == 0) {
        int red_o_16 = (ge + 4) * 64 + gt;
        smem_1[red_o_16] = acc[16];
    }
    if (j == 1) {
        int red_o_17 = (ge + 4) * 64 + gt + 8;
        smem_1[red_o_17] = acc[17];
    }
    if (j == 2) {
        int red_o_18 = (ge + 4) * 64 + gt + 16;
        smem_1[red_o_18] = acc[18];
    }
    if (j == 3) {
        int red_o_19 = (ge + 4) * 64 + gt + 24;
        smem_1[red_o_19] = acc[19];
    }
    if (j == 4) {
        int red_o_20 = (ge + 4) * 64 + gt + 32;
        smem_1[red_o_20] = acc[20];
    }
    if (j == 5) {
        int red_o_21 = (ge + 4) * 64 + gt + 40;
        smem_1[red_o_21] = acc[21];
    }
    if (j == 6) {
        int red_o_22 = (ge + 4) * 64 + gt + 48;
        smem_1[red_o_22] = acc[22];
    }
    if (j == 7) {
        int red_o_23 = (ge + 4) * 64 + gt + 56;
        smem_1[red_o_23] = acc[23];
    }
    if (j == 8) {
        int red_o_24 = (ge + 6) * 64 + gt;
        smem_1[red_o_24] = acc[24];
    }
    if (j == 9) {
        int red_o_25 = (ge + 6) * 64 + gt + 8;
        smem_1[red_o_25] = acc[25];
    }
    if (j == 10) {
        int red_o_26 = (ge + 6) * 64 + gt + 16;
        smem_1[red_o_26] = acc[26];
    }
    if (j == 11) {
        int red_o_27 = (ge + 6) * 64 + gt + 24;
        smem_1[red_o_27] = acc[27];
    }
    if (j == 12) {
        int red_o_28 = (ge + 6) * 64 + gt + 32;
        smem_1[red_o_28] = acc[28];
    }
    if (j == 13) {
        int red_o_29 = (ge + 6) * 64 + gt + 40;
        smem_1[red_o_29] = acc[29];
    }
    if (j == 14) {
        int red_o_30 = (ge + 6) * 64 + gt + 48;
        smem_1[red_o_30] = acc[30];
    }
    if (j == 15) {
        int red_o_31 = (ge + 6) * 64 + gt + 56;
        smem_1[red_o_31] = acc[31];
    }
    if (j == 0) {
        int red_o_32 = (ge + 8) * 64 + gt;
        smem_1[red_o_32] = acc[32];
    }
    if (j == 1) {
        int red_o_33 = (ge + 8) * 64 + gt + 8;
        smem_1[red_o_33] = acc[33];
    }
    if (j == 2) {
        int red_o_34 = (ge + 8) * 64 + gt + 16;
        smem_1[red_o_34] = acc[34];
    }
    if (j == 3) {
        int red_o_35 = (ge + 8) * 64 + gt + 24;
        smem_1[red_o_35] = acc[35];
    }
    if (j == 4) {
        int red_o_36 = (ge + 8) * 64 + gt + 32;
        smem_1[red_o_36] = acc[36];
    }
    if (j == 5) {
        int red_o_37 = (ge + 8) * 64 + gt + 40;
        smem_1[red_o_37] = acc[37];
    }
    if (j == 6) {
        int red_o_38 = (ge + 8) * 64 + gt + 48;
        smem_1[red_o_38] = acc[38];
    }
    if (j == 7) {
        int red_o_39 = (ge + 8) * 64 + gt + 56;
        smem_1[red_o_39] = acc[39];
    }
    if (j == 8) {
        int red_o_40 = (ge + 10) * 64 + gt;
        smem_1[red_o_40] = acc[40];
    }
    if (j == 9) {
        int red_o_41 = (ge + 10) * 64 + gt + 8;
        smem_1[red_o_41] = acc[41];
    }
    if (j == 10) {
        int red_o_42 = (ge + 10) * 64 + gt + 16;
        smem_1[red_o_42] = acc[42];
    }
    if (j == 11) {
        int red_o_43 = (ge + 10) * 64 + gt + 24;
        smem_1[red_o_43] = acc[43];
    }
    if (j == 12) {
        int red_o_44 = (ge + 10) * 64 + gt + 32;
        smem_1[red_o_44] = acc[44];
    }
    if (j == 13) {
        int red_o_45 = (ge + 10) * 64 + gt + 40;
        smem_1[red_o_45] = acc[45];
    }
    if (j == 14) {
        int red_o_46 = (ge + 10) * 64 + gt + 48;
        smem_1[red_o_46] = acc[46];
    }
    if (j == 15) {
        int red_o_47 = (ge + 10) * 64 + gt + 56;
        smem_1[red_o_47] = acc[47];
    }
    if (j == 0) {
        int red_o_48 = (ge + 12) * 64 + gt;
        smem_1[red_o_48] = acc[48];
    }
    if (j == 1) {
        int red_o_49 = (ge + 12) * 64 + gt + 8;
        smem_1[red_o_49] = acc[49];
    }
    if (j == 2) {
        int red_o_50 = (ge + 12) * 64 + gt + 16;
        smem_1[red_o_50] = acc[50];
    }
    if (j == 3) {
        int red_o_51 = (ge + 12) * 64 + gt + 24;
        smem_1[red_o_51] = acc[51];
    }
    if (j == 4) {
        int red_o_52 = (ge + 12) * 64 + gt + 32;
        smem_1[red_o_52] = acc[52];
    }
    if (j == 5) {
        int red_o_53 = (ge + 12) * 64 + gt + 40;
        smem_1[red_o_53] = acc[53];
    }
    if (j == 6) {
        int red_o_54 = (ge + 12) * 64 + gt + 48;
        smem_1[red_o_54] = acc[54];
    }
    if (j == 7) {
        int red_o_55 = (ge + 12) * 64 + gt + 56;
        smem_1[red_o_55] = acc[55];
    }
    if (j == 8) {
        int red_o_56 = (ge + 14) * 64 + gt;
        smem_1[red_o_56] = acc[56];
    }
    if (j == 9) {
        int red_o_57 = (ge + 14) * 64 + gt + 8;
        smem_1[red_o_57] = acc[57];
    }
    if (j == 10) {
        int red_o_58 = (ge + 14) * 64 + gt + 16;
        smem_1[red_o_58] = acc[58];
    }
    if (j == 11) {
        int red_o_59 = (ge + 14) * 64 + gt + 24;
        smem_1[red_o_59] = acc[59];
    }
    if (j == 12) {
        int red_o_60 = (ge + 14) * 64 + gt + 32;
        smem_1[red_o_60] = acc[60];
    }
    if (j == 13) {
        int red_o_61 = (ge + 14) * 64 + gt + 40;
        smem_1[red_o_61] = acc[61];
    }
    if (j == 14) {
        int red_o_62 = (ge + 14) * 64 + gt + 48;
        smem_1[red_o_62] = acc[62];
    }
    if (j == 15) {
        int red_o_63 = (ge + 14) * 64 + gt + 56;
        smem_1[red_o_63] = acc[63];
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    float peer[8];
    float res[1];
    int idx_0 = tid;
    if (idx_0 < 128) {
        int el_0 = rank * 2 + idx_0 % 2;
        int tl_0 = idx_0 / 2;
        unsigned int red_byte_0 = (unsigned int)((el_0 * 64 + tl_0) * 4);
        uint32_t _cluster_mapa_6;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_6) : "r"(smem_addr + red_byte_0), "r"(0));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[0])) : "r"(_cluster_mapa_6));
        uint32_t _cluster_mapa_7;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_7) : "r"(smem_addr + red_byte_0), "r"(1));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[1])) : "r"(_cluster_mapa_7));
        uint32_t _cluster_mapa_8;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_8) : "r"(smem_addr + red_byte_0), "r"(2));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[2])) : "r"(_cluster_mapa_8));
        uint32_t _cluster_mapa_9;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_9) : "r"(smem_addr + red_byte_0), "r"(3));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[3])) : "r"(_cluster_mapa_9));
        uint32_t _cluster_mapa_10;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_10) : "r"(smem_addr + red_byte_0), "r"(4));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[4])) : "r"(_cluster_mapa_10));
        uint32_t _cluster_mapa_11;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_11) : "r"(smem_addr + red_byte_0), "r"(5));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[5])) : "r"(_cluster_mapa_11));
        uint32_t _cluster_mapa_12;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_12) : "r"(smem_addr + red_byte_0), "r"(6));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[6])) : "r"(_cluster_mapa_12));
        uint32_t _cluster_mapa_13;
        asm volatile(
            "mapa.shared::cluster.u32 %0, %1, %2;"
            : "=r"(_cluster_mapa_13) : "r"(smem_addr + red_byte_0), "r"(7));
        asm volatile("ld.shared::cluster.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&peer[7])) : "r"(_cluster_mapa_13));
        res[0] = peer[0];
        res[0] = res[0] + peer[1];
        res[0] = res[0] + peer[2];
        res[0] = res[0] + peer[3];
        res[0] = res[0] + peer[4];
        res[0] = res[0] + peer[5];
        res[0] = res[0] + peer[6];
        res[0] = res[0] + peer[7];
        if (t0 + tl_0 < T) {
            *(reinterpret_cast<float*>(out + ((long long)(t0 + tl_0) * (long long)ldo + (long long)(e0 + el_0))) + (0)) = res[0];
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

} // extern "C"
