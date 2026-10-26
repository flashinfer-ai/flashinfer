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
#define SMEM_RED_OFF 0
#define SMEM_RED_STAGE_BYTES 2048
#define SMEM_RED_STRIDE 2048
#define SMEM_TOTAL 2048
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 1

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_86b5ecfbd90aca4b52e5(float* __restrict__ partial, float* __restrict__ dw, int C, int main_ctas, int n_ch_tiles, int ct_vec)
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
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;

    // === Task calls (declaration order) ===
    int lane_1 = tid / 4;
    int cl = tid % 4;
    int ch = bid * 4 + cl;
    long long C64 = (long long)C;
    long long ch64 = (long long)ch;
    int ctc = ch / 8 / ct_vec;
    int G = (main_ctas - ctc + n_ch_tiles - 1) / n_ch_tiles;
    int n_trips = G + 32 - 1 >> 5;
    float a4[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        a4[e] = 0.0f;
    }
    float pv[32];
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (ch < C) {
        #pragma unroll 1
        for (int k0 = 0; k0 < n_trips; k0 += 8) {
            #pragma unroll
            for (int u = 0; u < 8; u++) {
                int r = lane_1 + (k0 + u << 5);
                if (r < G) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial + ((long long)r * C64 + ch64) * 4);
                        pv[u * 4 + 0] = _v4.x;
                        pv[u * 4 + 1] = _v4.y;
                        pv[u * 4 + 2] = _v4.z;
                        pv[u * 4 + 3] = _v4.w;
                    }
                }
            }
            #pragma unroll
            for (int u_1 = 0; u_1 < 8; u_1++) {
                int r_1 = lane_1 + (k0 + u_1 << 5);
                if (r_1 < G) {
                    #pragma unroll
                    for (int e_1 = 0; e_1 < 4; e_1++) {
                        a4[e_1] = a4[e_1] + pv[u_1 * 4 + e_1];
                    }
                }
            }
        }
    }
    int slot = (lane_1 * 4 + cl) * 4;
    #pragma unroll
    for (int e_2 = 0; e_2 < 4; e_2++) {
        red[slot + e_2] = a4[e_2];
    }
    __syncthreads();
    if (tid < 16) {
        int cl2 = tid / 4;
        int chain = tid % 4;
        int ch2 = bid * 4 + cl2;
        if (ch2 < C) {
            float v[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                v[j] = red[(j * 4 + cl2) * 4 + chain];
            }
            v[0] = v[0] + v[16];
            v[1] = v[1] + v[17];
            v[2] = v[2] + v[18];
            v[3] = v[3] + v[19];
            v[4] = v[4] + v[20];
            v[5] = v[5] + v[21];
            v[6] = v[6] + v[22];
            v[7] = v[7] + v[23];
            v[8] = v[8] + v[24];
            v[9] = v[9] + v[25];
            v[10] = v[10] + v[26];
            v[11] = v[11] + v[27];
            v[12] = v[12] + v[28];
            v[13] = v[13] + v[29];
            v[14] = v[14] + v[30];
            v[15] = v[15] + v[31];
            v[0] = v[0] + v[8];
            v[1] = v[1] + v[9];
            v[2] = v[2] + v[10];
            v[3] = v[3] + v[11];
            v[4] = v[4] + v[12];
            v[5] = v[5] + v[13];
            v[6] = v[6] + v[14];
            v[7] = v[7] + v[15];
            v[0] = v[0] + v[4];
            v[1] = v[1] + v[5];
            v[2] = v[2] + v[6];
            v[3] = v[3] + v[7];
            v[0] = v[0] + v[2];
            v[1] = v[1] + v[3];
            v[0] = v[0] + v[1];
            float total = v[0];
            long long ch64b = (long long)ch2;
            *(reinterpret_cast<float*>(dw) + (ch64b * 4 + (long long)chain)) = total;
        }
    }
}

} // extern "C"
