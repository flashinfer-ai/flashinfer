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
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_lm_head_loss_7d065096a4d043d032d1(long long* __restrict__ keys, float* __restrict__ partial, __nv_bfloat16* __restrict__ out, int H, int V, int n_keys, int split, int span)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int col = blockIdx.y * 1024 + warp * 256 + lane * 8;
    int live = ((col < H) ? 1 : 0);
    int lo = blockIdx.x * span;
    if (lo < n_keys) {
        int _min_0 = ((span) < (n_keys - lo) ? (span) : (n_keys - lo));
        int hi = lo + _min_0;
        #pragma unroll 1
        for (int base = lo; base < hi; base += 32) {
            int pos_l = base + lane;
            int first_l = 0;
            if (pos_l < hi) {
                int row_l = (int)(keys[pos_l] >> 32);
                if (row_l >= 0 && row_l < V && pos_l < n_keys - split) {
                    if (pos_l == 0 || (int)(keys[pos_l - 1] >> 32) != row_l) {
                        if ((int)(keys[pos_l + split] >> 32) == row_l) {
                            first_l = 1;
                        }
                    }
                }
            }
            unsigned int starts = __ballot_sync(0xFFFFFFFF, first_l != 0);
            int segments = (int)(unsigned int)__popc(starts);
            #pragma unroll 1
            for (int _segment = 0; _segment < segments; _segment++) {
                int _find_first_set_1 = __ffs(starts);
                int lane_s = _find_first_set_1 - 1;
                starts = starts & starts - 1;
                int seg = base + lane_s;
                int row = (int)(keys[seg] >> 32);
                int parts = 2;
                #pragma unroll 1
                for (int q = 2; q < (n_keys - seg - 1) / split + 1; q++) {
                    if ((int)(keys[seg + q * split] >> 32) != row) {
                        break;
                    }
                    parts = q + 1;
                }
                if (live != 0) {
                    float acc[8];
                    int first_slot = 2 * (seg / split) + 1;
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)first_slot * (unsigned long long)H + (unsigned long long)col);
                        acc[0 + 0] = _v4.x;
                        acc[0 + 1] = _v4.y;
                        acc[0 + 2] = _v4.z;
                        acc[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)first_slot * (unsigned long long)H + (unsigned long long)(col + 4));
                        acc[4 + 0] = _v4.x;
                        acc[4 + 1] = _v4.y;
                        acc[4 + 2] = _v4.z;
                        acc[4 + 3] = _v4.w;
                    }
                    #pragma unroll 1
                    for (int q_1 = 1; q_1 < parts; q_1++) {
                        float part[8];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)(2 * ((seg + q_1 * split) / split) + ((q_1 + 1 < parts) ? 1 : 0)) * (unsigned long long)H + (unsigned long long)col);
                            part[0 + 0] = _v4.x;
                            part[0 + 1] = _v4.y;
                            part[0 + 2] = _v4.z;
                            part[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (unsigned long long)(2 * ((seg + q_1 * split) / split) + ((q_1 + 1 < parts) ? 1 : 0)) * (unsigned long long)H + (unsigned long long)(col + 4));
                            part[4 + 0] = _v4.x;
                            part[4 + 1] = _v4.y;
                            part[4 + 2] = _v4.z;
                            part[4 + 3] = _v4.w;
                        }
                        acc[0] = acc[0] + part[0];
                        acc[1] = acc[1] + part[1];
                        acc[2] = acc[2] + part[2];
                        acc[3] = acc[3] + part[3];
                        acc[4] = acc[4] + part[4];
                        acc[5] = acc[5] + part[5];
                        acc[6] = acc[6] + part[6];
                        acc[7] = acc[7] + part[7];
                    }
                    unsigned int packed[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(out + ((unsigned long long)row * (unsigned long long)H + (unsigned long long)col))[0] = reinterpret_cast<int4*>(packed)[0];
                }
            }
        }
    }
}

} // extern "C"
