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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
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

__global__ __launch_bounds__(128, 8) void
kernel_cake_lm_head_loss_9fb02bcaacbe1871aae8(__nv_bfloat16* __restrict__ z, long long* __restrict__ labels, float* __restrict__ lse, float* __restrict__ d, int row0, int d_off, int V)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int r = blockIdx.y;
    int t = row0 + r;
    long long lbl = labels[t];
    float lse_t = lse[t];
    float d_raw = d[d_off + r];
    unsigned long long row_base = (unsigned long long)r * (unsigned long long)V;
    int num_vecs = V / 8;
    int vi0 = blockIdx.x * 1024 + tid;
    if (lbl < 0) {
        float zero_out[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            zero_out[j] = 0.0f;
        }
        #pragma unroll
        for (int kz = 0; kz < 8; kz++) {
            int vi_z = vi0 + kz * 128;
            if (vi_z < num_vecs) {
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(zero_out[0 + 0], zero_out[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(zero_out[0 + 2], zero_out[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(zero_out[0 + 4], zero_out[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(zero_out[0 + 6], zero_out[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(z + (row_base + (unsigned long long)vi_z * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
    } else {
        int lbl32 = (int)lbl;
        int lbl_vec = lbl32 / 8;
        float nlse2 = (-lse_t) * 1.4426950408889634f;
        unsigned int zero32 = 0;
        unsigned int zw[32];
        float zf[64];
        #pragma unroll
        for (int kl = 0; kl < 8; kl++) {
            int vi_l = vi0 + kl * 128;
            if (vi_l < num_vecs) {
                {
                    uint4 _uv4_0 = *reinterpret_cast<const uint4*>(z + (row_base + (unsigned long long)vi_l * 8) + 0);
                    zw[kl * 4 + 0] = _uv4_0.x;
                    zw[kl * 4 + 1] = _uv4_0.y;
                    zw[kl * 4 + 2] = _uv4_0.z;
                    zw[kl * 4 + 3] = _uv4_0.w;
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int vi = vi0 + k * 128;
            if (vi < num_vecs) {
                float out[8];
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    uint32_t _prmt_b32_0;
                    asm("prmt.b32 %0, %1, %2, 0x1044;" : "=r"(_prmt_b32_0) : "r"(zw[k * 4 + w]), "r"(zero32));
                    float _exp_0 = expf(__uint_as_float(_prmt_b32_0) - lse_t);
                    uint32_t _prmt_b32_1;
                    asm("prmt.b32 %0, %1, %2, 0x3244;" : "=r"(_prmt_b32_1) : "r"(zw[k * 4 + w]), "r"(zero32));
                    float _exp_1 = expf(__uint_as_float(_prmt_b32_1) - lse_t);
                    out[2 * w] = d_raw * (-_exp_0);
                    out[2 * w + 1] = d_raw * (-_exp_1);
                }
                if (vi == lbl_vec) {
                    int col0 = vi * 8;
                    #pragma unroll
                    for (int wl = 0; wl < 4; wl++) {
                        if (col0 + 2 * wl == lbl32) {
                            uint32_t _prmt_b32_2;
                            asm("prmt.b32 %0, %1, %2, 0x1044;" : "=r"(_prmt_b32_2) : "r"(zw[k * 4 + wl]), "r"(zero32));
                            float _exp_2 = expf(__uint_as_float(_prmt_b32_2) - lse_t);
                            out[2 * wl] = d_raw * (1.0f - _exp_2);
                        }
                        if (col0 + 2 * wl + 1 == lbl32) {
                            uint32_t _prmt_b32_3;
                            asm("prmt.b32 %0, %1, %2, 0x3244;" : "=r"(_prmt_b32_3) : "r"(zw[k * 4 + wl]), "r"(zero32));
                            float _exp_3 = expf(__uint_as_float(_prmt_b32_3) - lse_t);
                            out[2 * wl + 1] = d_raw * (1.0f - _exp_3);
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(z + (row_base + (unsigned long long)vi * 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        }
    }
}

} // extern "C"
