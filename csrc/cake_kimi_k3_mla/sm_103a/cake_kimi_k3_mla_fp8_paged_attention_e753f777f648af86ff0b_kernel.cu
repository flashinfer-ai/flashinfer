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
#define SMEM_SMEM_W_OFF 0
#define SMEM_SMEM_W_STAGE_BYTES 1024
#define SMEM_SMEM_W_STRIDE 1024
#define SMEM_SMEM_RED_OFF 1024
#define SMEM_SMEM_RED_STAGE_BYTES 64
#define SMEM_SMEM_RED_STRIDE 64
#define SMEM_SMEM_ACC_OFF 1088
#define SMEM_SMEM_ACC_STAGE_BYTES 4096
#define SMEM_SMEM_ACC_STRIDE 4096
#define SMEM_TOTAL 5248
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_kimi_k3_mla_fp8_paged_attention_e753f777f648af86ff0b(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, __nv_bfloat16* __restrict__ O, int* __restrict__ cum_seq_lens_q, int batch, int num_heads, int num_split, float bmm2_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_w = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_w_addr = smem + 0;
    float* smem_red = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_red_addr = smem + 1024;
    float* smem_acc = reinterpret_cast<float*>(smem_raw + 1088);
    const int smem_acc_addr = smem + 1088;

    // === Task calls (dependency order) ===
    int row = blockIdx.x;
    int chunk = blockIdx.y;
    int rows_total = cum_seq_lens_q[batch] * num_heads;
    if (row < rows_total) {
        int stat_base = row * num_split;
        int last_split = num_split - 1;
        int s_idx = tid;
        int s_ld = ((s_idx > last_split) ? last_split : s_idx);
        float m_raw = partial_max[stat_base + s_ld];
        float sum_raw = partial_sum[stat_base + s_ld];
        float m_s = ((s_idx > last_split) ? -CAKE_INF : m_raw);
        float _warp_reduce_0 = m_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float m_warp = _warp_reduce_0;
        if (lane == 0) {
            smem_red[warp] = m_warp;
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        float max_m = smem_red[0];
        #pragma unroll
        for (int w = 1; w < 8; w++) {
            float m_w = smem_red[w];
            float _max_0 = max_noftz(max_m, m_w);
            max_m = _max_0;
        }
        float w_s = 0.0f;
        if (m_s > -CAKE_INF) {
            float _exp2_0 = approx_exp2(m_s - max_m);
            w_s = _exp2_0 * sum_raw;
        }
        smem_w[s_idx] = w_s;
        float _warp_reduce_1 = w_s;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        float sum_warp = _warp_reduce_1;
        if (lane == 0) {
            smem_red[8 + warp] = sum_warp;
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        float sum_w = 0.0f;
        #pragma unroll
        for (int w_1 = 0; w_1 < 8; w_1++) {
            float s_w = smem_red[8 + w_1];
            sum_w = sum_w + s_w;
        }
        float inv_sum = 0.0f;
        if (sum_w > 0.0f) {
            float _rcp_0 = approx_rcp(sum_w);
            inv_sum = _rcp_0 * bmm2_scale;
        }
        int half = lane >> 4;
        int d0 = chunk * 128 + (lane & 15) * 8;
        float acc[8];
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            acc[e] = 0.0f;
        }
        int n_iter_w = (num_split + 16 - 1) / 16;
        #pragma unroll 8
        for (int k = 0; k < n_iter_w; k++) {
            int s_raw = (k * 8 + warp) * 2 + half;
            int s_w_idx = ((s_raw > last_split) ? last_split : s_raw);
            float w_raw = smem_w[s_w_idx];
            float w_k = ((s_raw > last_split) ? 0.0f : w_raw);
            float _vec_load_0[8];
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_O + (stat_base + s_w_idx) * 512 + d0);
                uint4 _vld_0[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_0[_blk] = _vptr_0[_blk];
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_0[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int e_1 = 0; e_1 < 8; e_1++) {
                float contrib = w_k * _vec_load_0[e_1];
                float safe = ((w_k > 0.0f) ? contrib : 0.0f);
                acc[e_1] = acc[e_1] + safe;
            }
        }
        #pragma unroll
        for (int e_2 = 0; e_2 < 8; e_2++) {
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, acc[e_2], 16);
            float other = _shfl_xor_0;
            acc[e_2] = acc[e_2] + other;
        }
        if (lane < 16) {
            #pragma unroll
            for (int e_3 = 0; e_3 < 8; e_3++) {
                smem_acc[warp * 128 + lane * 8 + e_3] = acc[e_3];
            }
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        if (warp == 0) {
            int d_out = chunk * 128 + lane * 4;
            float out[4];
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                float tot = 0.0f;
                #pragma unroll
                for (int w_2 = 0; w_2 < 8; w_2++) {
                    float a_w = smem_acc[w_2 * 128 + lane * 4 + e_4];
                    tot = tot + a_w;
                }
                out[e_4] = tot * inv_sum;
            }
            #pragma unroll
            for (int e_5 = 0; e_5 < 4; e_5++) {
                *(reinterpret_cast<__nv_bfloat16*>(O + (row * 512 + d_out + e_5)) + (0)) = __float2bfloat16_rn(out[e_5]);
            }
        }
    }
}

} // extern "C"
