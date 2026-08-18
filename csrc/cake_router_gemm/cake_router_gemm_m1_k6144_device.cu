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

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_RED_OFF 0
#define SMEM_RED_STAGE_BYTES 16
#define SMEM_RED_STRIDE 16
#define SMEM_TOTAL 128
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_cake_blackwell_router_gemm_m1_k6144(__nv_bfloat16* __restrict__ mat_a, __nv_bfloat16* __restrict__ mat_b, float* __restrict__ out_f32, __nv_bfloat16* __restrict__ out_bf16, int num_experts, int out_is_bf16)
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
    float* red = reinterpret_cast<float*>(smem_raw + 0);
    const int red_addr = smem + 0;

    // === Task calls (dependency order) ===
    int expert = blockIdx.x;
    float acc[1];
    #pragma unroll
    for (int m = 0; m < 1; m++) {
        acc[m] = 0.0f;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    #pragma unroll
    for (int ki = 0; ki < 6; ki++) {
        int k_base = ki * 1024 + tid * 8;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(mat_b + (expert * 6144 + k_base) + 0);
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
        for (int m_1 = 0; m_1 < 1; m_1++) {
            float _vec_load_1[8];
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(mat_a + (m_1 * 6144 + k_base) + 0);
                uint4 _vld_1[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_1[_blk] = _vptr_1[_blk];
                    uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_1[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                float _fma_0 = __fmaf_rn(_vec_load_1[j], _vec_load_0[j], acc[m_1]);
                acc[m_1] = _fma_0;
            }
        }
    }
    #pragma unroll
    for (int m_2 = 0; m_2 < 1; m_2++) {
        float _warp_reduce_0 = acc[m_2];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        float warp_sum = _warp_reduce_0;
        if (lane == 0) {
            red[m_2 * 4 + warp] = warp_sum;
        }
    }
    asm volatile("barrier.sync 2, 128;" ::: "memory");
    if (warp == 0) {
        if (lane < 1) {
            int m_3 = lane;
            float total = red[m_3 * 4];
            #pragma unroll
            for (int source_warp = 1; source_warp < 4; source_warp++) {
                total = total + red[m_3 * 4 + source_warp];
            }
            int offset = m_3 * num_experts + expert;
            if (out_is_bf16 == 0) {
                *(reinterpret_cast<float*>(out_f32 + offset) + (0)) = total;
            } else {
                *(reinterpret_cast<__nv_bfloat16*>(out_bf16 + offset) + (0)) = __float2bfloat16_rn(total);
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
