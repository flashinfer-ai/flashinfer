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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 3584
#define SMEM_PARTIALS_STRIDE 3584
#define SMEM_TOTAL 3584
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_b65033ed0546ce6d229f(__nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ w_slice, float* __restrict__ out, int num_tokens)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (dependency order) ===
    int row0 = blockIdx.x * 4;
    int k0 = tid * 8;
    unsigned int wv[16];
    for (int r = 0; r < 4; r++) {
        unsigned int _vec_load_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(w_slice) + (((row0 + r) * 3584 + k0) / 2) + 0);
            _vec_load_0[0 + 0] = _uv4_0.x;
            _vec_load_0[0 + 1] = _uv4_0.y;
            _vec_load_0[0 + 2] = _uv4_0.z;
            _vec_load_0[0 + 3] = _uv4_0.w;
        }
        for (int q = 0; q < 4; q++) {
            wv[r * 4 + q] = _vec_load_0[q];
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float acc[64];
    for (int i = 0; i < 64; i++) {
        acc[i] = 0.0f;
    }
    for (int m = 0; m < 16; m++) {
        if (m < num_tokens) {
            unsigned int _vec_load_1[4];
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(y) + ((m * 3584 + k0) / 2) + 0);
                _vec_load_1[0 + 0] = _uv4_1.x;
                _vec_load_1[0 + 1] = _uv4_1.y;
                _vec_load_1[0 + 2] = _uv4_1.z;
                _vec_load_1[0 + 3] = _uv4_1.w;
            }
            for (int r_1 = 0; r_1 < 4; r_1++) {
                float _bf16x2_dot_f32_0;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_0) : "r"(wv[r_1 * 4]), "r"(_vec_load_1[0]), "f"(0.0f));
                float _bf16x2_dot_f32_1;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_1) : "r"(wv[r_1 * 4 + 1]), "r"(_vec_load_1[1]), "f"(_bf16x2_dot_f32_0));
                float _bf16x2_dot_f32_2;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_2) : "r"(wv[r_1 * 4 + 2]), "r"(_vec_load_1[2]), "f"(_bf16x2_dot_f32_1));
                float _bf16x2_dot_f32_3;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_3) : "r"(wv[r_1 * 4 + 3]), "r"(_vec_load_1[3]), "f"(_bf16x2_dot_f32_2));
                acc[r_1 * 16 + m] = _bf16x2_dot_f32_3;
            }
        }
    }
    for (int m_1 = 0; m_1 < 16; m_1++) {
        if (m_1 < num_tokens) {
            for (int r_2 = 0; r_2 < 4; r_2++) {
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 16 + m_1], 16);
                acc[r_2 * 16 + m_1] = acc[r_2 * 16 + m_1] + _shfl_xor_0;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 16 + m_1], 8);
                acc[r_2 * 16 + m_1] = acc[r_2 * 16 + m_1] + _shfl_xor_1;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 16 + m_1], 4);
                acc[r_2 * 16 + m_1] = acc[r_2 * 16 + m_1] + _shfl_xor_2;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 16 + m_1], 2);
                acc[r_2 * 16 + m_1] = acc[r_2 * 16 + m_1] + _shfl_xor_3;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 16 + m_1], 1);
                acc[r_2 * 16 + m_1] = acc[r_2 * 16 + m_1] + _shfl_xor_4;
            }
            if (lane == 0) {
                for (int r_3 = 0; r_3 < 4; r_3++) {
                    partials[warp * 64 + r_3 * 16 + m_1] = acc[r_3 * 16 + m_1];
                }
            }
        }
    }
    asm volatile("barrier.sync 1, 448;" ::: "memory");
    if (tid < 64) {
        int r_out = tid / 16;
        int m_out = tid % 16;
        if (m_out < num_tokens) {
            float total = 0.0f;
            for (int wgt = 0; wgt < 14; wgt++) {
                total = total + partials[wgt * 64 + tid];
            }
            *(reinterpret_cast<float*>(out + (m_out * 512 + row0 + r_out)) + (0)) = total;
        }
    }
}

} // extern "C"
