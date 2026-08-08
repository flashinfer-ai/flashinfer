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
// Frozen from Cake commit f120b38798f4481a116127e7cdecbe4ed9d2cf5f.
// Measured kernel commit: ca826e6b97b2cd696e208c842eb4008af2328d65.
// Weave source sha256:ce4bf0aba8a398979b70df863658baa9b3700fa840b4be957997fd0bee64cf0b.
// Original generated payload before FlashInfer type adaptation: 7194 bytes,
// sha256:8ca2513545faf9cb960c6393fb48f4c9f4598a781ad9c929e8d88ebda738a89b.
// This integration uses standard integer types and adds sm_103a-only PDL
// wait/trigger control. The raw payload hash above remains the export identity.
// Specialization: sm_103a, vocab=32000, t512, vec4, TEMP_KIND=0,
// MATERIALIZE_EXP=0, grid=(rows, 1, 1), dynamic shared memory=128 bytes.

// clang-format off
#include <stdint.h>
#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 128
#define SMEM_REDUCE_SMEM_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 512
#define VOCAB_SIZE 32000
#define VECTORS_PER_THREAD 16
#define TEMP_KIND 0
#define MATERIALIZE_EXP 0

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_mr474_manual_softmax_exp2_t512_vec4(float* __restrict__ x, float* __restrict__ temperature, float* __restrict__ y, float temperature_scalar)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    // The integration passes nullptr only to encode PDL for this frozen
    // TEMP_KIND==0 specialization; the temperature slot is never dereferenced.
    const bool enable_pdl = temperature == nullptr;
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* reduce_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_smem_addr = smem + 0;

    // === Task calls (dependency order) ===
    float inv_temperature = 1.0f;
    {
    }
    float local_max = -LOOM_INF;
    #pragma unroll
    for (int tile = 0; tile < VECTORS_PER_THREAD; tile++) {
        if ((tile * 512 + tid) * 4 < VOCAB_SIZE) {
            float _vec_load_0[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x + ((unsigned long long)bid * (unsigned long long)VOCAB_SIZE + (unsigned long long)((tile * 512 + tid) * 4)) + 0);
                _vec_load_0[0 + 0] = _v4.x;
                _vec_load_0[0 + 1] = _v4.y;
                _vec_load_0[0 + 2] = _v4.z;
                _vec_load_0[0 + 3] = _v4.w;
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float value = _vec_load_0[j];
                float _max_0 = max_noftz(local_max, value);
                local_max = _max_0;
            }
        }
    }
    float _warp_reduce_0 = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
    local_max = _warp_reduce_0;
    if (lane == 0) {
        reduce_smem[warp] = local_max;
    }
    __syncthreads();
    float block_max = ((lane < 16) ? reduce_smem[lane] : -LOOM_INF);
    float _warp_reduce_1 = block_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 = max_noftz(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
    block_max = _warp_reduce_1;
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            reduce_smem[0] = block_max;
        }
    }
    __syncthreads();
    float row_max = reduce_smem[0];
    __syncthreads();
    float local_sum = 0.0f;
    #pragma unroll
    for (int tile_1 = 0; tile_1 < VECTORS_PER_THREAD; tile_1++) {
        if ((tile_1 * 512 + tid) * 4 < VOCAB_SIZE) {
            float _vec_load_1[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(x + ((unsigned long long)bid * (unsigned long long)VOCAB_SIZE + (unsigned long long)((tile_1 * 512 + tid) * 4)) + 0);
                _vec_load_1[0 + 0] = _v4.x;
                _vec_load_1[0 + 1] = _v4.y;
                _vec_load_1[0 + 2] = _v4.z;
                _vec_load_1[0 + 3] = _v4.w;
            }
            float exp_values[4];
            #pragma unroll
            for (int j_1 = 0; j_1 < 4; j_1++) {
                float value_1 = _vec_load_1[j_1];
                float _exp2_0 = approx_exp2((value_1 - row_max) * 1.4426950408889634f);
                float exp_value = ((row_max == -LOOM_INF) ? 0.0f : _exp2_0);
                exp_values[j_1] = exp_value;
                local_sum += exp_value;
            }
        }
    }
    float _warp_reduce_2 = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_2 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_2, offset);
    local_sum = _warp_reduce_2;
    if (lane == 0) {
        reduce_smem[warp] = local_sum;
    }
    __syncthreads();
    float block_sum = ((lane < 16) ? reduce_smem[lane] : 0.0f);
    float _warp_reduce_3 = block_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_3 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_3, offset);
    block_sum = _warp_reduce_3;
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            reduce_smem[0] = block_sum;
        }
    }
    __syncthreads();
    float inv_sum = ((reduce_smem[0] == 0.0f) ? 0.0f : 1.0f / reduce_smem[0]);
    #pragma unroll
    for (int tile_2 = 0; tile_2 < VECTORS_PER_THREAD; tile_2++) {
        if ((tile_2 * 512 + tid) * 4 < VOCAB_SIZE) {
            float out_values[4];
            {
                float _vec_load_3[4];
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(x + ((unsigned long long)bid * (unsigned long long)VOCAB_SIZE + (unsigned long long)((tile_2 * 512 + tid) * 4)) + 0);
                    _vec_load_3[0 + 0] = _v4.x;
                    _vec_load_3[0 + 1] = _v4.y;
                    _vec_load_3[0 + 2] = _v4.z;
                    _vec_load_3[0 + 3] = _v4.w;
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 4; j_2++) {
                    float value_2 = _vec_load_3[j_2];
                    float _exp2_1 = approx_exp2((value_2 - row_max) * 1.4426950408889634f);
                    out_values[j_2] = _exp2_1 * inv_sum;
                }
            }
            {
                float4 _v4 = make_float4(out_values[0 + 0], out_values[0 + 1], out_values[0 + 2], out_values[0 + 3]);
                *reinterpret_cast<float4*>(y + ((unsigned long long)bid * (unsigned long long)VOCAB_SIZE + (unsigned long long)((tile_2 * 512 + tid) * 4)) + 0) = _v4;
            }
        }
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    if (enable_pdl) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

} // extern "C"

// clang-format on
