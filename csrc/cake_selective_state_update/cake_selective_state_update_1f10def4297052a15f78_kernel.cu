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
#define SMEM_S_B_OFF 0
#define SMEM_S_B_STAGE_BYTES 2048
#define SMEM_S_B_STRIDE 2048
#define SMEM_S_C_OFF 2048
#define SMEM_S_C_STAGE_BYTES 2048
#define SMEM_S_C_STRIDE 2048
#define SMEM_S_X_OFF 4096
#define SMEM_S_X_STAGE_BYTES 128
#define SMEM_S_X_STRIDE 128
#define SMEM_S_DT_OFF 4352
#define SMEM_S_DT_STAGE_BYTES 32
#define SMEM_S_DT_STRIDE 32
#define SMEM_S_DST_OFF 4384
#define SMEM_S_DST_STAGE_BYTES 64
#define SMEM_S_DST_STRIDE 64
#define SMEM_S_STATE_OFF 4480
#define SMEM_S_STATE_STAGE_BYTES 4096
#define SMEM_S_STATE_STRIDE 4096
#define SMEM_TOTAL 12672
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 8

#include <math_constants.h>


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_selective_state_update_1f10def4297052a15f78(float* __restrict__ state, __nv_bfloat16* __restrict__ x, float* __restrict__ dt, float* __restrict__ A, __nv_bfloat16* __restrict__ B, __nv_bfloat16* __restrict__ C, float* __restrict__ D, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ output, long long* __restrict__ state_batch_indices, long long* __restrict__ dst_state_batch_indices, int batch_size, int token_steps, unsigned long long state_stride_slot, long long pad_slot_id)
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
    __nv_bfloat16* s_B = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_B_addr = smem + 0;
    __nv_bfloat16* s_C = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int s_C_addr = smem + 2048;
    __nv_bfloat16* s_x = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int s_x_addr = smem + 4096;
    float* s_dt = reinterpret_cast<float*>(smem_raw + 4352);
    const int s_dt_addr = smem + 4352;
    long long* s_dst = reinterpret_cast<long long*>(smem_raw + 4384);
    const int s_dst_addr = smem + 4384;
    float* s_state = reinterpret_cast<float*>(smem_raw + 4480);
    const int s_state_addr = smem + 4480;

    // === Task calls (declaration order) ===
    int batch = blockIdx.x;
    int head = blockIdx.y;
    int dim_tile = blockIdx.z;
    int dim_base = dim_tile * 8;
    int token_base = batch * token_steps;
    long long source_slot = state_batch_indices[batch];
    if (warp == 0) {
        #pragma unroll
        for (int pack_turn = 0; pack_turn < 4; pack_turn++) {
            int pack = lane + pack_turn * 32;
            int step = pack / 16;
            int col = pack % 16 * 8;
            int source_step = step;
            if (source_step >= token_steps) {
                source_step = 0;
            }
            int src_index = (token_base + source_step) * 128 + col;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_B_addr + (unsigned int)((step * 128 + col) * 2)), "l"(B + src_index));
        }
    }
    if (warp == 1) {
        #pragma unroll
        for (int pack_turn_1 = 0; pack_turn_1 < 4; pack_turn_1++) {
            int pack_1 = lane + pack_turn_1 * 32;
            int step_1 = pack_1 / 16;
            int col_1 = pack_1 % 16 * 8;
            int source_step_1 = step_1;
            if (source_step_1 >= token_steps) {
                source_step_1 = 0;
            }
            int src_index_1 = (token_base + source_step_1) * 128 + col_1;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_C_addr + (unsigned int)((step_1 * 128 + col_1) * 2)), "l"(C + src_index_1));
        }
    }
    #pragma unroll
    for (int token_turn = 0; token_turn < 2; token_turn++) {
        int step_2 = warp + token_turn * 4;
        int source_step_2 = step_2;
        if (source_step_2 >= token_steps) {
            source_step_2 = 0;
        }
        if (lane < 1) {
            int col_2 = lane * 8;
            int src_index_2 = ((token_base + source_step_2) * 16 + head) * 64 + dim_base + col_2;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(s_x_addr + (unsigned int)((step_2 * 8 + col_2) * 2)), "l"(x + src_index_2));
        }
    }
    float direct_values[8];
    int direct_member = lane % 16;
    int direct_row = warp * 2 + lane / 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        direct_values[i] = 0.0f;
    }
    if (source_slot != pad_slot_id) {
        #pragma unroll
        for (int tile = 0; tile < 2; tile++) {
            int direct_col = tile * 64 + direct_member * 4;
            unsigned long long direct_index = (unsigned long long)source_slot * state_stride_slot + (unsigned long long)((head * 64 + dim_base + direct_row) * 128 + direct_col);
            {
                float4 _v4 = *reinterpret_cast<const float4*>(state + direct_index);
                direct_values[tile * 4 + 0] = _v4.x;
                direct_values[tile * 4 + 1] = _v4.y;
                direct_values[tile * 4 + 2] = _v4.z;
                direct_values[tile * 4 + 3] = _v4.w;
            }
        }
    }
    if (tid < token_steps) {
        int step_3 = tid;
        float dt_value = dt[(token_base + step_3) * 16 + head];
        dt_value += dt_bias[head];
        float _exp_2 = expf(fminf(dt_value, 20.0f));
        float _log1p_1 = log1pf(_exp_2);
        s_dt[step_3] = fmaxf(_log1p_1, dt_value);
    }
    if (tid < token_steps) {
        s_dst[tid] = dst_state_batch_indices[batch * token_steps + tid];
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int member = lane % 16;
    int subgroup = lane / 16;
    int local_row = warp * 2 + subgroup;
    int dim_index = dim_base + local_row;
    float state_values[8];
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        state_values[i_1] = direct_values[i_1];
    }
    float a_value = A[head];
    float d_value = D[head];
    float pending_partial = 0.0f;
    float pending_x = 0.0f;
    #pragma unroll
    for (int step_4 = 0; step_4 < 8; step_4++) {
        if (step_4 < token_steps) {
            float dt_value_1 = s_dt[step_4];
            float decay = expf(a_value * dt_value_1);
            float x_value = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_x) + ((step_4 * 8 + local_row) * 2))[0];
            float dtx = dt_value_1 * x_value;
            float partial = 0.0f;
            #pragma unroll
            for (int tile_1 = 0; tile_1 < 2; tile_1++) {
                int state_col = tile_1 * 64 + member * 4;
                float b_quad[4];
                float c_quad[4];
                #pragma unroll
                for (int element = 0; element < 4; element++) {
                    b_quad[element] = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_B) + ((step_4 * 128 + state_col + element) * 2))[0];
                    c_quad[element] = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_C) + ((step_4 * 128 + state_col + element) * 2))[0];
                }
                #pragma unroll
                for (int element_1 = 0; element_1 < 4; element_1++) {
                    float _add_3;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(mul_f32_rn_noftz(state_values[tile_1 * 4 + element_1], decay)), "f"(mul_f32_rn_noftz(b_quad[element_1], dtx)));
                    state_values[tile_1 * 4 + element_1] = _add_3;
                    float _add_4;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(partial), "f"(mul_f32_rn_noftz(state_values[tile_1 * 4 + element_1], c_quad[element_1])));
                    partial = _add_4;
                }
            }
            if (step_4 > 0) {
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 8);
                pending_partial += _shfl_xor_sync_5;
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 4);
                pending_partial += _shfl_xor_sync_6;
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 2);
                pending_partial += _shfl_xor_sync_7;
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 1);
                pending_partial += _shfl_xor_sync_8;
                if (member == 0) {
                    int previous_index = ((token_base + step_4 - 1) * 16 + head) * 64 + dim_index;
                    float _add_9;
                    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(pending_partial), "f"(mul_f32_rn_noftz(d_value, pending_x)));
                    output[previous_index] = _add_9;
                }
            }
            pending_partial = partial;
            pending_x = x_value;
            long long destination_slot = s_dst[step_4];
            if (source_slot != pad_slot_id) {
                if (destination_slot != pad_slot_id) {
                    #pragma unroll
                    for (int tile_2 = 0; tile_2 < 2; tile_2++) {
                        int state_col_1 = tile_2 * 64 + member * 4;
                        int destination_row_i32 = (head * 64 + dim_index) * 128 + state_col_1;
                        unsigned long long destination_index = (unsigned long long)destination_slot * state_stride_slot + (unsigned long long)destination_row_i32;
                        {
                            float4 _v4 = make_float4(state_values[tile_2 * 4 + 0], state_values[tile_2 * 4 + 1], state_values[tile_2 * 4 + 2], state_values[tile_2 * 4 + 3]);
                            *reinterpret_cast<float4*>(state + destination_index) = _v4;
                        }
                    }
                }
            }
        }
    }
    if (token_steps > 0) {
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 8);
        pending_partial += _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 4);
        pending_partial += _shfl_xor_sync_11;
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 2);
        pending_partial += _shfl_xor_sync_12;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, pending_partial, 1);
        pending_partial += _shfl_xor_sync_13;
        if (member == 0) {
            int last_index = ((token_base + token_steps - 1) * 16 + head) * 64 + dim_index;
            float _add_14;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(pending_partial), "f"(mul_f32_rn_noftz(d_value, pending_x)));
            output[last_index] = _add_14;
        }
    }
}

} // extern "C"
