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

#define CAKE_INF CUDART_INF_F
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 132
#define SMEM_REDUCE_SMEM_STRIDE 132
#define SMEM_RMS_SCALAR_OFF 132
#define SMEM_RMS_SCALAR_STAGE_BYTES 4
#define SMEM_RMS_SCALAR_STRIDE 4
#define SMEM_RMS_SLOTS_OFF 136
#define SMEM_RMS_SLOTS_STAGE_BYTES 32
#define SMEM_RMS_SLOTS_STRIDE 32
#define SMEM_TOTAL 256
#define THREADS 224

#include <math_constants.h>

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


namespace flashinfer_cake_moe_finalize_allreduce_fusion {
__device__ inline unsigned int laneId()
{
    uint32_t id;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}
} // namespace flashinfer_cake_moe_finalize_allreduce_fusion

extern "C" {

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws2_o110_small(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = flashinfer_cake_moe_finalize_allreduce_fusion::laneId();

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 168;
    #define rms_bar_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* reduce_smem = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_smem_addr = smem + 0;
    float* rms_scalar = reinterpret_cast<float*>(smem_raw + 132);
    const int rms_scalar_addr = smem + 132;
    float* rms_slots = reinterpret_cast<float*>(smem_raw + 136);
    const int rms_slots_addr = smem + 136;

    // === Task calls (declaration order) ===
    int rank = world_rank;
    long long control_address = workspace_tensor[6];
    int* control_0 = reinterpret_cast<int*>(control_address);
    int control_bytes_per_buffer_word = control_0[3];
    long long workspace_address = workspace_tensor[4 + rank];
    __half* workspace_local = reinterpret_cast<__half*>(workspace_address);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int first_token = cluster_id;
    int first_thread = cta_rank * 224 + tid;
    int first_index[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    unsigned int first_shared[4] = {0};
    if (first_token < tokens) {
        if (has_shared_expert != 0) {
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(shared_expert_output + (first_token * 7168 + first_thread * 8) + 0);
                first_shared[0 + 0] = _uv4_0.x;
                first_shared[0 + 1] = _uv4_0.y;
                first_shared[0 + 2] = _uv4_0.z;
                first_shared[0 + 3] = _uv4_0.w;
            }
        }
        if (top_k > 0) {
            int _vec_load_12[1];
            {
                _vec_load_12[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k));
            }
            first_index[0] = _vec_load_12[0];
        }
        if (top_k > 1) {
            int _vec_load_13[1];
            {
                _vec_load_13[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 1));
            }
            first_index[1] = _vec_load_13[0];
        }
        if (top_k > 2) {
            int _vec_load_14[1];
            {
                _vec_load_14[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 2));
            }
            first_index[2] = _vec_load_14[0];
        }
        if (top_k > 3) {
            int _vec_load_15[1];
            {
                _vec_load_15[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 3));
            }
            first_index[3] = _vec_load_15[0];
        }
        if (top_k > 4) {
            int _vec_load_16[1];
            {
                _vec_load_16[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 4));
            }
            first_index[4] = _vec_load_16[0];
        }
        if (top_k > 5) {
            int _vec_load_19[1];
            {
                _vec_load_19[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 5));
            }
            first_index[5] = _vec_load_19[0];
        }
        if (top_k > 6) {
            int _vec_load_20[1];
            {
                _vec_load_20[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 6));
            }
            first_index[6] = _vec_load_20[0];
        }
        if (top_k > 7) {
            int _vec_load_21[1];
            {
                _vec_load_21[0] = *reinterpret_cast<const int*>(inverse_indices + (first_token * top_k + 7));
            }
            first_index[7] = _vec_load_21[0];
        }
    }
    asm volatile("" :: "r"((uint32_t)(first_index[0])), "r"((uint32_t)(first_index[1])), "r"((uint32_t)(first_index[2])), "r"((uint32_t)(first_index[3])), "r"((uint32_t)(first_index[4])), "r"((uint32_t)(first_index[5])), "r"((uint32_t)(first_index[6])), "r"((uint32_t)(first_index[7])), "r"((uint32_t)(first_shared[0])), "r"((uint32_t)(first_shared[1])), "r"((uint32_t)(first_shared[2])), "r"((uint32_t)(first_shared[3])));
    int control_current_word = control_0[2];
    int control_clear_word = control_0[4];
    int control_current = control_current_word % 3;
    __half* peer0 = reinterpret_cast<__half*>(workspace_tensor[4]);
    __half* peer1 = reinterpret_cast<__half*>(workspace_tensor[5]);
    __syncthreads();
    if (tid == 0) {
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(&reinterpret_cast<unsigned int*>(control_0)[0], 1);
    }
    if (tid == 0) {
        mbarrier_init(rms_bar_addr, 4);
        mbarrier_init(rms_bar_addr + 8, 4);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    int cluster_thread = cta_rank * 224 + tid;
    int token_stride = num_clusters;
    int access_stride = token_stride * 896;
    int first_access = cluster_id * 896 + (unsigned int)cluster_thread;
    int token_begin = cluster_id;
    int total_access = tokens * 896;
    int token_end = tokens;
    int route_end = top_k;
    #pragma unroll 1
    for (int token = token_begin; token < token_end; token += token_stride) {
        float acc[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[j] = 0.0f;
        }
        float shared_early[8];
        if (has_shared_expert != 0) {
            int shared_early_elem = token * 7168 + cluster_thread * 8;
            float _vec_load_0[8];
            unsigned int shared_words[4];
            if (token == token_begin) {
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    shared_words[w] = first_shared[w];
                }
            } else {
                {
                    uint4 _uv4_1 = *reinterpret_cast<const uint4*>(shared_expert_output + shared_early_elem + 0);
                    shared_words[0 + 0] = _uv4_1.x;
                    shared_words[0 + 1] = _uv4_1.y;
                    shared_words[0 + 2] = _uv4_1.z;
                    shared_words[0 + 3] = _uv4_1.w;
                }
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_0[_pair * 2]))
                    : "r"(shared_words[_pair]));
            }
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                shared_early[j_1] = _vec_load_0[j_1];
            }
        }
        int oct_index[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        if (top_k > 0) {
            int _vec_load_22[1];
            if (token == token_begin) {
                _vec_load_22[0] = first_index[0];
            } else {
                {
                    _vec_load_22[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k));
                }
            }
            oct_index[0] = _vec_load_22[0];
        }
        if (top_k > 1) {
            int _vec_load_23[1];
            if (token == token_begin) {
                _vec_load_23[0] = first_index[1];
            } else {
                {
                    _vec_load_23[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 1));
                }
            }
            oct_index[1] = _vec_load_23[0];
        }
        if (top_k > 2) {
            int _vec_load_24[1];
            if (token == token_begin) {
                _vec_load_24[0] = first_index[2];
            } else {
                {
                    _vec_load_24[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 2));
                }
            }
            oct_index[2] = _vec_load_24[0];
        }
        if (top_k > 3) {
            int _vec_load_25[1];
            if (token == token_begin) {
                _vec_load_25[0] = first_index[3];
            } else {
                {
                    _vec_load_25[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 3));
                }
            }
            oct_index[3] = _vec_load_25[0];
        }
        if (top_k > 4) {
            int _vec_load_26[1];
            if (token == token_begin) {
                _vec_load_26[0] = first_index[4];
            } else {
                {
                    _vec_load_26[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 4));
                }
            }
            oct_index[4] = _vec_load_26[0];
        }
        if (top_k > 5) {
            int _vec_load_27[1];
            if (token == token_begin) {
                _vec_load_27[0] = first_index[5];
            } else {
                {
                    _vec_load_27[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 5));
                }
            }
            oct_index[5] = _vec_load_27[0];
        }
        if (top_k > 6) {
            int _vec_load_28[1];
            if (token == token_begin) {
                _vec_load_28[0] = first_index[6];
            } else {
                {
                    _vec_load_28[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 6));
                }
            }
            oct_index[6] = _vec_load_28[0];
        }
        if (top_k > 7) {
            int _vec_load_29[1];
            if (token == token_begin) {
                _vec_load_29[0] = first_index[7];
            } else {
                {
                    _vec_load_29[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 7));
                }
            }
            oct_index[7] = _vec_load_29[0];
        }
        #pragma unroll 1
        for (int route_base = 0; route_base < route_end; route_base += 8) {
            int next_oct_index[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
            unsigned int oct_words[32];
            float oct_scales[8];
            if (route_base + 8 < top_k) {
                int _vec_load_30[1];
                {
                    _vec_load_30[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8));
                }
                next_oct_index[0] = _vec_load_30[0];
            }
            if (route_base + 8 + 1 < top_k) {
                int _vec_load_31[1];
                {
                    _vec_load_31[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 1));
                }
                next_oct_index[1] = _vec_load_31[0];
            }
            if (route_base + 8 + 2 < top_k) {
                int _vec_load_32[1];
                {
                    _vec_load_32[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 2));
                }
                next_oct_index[2] = _vec_load_32[0];
            }
            if (route_base + 8 + 3 < top_k) {
                int _vec_load_35[1];
                {
                    _vec_load_35[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 3));
                }
                next_oct_index[3] = _vec_load_35[0];
            }
            if (route_base + 8 + 4 < top_k) {
                int _vec_load_36[1];
                {
                    _vec_load_36[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 4));
                }
                next_oct_index[4] = _vec_load_36[0];
            }
            if (route_base + 8 + 5 < top_k) {
                int _vec_load_37[1];
                {
                    _vec_load_37[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 5));
                }
                next_oct_index[5] = _vec_load_37[0];
            }
            if (route_base + 8 + 6 < top_k) {
                int _vec_load_38[1];
                {
                    _vec_load_38[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 6));
                }
                next_oct_index[6] = _vec_load_38[0];
            }
            if (route_base + 8 + 7 < top_k) {
                int _vec_load_39[1];
                {
                    _vec_load_39[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 8 + 7));
                }
                next_oct_index[7] = _vec_load_39[0];
            }
            if (oct_index[0] >= 0) {
                long long expert_elem = (long long)oct_index[0] * 7168 + (long long)(cluster_thread * 8);
                unsigned int _vec_load_17[4];
                {
                    uint4 _uv4_2 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem + 0);
                    _vec_load_17[0 + 0] = _uv4_2.x;
                    _vec_load_17[0 + 1] = _uv4_2.y;
                    _vec_load_17[0 + 2] = _uv4_2.z;
                    _vec_load_17[0 + 3] = _uv4_2.w;
                }
                #pragma unroll
                for (int w_1 = 0; w_1 < 4; w_1++) {
                    oct_words[w_1] = _vec_load_17[w_1];
                }
                float _vec_load_18[1];
                {
                    __half _f16_3 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base);
                    _vec_load_18[0] = __half2float(_f16_3);
                }
                oct_scales[0] = _vec_load_18[0];
            }
            if (oct_index[1] >= 0) {
                long long expert_elem_0 = (long long)oct_index[1] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_0[4];
                {
                    uint4 _uv4_4 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_0 + 0);
                    vec_load_0[0 + 0] = _uv4_4.x;
                    vec_load_0[0 + 1] = _uv4_4.y;
                    vec_load_0[0 + 2] = _uv4_4.z;
                    vec_load_0[0 + 3] = _uv4_4.w;
                }
                #pragma unroll
                for (int w_0 = 0; w_0 < 4; w_0++) {
                    oct_words[4 + w_0] = vec_load_0[w_0];
                }
                float vec_load_r_0[1];
                {
                    __half _f16_5 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 1);
                    vec_load_r_0[0] = __half2float(_f16_5);
                }
                oct_scales[1] = vec_load_r_0[0];
            }
            if (oct_index[2] >= 0) {
                long long expert_elem_1 = (long long)oct_index[2] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_1[4];
                {
                    uint4 _uv4_6 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_1 + 0);
                    vec_load_1[0 + 0] = _uv4_6.x;
                    vec_load_1[0 + 1] = _uv4_6.y;
                    vec_load_1[0 + 2] = _uv4_6.z;
                    vec_load_1[0 + 3] = _uv4_6.w;
                }
                #pragma unroll
                for (int w_1_1 = 0; w_1_1 < 4; w_1_1++) {
                    oct_words[8 + w_1_1] = vec_load_1[w_1_1];
                }
                float vec_load_r_1[1];
                {
                    __half _f16_7 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 2);
                    vec_load_r_1[0] = __half2float(_f16_7);
                }
                oct_scales[2] = vec_load_r_1[0];
            }
            if (oct_index[3] >= 0) {
                long long expert_elem_2 = (long long)oct_index[3] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_2[4];
                {
                    uint4 _uv4_8 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_2 + 0);
                    vec_load_2[0 + 0] = _uv4_8.x;
                    vec_load_2[0 + 1] = _uv4_8.y;
                    vec_load_2[0 + 2] = _uv4_8.z;
                    vec_load_2[0 + 3] = _uv4_8.w;
                }
                #pragma unroll
                for (int w_2 = 0; w_2 < 4; w_2++) {
                    oct_words[12 + w_2] = vec_load_2[w_2];
                }
                float vec_load_r_2[1];
                {
                    __half _f16_9 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 3);
                    vec_load_r_2[0] = __half2float(_f16_9);
                }
                oct_scales[3] = vec_load_r_2[0];
            }
            if (oct_index[4] >= 0) {
                long long expert_elem_3 = (long long)oct_index[4] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_3[4];
                {
                    uint4 _uv4_10 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_3 + 0);
                    vec_load_3[0 + 0] = _uv4_10.x;
                    vec_load_3[0 + 1] = _uv4_10.y;
                    vec_load_3[0 + 2] = _uv4_10.z;
                    vec_load_3[0 + 3] = _uv4_10.w;
                }
                #pragma unroll
                for (int w_3 = 0; w_3 < 4; w_3++) {
                    oct_words[16 + w_3] = vec_load_3[w_3];
                }
                float vec_load_r_3[1];
                {
                    __half _f16_11 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 4);
                    vec_load_r_3[0] = __half2float(_f16_11);
                }
                oct_scales[4] = vec_load_r_3[0];
            }
            if (oct_index[5] >= 0) {
                long long expert_elem_4 = (long long)oct_index[5] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_4[4];
                {
                    uint4 _uv4_12 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_4 + 0);
                    vec_load_4[0 + 0] = _uv4_12.x;
                    vec_load_4[0 + 1] = _uv4_12.y;
                    vec_load_4[0 + 2] = _uv4_12.z;
                    vec_load_4[0 + 3] = _uv4_12.w;
                }
                #pragma unroll
                for (int w_4 = 0; w_4 < 4; w_4++) {
                    oct_words[20 + w_4] = vec_load_4[w_4];
                }
                float vec_load_r_4[1];
                {
                    __half _f16_13 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 5);
                    vec_load_r_4[0] = __half2float(_f16_13);
                }
                oct_scales[5] = vec_load_r_4[0];
            }
            if (oct_index[6] >= 0) {
                long long expert_elem_5 = (long long)oct_index[6] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_5[4];
                {
                    uint4 _uv4_14 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_5 + 0);
                    vec_load_5[0 + 0] = _uv4_14.x;
                    vec_load_5[0 + 1] = _uv4_14.y;
                    vec_load_5[0 + 2] = _uv4_14.z;
                    vec_load_5[0 + 3] = _uv4_14.w;
                }
                #pragma unroll
                for (int w_5 = 0; w_5 < 4; w_5++) {
                    oct_words[24 + w_5] = vec_load_5[w_5];
                }
                float vec_load_r_5[1];
                {
                    __half _f16_15 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 6);
                    vec_load_r_5[0] = __half2float(_f16_15);
                }
                oct_scales[6] = vec_load_r_5[0];
            }
            if (oct_index[7] >= 0) {
                long long expert_elem_6 = (long long)oct_index[7] * 7168 + (long long)(cluster_thread * 8);
                unsigned int vec_load_6[4];
                {
                    uint4 _uv4_16 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_6 + 0);
                    vec_load_6[0 + 0] = _uv4_16.x;
                    vec_load_6[0 + 1] = _uv4_16.y;
                    vec_load_6[0 + 2] = _uv4_16.z;
                    vec_load_6[0 + 3] = _uv4_16.w;
                }
                #pragma unroll
                for (int w_6 = 0; w_6 < 4; w_6++) {
                    oct_words[28 + w_6] = vec_load_6[w_6];
                }
                float vec_load_r_6[1];
                {
                    __half _f16_17 = *reinterpret_cast<const __half*>(expert_scales + token * top_k + route_base + 7);
                    vec_load_r_6[0] = __half2float(_f16_17);
                }
                oct_scales[7] = vec_load_r_6[0];
            }
            if (oct_index[0] >= 0) {
                float scaled[8];
                float unpack_f16[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16[_pair * 2]))
                        : "r"(oct_words[_pair]));
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    scaled[j_2] = unpack_f16[j_2] * oct_scales[0];
                }
                unsigned int scaled_f16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_f16[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_0[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_0[_pair * 2]))
                        : "r"(scaled_f16[_pair]));
                }
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    acc[j_3] = acc[j_3] + unpack_f16_0[j_3];
                }
                unsigned int scaled_f16_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_0[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_0[_pair]));
                }
            }
            if (oct_index[1] >= 0) {
                float scaled_1[8];
                float unpack_f16_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_1[_pair * 2]))
                        : "r"(oct_words[4 + _pair]));
                }
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    scaled_1[j_4] = unpack_f16_1[j_4] * oct_scales[1];
                }
                unsigned int scaled_f16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
                    scaled_f16_1[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_1_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_1_1[_pair * 2]))
                        : "r"(scaled_f16_1[_pair]));
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    acc[j_5] = acc[j_5] + unpack_f16_1_1[j_5];
                }
                unsigned int scaled_f16_1_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_1_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_1_1[_pair]));
                }
            }
            if (oct_index[2] >= 0) {
                float scaled_2[8];
                float unpack_f16_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_2[_pair * 2]))
                        : "r"(oct_words[8 + _pair]));
                }
                #pragma unroll
                for (int j_6 = 0; j_6 < 8; j_6++) {
                    scaled_2[j_6] = unpack_f16_2[j_6] * oct_scales[2];
                }
                unsigned int scaled_f16_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_f16_2[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_2_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_2_1[_pair * 2]))
                        : "r"(scaled_f16_2[_pair]));
                }
                #pragma unroll
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    acc[j_7] = acc[j_7] + unpack_f16_2_1[j_7];
                }
                unsigned int scaled_f16_2_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_2_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_2_1[_pair]));
                }
            }
            if (oct_index[3] >= 0) {
                float scaled_0[8];
                float unpack_f16_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_3[_pair * 2]))
                        : "r"(oct_words[12 + _pair]));
                }
                #pragma unroll
                for (int j_0 = 0; j_0 < 8; j_0++) {
                    scaled_0[j_0] = unpack_f16_3[j_0] * oct_scales[3];
                }
                unsigned int scaled_f16_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_0[_lp*2 + 0], scaled_0[_lp*2+1 + 0]));
                    scaled_f16_3[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_3_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_3_1[_pair * 2]))
                        : "r"(scaled_f16_3[_pair]));
                }
                #pragma unroll
                for (int j_1_1 = 0; j_1_1 < 8; j_1_1++) {
                    acc[j_1_1] = acc[j_1_1] + unpack_f16_3_1[j_1_1];
                }
                unsigned int scaled_f16_3_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_3_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_3_1[_pair]));
                }
            }
            if (oct_index[4] >= 0) {
                float scaled_1_1[8];
                float unpack_f16_4[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_4[_pair * 2]))
                        : "r"(oct_words[16 + _pair]));
                }
                #pragma unroll
                for (int j_2_1 = 0; j_2_1 < 8; j_2_1++) {
                    scaled_1_1[j_2_1] = unpack_f16_4[j_2_1] * oct_scales[4];
                }
                unsigned int scaled_f16_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_1_1[_lp*2 + 0], scaled_1_1[_lp*2+1 + 0]));
                    scaled_f16_4[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_4_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_4_1[_pair * 2]))
                        : "r"(scaled_f16_4[_pair]));
                }
                #pragma unroll
                for (int j_3_1 = 0; j_3_1 < 8; j_3_1++) {
                    acc[j_3_1] = acc[j_3_1] + unpack_f16_4_1[j_3_1];
                }
                unsigned int scaled_f16_4_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_4_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_4_1[_pair]));
                }
            }
            if (oct_index[5] >= 0) {
                float scaled_2_1[8];
                float unpack_f16_5[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_5[_pair * 2]))
                        : "r"(oct_words[20 + _pair]));
                }
                #pragma unroll
                for (int j_4_1 = 0; j_4_1 < 8; j_4_1++) {
                    scaled_2_1[j_4_1] = unpack_f16_5[j_4_1] * oct_scales[5];
                }
                unsigned int scaled_f16_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_2_1[_lp*2 + 0], scaled_2_1[_lp*2+1 + 0]));
                    scaled_f16_5[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_5_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_5_1[_pair * 2]))
                        : "r"(scaled_f16_5[_pair]));
                }
                #pragma unroll
                for (int j_5_1 = 0; j_5_1 < 8; j_5_1++) {
                    acc[j_5_1] = acc[j_5_1] + unpack_f16_5_1[j_5_1];
                }
                unsigned int scaled_f16_5_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_5_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_5_1[_pair]));
                }
            }
            if (oct_index[6] >= 0) {
                float scaled_3[8];
                float unpack_f16_6[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_6[_pair * 2]))
                        : "r"(oct_words[24 + _pair]));
                }
                #pragma unroll
                for (int j_6_1 = 0; j_6_1 < 8; j_6_1++) {
                    scaled_3[j_6_1] = unpack_f16_6[j_6_1] * oct_scales[6];
                }
                unsigned int scaled_f16_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_3[_lp*2 + 0], scaled_3[_lp*2+1 + 0]));
                    scaled_f16_6[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_6_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_6_1[_pair * 2]))
                        : "r"(scaled_f16_6[_pair]));
                }
                #pragma unroll
                for (int j_7_1 = 0; j_7_1 < 8; j_7_1++) {
                    acc[j_7_1] = acc[j_7_1] + unpack_f16_6_1[j_7_1];
                }
                unsigned int scaled_f16_6_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_6_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_6_1[_pair]));
                }
            }
            if (oct_index[7] >= 0) {
                float scaled_4[8];
                float unpack_f16_7[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_7[_pair * 2]))
                        : "r"(oct_words[28 + _pair]));
                }
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    scaled_4[j_8] = unpack_f16_7[j_8] * oct_scales[7];
                }
                unsigned int scaled_f16_7[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled_4[_lp*2 + 0], scaled_4[_lp*2+1 + 0]));
                    scaled_f16_7[_lp] = *(uint32_t*)&_h2;
                }
                float unpack_f16_7_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_7_1[_pair * 2]))
                        : "r"(scaled_f16_7[_pair]));
                }
                #pragma unroll
                for (int j_9 = 0; j_9 < 8; j_9++) {
                    acc[j_9] = acc[j_9] + unpack_f16_7_1[j_9];
                }
                unsigned int scaled_f16_7_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    scaled_f16_7_1[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                        : "r"(scaled_f16_7_1[_pair]));
                }
            }
            oct_index[0] = next_oct_index[0];
            oct_index[1] = next_oct_index[1];
            oct_index[2] = next_oct_index[2];
            oct_index[3] = next_oct_index[3];
            oct_index[4] = next_oct_index[4];
            oct_index[5] = next_oct_index[5];
            oct_index[6] = next_oct_index[6];
            oct_index[7] = next_oct_index[7];
        }
        if (routed_scaling_factor != 1.0f) {
            #pragma unroll
            for (int j_10 = 0; j_10 < 8; j_10++) {
                acc[j_10] = acc[j_10] * routed_scaling_factor;
            }
            unsigned int scaled_f16_8[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                scaled_f16_8[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                    : "r"(scaled_f16_8[_pair]));
            }
        }
        if (has_shared_expert != 0) {
            int shared_elem = token * 7168 + cluster_thread * 8;
            #pragma unroll
            for (int j_11 = 0; j_11 < 8; j_11++) {
                acc[j_11] = acc[j_11] + shared_early[j_11];
            }
            unsigned int scaled_f16_9[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                scaled_f16_9[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&acc[_pair * 2]))
                    : "r"(scaled_f16_9[_pair]));
            }
        }
        #pragma unroll
        for (int j_12 = 0; j_12 < 8; j_12++) {
            acc[j_12] = ((acc[j_12] == 0.0f) ? 0.0f : acc[j_12]);
        }
        unsigned int scaled_f16_10[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            scaled_f16_10[_lp] = *(uint32_t*)&_h2;
        }
        int publish_current_word = control_current_word;
        asm("" : "+r"(publish_current_word));
        int publish_current = publish_current_word % 3;
        int publish_bytes_word = control_bytes_per_buffer_word;
        asm("" : "+r"(publish_bytes_word));
        long long publish_stride = (long long)publish_bytes_word / 2;
        long long token_data_base = (long long)publish_current * publish_stride;
        int access = token * 896 + cluster_thread;
        long long slot = (long long)rank * (long long)total_access * 8 + (long long)access * 8;
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer0 + token_data_base + slot), "r"((scaled_f16_10)[0]), "r"((scaled_f16_10)[1]), "r"((scaled_f16_10)[2]), "r"((scaled_f16_10)[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer1 + token_data_base + slot), "r"((scaled_f16_10)[0]), "r"((scaled_f16_10)[1]), "r"((scaled_f16_10)[2]), "r"((scaled_f16_10)[3]) : "memory");
        first_index[0] = -1;
        first_index[1] = -1;
        first_index[2] = -1;
        first_index[3] = -1;
        first_index[4] = -1;
        first_index[5] = -1;
        first_index[6] = -1;
        first_index[7] = -1;
        first_shared[0] = 0;
        first_shared[1] = 0;
        first_shared[2] = 0;
        first_shared[3] = 0;
    }
    int late_current_word = control_current_word;
    asm("" : "+r"(late_current_word));
    int late_current = late_current_word % 3;
    int late_dirty = (late_current_word + 2) % 3;
    int late_bytes_word = control_bytes_per_buffer_word;
    asm("" : "+r"(late_bytes_word));
    long long comm_stride_elems = (long long)late_bytes_word / 2;
    long long data_base = (long long)late_current * comm_stride_elems;
    unsigned int clear_words[4];
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        clear_words[word] = 2147516416;
    }
    long long clear_base = (long long)late_dirty * comm_stride_elems;
    #pragma unroll 4
    for (int clear_access = first_access; clear_access < control_clear_word / 8; clear_access += access_stride) {
        {
            auto _gst_18 = workspace_local + (clear_base + (long long)clear_access * 8);
            __builtin_assume(__isGlobal(_gst_18));
            reinterpret_cast<int4*>(_gst_18)[0] = reinterpret_cast<int4*>(clear_words)[0];
        }
    }
    int access_1 = first_access;
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int rms_parity = 0;
    unsigned int _phase_rms_bar = 0;
    #pragma unroll 1
    for (int token_1 = token_begin; token_1 < token_end; token_1 += token_stride) {
        int elem_early = access_1 * 8;
        float _vec_load_33[8];
        unsigned int residual_words[4];
        {
            uint4 _uv4_19 = *reinterpret_cast<const uint4*>(residual + elem_early + 0);
            residual_words[0 + 0] = _uv4_19.x;
            residual_words[0 + 1] = _uv4_19.y;
            residual_words[0 + 2] = _uv4_19.z;
            residual_words[0 + 3] = _uv4_19.w;
        }
        unsigned int norm_weight_words[4];
        {
            uint4 _uv4_20 = *reinterpret_cast<const uint4*>(norm_weight + (cluster_thread * 8) + 0);
            norm_weight_words[0 + 0] = _uv4_20.x;
            norm_weight_words[0 + 1] = _uv4_20.y;
            norm_weight_words[0 + 2] = _uv4_20.z;
            norm_weight_words[0 + 3] = _uv4_20.w;
        }
        asm volatile("" :: "r"((uint32_t)(residual_words[0])), "r"((uint32_t)(residual_words[1])), "r"((uint32_t)(residual_words[2])), "r"((uint32_t)(residual_words[3])), "r"((uint32_t)(norm_weight_words[0])), "r"((uint32_t)(norm_weight_words[1])), "r"((uint32_t)(norm_weight_words[2])), "r"((uint32_t)(norm_weight_words[3])));
        uint32_t _sysv_poll_group_0[8];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(workspace_local + (data_base + (long long)(access_1 * 8))) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(workspace_local + (data_base + (long long)(total_access * 8) + (long long)(access_1 * 8))) : "memory");
        } while (
            (((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
            (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u)
        );
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_33[_pair * 2]))
                : "r"(residual_words[_pair]));
        }
        float unpack_f16_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_8[_pair * 2]))
                : "r"(_sysv_poll_group_0[_pair]));
        }
        float sum_value[8];
        #pragma unroll
        for (int j_13 = 0; j_13 < 8; j_13++) {
            sum_value[j_13] = unpack_f16_8[j_13];
        }
        float unpack_f16_8_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&unpack_f16_8_1[_pair * 2]))
                : "r"(_sysv_poll_group_0[4 + _pair]));
        }
        #pragma unroll
        for (int j_14 = 0; j_14 < 8; j_14++) {
            sum_value[j_14] = sum_value[j_14] + unpack_f16_8_1[j_14];
        }
        unsigned int scaled_f16_11[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
            scaled_f16_11[_lp] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&sum_value[_pair * 2]))
                : "r"(scaled_f16_11[_pair]));
        }
        int access_in_token = cluster_thread;
        int elem = access_1 * 8;
        float _vec_load_34[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_34[_pair * 2]))
                : "r"(norm_weight_words[_pair]));
        }
        #pragma unroll
        for (int j_15 = 0; j_15 < 8; j_15++) {
            _vec_load_33[j_15] = _vec_load_33[j_15] + sum_value[j_15];
        }
        unsigned int scaled_f16_8_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(_vec_load_33[_lp*2 + 0], _vec_load_33[_lp*2+1 + 0]));
            scaled_f16_8_1[_lp] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                ".reg .b16 h_lo, h_hi;\n\t"
                ".reg .b32 f_lo, f_hi;\n\t"
                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                "cvt.f32.f16 f_lo, h_lo;\n\t"
                "cvt.f32.f16 f_hi, h_hi;\n\t"
                "mov.b64 %0, {f_lo, f_hi};\n\t"
                "}\n"
                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_33[_pair * 2]))
                : "r"(scaled_f16_8_1[_pair]));
        }
        unsigned int _packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(_vec_load_33[_lp*2 + 0], _vec_load_33[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(_vec_load_33[_lp*2 + 2], _vec_load_33[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(_vec_load_33[_lp*2 + 4], _vec_load_33[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_h2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(_vec_load_33[_lp*2 + 6], _vec_load_33[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_h2;
        }
        reinterpret_cast<int4*>(residual_out + elem)[0] = reinterpret_cast<int4*>(_packed_0)[0];
        float square_sum = 0.0f;
        #pragma unroll
        for (int j_16 = 0; j_16 < 8; j_16++) {
            square_sum = __fmaf_rn(_vec_load_33[j_16], _vec_load_33[j_16], square_sum);
        }
        float _warp_reduce_21 = square_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_21 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_21, offset);
        square_sum = _warp_reduce_21;
        if (lane == 0) {
            reduce_smem[warp] = square_sum;
        }
        __syncthreads();
        if (tid == 0) {
            float block_sum = reduce_smem[0] + reduce_smem[4] + (reduce_smem[2] + reduce_smem[6]) + (reduce_smem[1] + reduce_smem[5] + reduce_smem[3]);
            mbarrier_arrive_expect_tx(rms_bar_addr + (rms_parity) * 8, 16);
            int slot_addr = rms_slots_addr + (unsigned int)(rms_parity * 16) + (unsigned int)(cta_rank * 4);
            uint32_t _cluster_mapa_22;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_22) : "r"(slot_addr), "r"(0));
            uint32_t _mbar_peer_23;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_23) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(0));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_22), "f"(block_sum), "r"(_mbar_peer_23) : "memory");
            uint32_t _cluster_mapa_24;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_24) : "r"(slot_addr), "r"(1));
            uint32_t _mbar_peer_25;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_25) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(1));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_24), "f"(block_sum), "r"(_mbar_peer_25) : "memory");
            uint32_t _cluster_mapa_26;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_26) : "r"(slot_addr), "r"(2));
            uint32_t _mbar_peer_27;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_27) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(2));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_26), "f"(block_sum), "r"(_mbar_peer_27) : "memory");
            uint32_t _cluster_mapa_28;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_cluster_mapa_28) : "r"(slot_addr), "r"(3));
            uint32_t _mbar_peer_29;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_29) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(3));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_cluster_mapa_28), "f"(block_sum), "r"(_mbar_peer_29) : "memory");
            asm volatile("fence.release.sync_restrict::shared::cta.cluster;" ::: "memory");
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(rms_bar_addr + (rms_parity) * 8), "r"(cta_rank ^ 1) : "memory");
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(rms_bar_addr + (rms_parity) * 8), "r"(cta_rank ^ 2) : "memory");
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(rms_bar_addr + (rms_parity) * 8), "r"(cta_rank ^ 3) : "memory");
        }
        mbarrier_wait_cluster_hint(rms_bar_addr + (rms_parity) * 8, _phase_rms_bar, 10000000);
        float cluster_sum = 0.0f;
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4];
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 1];
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 2];
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 3];
        float rstd = rsqrtf(cluster_sum / 7168.0f + epsilon);
        float norm_value[8];
        #pragma unroll
        for (int j_17 = 0; j_17 < 8; j_17++) {
            norm_value[j_17] = _vec_load_33[j_17] * rstd * (_vec_load_34[j_17] + weight_bias);
        }
        unsigned int scaled_f16_9_1[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
            scaled_f16_9_1[_lp] = *(uint32_t*)&_h2;
        }
        reinterpret_cast<int4*>(norm_out + elem)[0] = reinterpret_cast<int4*>(scaled_f16_9_1)[0];
        access_1 += access_stride;
        rms_parity += 1;
        if (rms_parity == 2) { rms_parity = 0; _phase_rms_bar ^= 1; }
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _sysv_wait_p_30 = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned int*>(control_0)) + (0);
                while (*_sysv_wait_p_30 != static_cast<int>(num_bids)) {}
            }
            control_0[2] = (control_current_word + 1) % 3;
            control_0[4] = token_end * 7168 * 2;
            reinterpret_cast<unsigned int*>(control_0)[0] = 0;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
