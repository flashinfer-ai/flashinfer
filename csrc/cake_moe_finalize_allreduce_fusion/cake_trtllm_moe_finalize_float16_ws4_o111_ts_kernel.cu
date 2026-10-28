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
#define NUM_MAIN_STAGES 1
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
#define LAUNCH_MIN_BLOCKS 4

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


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__global__ __launch_bounds__(224, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws4_o111_ts(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
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
    long long control_address = workspace_tensor[12];
    int* control_0 = reinterpret_cast<int*>(control_address);
    int control_bytes_per_buffer_word = control_0[3];
    long long comm_stride_elems = (long long)control_bytes_per_buffer_word / 2;
    long long workspace_address = workspace_tensor[8 + rank];
    __half* workspace_local = reinterpret_cast<__half*>(workspace_address);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int control_current_word = control_0[2];
    int control_clear_word = control_0[4];
    int control_current = control_current_word % 3;
    int control_dirty = (control_current_word + 2) % 3;
    long long data_base = (long long)control_current * comm_stride_elems;
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
    int local_tokens = ((token_begin < token_end) ? (token_end - token_begin + token_stride - 1) / token_stride : 0);
    unsigned int clear_words[4];
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        clear_words[word] = 2147516416;
    }
    long long clear_base = (long long)control_dirty * comm_stride_elems;
    int access = first_access;
    int rms_parity = 0;
    int clear_limit = control_clear_word / 8;
    int clear_total = ((clear_limit > first_access) ? (clear_limit - first_access + access_stride - 1) / access_stride : 0);
    int clear_per_step = (clear_total + local_tokens + 1) / (local_tokens + 2);
    int clear_next = first_access;
    unsigned int _phase_rms_bar = 0;
    #pragma unroll 1
    for (int step = 0; step < local_tokens + 2; step++) {
        if (local_tokens > step) {
            int published_token = token_begin + step * token_stride;
            float acc[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[j] = 0.0f;
            }
            int route_index_qa = -1;
            int route_index_qb = -1;
            int route_index_qc = -1;
            int route_index_qd = -1;
            if (top_k > 0) {
                int _vec_load_0[1];
                {
                    _vec_load_0[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k));
                }
                route_index_qa = _vec_load_0[0];
            }
            if (top_k > 1) {
                int _vec_load_1[1];
                {
                    _vec_load_1[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + 1));
                }
                route_index_qb = _vec_load_1[0];
            }
            if (top_k > 2) {
                int _vec_load_2[1];
                {
                    _vec_load_2[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + 2));
                }
                route_index_qc = _vec_load_2[0];
            }
            if (top_k > 3) {
                int _vec_load_3[1];
                {
                    _vec_load_3[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + 3));
                }
                route_index_qd = _vec_load_3[0];
            }
            #pragma unroll 1
            for (int route_base = 0; route_base < route_end; route_base += 4) {
                int next_index_qa = -1;
                int next_index_qb = -1;
                int next_index_qc = -1;
                int next_index_qd = -1;
                unsigned int route_words[16];
                float route_scales[4];
                if (route_base + 4 < top_k) {
                    int _vec_load_4[1];
                    {
                        _vec_load_4[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + route_base + 4));
                    }
                    next_index_qa = _vec_load_4[0];
                }
                if (route_base + 5 < top_k) {
                    int _vec_load_5[1];
                    {
                        _vec_load_5[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + route_base + 5));
                    }
                    next_index_qb = _vec_load_5[0];
                }
                if (route_base + 6 < top_k) {
                    int _vec_load_6[1];
                    {
                        _vec_load_6[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + route_base + 6));
                    }
                    next_index_qc = _vec_load_6[0];
                }
                if (route_base + 7 < top_k) {
                    int _vec_load_7[1];
                    {
                        _vec_load_7[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + route_base + 7));
                    }
                    next_index_qd = _vec_load_7[0];
                }
                if (route_index_qa >= 0) {
                    long long expert_elem = (long long)route_index_qa * 7168 + (long long)(cluster_thread * 8);
                    unsigned int _vec_load_8[4];
                    {
                        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem + 0);
                        _vec_load_8[0 + 0] = _uv4_0.x;
                        _vec_load_8[0 + 1] = _uv4_0.y;
                        _vec_load_8[0 + 2] = _uv4_0.z;
                        _vec_load_8[0 + 3] = _uv4_0.w;
                    }
                    #pragma unroll
                    for (int w = 0; w < 4; w++) {
                        route_words[w] = _vec_load_8[w];
                    }
                    float _vec_load_9[1];
                    {
                        __half _f16_1 = *reinterpret_cast<const __half*>(expert_scales + published_token * top_k + route_base);
                        _vec_load_9[0] = __half2float(_f16_1);
                    }
                    route_scales[0] = _vec_load_9[0];
                }
                if (route_index_qb >= 0) {
                    long long expert_elem_1 = (long long)route_index_qb * 7168 + (long long)(cluster_thread * 8);
                    unsigned int _vec_load_10[4];
                    {
                        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_1 + 0);
                        _vec_load_10[0 + 0] = _uv4_2.x;
                        _vec_load_10[0 + 1] = _uv4_2.y;
                        _vec_load_10[0 + 2] = _uv4_2.z;
                        _vec_load_10[0 + 3] = _uv4_2.w;
                    }
                    #pragma unroll
                    for (int w_1 = 0; w_1 < 4; w_1++) {
                        route_words[4 + w_1] = _vec_load_10[w_1];
                    }
                    float _vec_load_11[1];
                    {
                        __half _f16_3 = *reinterpret_cast<const __half*>(expert_scales + published_token * top_k + route_base + 1);
                        _vec_load_11[0] = __half2float(_f16_3);
                    }
                    route_scales[1] = _vec_load_11[0];
                }
                if (route_index_qc >= 0) {
                    long long expert_elem_2 = (long long)route_index_qc * 7168 + (long long)(cluster_thread * 8);
                    unsigned int _vec_load_12[4];
                    {
                        uint4 _uv4_4 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_2 + 0);
                        _vec_load_12[0 + 0] = _uv4_4.x;
                        _vec_load_12[0 + 1] = _uv4_4.y;
                        _vec_load_12[0 + 2] = _uv4_4.z;
                        _vec_load_12[0 + 3] = _uv4_4.w;
                    }
                    #pragma unroll
                    for (int w_2 = 0; w_2 < 4; w_2++) {
                        route_words[8 + w_2] = _vec_load_12[w_2];
                    }
                    float _vec_load_13[1];
                    {
                        __half _f16_5 = *reinterpret_cast<const __half*>(expert_scales + published_token * top_k + route_base + 2);
                        _vec_load_13[0] = __half2float(_f16_5);
                    }
                    route_scales[2] = _vec_load_13[0];
                }
                if (route_index_qd >= 0) {
                    long long expert_elem_3 = (long long)route_index_qd * 7168 + (long long)(cluster_thread * 8);
                    unsigned int _vec_load_14[4];
                    {
                        uint4 _uv4_6 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_3 + 0);
                        _vec_load_14[0 + 0] = _uv4_6.x;
                        _vec_load_14[0 + 1] = _uv4_6.y;
                        _vec_load_14[0 + 2] = _uv4_6.z;
                        _vec_load_14[0 + 3] = _uv4_6.w;
                    }
                    #pragma unroll
                    for (int w_3 = 0; w_3 < 4; w_3++) {
                        route_words[12 + w_3] = _vec_load_14[w_3];
                    }
                    float _vec_load_15[1];
                    {
                        __half _f16_7 = *reinterpret_cast<const __half*>(expert_scales + published_token * top_k + route_base + 3);
                        _vec_load_15[0] = __half2float(_f16_7);
                    }
                    route_scales[3] = _vec_load_15[0];
                }
                if (route_index_qa >= 0) {
                    float scaled[8];
                    float _unpack_f16_0[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_0[_pair * 2]))
                            : "r"(route_words[_pair]));
                    }
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        scaled[j_1] = _unpack_f16_0[j_1] * route_scales[0];
                    }
                    unsigned int scaled_f16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                        scaled_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    float _unpack_f16_1[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_1[_pair * 2]))
                            : "r"(scaled_f16[_pair]));
                    }
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        acc[j_2] = acc[j_2] + _unpack_f16_1[j_2];
                    }
                    unsigned int acc_f16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_f16[_lp] = *(uint32_t*)&_h2;
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
                            : "r"(acc_f16[_pair]));
                    }
                }
                if (route_index_qb >= 0) {
                    float scaled_1[8];
                    float _unpack_f16_2[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_2[_pair * 2]))
                            : "r"(route_words[4 + _pair]));
                    }
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        scaled_1[j_3] = _unpack_f16_2[j_3] * route_scales[1];
                    }
                    unsigned int scaled_f16_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
                        scaled_f16_1[_lp] = *(uint32_t*)&_h2;
                    }
                    float _unpack_f16_3[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_3[_pair * 2]))
                            : "r"(scaled_f16_1[_pair]));
                    }
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 8; j_4++) {
                        acc[j_4] = acc[j_4] + _unpack_f16_3[j_4];
                    }
                    unsigned int acc_f16_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_f16_1[_lp] = *(uint32_t*)&_h2;
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
                            : "r"(acc_f16_1[_pair]));
                    }
                }
                if (route_index_qc >= 0) {
                    float scaled_2[8];
                    float _unpack_f16_4[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_4[_pair * 2]))
                            : "r"(route_words[8 + _pair]));
                    }
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 8; j_5++) {
                        scaled_2[j_5] = _unpack_f16_4[j_5] * route_scales[2];
                    }
                    unsigned int scaled_f16_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                        scaled_f16_2[_lp] = *(uint32_t*)&_h2;
                    }
                    float _unpack_f16_5[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_5[_pair * 2]))
                            : "r"(scaled_f16_2[_pair]));
                    }
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        acc[j_6] = acc[j_6] + _unpack_f16_5[j_6];
                    }
                    unsigned int acc_f16_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_f16_2[_lp] = *(uint32_t*)&_h2;
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
                            : "r"(acc_f16_2[_pair]));
                    }
                }
                if (route_index_qd >= 0) {
                    float scaled_3[8];
                    float _unpack_f16_6[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_6[_pair * 2]))
                            : "r"(route_words[12 + _pair]));
                    }
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 8; j_7++) {
                        scaled_3[j_7] = _unpack_f16_6[j_7] * route_scales[3];
                    }
                    unsigned int scaled_f16_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(scaled_3[_lp*2 + 0], scaled_3[_lp*2+1 + 0]));
                        scaled_f16_3[_lp] = *(uint32_t*)&_h2;
                    }
                    float _unpack_f16_7[8];
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_7[_pair * 2]))
                            : "r"(scaled_f16_3[_pair]));
                    }
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 8; j_8++) {
                        acc[j_8] = acc[j_8] + _unpack_f16_7[j_8];
                    }
                    unsigned int acc_f16_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_f16_3[_lp] = *(uint32_t*)&_h2;
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
                            : "r"(acc_f16_3[_pair]));
                    }
                }
                route_index_qa = next_index_qa;
                route_index_qb = next_index_qb;
                route_index_qc = next_index_qc;
                route_index_qd = next_index_qd;
            }
            if (routed_scaling_factor != 1.0f) {
                #pragma unroll
                for (int j_9 = 0; j_9 < 8; j_9++) {
                    acc[j_9] = acc[j_9] * routed_scaling_factor;
                }
                unsigned int acc_f16_4[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_f16_4[_lp] = *(uint32_t*)&_h2;
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
                        : "r"(acc_f16_4[_pair]));
                }
            }
            if (has_shared_expert != 0) {
                int shared_elem = published_token * 7168 + cluster_thread * 8;
                float _vec_load_16[8];
                {
                    const uint4* _vptr_8 = reinterpret_cast<const uint4*>(shared_expert_output + shared_elem + 0);
                    uint4 _vld_8[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_8[_blk] = _vptr_8[_blk];
                        uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
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
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_16[0 + _blk * 8 + _pair * 2]))
                                : "r"(_vpairs_8[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_10 = 0; j_10 < 8; j_10++) {
                    acc[j_10] = acc[j_10] + _vec_load_16[j_10];
                }
                unsigned int acc_f16_5[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_f16_5[_lp] = *(uint32_t*)&_h2;
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
                        : "r"(acc_f16_5[_pair]));
                }
            }
            #pragma unroll
            for (int j_11 = 0; j_11 < 8; j_11++) {
                acc[j_11] = ((acc[j_11] == 0.0f) ? 0.0f : acc[j_11]);
            }
            unsigned int acc_f16_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                acc_f16_6[_lp] = *(uint32_t*)&_h2;
            }
            int chunk = cluster_thread / 224;
            int owner = published_token + chunk & 3;
            long long scatter_offset = ((long long)rank * (long long)tokens + (long long)published_token) * 1792 + (long long)(cluster_thread - chunk * 224) * 8;
            __half* scatter_dst = reinterpret_cast<__half*>(workspace_tensor[8 + owner]) + data_base + scatter_offset;
            asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(scatter_dst), "r"((acc_f16_6)[0]), "r"((acc_f16_6)[1]), "r"((acc_f16_6)[2]), "r"((acc_f16_6)[3]) : "memory");
        }
        int clear_end = clear_next + clear_per_step * access_stride;
        clear_end = ((clear_end > clear_limit) ? clear_limit : clear_end);
        #pragma unroll 1
        for (int clear_access = clear_next; clear_access < clear_end; clear_access += access_stride) {
            {
                auto _gst_9 = workspace_local + (clear_base + (long long)clear_access * 8);
                __builtin_assume(__isGlobal(_gst_9));
                reinterpret_cast<int4*>(_gst_9)[0] = reinterpret_cast<int4*>(clear_words)[0];
            }
        }
        clear_next = clear_end;
        if (step == 2) {
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
        if (step >= 1 && step <= local_tokens) {
            int owned_token = token_begin + (step - 1) * token_stride;
            int owned_chunk = cluster_thread / 224;
            if ((owned_token + owned_chunk & 3) == rank) {
                long long owned_offset = (long long)owned_token * 1792 + (long long)(cluster_thread - owned_chunk * 224) * 8;
                long long sender_stride = (long long)tokens * 1792;
                int owned_access = owned_token * 896 + cluster_thread;
                uint32_t _sysv_poll_group_0[16];
                do {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(workspace_local + (data_base + owned_offset)) : "memory");
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(workspace_local + (data_base + sender_stride + owned_offset)) : "memory");
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(workspace_local + (data_base + 2 * sender_stride + owned_offset)) : "memory");
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(workspace_local + (data_base + 3 * sender_stride + owned_offset)) : "memory");
                } while (
                    (((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 16) & 0xffffu) == 0x8000u) ||
                    (((_sysv_poll_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 16) & 0xffffu) == 0x8000u)
                );
                float _unpack_f16_8[8];
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
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_8[_pair * 2]))
                        : "r"(_sysv_poll_group_0[_pair]));
                }
                float sum_value[8];
                #pragma unroll
                for (int j_12 = 0; j_12 < 8; j_12++) {
                    sum_value[j_12] = _unpack_f16_8[j_12];
                }
                float _unpack_f16_9[8];
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
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_9[_pair * 2]))
                        : "r"(_sysv_poll_group_0[4 + _pair]));
                }
                #pragma unroll
                for (int j_13 = 0; j_13 < 8; j_13++) {
                    sum_value[j_13] = sum_value[j_13] + _unpack_f16_9[j_13];
                }
                unsigned int sum_value_f16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                    sum_value_f16[_lp] = *(uint32_t*)&_h2;
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
                        : "r"(sum_value_f16[_pair]));
                }
                float _unpack_f16_10[8];
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
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_10[_pair * 2]))
                        : "r"(_sysv_poll_group_0[8 + _pair]));
                }
                #pragma unroll
                for (int j_14 = 0; j_14 < 8; j_14++) {
                    sum_value[j_14] = sum_value[j_14] + _unpack_f16_10[j_14];
                }
                unsigned int sum_value_f16_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                    sum_value_f16_0[_lp] = *(uint32_t*)&_h2;
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
                        : "r"(sum_value_f16_0[_pair]));
                }
                float _unpack_f16_11[8];
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
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_11[_pair * 2]))
                        : "r"(_sysv_poll_group_0[12 + _pair]));
                }
                #pragma unroll
                for (int j_15 = 0; j_15 < 8; j_15++) {
                    sum_value[j_15] = sum_value[j_15] + _unpack_f16_11[j_15];
                }
                unsigned int sum_value_f16_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                    sum_value_f16_1[_lp] = *(uint32_t*)&_h2;
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
                        : "r"(sum_value_f16_1[_pair]));
                }
                long long gather_offset = (long long)tokens * 7168 + (long long)owned_access * 8;
                long long peer_base_0[1];
                {
                    uint64_t _scalar_bits_10;
                    asm volatile("ld.global.b64 %0, [%1];"
                        : "=l"(_scalar_bits_10) : "l"((const void*)(workspace_tensor + (8))) : "memory");
                    peer_base_0[0] = (long long)_scalar_bits_10;
                }
                __half* peer_0 = reinterpret_cast<__half*>(peer_base_0[0]) + data_base;
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_0 + gather_offset), "r"((sum_value_f16_1)[0]), "r"((sum_value_f16_1)[1]), "r"((sum_value_f16_1)[2]), "r"((sum_value_f16_1)[3]) : "memory");
                long long peer_base_1[1];
                {
                    uint64_t _scalar_bits_11;
                    asm volatile("ld.global.b64 %0, [%1];"
                        : "=l"(_scalar_bits_11) : "l"((const void*)(workspace_tensor + (9))) : "memory");
                    peer_base_1[0] = (long long)_scalar_bits_11;
                }
                __half* peer_1 = reinterpret_cast<__half*>(peer_base_1[0]) + data_base;
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_1 + gather_offset), "r"((sum_value_f16_1)[0]), "r"((sum_value_f16_1)[1]), "r"((sum_value_f16_1)[2]), "r"((sum_value_f16_1)[3]) : "memory");
                long long peer_base_2[1];
                {
                    uint64_t _scalar_bits_12;
                    asm volatile("ld.global.b64 %0, [%1];"
                        : "=l"(_scalar_bits_12) : "l"((const void*)(workspace_tensor + (10))) : "memory");
                    peer_base_2[0] = (long long)_scalar_bits_12;
                }
                __half* peer_2 = reinterpret_cast<__half*>(peer_base_2[0]) + data_base;
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_2 + gather_offset), "r"((sum_value_f16_1)[0]), "r"((sum_value_f16_1)[1]), "r"((sum_value_f16_1)[2]), "r"((sum_value_f16_1)[3]) : "memory");
                long long peer_base_3[1];
                {
                    uint64_t _scalar_bits_13;
                    asm volatile("ld.global.b64 %0, [%1];"
                        : "=l"(_scalar_bits_13) : "l"((const void*)(workspace_tensor + (11))) : "memory");
                    peer_base_3[0] = (long long)_scalar_bits_13;
                }
                __half* peer_3 = reinterpret_cast<__half*>(peer_base_3[0]) + data_base;
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_3 + gather_offset), "r"((sum_value_f16_1)[0]), "r"((sum_value_f16_1)[1]), "r"((sum_value_f16_1)[2]), "r"((sum_value_f16_1)[3]) : "memory");
            }
        }
        if (step >= 2) {
            int reduced_token = token_begin + (step - 2) * token_stride;
            int elem_early = access * 8;
            float _vec_load_17[8];
            {
                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(residual + elem_early + 0);
                uint4 _vld_14[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_14[_blk] = _vptr_14[_blk];
                    uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_17[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_14[_pair]));
                    }
                }
            }
            float sum_value_1[8];
            long long gather_in = data_base + (long long)tokens * 7168 + (long long)access * 8;
            uint32_t gathered[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(gathered[0]), "=r"(gathered[1]), "=r"(gathered[2]), "=r"(gathered[3]) : "l"(workspace_local + gather_in) : "memory");
            } while (
                (((gathered[0] >> 0) & 0xffffu) == 0x8000u) || (((gathered[0] >> 16) & 0xffffu) == 0x8000u) || (((gathered[1] >> 0) & 0xffffu) == 0x8000u) || (((gathered[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((gathered[2] >> 0) & 0xffffu) == 0x8000u) || (((gathered[2] >> 16) & 0xffffu) == 0x8000u) || (((gathered[3] >> 0) & 0xffffu) == 0x8000u) || (((gathered[3] >> 16) & 0xffffu) == 0x8000u)
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&sum_value_1[_pair * 2]))
                    : "r"(gathered[_pair]));
            }
            int access_in_token = cluster_thread;
            int elem = access * 8;
            float _vec_load_18[8];
            {
                const uint4* _vptr_15 = reinterpret_cast<const uint4*>(norm_weight + (access_in_token * 8) + 0);
                uint4 _vld_15[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_15[_blk] = _vptr_15[_blk];
                    uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
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
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_18[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_15[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int j_16 = 0; j_16 < 8; j_16++) {
                _vec_load_17[j_16] = _vec_load_17[j_16] + sum_value_1[j_16];
            }
            unsigned int _vec_load_17_f16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_17[_lp*2 + 0], _vec_load_17[_lp*2+1 + 0]));
                _vec_load_17_f16[_lp] = *(uint32_t*)&_h2;
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_17[_pair * 2]))
                    : "r"(_vec_load_17_f16[_pair]));
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_17[_lp*2 + 0], _vec_load_17[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_17[_lp*2 + 2], _vec_load_17[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_17[_lp*2 + 4], _vec_load_17[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_17[_lp*2 + 6], _vec_load_17[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_h2;
            }
            reinterpret_cast<int4*>(residual_out + elem)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            float square_sum = 0.0f;
            #pragma unroll
            for (int j_17 = 0; j_17 < 8; j_17++) {
                square_sum = __fmaf_rn(_vec_load_17[j_17], _vec_load_17[j_17], square_sum);
            }
            float _warp_reduce_16 = square_sum;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_16 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_16, offset);
            square_sum = _warp_reduce_16;
            if (lane == 0) {
                reduce_smem[warp] = square_sum;
            }
            __syncthreads();
            if (tid == 0) {
                float block_sum = reduce_smem[0] + reduce_smem[4] + (reduce_smem[2] + reduce_smem[6]) + (reduce_smem[1] + reduce_smem[5] + reduce_smem[3]);
                mbarrier_arrive_expect_tx(rms_bar_addr + (rms_parity) * 8, 16);
                int slot_addr = rms_slots_addr + (unsigned int)(rms_parity * 16) + (unsigned int)(cta_rank * 4);
                uint32_t _cluster_mapa_17;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_17) : "r"(slot_addr), "r"(0));
                uint32_t _mbar_peer_18;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_18) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(0));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_17), "f"(block_sum), "r"(_mbar_peer_18) : "memory");
                uint32_t _cluster_mapa_19;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_19) : "r"(slot_addr), "r"(1));
                uint32_t _mbar_peer_20;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_20) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(1));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_19), "f"(block_sum), "r"(_mbar_peer_20) : "memory");
                uint32_t _cluster_mapa_21;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_21) : "r"(slot_addr), "r"(2));
                uint32_t _mbar_peer_22;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_22) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(2));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_21), "f"(block_sum), "r"(_mbar_peer_22) : "memory");
                uint32_t _cluster_mapa_23;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_23) : "r"(slot_addr), "r"(3));
                uint32_t _mbar_peer_24;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_24) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(3));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_23), "f"(block_sum), "r"(_mbar_peer_24) : "memory");
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
            for (int j_18 = 0; j_18 < 8; j_18++) {
                norm_value[j_18] = _vec_load_17[j_18] * rstd * (_vec_load_18[j_18] + weight_bias);
            }
            unsigned int norm_value_f16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
                norm_value_f16[_lp] = *(uint32_t*)&_h2;
            }
            reinterpret_cast<int4*>(norm_out + elem)[0] = reinterpret_cast<int4*>(norm_value_f16)[0];
            uint32_t _f16x2_abs_0;
            asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_0) : "r"(norm_value_f16[0]));
            uint32_t _f16x2_abs_1;
            asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_1) : "r"(norm_value_f16[1]));
            uint32_t _f16x2_max_0;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_0) : "r"(_f16x2_abs_0), "r"(_f16x2_abs_1));
            uint32_t _f16x2_abs_2;
            asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_2) : "r"(norm_value_f16[2]));
            uint32_t _f16x2_max_1;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_1) : "r"(_f16x2_max_0), "r"(_f16x2_abs_2));
            uint32_t _f16x2_abs_3;
            asm("abs.f16x2 %0, %1;" : "=r"(_f16x2_abs_3) : "r"(norm_value_f16[3]));
            uint32_t _f16x2_max_2;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_2) : "r"(_f16x2_max_1), "r"(_f16x2_abs_3));
            unsigned int _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, _f16x2_max_2, 1);
            uint32_t _f16x2_max_3;
            asm("max.f16x2 %0, %1, %2;" : "=r"(_f16x2_max_3) : "r"(_shfl_xor_sync_25), "r"(_f16x2_max_2));
            uint16_t _f16_max_0;
            asm("max.f16 %0, %1, %2;" : "=h"(_f16_max_0) : "h"((uint16_t)(_f16x2_max_3 & 65535)), "h"((uint16_t)(_f16x2_max_3 >> 16)));
            float vector_max;
            asm("cvt.f32.f16 %0, %1;" : "=f"(vector_max) : "h"((uint16_t)(_f16_max_0)));
            float _unpack_f16_12[8];
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_12[_pair * 2]))
                    : "r"(norm_value_f16[_pair]));
            }
            float _rcp_26 = approx_rcp(6.0f);
            float sf_value = scale_factor * (vector_max * _rcp_26);
            uint16_t _e4m3x2_28;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_28) : "f"(0.0f), "f"(sf_value));
            uint8_t _cvt_satfinite_e4m3_f32_27 = (uint8_t)_e4m3x2_28;
            uint32_t _f16x2_decode_29;
            float sf_rounded;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_29) : "h"(_e4m3x2_28));
            uint16_t _f16_decode_29 = (uint16_t)_f16x2_decode_29;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded) : "h"(_f16_decode_29));
            float _rcp_31 = approx_rcp(scale_factor);
            float _rcp_30 = approx_rcp(sf_rounded * _rcp_31);
            float output_scale = ((sf_rounded != 0.0f) ? _rcp_30 : 0.0f);
            #pragma unroll
            for (int j_19 = 0; j_19 < 8; j_19++) {
                _unpack_f16_12[j_19] = _unpack_f16_12[j_19] * output_scale;
            }
            unsigned int _fp4_0[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(_unpack_f16_12[0]), "f"(_unpack_f16_12[1]), "f"(_unpack_f16_12[2]), "f"(_unpack_f16_12[3]), "f"(_unpack_f16_12[4]), "f"(_unpack_f16_12[5]), "f"(_unpack_f16_12[6]), "f"(_unpack_f16_12[7]));
            *(reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(quant_out)) + (access)) = _fp4_0[0];
            if (lane % 2 == 0) {
                int scale_col = access_in_token / 2;
                int inner_k = scale_col % 4;
                int k_tile = scale_col / 4;
                int inner_m = reduced_token % 128 / 32;
                int outer_m = reduced_token % 32;
                int m_tile = reduced_token / 128;
                int scale_index = m_tile * 112 * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k;
                uint16_t _cvt_satfinite_e4m3x2_f32_32;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_32) : "f"(0.0f), "f"(sf_value));
                *reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(scale_out) + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_32 & 127));
            }
            access += access_stride;
            rms_parity += 1;
            if (rms_parity == 2) { rms_parity = 0; _phase_rms_bar ^= 1; }
        }
    }
    if (local_tokens == 0) {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    #pragma unroll 1
    for (int clear_access_1 = clear_next; clear_access_1 < clear_limit; clear_access_1 += access_stride) {
        {
            auto _gst_33 = workspace_local + (clear_base + (long long)clear_access_1 * 8);
            __builtin_assume(__isGlobal(_gst_33));
            reinterpret_cast<int4*>(_gst_33)[0] = reinterpret_cast<int4*>(clear_words)[0];
        }
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _sysv_wait_p_34 = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned int*>(control_0)) + (0);
                while (*_sysv_wait_p_34 != static_cast<int>(num_bids)) {}
            }
            control_0[2] = (control_current_word + 1) % 3;
            control_0[4] = token_end * 14336;
            reinterpret_cast<unsigned int*>(control_0)[0] = 0;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
