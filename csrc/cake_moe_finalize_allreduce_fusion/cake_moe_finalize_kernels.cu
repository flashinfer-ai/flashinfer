/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#include <utility>

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

template <typename F, size_t... _i> __device__ __forceinline__ void poll_expand(F&& f, std::index_sequence<_i...>) {
    (f(std::integral_constant<int, static_cast<int>(_i)>{}), ...);
}

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

namespace cake_trtllm_moe_finalize {
__device__ inline unsigned int laneId()
{
    uint32_t id;
    asm("mov.u32 %0, %%laneid;" : "=r"(id));
    return id;
}
} // namespace cake_trtllm_moe_finalize
template <typename W, size_t... _i> __device__ __forceinline__ bool poll_any_0(const W* _sysv_poll_group_0, std::index_sequence<_i...>) {
    return (... || ((((_sysv_poll_group_0[_i] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[_i] >> 16) & 0xffffu) == 0x8000u)));
}


namespace cake_trtllm_moe_finalize {
template <typename T> struct dtype_traits;
template <> struct dtype_traits<__nv_bfloat16> {
    using vec2 = __nv_bfloat162;
    template <typename... Args>
    static __device__ __forceinline__ auto elem2float(Args&&... args) {
        return __bfloat162float(static_cast<Args&&>(args)...);
    }
    static __device__ __forceinline__ void unpack2(float* out, uint32_t in0) {
        asm volatile("{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n" : "=f"(out[0]), "=f"(out[1]) : "r"(in0));
    }
    template <typename... Args>
    static __device__ __forceinline__ auto float22elem2_rn(Args&&... args) {
        return __float22bfloat162_rn(static_cast<Args&&>(args)...);
    }
    template <typename... Args>
    static __device__ __forceinline__ auto floats2elem2_rn(Args&&... args) {
        return __floats2bfloat162_rn(static_cast<Args&&>(args)...);
    }
    static __device__ __forceinline__ void abs_elemx2(uint32_t& out0, uint32_t in0) {
        asm("abs.bf16x2 %0, %1;" : "=r"(out0) : "r"(in0));
    }
    static __device__ __forceinline__ void max_elemx2(uint32_t& out0, uint32_t in0, uint32_t in1) {
        asm("max.bf16x2 %0, %1, %2;" : "=r"(out0) : "r"(in0), "r"(in1));
    }
    static __device__ __forceinline__ void max_elem(uint16_t& out0, uint16_t in0, uint16_t in1) {
        asm("max.bf16 %0, %1, %2;" : "=h"(out0) : "h"(in0), "h"(in1));
    }
    static __device__ __forceinline__ void cvt_f32_elem(float& out0, uint16_t in0) {
        asm("cvt.f32.bf16 %0, %1;" : "=f"(out0) : "h"(in0));
    }
};
template <> struct dtype_traits<__half> {
    using vec2 = __half2;
    template <typename... Args>
    static __device__ __forceinline__ auto elem2float(Args&&... args) {
        return __half2float(static_cast<Args&&>(args)...);
    }
    static __device__ __forceinline__ void unpack2(float* out, uint32_t in0) {
        asm volatile("{\n\t"
            ".reg .b16 h_lo, h_hi;\n\t"
            ".reg .b32 f_lo, f_hi;\n\t"
            "mov.b32 {h_lo, h_hi}, %1;\n\t"
            "cvt.f32.f16 f_lo, h_lo;\n\t"
            "cvt.f32.f16 f_hi, h_hi;\n\t"
            "mov.b64 %0, {f_lo, f_hi};\n\t"
            "}\n" : "=l"(*reinterpret_cast<unsigned long long*>(out)) : "r"(in0));
    }
    template <typename... Args>
    static __device__ __forceinline__ auto float22elem2_rn(Args&&... args) {
        return __float22half2_rn(static_cast<Args&&>(args)...);
    }
    template <typename... Args>
    static __device__ __forceinline__ auto floats2elem2_rn(Args&&... args) {
        return __floats2half2_rn(static_cast<Args&&>(args)...);
    }
    static __device__ __forceinline__ void abs_elemx2(uint32_t& out0, uint32_t in0) {
        asm("abs.f16x2 %0, %1;" : "=r"(out0) : "r"(in0));
    }
    static __device__ __forceinline__ void max_elemx2(uint32_t& out0, uint32_t in0, uint32_t in1) {
        asm("max.f16x2 %0, %1, %2;" : "=r"(out0) : "r"(in0), "r"(in1));
    }
    static __device__ __forceinline__ void max_elem(uint16_t& out0, uint16_t in0, uint16_t in1) {
        asm("max.f16 %0, %1, %2;" : "=h"(out0) : "h"(in0), "h"(in1));
    }
    static __device__ __forceinline__ void cvt_f32_elem(float& out0, uint16_t in0) {
        asm("cvt.f32.f16 %0, %1;" : "=f"(out0) : "h"(in0));
    }
};

template <typename T, int WS, bool QUANT>
__device__ __forceinline__ void finalize_body(T* __restrict__ allreduce_in, int* __restrict__ inverse_indices, T* __restrict__ expert_scales, T* __restrict__ shared_expert_output, T* __restrict__ residual, T* __restrict__ norm_weight, T* __restrict__ residual_out, T* __restrict__ norm_out, T* __restrict__ quant_out, T* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = cake_trtllm_moe_finalize::laneId();
    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
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
    // === Task calls (dependency order) ===
    int rank = world_rank;
    long long control_address = workspace_tensor[3 * WS];
    int* control = reinterpret_cast<int*>(control_address);
    unsigned int* completion = reinterpret_cast<unsigned int*>(control_address);
    int* flag_addr = control + 2;
    int* comm_size_addr = control + 3;
    int* clear_addr = control + 4;
    long long comm_stride_elems = (long long)*comm_size_addr / 2;
    long long workspace_address = workspace_tensor[2 * WS + rank];
    T* workspace_local = reinterpret_cast<T*>(workspace_address);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int flag = *flag_addr;
    int clear_size = *clear_addr;
    int data_epoch = flag % 3;
    int clear_epoch = (flag + 2) % 3;
    long long data_base = (long long)data_epoch * comm_stride_elems;
    T* peer[WS];
    #pragma unroll
    for (int p = 0; p < WS; p++) {
        peer[p] = reinterpret_cast<T*>(workspace_tensor[(p + 2 * WS)]) + data_base;
    }
    __syncthreads();
    if (tid == 0) {
        {
            unsigned int* _lca_p_0 = reinterpret_cast<unsigned int*>(completion) + (0);
            atomicAdd(_lca_p_0, 1u);
        }
    }
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
        int route_index_qa = -1;
        int route_index_qb = -1;
        int route_index_qc = -1;
        int route_index_qd = -1;
        if (top_k > 0) {
            int _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k));
            }
            route_index_qa = _vec_load_0[0];
        }
        if (top_k > 1) {
            int _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 1));
            }
            route_index_qb = _vec_load_1[0];
        }
        if (top_k > 2) {
            int _vec_load_2[1];
            {
                _vec_load_2[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 2));
            }
            route_index_qc = _vec_load_2[0];
        }
        if (top_k > 3) {
            int _vec_load_3[1];
            {
                _vec_load_3[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + 3));
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
                    _vec_load_4[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 4));
                }
                next_index_qa = _vec_load_4[0];
            }
            if (route_base + 5 < top_k) {
                int _vec_load_5[1];
                {
                    _vec_load_5[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 5));
                }
                next_index_qb = _vec_load_5[0];
            }
            if (route_base + 6 < top_k) {
                int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 6));
                }
                next_index_qc = _vec_load_6[0];
            }
            if (route_base + 7 < top_k) {
                int _vec_load_7[1];
                {
                    _vec_load_7[0] = *reinterpret_cast<const int*>(inverse_indices + (token * top_k + route_base + 7));
                }
                next_index_qd = _vec_load_7[0];
            }
            if (route_index_qa >= 0) {
                long long expert_elem = (long long)route_index_qa * 7168 + (long long)(cluster_thread * 8);
                unsigned int _vec_load_8[4];
                {
                    uint4 _uv4_1 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem + 0);
                    _vec_load_8[0 + 0] = _uv4_1.x;
                    _vec_load_8[0 + 1] = _uv4_1.y;
                    _vec_load_8[0 + 2] = _uv4_1.z;
                    _vec_load_8[0 + 3] = _uv4_1.w;
                }
                #pragma unroll
                for (int w = 0; w < 4; w++) {
                    route_words[w] = _vec_load_8[w];
                }
                float _vec_load_9[1];
                {
                    T _elem_2 = *reinterpret_cast<const T*>(expert_scales + token * top_k + route_base);
                    _vec_load_9[0] = dtype_traits<T>::elem2float(_elem_2);
                }
                route_scales[0] = _vec_load_9[0];
            }
            if (route_index_qb >= 0) {
                long long expert_elem_1 = (long long)route_index_qb * 7168 + (long long)(cluster_thread * 8);
                unsigned int _vec_load_10[4];
                {
                    uint4 _uv4_3 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_1 + 0);
                    _vec_load_10[0 + 0] = _uv4_3.x;
                    _vec_load_10[0 + 1] = _uv4_3.y;
                    _vec_load_10[0 + 2] = _uv4_3.z;
                    _vec_load_10[0 + 3] = _uv4_3.w;
                }
                #pragma unroll
                for (int w_1 = 0; w_1 < 4; w_1++) {
                    route_words[4 + w_1] = _vec_load_10[w_1];
                }
                float _vec_load_11[1];
                {
                    T _elem_4 = *reinterpret_cast<const T*>(expert_scales + token * top_k + route_base + 1);
                    _vec_load_11[0] = dtype_traits<T>::elem2float(_elem_4);
                }
                route_scales[1] = _vec_load_11[0];
            }
            if (route_index_qc >= 0) {
                long long expert_elem_2 = (long long)route_index_qc * 7168 + (long long)(cluster_thread * 8);
                unsigned int _vec_load_12[4];
                {
                    uint4 _uv4_5 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_2 + 0);
                    _vec_load_12[0 + 0] = _uv4_5.x;
                    _vec_load_12[0 + 1] = _uv4_5.y;
                    _vec_load_12[0 + 2] = _uv4_5.z;
                    _vec_load_12[0 + 3] = _uv4_5.w;
                }
                #pragma unroll
                for (int w_2 = 0; w_2 < 4; w_2++) {
                    route_words[8 + w_2] = _vec_load_12[w_2];
                }
                float _vec_load_13[1];
                {
                    T _elem_6 = *reinterpret_cast<const T*>(expert_scales + token * top_k + route_base + 2);
                    _vec_load_13[0] = dtype_traits<T>::elem2float(_elem_6);
                }
                route_scales[2] = _vec_load_13[0];
            }
            if (route_index_qd >= 0) {
                long long expert_elem_3 = (long long)route_index_qd * 7168 + (long long)(cluster_thread * 8);
                unsigned int _vec_load_14[4];
                {
                    uint4 _uv4_7 = *reinterpret_cast<const uint4*>(allreduce_in + expert_elem_3 + 0);
                    _vec_load_14[0 + 0] = _uv4_7.x;
                    _vec_load_14[0 + 1] = _uv4_7.y;
                    _vec_load_14[0 + 2] = _uv4_7.z;
                    _vec_load_14[0 + 3] = _uv4_7.w;
                }
                #pragma unroll
                for (int w_3 = 0; w_3 < 4; w_3++) {
                    route_words[12 + w_3] = _vec_load_14[w_3];
                }
                float _vec_load_15[1];
                {
                    T _elem_8 = *reinterpret_cast<const T*>(expert_scales + token * top_k + route_base + 3);
                    _vec_load_15[0] = dtype_traits<T>::elem2float(_elem_8);
                }
                route_scales[3] = _vec_load_15[0];
            }
            if (route_index_qa >= 0) {
                float scaled[8];
                float route_words_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&route_words_f32[_pair * 2], route_words[_pair]);
                }
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    scaled[j_1] = route_words_f32[j_1] * route_scales[0];
                }
                uint32_t scaled_elem[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_elem[_lp] = *(uint32_t*)&_bf2;
                }
                float scaled_elem_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&scaled_elem_f32[_pair * 2], scaled_elem[_pair]);
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    acc[j_2] = acc[j_2] + scaled_elem_f32[j_2];
                }
                uint32_t acc_elem[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_elem[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&acc[_pair * 2], acc_elem[_pair]);
                }
            }
            if (route_index_qb >= 0) {
                float scaled_1[8];
                float route_words_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&route_words_f32_1[_pair * 2], route_words[4 + _pair]);
                }
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    scaled_1[j_3] = route_words_f32_1[j_3] * route_scales[1];
                }
                uint32_t scaled_elem_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
                    scaled_elem_1[_lp] = *(uint32_t*)&_bf2;
                }
                float scaled_elem_f32_1[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&scaled_elem_f32_1[_pair * 2], scaled_elem_1[_pair]);
                }
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    acc[j_4] = acc[j_4] + scaled_elem_f32_1[j_4];
                }
                uint32_t acc_elem_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_elem_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&acc[_pair * 2], acc_elem_1[_pair]);
                }
            }
            if (route_index_qc >= 0) {
                float scaled_2[8];
                float route_words_f32_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&route_words_f32_2[_pair * 2], route_words[8 + _pair]);
                }
                #pragma unroll
                for (int j_5 = 0; j_5 < 8; j_5++) {
                    scaled_2[j_5] = route_words_f32_2[j_5] * route_scales[2];
                }
                uint32_t scaled_elem_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(scaled_2[_lp*2 + 0], scaled_2[_lp*2+1 + 0]));
                    scaled_elem_2[_lp] = *(uint32_t*)&_bf2;
                }
                float scaled_elem_f32_2[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&scaled_elem_f32_2[_pair * 2], scaled_elem_2[_pair]);
                }
                #pragma unroll
                for (int j_6 = 0; j_6 < 8; j_6++) {
                    acc[j_6] = acc[j_6] + scaled_elem_f32_2[j_6];
                }
                uint32_t acc_elem_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_elem_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&acc[_pair * 2], acc_elem_2[_pair]);
                }
            }
            if (route_index_qd >= 0) {
                float scaled_3[8];
                float route_words_f32_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&route_words_f32_3[_pair * 2], route_words[12 + _pair]);
                }
                #pragma unroll
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    scaled_3[j_7] = route_words_f32_3[j_7] * route_scales[3];
                }
                uint32_t scaled_elem_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(scaled_3[_lp*2 + 0], scaled_3[_lp*2+1 + 0]));
                    scaled_elem_3[_lp] = *(uint32_t*)&_bf2;
                }
                float scaled_elem_f32_3[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&scaled_elem_f32_3[_pair * 2], scaled_elem_3[_pair]);
                }
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    acc[j_8] = acc[j_8] + scaled_elem_f32_3[j_8];
                }
                uint32_t acc_elem_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_elem_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&acc[_pair * 2], acc_elem_3[_pair]);
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
            uint32_t acc_elem_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                acc_elem_4[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                dtype_traits<T>::unpack2(&acc[_pair * 2], acc_elem_4[_pair]);
            }
        }
        if (has_shared_expert != 0) {
            int shared_elem = token * 7168 + cluster_thread * 8;
            float _vec_load_16[8];
            {
                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(shared_expert_output + shared_elem + 0);
                uint4 _vld_9[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_9[_blk] = _vptr_9[_blk];
                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        dtype_traits<T>::unpack2(&_vec_load_16[0 + _blk * 8 + _pair * 2], _vpairs_9[_pair]);
                    }
                }
            }
            #pragma unroll
            for (int j_10 = 0; j_10 < 8; j_10++) {
                acc[j_10] = acc[j_10] + _vec_load_16[j_10];
            }
            uint32_t acc_elem_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                acc_elem_5[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                dtype_traits<T>::unpack2(&acc[_pair * 2], acc_elem_5[_pair]);
            }
        }
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            acc[j_11] = ((acc[j_11] == 0.0f) ? 0.0f : acc[j_11]);
        }
        uint32_t acc_elem_6[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            acc_elem_6[_lp] = *(uint32_t*)&_bf2;
        }
        int access = token * 896 + cluster_thread;
        long long slot = (long long)rank * (long long)total_access * 8 + (long long)access * 8;
        #pragma unroll
        for (int p = 0; p < WS; p++) {
            asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer[p] + slot), "r"((acc_elem_6)[0]), "r"((acc_elem_6)[1]), "r"((acc_elem_6)[2]), "r"((acc_elem_6)[3]) : "memory");
        }
    }
    unsigned int clear_words[4];
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        clear_words[word] = 2147516416;
    }
    long long clear_base = (long long)clear_epoch * comm_stride_elems;
    #pragma unroll 4
    for (int clear_access = first_access; clear_access < clear_size / 8; clear_access += access_stride) {
        reinterpret_cast<int4*>(workspace_local + (clear_base + (long long)clear_access * 8))[0] = reinterpret_cast<int4*>(clear_words)[0];
    }
    int access_1 = first_access;
    int rms_parity = 0;
    #pragma unroll 1
    for (int token_1 = token_begin; token_1 < token_end; token_1 += token_stride) {
        int elem_early = access_1 * 8;
        float residual_early[8];
        int hoist_on = 1;
        int hoist_ws_max_lit = WS;
        if (hoist_ws_max_lit > 4) {
            hoist_on = 0;
        }
        if (hoist_on == 1) {
            float _vec_load_17[8];
            {
                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(residual + elem_early + 0);
                uint4 _vld_10[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_10[_blk] = _vptr_10[_blk];
                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        dtype_traits<T>::unpack2(&_vec_load_17[0 + _blk * 8 + _pair * 2], _vpairs_10[_pair]);
                    }
                }
            }
            #pragma unroll
            for (int j_12 = 0; j_12 < 8; j_12++) {
                residual_early[j_12] = _vec_load_17[j_12];
            }
        }
        uint32_t _sysv_poll_group_0[4 * WS];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(workspace_local + (data_base + (long long)(access_1 * 8))) : "memory");
            poll_expand([&](auto _pc) {
                constexpr int p = 1 + decltype(_pc)::value;
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4 * p]), "=r"(_sysv_poll_group_0[(4 * p + 1)]), "=r"(_sysv_poll_group_0[(4 * p + 2)]), "=r"(_sysv_poll_group_0[(4 * p + 3)]) : "l"(workspace_local + (data_base + (long long)(p * total_access * 8) + (long long)(access_1 * 8))) : "memory");
            }, std::make_index_sequence<WS - 1>{});
        } while (poll_any_0(_sysv_poll_group_0, std::make_index_sequence<4 * WS>{}));
        float _sysv_poll_group_0_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            dtype_traits<T>::unpack2(&_sysv_poll_group_0_f32[_pair * 2], _sysv_poll_group_0[_pair]);
        }
        float sum_value[8];
        #pragma unroll
        for (int j_13 = 0; j_13 < 8; j_13++) {
            sum_value[j_13] = _sysv_poll_group_0_f32[j_13];
        }
        #pragma unroll
        for (int p = 1; p < WS; p++) {
            float _sysv_poll_group_0_f32_0[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                dtype_traits<T>::unpack2(&_sysv_poll_group_0_f32_0[_pair * 2], _sysv_poll_group_0[4 * p + _pair]);
            }
            #pragma unroll
            for (int j_14 = 0; j_14 < 8; j_14++) {
                sum_value[j_14] = sum_value[j_14] + _sysv_poll_group_0_f32_0[j_14];
            }
            uint32_t sum_value_elem[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_elem[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                dtype_traits<T>::unpack2(&sum_value[_pair * 2], sum_value_elem[_pair]);
            }
        }
        int access_in_token = cluster_thread;
        int elem = access_1 * 8;
        float residual_value[8];
        if (hoist_on == 1) {
            #pragma unroll
            for (int j_17 = 0; j_17 < 8; j_17++) {
                residual_value[j_17] = residual_early[j_17];
            }
        }
        if (hoist_on == 0) {
            float _vec_load_18[8];
            {
                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(residual + elem + 0);
                uint4 _vld_11[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_11[_blk] = _vptr_11[_blk];
                    uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        dtype_traits<T>::unpack2(&_vec_load_18[0 + _blk * 8 + _pair * 2], _vpairs_11[_pair]);
                    }
                }
            }
            #pragma unroll
            for (int j_18 = 0; j_18 < 8; j_18++) {
                residual_value[j_18] = _vec_load_18[j_18];
            }
        }
        float _vec_load_19[8];
        {
            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(norm_weight + (access_in_token * 8) + 0);
            uint4 _vld_12[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_12[_blk] = _vptr_12[_blk];
                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    dtype_traits<T>::unpack2(&_vec_load_19[0 + _blk * 8 + _pair * 2], _vpairs_12[_pair]);
                }
            }
        }
        #pragma unroll
        for (int j_19 = 0; j_19 < 8; j_19++) {
            residual_value[j_19] = residual_value[j_19] + sum_value[j_19];
        }
        uint32_t residual_value_elem[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(residual_value[_lp*2 + 0], residual_value[_lp*2+1 + 0]));
            residual_value_elem[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            dtype_traits<T>::unpack2(&residual_value[_pair * 2], residual_value_elem[_pair]);
        }
        {
            typename dtype_traits<T>::vec2 _pk[4];
            _pk[0] = dtype_traits<T>::floats2elem2_rn(residual_value[0 + 0], residual_value[0 + 1]);
            _pk[1] = dtype_traits<T>::floats2elem2_rn(residual_value[0 + 2], residual_value[0 + 3]);
            _pk[2] = dtype_traits<T>::floats2elem2_rn(residual_value[0 + 4], residual_value[0 + 5]);
            _pk[3] = dtype_traits<T>::floats2elem2_rn(residual_value[0 + 6], residual_value[0 + 7]);
            *reinterpret_cast<uint4*>(&((T*)(residual_out + elem))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        float square_sum = 0.0f;
        #pragma unroll
        for (int j_20 = 0; j_20 < 8; j_20++) {
            square_sum = square_sum + residual_value[j_20] * residual_value[j_20];
        }
        float _warp_reduce_0 = square_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        square_sum = _warp_reduce_0;
        if (lane == 0) {
            reduce_smem[warp] = square_sum;
        }
        __syncthreads();
        float block_sum = ((tid < 7) ? reduce_smem[lane] : 0.0f);
        float _warp_reduce_1 = block_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        block_sum = _warp_reduce_1;
        if (tid == 0) {
            int slot_addr = rms_slots_addr + (unsigned int)(rms_parity * 16) + (unsigned int)(cta_rank * 4);
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(slot_addr), "r"(0));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_0), "f"(block_sum) : "memory");
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(slot_addr), "r"(1));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_1), "f"(block_sum) : "memory");
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(slot_addr), "r"(2));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_2), "f"(block_sum) : "memory");
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(slot_addr), "r"(3));
            asm volatile(
                "st.shared::cluster.f32 [%0], %1;"
                :: "r"(_mapa_3), "f"(block_sum) : "memory");
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float cluster_sum = 0.0f;
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4];
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 1];
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 2];
        cluster_sum = cluster_sum + rms_slots[rms_parity * 4 + 3];
        float _rsqrt_0 = rsqrtf(cluster_sum / 7168.0f + epsilon);
        float rstd = _rsqrt_0;
        float norm_value[8];
        #pragma unroll
        for (int j_21 = 0; j_21 < 8; j_21++) {
            norm_value[j_21] = residual_value[j_21] * rstd * (_vec_load_19[j_21] + weight_bias);
        }
        uint32_t norm_value_elem[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            typename dtype_traits<T>::vec2 _bf2 = dtype_traits<T>::float22elem2_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
            norm_value_elem[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(norm_out + elem)[0] = reinterpret_cast<int4*>(norm_value_elem)[0];
        if constexpr (QUANT) {
            uint32_t _bf16x2_abs_0;
            dtype_traits<T>::abs_elemx2(_bf16x2_abs_0, norm_value_elem[0]);
            uint32_t _bf16x2_abs_1;
            dtype_traits<T>::abs_elemx2(_bf16x2_abs_1, norm_value_elem[1]);
            uint32_t _bf16x2_max_0;
            dtype_traits<T>::max_elemx2(_bf16x2_max_0, _bf16x2_abs_0, _bf16x2_abs_1);
            uint32_t _bf16x2_abs_2;
            dtype_traits<T>::abs_elemx2(_bf16x2_abs_2, norm_value_elem[2]);
            uint32_t _bf16x2_max_1;
            dtype_traits<T>::max_elemx2(_bf16x2_max_1, _bf16x2_max_0, _bf16x2_abs_2);
            uint32_t _bf16x2_abs_3;
            dtype_traits<T>::abs_elemx2(_bf16x2_abs_3, norm_value_elem[3]);
            uint32_t _bf16x2_max_2;
            dtype_traits<T>::max_elemx2(_bf16x2_max_2, _bf16x2_max_1, _bf16x2_abs_3);
            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, _bf16x2_max_2, 1);
            uint32_t _bf16x2_max_3;
            dtype_traits<T>::max_elemx2(_bf16x2_max_3, _shfl_xor_0, _bf16x2_max_2);
            uint16_t _elem_max_0;
            dtype_traits<T>::max_elem(_elem_max_0, (uint16_t)(_bf16x2_max_3 & 65535), (uint16_t)(_bf16x2_max_3 >> 16));
            float _cvt_f32_elem_0;
            dtype_traits<T>::cvt_f32_elem(_cvt_f32_elem_0, (uint16_t)(_elem_max_0));
            float vector_max = _cvt_f32_elem_0;
            float norm_value_elem_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                dtype_traits<T>::unpack2(&norm_value_elem_f32[_pair * 2], norm_value_elem[_pair]);
            }
            float _rcp_0 = approx_rcp(6.0f);
            float sf_value = scale_factor * (vector_max * _rcp_0);
            float _fp8_rt_0;
            uint16_t _e4m3x2_13;
            uint32_t _f16x2_13;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_13) : "f"(0.0f), "f"(sf_value));
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_13) : "h"(_e4m3x2_13));
            uint16_t _fp8_h0_13 = (uint16_t)(_f16x2_13 & 0xFFFFu);
            asm("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_13));
            float sf_rounded = _fp8_rt_0;
            float _rcp_1 = approx_rcp(scale_factor);
            float _rcp_2 = approx_rcp(sf_rounded * _rcp_1);
            float output_scale = ((sf_rounded != 0.0f) ? _rcp_2 : 0.0f);
            #pragma unroll
            for (int j_22 = 0; j_22 < 8; j_22++) {
                norm_value_elem_f32[j_22] = norm_value_elem_f32[j_22] * output_scale;
            }
            uint32_t _fp4_0[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(norm_value_elem_f32[0]), "f"(norm_value_elem_f32[1]), "f"(norm_value_elem_f32[2]), "f"(norm_value_elem_f32[3]), "f"(norm_value_elem_f32[4]), "f"(norm_value_elem_f32[5]), "f"(norm_value_elem_f32[6]), "f"(norm_value_elem_f32[7]));
            *(reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(quant_out)) + (access_1)) = _fp4_0[0];
            if (lane % 2 == 0) {
                int scale_col = access_in_token / 2;
                int inner_k = scale_col % 4;
                int k_tile = scale_col / 4;
                int inner_m = token_1 % 128 / 32;
                int outer_m = token_1 % 32;
                int m_tile = token_1 / 128;
                int scale_index = m_tile * 112 * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k;
                {
                    unsigned short _sf_pair;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                    *(reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(scale_out) + scale_index) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                }
            }
        }
        access_1 += access_stride;
        rms_parity = 1 - rms_parity;
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _lcv_p_13 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_lcv_p_13 != static_cast<int>(num_bids)) {}
                *reinterpret_cast<int*>(flag_addr) = static_cast<int>((flag + 1) % 3);
                *reinterpret_cast<int*>(clear_addr) = static_cast<int>(token_end * 7168 * WS);
                *(reinterpret_cast<int*>(completion) + (0)) = 0;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
} // namespace cake_trtllm_moe_finalize

extern "C" {

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o110(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__nv_bfloat16, 2, false>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o111(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__nv_bfloat16, 2, true>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o110(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__nv_bfloat16, 4, false>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws4_o111(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__nv_bfloat16, 4, true>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__nv_bfloat16, 8, false>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o111(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__nv_bfloat16, 8, true>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws2_o110(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__half, 2, false>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws2_o111(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__half, 2, true>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws4_o110(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__half, 4, false>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws4_o111(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__half, 4, true>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws8_o110(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__half, 8, false>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_float16_ws8_o111(__half* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __half* __restrict__ expert_scales, __half* __restrict__ shared_expert_output, __half* __restrict__ residual, __half* __restrict__ norm_weight, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
{
    cake_trtllm_moe_finalize::finalize_body<__half, 8, true>(allreduce_in, inverse_indices, expert_scales, shared_expert_output, residual, norm_weight, residual_out, norm_out, quant_out, scale_out, workspace_tensor, world_rank, tokens, top_k, has_shared_expert, routed_scaling_factor, epsilon, weight_bias, scale_factor);
}

} // extern "C"
