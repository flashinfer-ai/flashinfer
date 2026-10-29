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
#define SMEM_OWN_RING_OFF 256
#define SMEM_OWN_RING_STAGE_BYTES 7168
#define SMEM_OWN_RING_STRIDE 7168
#define SMEM_TOTAL 7424
#define THREADS 224
#define LAUNCH_MIN_BLOCKS 5

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
kernel_cake_trtllm_moe_finalize_bfloat16_ws2_o111_lt(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
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
    unsigned int* own_ring = reinterpret_cast<unsigned int*>(smem_raw + 256);
    const int own_ring_addr = smem + 256;

    // === Task calls (declaration order) ===
    int rank = world_rank;
    long long control_address = workspace_tensor[6];
    int* control_0 = reinterpret_cast<int*>(control_address);
    int control_bytes_per_buffer_word = control_0[3];
    long long comm_stride_elems = (long long)control_bytes_per_buffer_word / 2;
    long long workspace_address = workspace_tensor[4 + rank];
    __nv_bfloat16* workspace_local = reinterpret_cast<__nv_bfloat16*>(workspace_address);
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int control_current_word = control_0[2];
    int control_clear_word = control_0[4];
    int control_current = control_current_word % 3;
    int control_dirty = (control_current_word + 2) % 3;
    long long data_base = (long long)control_current * comm_stride_elems;
    __nv_bfloat16* peer0 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[4]) + data_base;
    __nv_bfloat16* peer1 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[5]) + data_base;
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
    unsigned int clear_words[4];
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        clear_words[word] = 2147516416;
    }
    long long clear_base = (long long)control_dirty * comm_stride_elems;
    int access = first_access;
    int rms_parity = 0;
    int interleaved_tokens = ((token_begin < token_end) ? (token_end - token_begin + token_stride - 1) / token_stride : 0);
    int clear_limit = control_clear_word / 8;
    int clear_total = ((clear_limit > first_access) ? (clear_limit - first_access + access_stride - 1) / access_stride : 0);
    int clear_per_step = (clear_total + interleaved_tokens + 1) / (interleaved_tokens + 1 + 1);
    int clear_next = first_access;
    unsigned int _phase_rms_bar = 0;
    #pragma unroll 1
    for (int step = 0; step < interleaved_tokens + 1; step++) {
        if (step < interleaved_tokens) {
            int published_token = token_begin + step * token_stride;
            float acc[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[j] = 0.0f;
            }
            int route_index_lo = -1;
            int route_index_hi = -1;
            if (top_k > 0) {
                int _vec_load_0[1];
                {
                    _vec_load_0[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k));
                }
                route_index_lo = _vec_load_0[0];
            }
            if (top_k > 1) {
                int _vec_load_1[1];
                {
                    _vec_load_1[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + 1));
                }
                route_index_hi = _vec_load_1[0];
            }
            #pragma unroll 1
            for (int route_base = 0; route_base < route_end; route_base += 2) {
                int next_index_lo = -1;
                int next_index_hi = -1;
                float route_values[16];
                float route_scales[2];
                if (route_base + 2 < top_k) {
                    int _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + route_base + 2));
                    }
                    next_index_lo = _vec_load_2[0];
                }
                if (route_base + 3 < top_k) {
                    int _vec_load_3[1];
                    {
                        _vec_load_3[0] = *reinterpret_cast<const int*>(inverse_indices + (published_token * top_k + route_base + 3));
                    }
                    next_index_hi = _vec_load_3[0];
                }
                if (route_index_lo >= 0) {
                    long long expert_elem = (long long)route_index_lo * 7168 + (long long)(cluster_thread * 8);
                    float _vec_load_4[8];
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(allreduce_in + expert_elem + 0);
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
                                    : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_0[_pair]));
                            }
                        }
                    }
                    float _vec_load_5[1];
                    {
                        __nv_bfloat16 _bf16_1 = *reinterpret_cast<const __nv_bfloat16*>(expert_scales + published_token * top_k + route_base);
                        _vec_load_5[0] = __bfloat162float(_bf16_1);
                    }
                    route_scales[0] = _vec_load_5[0];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        route_values[j_1] = _vec_load_4[j_1];
                    }
                }
                if (route_index_hi >= 0) {
                    long long expert_elem_1 = (long long)route_index_hi * 7168 + (long long)(cluster_thread * 8);
                    float _vec_load_6[8];
                    {
                        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(allreduce_in + expert_elem_1 + 0);
                        uint4 _vld_2[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_2[_blk] = _vptr_2[_blk];
                            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_2[_pair]));
                            }
                        }
                    }
                    float _vec_load_7[1];
                    {
                        __nv_bfloat16 _bf16_3 = *reinterpret_cast<const __nv_bfloat16*>(expert_scales + published_token * top_k + route_base + 1);
                        _vec_load_7[0] = __bfloat162float(_bf16_3);
                    }
                    route_scales[1] = _vec_load_7[0];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 8; j_2++) {
                        route_values[8 + j_2] = _vec_load_6[j_2];
                    }
                }
                if (route_index_lo >= 0) {
                    float scaled[8];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        scaled[j_3] = route_values[j_3] * route_scales[0];
                    }
                    unsigned int scaled_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                        scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_0[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_unpack_bf16_0[_pair * 2])[0]), "=f"((&_unpack_bf16_0[_pair * 2])[1])
                            : "r"(scaled_bf16[_pair]));
                    }
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 8; j_4++) {
                        acc[j_4] = acc[j_4] + _unpack_bf16_0[j_4];
                    }
                    unsigned int acc_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&acc[_pair * 2])[0]), "=f"((&acc[_pair * 2])[1])
                            : "r"(acc_bf16[_pair]));
                    }
                }
                if (route_index_hi >= 0) {
                    float scaled_1[8];
                    #pragma unroll
                    for (int j_5 = 0; j_5 < 8; j_5++) {
                        scaled_1[j_5] = route_values[8 + j_5] * route_scales[1];
                    }
                    unsigned int scaled_bf16_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled_1[_lp*2 + 0], scaled_1[_lp*2+1 + 0]));
                        scaled_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    float _unpack_bf16_1[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_unpack_bf16_1[_pair * 2])[0]), "=f"((&_unpack_bf16_1[_pair * 2])[1])
                            : "r"(scaled_bf16_1[_pair]));
                    }
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        acc[j_6] = acc[j_6] + _unpack_bf16_1[j_6];
                    }
                    unsigned int acc_bf16_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                        acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&acc[_pair * 2])[0]), "=f"((&acc[_pair * 2])[1])
                            : "r"(acc_bf16_1[_pair]));
                    }
                }
                route_index_lo = next_index_lo;
                route_index_hi = next_index_hi;
            }
            if (routed_scaling_factor != 1.0f) {
                #pragma unroll
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    acc[j_7] = acc[j_7] * routed_scaling_factor;
                }
                unsigned int acc_bf16_2[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_bf16_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&acc[_pair * 2])[0]), "=f"((&acc[_pair * 2])[1])
                        : "r"(acc_bf16_2[_pair]));
                }
            }
            if (has_shared_expert != 0) {
                int shared_elem = published_token * 7168 + cluster_thread * 8;
                float _vec_load_8[8];
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(shared_expert_output + shared_elem + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_8[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_4[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_8 = 0; j_8 < 8; j_8++) {
                    acc[j_8] = acc[j_8] + _vec_load_8[j_8];
                }
                unsigned int acc_bf16_3[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_bf16_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&acc[_pair * 2])[0]), "=f"((&acc[_pair * 2])[1])
                        : "r"(acc_bf16_3[_pair]));
                }
            }
            #pragma unroll
            for (int j_9 = 0; j_9 < 8; j_9++) {
                acc[j_9] = ((acc[j_9] == 0.0f) ? 0.0f : acc[j_9]);
            }
            unsigned int acc_bf16_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                acc_bf16_4[_lp] = *(uint32_t*)&_bf2;
            }
            int access_0 = published_token * 896 + cluster_thread;
            long long slot = (long long)rank * (long long)total_access * 8 + (long long)access_0 * 8;
            reinterpret_cast<int4*>(own_ring + (((step & 1) * 224 + tid) * 4))[0] = reinterpret_cast<int4*>(acc_bf16_4)[0];
            if (rank == 0) {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer1 + slot), "r"((acc_bf16_4)[0]), "r"((acc_bf16_4)[1]), "r"((acc_bf16_4)[2]), "r"((acc_bf16_4)[3]) : "memory");
            } else {
                asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer0 + slot), "r"((acc_bf16_4)[0]), "r"((acc_bf16_4)[1]), "r"((acc_bf16_4)[2]), "r"((acc_bf16_4)[3]) : "memory");
            }
        }
        #pragma unroll 1
        for (int clear_step = 0; clear_step < clear_per_step; clear_step++) {
            if (clear_next >= clear_limit) {
                break;
            }
            {
                auto _gst_5 = workspace_local + (clear_base + (long long)clear_next * 8);
                __builtin_assume(__isGlobal(_gst_5));
                reinterpret_cast<int4*>(_gst_5)[0] = reinterpret_cast<int4*>(clear_words)[0];
            }
            clear_next += access_stride;
        }
        if (step == 1) {
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
        if (step >= 1) {
            int reduced_token = token_begin + (step - 1) * token_stride;
            unsigned int residual_words[4];
            unsigned int _vec_load_9[4];
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(residual + (access * 8) + 0);
                _vec_load_9[0 + 0] = _uv4_6.x;
                _vec_load_9[0 + 1] = _uv4_6.y;
                _vec_load_9[0 + 2] = _uv4_6.z;
                _vec_load_9[0 + 3] = _uv4_6.w;
            }
            #pragma unroll
            for (int word_1 = 0; word_1 < 4; word_1++) {
                residual_words[word_1] = _vec_load_9[word_1];
            }
            unsigned int rank_words[8];
            unsigned int own_words[4];
            reinterpret_cast<int4*>(own_words)[0] = reinterpret_cast<int4*>(own_ring + (((step - 1 & 1) * 224 + tid) * 4))[0];
            uint32_t peer_words[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(peer_words[0]), "=r"(peer_words[1]), "=r"(peer_words[2]), "=r"(peer_words[3]) : "l"(workspace_local + (data_base + (long long)(1 - rank) * (long long)(total_access * 8) + (long long)(access * 8))) : "memory");
            } while (
                (((peer_words[0] >> 0) & 0xffffu) == 0x8000u) || (((peer_words[0] >> 16) & 0xffffu) == 0x8000u) || (((peer_words[1] >> 0) & 0xffffu) == 0x8000u) || (((peer_words[1] >> 16) & 0xffffu) == 0x8000u) ||
                (((peer_words[2] >> 0) & 0xffffu) == 0x8000u) || (((peer_words[2] >> 16) & 0xffffu) == 0x8000u) || (((peer_words[3] >> 0) & 0xffffu) == 0x8000u) || (((peer_words[3] >> 16) & 0xffffu) == 0x8000u)
            );
            #pragma unroll
            for (int word_2 = 0; word_2 < 4; word_2++) {
                rank_words[word_2] = ((rank == 0) ? own_words[word_2] : peer_words[word_2]);
            }
            #pragma unroll
            for (int word_3 = 0; word_3 < 4; word_3++) {
                rank_words[4 + word_3] = ((rank == 0) ? peer_words[word_3] : own_words[word_3]);
            }
            float _unpack_bf16_2[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_2[_pair * 2])[0]), "=f"((&_unpack_bf16_2[_pair * 2])[1])
                    : "r"(rank_words[_pair]));
            }
            float sum_value[8];
            #pragma unroll
            for (int j_10 = 0; j_10 < 8; j_10++) {
                sum_value[j_10] = _unpack_bf16_2[j_10];
            }
            float _unpack_bf16_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_3[_pair * 2])[0]), "=f"((&_unpack_bf16_3[_pair * 2])[1])
                    : "r"(rank_words[4 + _pair]));
            }
            #pragma unroll
            for (int j_11 = 0; j_11 < 8; j_11++) {
                sum_value[j_11] = sum_value[j_11] + _unpack_bf16_3[j_11];
            }
            unsigned int sum_value_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16[_pair]));
            }
            int access_in_token = cluster_thread;
            int elem = access * 8;
            float _unpack_bf16_4[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                    : "r"(residual_words[_pair]));
            }
            float _vec_load_11[8];
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(norm_weight + (access_in_token * 8) + 0);
                uint4 _vld_7[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_7[_blk] = _vptr_7[_blk];
                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_11[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_7[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int j_12 = 0; j_12 < 8; j_12++) {
                _unpack_bf16_4[j_12] = _unpack_bf16_4[j_12] + sum_value[j_12];
            }
            unsigned int _unpack_bf16_4_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_unpack_bf16_4[_lp*2 + 0], _unpack_bf16_4[_lp*2+1 + 0]));
                _unpack_bf16_4_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_4[_pair * 2])[0]), "=f"((&_unpack_bf16_4[_pair * 2])[1])
                    : "r"(_unpack_bf16_4_bf16[_pair]));
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_unpack_bf16_4[_lp*2 + 0], _unpack_bf16_4[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_unpack_bf16_4[_lp*2 + 2], _unpack_bf16_4[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_unpack_bf16_4[_lp*2 + 4], _unpack_bf16_4[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_unpack_bf16_4[_lp*2 + 6], _unpack_bf16_4[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(residual_out + elem)[0] = reinterpret_cast<int4*>(_packed_0)[0];
            float square_sum = 0.0f;
            #pragma unroll
            for (int j_13 = 0; j_13 < 8; j_13++) {
                square_sum = __fmaf_rn(_unpack_bf16_4[j_13], _unpack_bf16_4[j_13], square_sum);
            }
            float _warp_reduce_8 = square_sum;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_8 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_8, offset);
            square_sum = _warp_reduce_8;
            if (lane == 0) {
                reduce_smem[warp] = square_sum;
            }
            __syncthreads();
            if (tid == 0) {
                float block_sum = reduce_smem[0] + reduce_smem[4] + (reduce_smem[2] + reduce_smem[6]) + (reduce_smem[1] + reduce_smem[5] + reduce_smem[3]);
                mbarrier_arrive_expect_tx(rms_bar_addr + (rms_parity) * 8, 16);
                int slot_addr = rms_slots_addr + (unsigned int)(rms_parity * 16) + (unsigned int)(cta_rank * 4);
                uint32_t _cluster_mapa_9;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_9) : "r"(slot_addr), "r"(0));
                uint32_t _mbar_peer_10;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_10) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(0));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_9), "f"(block_sum), "r"(_mbar_peer_10) : "memory");
                uint32_t _cluster_mapa_11;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_11) : "r"(slot_addr), "r"(1));
                uint32_t _mbar_peer_12;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_12) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(1));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_11), "f"(block_sum), "r"(_mbar_peer_12) : "memory");
                uint32_t _cluster_mapa_13;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_13) : "r"(slot_addr), "r"(2));
                uint32_t _mbar_peer_14;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_14) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(2));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_13), "f"(block_sum), "r"(_mbar_peer_14) : "memory");
                uint32_t _cluster_mapa_15;
                asm volatile(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_cluster_mapa_15) : "r"(slot_addr), "r"(3));
                uint32_t _mbar_peer_16;
                asm(
                    "mapa.shared::cluster.u32 %0, %1, %2;"
                    : "=r"(_mbar_peer_16) : "r"(rms_bar_addr + (rms_parity) * 8), "r"(3));
                asm volatile(
                    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                    :: "r"(_cluster_mapa_15), "f"(block_sum), "r"(_mbar_peer_16) : "memory");
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
            for (int j_14 = 0; j_14 < 8; j_14++) {
                norm_value[j_14] = _unpack_bf16_4[j_14] * rstd * (_vec_load_11[j_14] + weight_bias);
            }
            unsigned int norm_value_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
                norm_value_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(norm_out + elem)[0] = reinterpret_cast<int4*>(norm_value_bf16)[0];
            uint32_t _bf16x2_abs_0;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(norm_value_bf16[0]));
            uint32_t _bf16x2_abs_1;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(norm_value_bf16[1]));
            uint32_t _bf16x2_max_0;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
            uint32_t _bf16x2_abs_2;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(norm_value_bf16[2]));
            uint32_t _bf16x2_max_1;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_max_0), "r"(_bf16x2_abs_2));
            uint32_t _bf16x2_abs_3;
            asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(norm_value_bf16[3]));
            uint32_t _bf16x2_max_2;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_1), "r"(_bf16x2_abs_3));
            unsigned int _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, _bf16x2_max_2, 1);
            uint32_t _bf16x2_max_3;
            asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(_shfl_xor_sync_17), "r"(_bf16x2_max_2));
            uint16_t _bf16_max_0;
            asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(_bf16x2_max_3 & 65535)), "h"((uint16_t)(_bf16x2_max_3 >> 16)));
            float vector_max;
            asm("cvt.f32.bf16 %0, %1;" : "=f"(vector_max) : "h"((uint16_t)(_bf16_max_0)));
            float _unpack_bf16_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_unpack_bf16_5[_pair * 2])[0]), "=f"((&_unpack_bf16_5[_pair * 2])[1])
                    : "r"(norm_value_bf16[_pair]));
            }
            float _rcp_18 = approx_rcp(6.0f);
            float sf_value = scale_factor * (vector_max * _rcp_18);
            uint16_t _e4m3x2_20;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_20) : "f"(0.0f), "f"(sf_value));
            uint8_t _cvt_satfinite_e4m3_f32_19 = (uint8_t)_e4m3x2_20;
            uint32_t _f16x2_decode_21;
            float sf_rounded;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_21) : "h"(_e4m3x2_20));
            uint16_t _f16_decode_21 = (uint16_t)_f16x2_decode_21;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded) : "h"(_f16_decode_21));
            float _rcp_23 = approx_rcp(scale_factor);
            float _rcp_22 = approx_rcp(sf_rounded * _rcp_23);
            float output_scale = ((sf_rounded != 0.0f) ? _rcp_22 : 0.0f);
            #pragma unroll
            for (int j_15 = 0; j_15 < 8; j_15++) {
                _unpack_bf16_5[j_15] = _unpack_bf16_5[j_15] * output_scale;
            }
            unsigned int _fp4_0[1];
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(_unpack_bf16_5[0]), "f"(_unpack_bf16_5[1]), "f"(_unpack_bf16_5[2]), "f"(_unpack_bf16_5[3]), "f"(_unpack_bf16_5[4]), "f"(_unpack_bf16_5[5]), "f"(_unpack_bf16_5[6]), "f"(_unpack_bf16_5[7]));
            *(reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(quant_out)) + (access)) = _fp4_0[0];
            if (lane % 2 == 0) {
                int scale_col = access_in_token / 2;
                int inner_k = scale_col % 4;
                int k_tile = scale_col / 4;
                int inner_m = reduced_token % 128 / 32;
                int outer_m = reduced_token % 32;
                int m_tile = reduced_token / 128;
                int scale_index = m_tile * 112 * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k;
                uint16_t _cvt_satfinite_e4m3x2_f32_24;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_24) : "f"(0.0f), "f"(sf_value));
                *reinterpret_cast<unsigned char*>(reinterpret_cast<uint8_t*>(scale_out) + scale_index) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_24 & 127));
            }
            access += access_stride;
            rms_parity += 1;
            if (rms_parity == 2) { rms_parity = 0; _phase_rms_bar ^= 1; }
        }
    }
    if (interleaved_tokens == 0) {
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
    #pragma unroll 1
    for (int clear_access = clear_next; clear_access < clear_limit; clear_access += access_stride) {
        {
            auto _gst_25 = workspace_local + (clear_base + (long long)clear_access * 8);
            __builtin_assume(__isGlobal(_gst_25));
            reinterpret_cast<int4*>(_gst_25)[0] = reinterpret_cast<int4*>(clear_words)[0];
        }
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _sysv_wait_p_26 = reinterpret_cast<volatile int*>(reinterpret_cast<unsigned int*>(control_0)) + (0);
                while (*_sysv_wait_p_26 != static_cast<int>(num_bids)) {}
            }
            control_0[2] = (control_current_word + 1) % 3;
            control_0[4] = token_end * 7168 * 2;
            reinterpret_cast<unsigned int*>(control_0)[0] = 0;
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
