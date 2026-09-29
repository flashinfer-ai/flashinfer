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
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 132
#define SMEM_REDUCE_SMEM_STRIDE 132
#define SMEM_RMS_SCALAR_OFF 132
#define SMEM_RMS_SCALAR_STAGE_BYTES 4
#define SMEM_RMS_SCALAR_STRIDE 4
#define SMEM_RMS_PARTIALS_OFF 136
#define SMEM_RMS_PARTIALS_STAGE_BYTES 32
#define SMEM_RMS_PARTIALS_STRIDE 32
#define SMEM_TOTAL 256
#define THREADS 224

#include <math_constants.h>

__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
}

extern "C" {

__global__ __launch_bounds__(224, 5) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_allreduce_union_02f275d6fcf10cd4ade5(__nv_bfloat16* __restrict__ active_expert_tokens, float* __restrict__ expert_scales, __nv_bfloat16* __restrict__ token_input, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ gamma, __nv_bfloat16* __restrict__ moe_allreduce_out, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int* __restrict__ workspace_control, __nv_bfloat16* __restrict__ workspace_payload_0, __nv_bfloat16* __restrict__ workspace_payload_1, __nv_bfloat16* __restrict__ workspace_payload_2, __nv_bfloat16* __restrict__ workspace_payload_3, __nv_bfloat16* __restrict__ workspace_payload_4, __nv_bfloat16* __restrict__ workspace_payload_5, __nv_bfloat16* __restrict__ workspace_payload_6, __nv_bfloat16* __restrict__ workspace_payload_7, int world_rank, int tokens, int active_experts, float epsilon, float weight_bias, float scale_factor, int layout_code)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

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
    float* rms_partials = reinterpret_cast<float*>(smem_raw + 136);
    const int rms_partials_addr = smem + 136;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int rank = world_rank;
    long long control_address = workspace_tensor[24];
    int* control = reinterpret_cast<int*>(control_address);
    unsigned int* completion = reinterpret_cast<unsigned int*>(control_address);
    int* flag_addr = control + 2;
    int* comm_size_addr = control + 3;
    int* clear_addr = control + 4;
    long long comm_stride_elems = (long long)*comm_size_addr / 2;
    long long workspace_address = workspace_tensor[16 + rank];
    __nv_bfloat16* workspace_local = reinterpret_cast<__nv_bfloat16*>(workspace_address);
    int flag = *flag_addr;
    int clear_size = *clear_addr;
    int data_epoch = flag % 3;
    int clear_epoch = (flag + 2) % 3;
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
    int total_access = tokens * 896;
    int first_access = cluster_id * 896 + (unsigned int)cluster_thread;
    unsigned int clear_words[4];
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        clear_words[word] = 2147516416;
    }
    long long clear_base = (long long)clear_epoch * comm_stride_elems;
    #pragma unroll 4
    for (int access = first_access; access < clear_size / 8; access += access_stride) {
        {
            reinterpret_cast<int4*>(workspace_local + (clear_base + (long long)access * 8))[0] = reinterpret_cast<int4*>(clear_words)[0];
        }
    }
    int lane_access = cluster_thread;
    int rms_state[1];
    rms_state[0] = 0;
    int probe_valid[1];
    probe_valid[0] = 0;
    unsigned int probe_words[32];
    #pragma unroll
    for (int word_1 = 0; word_1 < 32; word_1++) {
        probe_words[word_1] = 2147516416;
    }
    long long candidate_data_base = (long long)data_epoch * comm_stride_elems;
    int local_tokens = ((unsigned int)tokens - cluster_id + num_clusters - 1) / num_clusters;
    #pragma unroll 1
    for (int step = 0; step < local_tokens + 1; step++) {
        if (local_tokens > step) {
            int publish_token = cluster_id + (unsigned int)step * num_clusters;
            int elem = publish_token * 7168 + cluster_thread * 8;
            float acc[8];
            unsigned int packed_acc[4];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                acc[j] = 0.0f;
            }
            #pragma unroll
            for (int pair = 0; pair < 4; pair++) {
                packed_acc[pair] = 0;
            }
            #pragma unroll 4
            for (int expert = 0; expert < active_experts; expert++) {
                long long expert_elem = (long long)expert * (long long)tokens * 7168 + (long long)elem;
                float _vec_load_0[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(active_expert_tokens + expert_elem + 0);
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
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                }
                float expert_scale = expert_scales[expert * tokens + publish_token];
                float scaled[8];
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    scaled[j_1] = _vec_load_0[j_1] * expert_scale;
                }
                uint32_t scaled_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                    uint32_t _bf16x2_add_0;
                    asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(packed_acc[pair_1]), "r"(scaled_bf16[pair_1]));
                    packed_acc[pair_1] = _bf16x2_add_0;
                }
            }
            float _vec_load_1[8];
            {
                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(token_input + elem + 0);
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
                            : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_2[_pair]));
                    }
                }
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&acc[_pair * 2])[0]), "=f"((&acc[_pair * 2])[1])
                    : "r"(packed_acc[_pair]));
            }
            #pragma unroll
            for (int j_2 = 0; j_2 < 8; j_2++) {
                acc[j_2] = acc[j_2] + _vec_load_1[j_2];
            }
            uint32_t acc_bf16[4];
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
            #pragma unroll
            for (int j_3 = 0; j_3 < 8; j_3++) {
                acc[j_3] = ((acc[j_3] == 0.0f) ? 0.0f : acc[j_3]);
            }
            int access_1 = publish_token * 896 + cluster_thread;
            long long slot = (long long)data_epoch * comm_stride_elems + (long long)rank * (long long)total_access * 8 + (long long)access_1 * 8;
            {
                uint32_t acc_bf16_0[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_bf16_0[_lp] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[16]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[17]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[18]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[19]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[20]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[21]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[22]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(workspace_tensor[23]) + slot)[0] = reinterpret_cast<int4*>(acc_bf16_0)[0];
            }
        }
        if (step >= 1) {
            int consume_token = cluster_id + (unsigned int)(step - 1) * num_clusters;
            int access_2 = consume_token * 896 + lane_access;
            int elem_1 = access_2 * 8;
            float _vec_load_2[8];
            {
                const uint4* _vptr_3 = reinterpret_cast<const uint4*>(residual + elem_1 + 0);
                uint4 _vld_3[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_3[_blk] = _vptr_3[_blk];
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_3[_pair]));
                    }
                }
            }
            float _vec_load_3[8];
            {
                const uint4* _vptr_4 = reinterpret_cast<const uint4*>(gamma + (lane_access * 8) + 0);
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
                            : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_4[_pair]));
                    }
                }
            }
            unsigned int rank_words[32];
            uint32_t _sysv_poll_group_0[32];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(workspace_local + (candidate_data_base + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(workspace_local + (candidate_data_base + (long long)(total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(workspace_local + (candidate_data_base + (long long)(2 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(workspace_local + (candidate_data_base + (long long)(3 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(workspace_local + (candidate_data_base + (long long)(4 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(workspace_local + (candidate_data_base + (long long)(5 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(workspace_local + (candidate_data_base + (long long)(6 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(workspace_local + (candidate_data_base + (long long)(7 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
            } while ((((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 16) & 0xffffu) == 0x8000u));
            #pragma unroll
            for (int word_2 = 0; word_2 < 32; word_2++) {
                rank_words[word_2] = _sysv_poll_group_0[word_2];
            }
            float sum_value[8];
            unsigned int sum_words[4];
            float rank_words_f32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32[_pair * 2])[0]), "=f"((&rank_words_f32[_pair * 2])[1])
                    : "r"(rank_words[_pair]));
            }
            #pragma unroll
            for (int j_4 = 0; j_4 < 8; j_4++) {
                sum_value[j_4] = rank_words_f32[j_4];
            }
            float rank_words_f32_0[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_0[_pair * 2])[0]), "=f"((&rank_words_f32_0[_pair * 2])[1])
                    : "r"(rank_words[4 + _pair]));
            }
            #pragma unroll
            for (int j_5 = 0; j_5 < 8; j_5++) {
                sum_value[j_5] = sum_value[j_5] + rank_words_f32_0[j_5];
            }
            uint32_t sum_value_bf16[4];
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
            float rank_words_f32_1[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_1[_pair * 2])[0]), "=f"((&rank_words_f32_1[_pair * 2])[1])
                    : "r"(rank_words[8 + _pair]));
            }
            #pragma unroll
            for (int j_6 = 0; j_6 < 8; j_6++) {
                sum_value[j_6] = sum_value[j_6] + rank_words_f32_1[j_6];
            }
            uint32_t sum_value_bf16_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16_2[_pair]));
            }
            float rank_words_f32_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_3[_pair * 2])[0]), "=f"((&rank_words_f32_3[_pair * 2])[1])
                    : "r"(rank_words[12 + _pair]));
            }
            #pragma unroll
            for (int j_7 = 0; j_7 < 8; j_7++) {
                sum_value[j_7] = sum_value[j_7] + rank_words_f32_3[j_7];
            }
            uint32_t sum_value_bf16_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16_4[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16_4[_pair]));
            }
            float rank_words_f32_5[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_5[_pair * 2])[0]), "=f"((&rank_words_f32_5[_pair * 2])[1])
                    : "r"(rank_words[16 + _pair]));
            }
            #pragma unroll
            for (int j_8 = 0; j_8 < 8; j_8++) {
                sum_value[j_8] = sum_value[j_8] + rank_words_f32_5[j_8];
            }
            uint32_t sum_value_bf16_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16_6[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16_6[_pair]));
            }
            float rank_words_f32_7[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_7[_pair * 2])[0]), "=f"((&rank_words_f32_7[_pair * 2])[1])
                    : "r"(rank_words[20 + _pair]));
            }
            #pragma unroll
            for (int j_9 = 0; j_9 < 8; j_9++) {
                sum_value[j_9] = sum_value[j_9] + rank_words_f32_7[j_9];
            }
            uint32_t sum_value_bf16_8[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16_8[_pair]));
            }
            float rank_words_f32_9[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_9[_pair * 2])[0]), "=f"((&rank_words_f32_9[_pair * 2])[1])
                    : "r"(rank_words[24 + _pair]));
            }
            #pragma unroll
            for (int j_10 = 0; j_10 < 8; j_10++) {
                sum_value[j_10] = sum_value[j_10] + rank_words_f32_9[j_10];
            }
            uint32_t sum_value_bf16_10[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16_10[_pair]));
            }
            float rank_words_f32_11[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rank_words_f32_11[_pair * 2])[0]), "=f"((&rank_words_f32_11[_pair * 2])[1])
                    : "r"(rank_words[28 + _pair]));
            }
            #pragma unroll
            for (int j_11 = 0; j_11 < 8; j_11++) {
                sum_value[j_11] = sum_value[j_11] + rank_words_f32_11[j_11];
            }
            uint32_t sum_value_bf16_12[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sum_value[_lp*2 + 0], sum_value[_lp*2+1 + 0]));
                sum_value_bf16_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&sum_value[_pair * 2])[0]), "=f"((&sum_value[_pair * 2])[1])
                    : "r"(sum_value_bf16_12[_pair]));
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(sum_value[0 + 0], sum_value[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(sum_value[0 + 2], sum_value[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(sum_value[0 + 4], sum_value[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(sum_value[0 + 6], sum_value[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(moe_allreduce_out + elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            #pragma unroll
            for (int j_12 = 0; j_12 < 8; j_12++) {
                _vec_load_2[j_12] = _vec_load_2[j_12] + sum_value[j_12];
            }
            uint32_t _vec_load_2_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_2[_lp*2 + 0], _vec_load_2[_lp*2+1 + 0]));
                _vec_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_2[_pair * 2])[0]), "=f"((&_vec_load_2[_pair * 2])[1])
                    : "r"(_vec_load_2_bf16[_pair]));
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_vec_load_2[0 + 0], _vec_load_2[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_vec_load_2[0 + 4], _vec_load_2[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_vec_load_2[0 + 6], _vec_load_2[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(residual_out + elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            float square_sum = 0.0f;
            #pragma unroll
            for (int j_13 = 0; j_13 < 8; j_13++) {
                square_sum = square_sum + _vec_load_2[j_13] * _vec_load_2[j_13];
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
            float rstd = 0.0f;
            {
                int slot_base = rms_state[0] * 4;
                if (tid < 4) {
                    uint32_t _mapa_0;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mapa_0) : "r"(rms_partials_addr), "r"(tid));
                    asm volatile(
                        "st.shared::cluster.f32 [%0], %1;"
                        :: "r"(_mapa_0 + (unsigned int)((slot_base + cta_rank) * 4)), "f"(block_sum) : "memory");
                }
                asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
                asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
                float cluster_sum = 0.0f;
                cluster_sum = cluster_sum + rms_partials[slot_base];
                cluster_sum = cluster_sum + rms_partials[slot_base + 1];
                cluster_sum = cluster_sum + rms_partials[slot_base + 2];
                cluster_sum = cluster_sum + rms_partials[slot_base + 3];
                float _rsqrt_0 = rsqrtf(cluster_sum / 7168.0f + epsilon);
                rstd = _rsqrt_0;
                rms_state[0] = 1 - rms_state[0];
            }
            float norm_value[8];
            #pragma unroll
            for (int j_14 = 0; j_14 < 8; j_14++) {
                norm_value[j_14] = _vec_load_2[j_14] * rstd * (_vec_load_3[j_14] + weight_bias);
            }
            uint32_t norm_value_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
                norm_value_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(norm_out + elem_1)[0] = reinterpret_cast<int4*>(norm_value_bf16)[0];
        }
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _lcv_p_5 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_lcv_p_5 != static_cast<int>(num_bids)) {}
                *reinterpret_cast<int*>(flag_addr) = static_cast<int>((flag + 1) % 3);
                *reinterpret_cast<int*>(clear_addr) = static_cast<int>(tokens * 7168 * 8);
                *(reinterpret_cast<int*>(completion) + (0)) = 0;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
