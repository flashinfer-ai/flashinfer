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
#define SMEM_REDUCE_SMEM_OFF 0
#define SMEM_REDUCE_SMEM_STAGE_BYTES 132
#define SMEM_REDUCE_SMEM_STRIDE 132
#define SMEM_RMS_SCALAR_OFF 132
#define SMEM_RMS_SCALAR_STAGE_BYTES 4
#define SMEM_RMS_SCALAR_STRIDE 4
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

__global__ __launch_bounds__(224) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_finalize_bfloat16_ws8_o110(__nv_bfloat16* __restrict__ allreduce_in, int* __restrict__ inverse_indices, __nv_bfloat16* __restrict__ expert_scales, __nv_bfloat16* __restrict__ shared_expert_output, __nv_bfloat16* __restrict__ residual, __nv_bfloat16* __restrict__ norm_weight, __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ norm_out, __nv_bfloat16* __restrict__ quant_out, __nv_bfloat16* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int world_rank, int tokens, int top_k, int has_shared_expert, float routed_scaling_factor, float epsilon, float weight_bias, float scale_factor)
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

    // === Task calls (dependency order) ===
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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int flag = *flag_addr;
    int clear_size = *clear_addr;
    int data_epoch = flag % 3;
    int clear_epoch = (flag + 2) % 3;
    long long data_base = (long long)data_epoch * comm_stride_elems;
    __nv_bfloat16* peer0 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[16]) + data_base;
    __nv_bfloat16* peer1 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[17]) + data_base;
    __nv_bfloat16* peer2 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[18]) + data_base;
    __nv_bfloat16* peer3 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[19]) + data_base;
    __nv_bfloat16* peer4 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[20]) + data_base;
    __nv_bfloat16* peer5 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[21]) + data_base;
    __nv_bfloat16* peer6 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[22]) + data_base;
    __nv_bfloat16* peer7 = reinterpret_cast<__nv_bfloat16*>(workspace_tensor[23]) + data_base;
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
        #pragma unroll 1
        for (int route = 0; route < route_end; route++) {
            int expanded_idx = token * top_k + route;
            int _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const int*>(inverse_indices + expanded_idx);
            }
            int permuted_idx = _vec_load_0[0];
            if (permuted_idx >= 0) {
                long long expert_elem = (long long)permuted_idx * 7168 + (long long)(cluster_thread * 8);
                float _vec_load_1[8];
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(allreduce_in + expert_elem + 0);
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
                float _vec_load_2[1];
                {
                    __nv_bfloat16 _bf16_2 = *reinterpret_cast<const __nv_bfloat16*>(expert_scales + expanded_idx);
                    _vec_load_2[0] = __bfloat162float(_bf16_2);
                }
                float route_scale = _vec_load_2[0];
                float scaled[8];
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    scaled[j_1] = _vec_load_1[j_1] * route_scale;
                }
                uint32_t scaled_bf16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float scaled_bf16_f32[8];
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&scaled_bf16_f32[_pair * 2])[0]), "=f"((&scaled_bf16_f32[_pair * 2])[1])
                        : "r"(scaled_bf16[_pair]));
                }
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    acc[j_2] = acc[j_2] + scaled_bf16_f32[j_2];
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
            }
        }
        {
            if (routed_scaling_factor != 1.0f) {
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    acc[j_3] = acc[j_3] * routed_scaling_factor;
                }
                uint32_t acc_bf16_1[4];
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
        }
        {
            if (has_shared_expert != 0) {
                int shared_elem = token * 7168 + cluster_thread * 8;
                float _vec_load_4[8];
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(shared_expert_output + shared_elem + 0);
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
                                : "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_4[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_3[_pair]));
                        }
                    }
                }
                #pragma unroll
                for (int j_4 = 0; j_4 < 8; j_4++) {
                    acc[j_4] = acc[j_4] + _vec_load_4[j_4];
                }
                uint32_t acc_bf16_2[4];
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
        }
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            acc[j_5] = ((acc[j_5] == 0.0f) ? 0.0f : acc[j_5]);
        }
        uint32_t acc_bf16_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
            acc_bf16_3[_lp] = *(uint32_t*)&_bf2;
        }
        int access = token * 896 + cluster_thread;
        long long slot = (long long)rank * (long long)total_access * 8 + (long long)access * 8;
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer0 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer1 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer2 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer3 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer4 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer5 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer6 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
        asm volatile("st.volatile.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer7 + slot), "r"(acc_bf16_3[0]), "r"(acc_bf16_3[1]), "r"(acc_bf16_3[2]), "r"(acc_bf16_3[3]) : "memory");
    }
    unsigned int clear_words[4];
    #pragma unroll
    for (int word = 0; word < 4; word++) {
        clear_words[word] = 2147516416;
    }
    long long clear_base = (long long)clear_epoch * comm_stride_elems;
    #pragma unroll 4
    for (int access_1 = first_access; access_1 < clear_size / 8; access_1 += access_stride) {
        reinterpret_cast<int4*>(workspace_local + (clear_base + (long long)access_1 * 8))[0] = reinterpret_cast<int4*>(clear_words)[0];
    }
    int access_2 = first_access;
    #pragma unroll 1
    for (int token_1 = token_begin; token_1 < token_end; token_1 += token_stride) {
        uint32_t _sysv_poll_group_0[32];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(peer0 + (access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(peer1 + (total_access * 8 + access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[8]), "=r"(_sysv_poll_group_0[9]), "=r"(_sysv_poll_group_0[10]), "=r"(_sysv_poll_group_0[11]) : "l"(peer2 + (2 * total_access * 8 + access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[12]), "=r"(_sysv_poll_group_0[13]), "=r"(_sysv_poll_group_0[14]), "=r"(_sysv_poll_group_0[15]) : "l"(peer3 + (3 * total_access * 8 + access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[16]), "=r"(_sysv_poll_group_0[17]), "=r"(_sysv_poll_group_0[18]), "=r"(_sysv_poll_group_0[19]) : "l"(peer4 + (4 * total_access * 8 + access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[20]), "=r"(_sysv_poll_group_0[21]), "=r"(_sysv_poll_group_0[22]), "=r"(_sysv_poll_group_0[23]) : "l"(peer5 + (5 * total_access * 8 + access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[24]), "=r"(_sysv_poll_group_0[25]), "=r"(_sysv_poll_group_0[26]), "=r"(_sysv_poll_group_0[27]) : "l"(peer6 + (6 * total_access * 8 + access_2 * 8)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[28]), "=r"(_sysv_poll_group_0[29]), "=r"(_sysv_poll_group_0[30]), "=r"(_sysv_poll_group_0[31]) : "l"(peer7 + (7 * total_access * 8 + access_2 * 8)) : "memory");
        } while ((((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[8] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[9] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[10] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[11] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[12] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[13] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[14] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[15] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[16] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[17] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[18] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[19] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[20] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[21] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[22] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[23] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[24] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[25] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[26] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[27] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[28] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[29] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[30] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[31] >> 16) & 0xffffu) == 0x8000u));
        float _sysv_poll_group_0_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[_pair]));
        }
        float sum_value[8];
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            sum_value[j_6] = _sysv_poll_group_0_f32[j_6];
        }
        float _sysv_poll_group_0_f32_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_0[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_0[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[4 + _pair]));
        }
        #pragma unroll
        for (int j_7 = 0; j_7 < 8; j_7++) {
            sum_value[j_7] = sum_value[j_7] + _sysv_poll_group_0_f32_0[j_7];
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
        float _sysv_poll_group_0_f32_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_1[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_1[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[8 + _pair]));
        }
        #pragma unroll
        for (int j_8 = 0; j_8 < 8; j_8++) {
            sum_value[j_8] = sum_value[j_8] + _sysv_poll_group_0_f32_1[j_8];
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
        float _sysv_poll_group_0_f32_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_3[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_3[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[12 + _pair]));
        }
        #pragma unroll
        for (int j_9 = 0; j_9 < 8; j_9++) {
            sum_value[j_9] = sum_value[j_9] + _sysv_poll_group_0_f32_3[j_9];
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
        float _sysv_poll_group_0_f32_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_5[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_5[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[16 + _pair]));
        }
        #pragma unroll
        for (int j_10 = 0; j_10 < 8; j_10++) {
            sum_value[j_10] = sum_value[j_10] + _sysv_poll_group_0_f32_5[j_10];
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
        float _sysv_poll_group_0_f32_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_7[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_7[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[20 + _pair]));
        }
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            sum_value[j_11] = sum_value[j_11] + _sysv_poll_group_0_f32_7[j_11];
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
        float _sysv_poll_group_0_f32_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_9[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_9[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[24 + _pair]));
        }
        #pragma unroll
        for (int j_12 = 0; j_12 < 8; j_12++) {
            sum_value[j_12] = sum_value[j_12] + _sysv_poll_group_0_f32_9[j_12];
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
        float _sysv_poll_group_0_f32_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_sysv_poll_group_0_f32_11[_pair * 2])[0]), "=f"((&_sysv_poll_group_0_f32_11[_pair * 2])[1])
                : "r"(_sysv_poll_group_0[28 + _pair]));
        }
        #pragma unroll
        for (int j_13 = 0; j_13 < 8; j_13++) {
            sum_value[j_13] = sum_value[j_13] + _sysv_poll_group_0_f32_11[j_13];
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
        int access_in_token = cluster_thread;
        int elem = access_2 * 8;
        float _vec_load_5[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(residual + elem + 0);
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
                        : "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_5[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_4[_pair]));
                }
            }
        }
        float _vec_load_6[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(norm_weight + (access_in_token * 8) + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_6[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
        }
        #pragma unroll
        for (int j_14 = 0; j_14 < 8; j_14++) {
            _vec_load_5[j_14] = _vec_load_5[j_14] + sum_value[j_14];
        }
        uint32_t _vec_load_5_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_vec_load_5[_lp*2 + 0], _vec_load_5[_lp*2+1 + 0]));
            _vec_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&_vec_load_5[_pair * 2])[0]), "=f"((&_vec_load_5[_pair * 2])[1])
                : "r"(_vec_load_5_bf16[_pair]));
        }
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(_vec_load_5[0 + 0], _vec_load_5[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(_vec_load_5[0 + 2], _vec_load_5[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(_vec_load_5[0 + 4], _vec_load_5[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(_vec_load_5[0 + 6], _vec_load_5[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(residual_out + elem))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
        float square_sum = 0.0f;
        #pragma unroll
        for (int j_15 = 0; j_15 < 8; j_15++) {
            square_sum = square_sum + _vec_load_5[j_15] * _vec_load_5[j_15];
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
            rms_scalar[0] = block_sum;
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        float cluster_sum = 0.0f;
        if (tid == 0) {
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(rms_scalar_addr), "r"(0));
            float _cluster_ld_0;
            asm volatile(
                "ld.shared::cluster.f32 %0, [%1];"
                : "=f"(_cluster_ld_0) : "r"(_mapa_0) : "memory");
            cluster_sum = cluster_sum + _cluster_ld_0;
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(rms_scalar_addr), "r"(1));
            float _cluster_ld_1;
            asm volatile(
                "ld.shared::cluster.f32 %0, [%1];"
                : "=f"(_cluster_ld_1) : "r"(_mapa_1) : "memory");
            cluster_sum = cluster_sum + _cluster_ld_1;
            uint32_t _mapa_2;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_2) : "r"(rms_scalar_addr), "r"(2));
            float _cluster_ld_2;
            asm volatile(
                "ld.shared::cluster.f32 %0, [%1];"
                : "=f"(_cluster_ld_2) : "r"(_mapa_2) : "memory");
            cluster_sum = cluster_sum + _cluster_ld_2;
            uint32_t _mapa_3;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_3) : "r"(rms_scalar_addr), "r"(3));
            float _cluster_ld_3;
            asm volatile(
                "ld.shared::cluster.f32 %0, [%1];"
                : "=f"(_cluster_ld_3) : "r"(_mapa_3) : "memory");
            cluster_sum = cluster_sum + _cluster_ld_3;
        }
        asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        if (tid == 0) {
            float _rsqrt_0 = rsqrtf(cluster_sum / 7168.0f + epsilon);
            rms_scalar[0] = _rsqrt_0;
        }
        __syncthreads();
        float rstd = rms_scalar[0];
        float norm_value[8];
        {
            #pragma unroll
            for (int j_16 = 0; j_16 < 8; j_16++) {
                norm_value[j_16] = _vec_load_5[j_16] * rstd * (_vec_load_6[j_16] + weight_bias);
            }
        }
        uint32_t norm_value_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
            norm_value_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(norm_out + elem)[0] = reinterpret_cast<int4*>(norm_value_bf16)[0];
        access_2 += access_stride;
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _lcv_p_6 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_lcv_p_6 != static_cast<int>(num_bids)) {}
                *reinterpret_cast<int*>(flag_addr) = static_cast<int>((flag + 1) % 3);
                *reinterpret_cast<int*>(clear_addr) = static_cast<int>(token_end * 7168 * 8);
                *(reinterpret_cast<int*>(completion) + (0)) = 0;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
