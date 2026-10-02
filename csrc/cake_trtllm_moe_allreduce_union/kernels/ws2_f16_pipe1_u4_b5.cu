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

#include "../cake_trtllm_moe_allreduce_union_common.cuh"

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

extern "C" {

__global__ __launch_bounds__(224, 5) __cluster_dims__(4,1,1) void
kernel_cake_trtllm_moe_allreduce_union_ws2_f16_pipe1_u4_b5(__half* __restrict__ active_expert_tokens, float* __restrict__ expert_scales, __half* __restrict__ token_input, __half* __restrict__ residual, __half* __restrict__ gamma, __half* __restrict__ moe_allreduce_out, __half* __restrict__ residual_out, __half* __restrict__ norm_out, __half* __restrict__ quant_out, __half* __restrict__ scale_out, long long* __restrict__ workspace_tensor, int* __restrict__ workspace_control, __half* __restrict__ workspace_payload_0, __half* __restrict__ workspace_payload_1, int world_rank, int tokens, int active_experts, float epsilon, float weight_bias, float scale_factor, int layout_code)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1030
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#else
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#endif

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
    long long control_address = workspace_tensor[6];
    int* control = reinterpret_cast<int*>(control_address);
    unsigned int* completion = reinterpret_cast<unsigned int*>(control_address);
    int* flag_addr = control + 2;
    int* comm_size_addr = control + 3;
    int* clear_addr = control + 4;
    long long comm_stride_elems = (long long)*comm_size_addr / 2;
    long long workspace_address = workspace_tensor[4 + rank];
    __half* workspace_local = reinterpret_cast<__half*>(workspace_address);
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
    unsigned int probe_words[8];
    #pragma unroll
    for (int word_1 = 0; word_1 < 8; word_1++) {
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
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_0[0 + _blk * 8 + _pair * 2]))
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
                uint32_t scaled_f16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                    scaled_f16[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                    uint32_t _f16x2_add_0;
                    asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_0) : "r"(packed_acc[pair_1]), "r"(scaled_f16[pair_1]));
                    packed_acc[pair_1] = _f16x2_add_0;
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
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_1[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_2[_pair]));
                    }
                }
            }
            uint32_t _vec_load_1_f16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_1[_lp*2 + 0], _vec_load_1[_lp*2+1 + 0]));
                _vec_load_1_f16[_lp] = *(uint32_t*)&_h2;
            }
            #pragma unroll
            for (int pair_2 = 0; pair_2 < 4; pair_2++) {
                uint32_t _f16x2_add_1;
                asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_1) : "r"(packed_acc[pair_2]), "r"(_vec_load_1_f16[pair_2]));
                packed_acc[pair_2] = _f16x2_add_1;
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
                    : "r"(packed_acc[_pair]));
            }
            #pragma unroll
            for (int j_2 = 0; j_2 < 8; j_2++) {
                acc[j_2] = ((acc[j_2] == 0.0f) ? 0.0f : acc[j_2]);
            }
            int access_1 = publish_token * 896 + cluster_thread;
            long long slot = (long long)data_epoch * comm_stride_elems + (long long)rank * (long long)total_access * 8 + (long long)access_1 * 8;
            {
                uint32_t acc_f16[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                    acc_f16[_lp] = *(uint32_t*)&_h2;
                }
                reinterpret_cast<int4*>(reinterpret_cast<__half*>(workspace_tensor[4]) + slot)[0] = reinterpret_cast<int4*>(acc_f16)[0];
                reinterpret_cast<int4*>(reinterpret_cast<__half*>(workspace_tensor[5]) + slot)[0] = reinterpret_cast<int4*>(acc_f16)[0];
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
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_2[0 + _blk * 8 + _pair * 2]))
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
                            ".reg .b16 h_lo, h_hi;\n\t"
                            ".reg .b32 f_lo, f_hi;\n\t"
                            "mov.b32 {h_lo, h_hi}, %1;\n\t"
                            "cvt.f32.f16 f_lo, h_lo;\n\t"
                            "cvt.f32.f16 f_hi, h_hi;\n\t"
                            "mov.b64 %0, {f_lo, f_hi};\n\t"
                            "}\n"
                            : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_3[0 + _blk * 8 + _pair * 2]))
                            : "r"(_vpairs_4[_pair]));
                    }
                }
            }
            int dividend = rank;
            int dividend_0 = rank + 1;
            unsigned int rank_words[8];
            uint32_t _sysv_poll_group_0[8];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(workspace_local + (candidate_data_base + (long long)(dividend % 2 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[4]), "=r"(_sysv_poll_group_0[5]), "=r"(_sysv_poll_group_0[6]), "=r"(_sysv_poll_group_0[7]) : "l"(workspace_local + (candidate_data_base + (long long)(dividend_0 % 2 * total_access * 8) + (long long)(access_2 * 8))) : "memory");
            } while ((((_sysv_poll_group_0[0] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[0] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[1] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[2] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[3] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[4] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[5] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[6] >> 16) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 0) & 0xffffu) == 0x8000u) || (((_sysv_poll_group_0[7] >> 16) & 0xffffu) == 0x8000u));
            #pragma unroll
            for (int word_2 = 0; word_2 < 8; word_2++) {
                rank_words[word_2] = _sysv_poll_group_0[word_2];
            }
            float sum_value[8];
            unsigned int sum_words[4];
            #pragma unroll
            for (int pair_3 = 0; pair_3 < 4; pair_3++) {
                sum_words[pair_3] = rank_words[pair_3];
            }
            #pragma unroll
            for (int pair_4 = 0; pair_4 < 4; pair_4++) {
                uint32_t _f16x2_add_2;
                asm("add.f16x2 %0, %1, %2;" : "=r"(_f16x2_add_2) : "r"(sum_words[pair_4]), "r"(rank_words[4 + pair_4]));
                sum_words[pair_4] = _f16x2_add_2;
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
                    : "r"(sum_words[_pair]));
            }
            {
                __half2 _pk[4];
                _pk[0] = __floats2half2_rn(sum_value[0 + 0], sum_value[0 + 1]);
                _pk[1] = __floats2half2_rn(sum_value[0 + 2], sum_value[0 + 3]);
                _pk[2] = __floats2half2_rn(sum_value[0 + 4], sum_value[0 + 5]);
                _pk[3] = __floats2half2_rn(sum_value[0 + 6], sum_value[0 + 7]);
                *reinterpret_cast<uint4*>(&((__half*)(moe_allreduce_out + elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            #pragma unroll
            for (int j_3 = 0; j_3 < 8; j_3++) {
                _vec_load_2[j_3] = _vec_load_2[j_3] + sum_value[j_3];
            }
            uint32_t _vec_load_2_f16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_vec_load_2[_lp*2 + 0], _vec_load_2[_lp*2+1 + 0]));
                _vec_load_2_f16[_lp] = *(uint32_t*)&_h2;
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&_vec_load_2[_pair * 2]))
                    : "r"(_vec_load_2_f16[_pair]));
            }
            {
                __half2 _pk[4];
                _pk[0] = __floats2half2_rn(_vec_load_2[0 + 0], _vec_load_2[0 + 1]);
                _pk[1] = __floats2half2_rn(_vec_load_2[0 + 2], _vec_load_2[0 + 3]);
                _pk[2] = __floats2half2_rn(_vec_load_2[0 + 4], _vec_load_2[0 + 5]);
                _pk[3] = __floats2half2_rn(_vec_load_2[0 + 6], _vec_load_2[0 + 7]);
                *reinterpret_cast<uint4*>(&((__half*)(residual_out + elem_1))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            float square_sum = 0.0f;
            #pragma unroll
            for (int j_4 = 0; j_4 < 8; j_4++) {
                square_sum = square_sum + _vec_load_2[j_4] * _vec_load_2[j_4];
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
            for (int j_5 = 0; j_5 < 8; j_5++) {
                norm_value[j_5] = _vec_load_2[j_5] * rstd * (_vec_load_3[j_5] + weight_bias);
            }
            uint32_t norm_value_f16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(norm_value[_lp*2 + 0], norm_value[_lp*2+1 + 0]));
                norm_value_f16[_lp] = *(uint32_t*)&_h2;
            }
            reinterpret_cast<int4*>(norm_out + elem_1)[0] = reinterpret_cast<int4*>(norm_value_f16)[0];
        }
    }
    if (bid == 0) {
        if (tid == 0) {
            {
                volatile int* _lcv_p_5 = reinterpret_cast<volatile int*>(completion) + (0);
                while (*_lcv_p_5 != static_cast<int>(num_bids)) {}
                *reinterpret_cast<int*>(flag_addr) = static_cast<int>((flag + 1) % 3);
                *reinterpret_cast<int*>(clear_addr) = static_cast<int>(tokens * 7168 * 2);
                *(reinterpret_cast<int*>(completion) + (0)) = 0;
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

} // extern "C"
