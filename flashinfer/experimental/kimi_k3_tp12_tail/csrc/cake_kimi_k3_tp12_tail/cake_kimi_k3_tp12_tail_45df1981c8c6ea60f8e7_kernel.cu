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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_kimi_k3_tp12_tail_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_REDUCE_SCRATCH_OFF 0
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 128
#define SMEM_REDUCE_SCRATCH_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 448

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_45df1981c8c6ea60f8e7(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ y_out, __nv_bfloat16* __restrict__ gamma, long long* __restrict__ peer_ptrs, __nv_bfloat16* __restrict__ mcast_ptr, unsigned int* __restrict__ buffer_flags, int num_tokens, int rank, float epsilon)
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
    float* reduce_scratch = reinterpret_cast<float*>(smem_raw + 0);
    const int reduce_scratch_addr = smem + 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int token = blockIdx.x;
    int element = tid * 8;
    int thread_offset = token * 3584 + element;
    int dest_rank = token % 12;
    int dest_token = token / 12;
    unsigned int _vec_load_0[4];
    {
        uint4 _uv4_0 = *reinterpret_cast<const uint4*>(buffer_flags + 0);
        _vec_load_0[0 + 0] = _uv4_0.x;
        _vec_load_0[0 + 1] = _uv4_0.y;
        _vec_load_0[0 + 2] = _uv4_0.z;
        _vec_load_0[0 + 3] = _uv4_0.w;
    }
    unsigned int _vec_load_1[4];
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(buffer_flags + 4);
        _vec_load_1[0 + 0] = _uv4_1.x;
        _vec_load_1[0 + 1] = _uv4_1.y;
        _vec_load_1[0 + 2] = _uv4_1.z;
        _vec_load_1[0 + 3] = _uv4_1.w;
    }
    unsigned int dirty_num_stages = _vec_load_0[3];
    unsigned int dirty_stage_bytes = 0;
    if (dirty_num_stages > 0) {
        dirty_stage_bytes = _vec_load_0[2] / dirty_num_stages;
    }
    unsigned long long current_epoch_elements = (unsigned long long)_vec_load_0[0] * (unsigned long long)_vec_load_0[2] / 2;
    unsigned long long stage_elements = (unsigned long long)_vec_load_0[2] / 2 / 2;
    uint32_t _packed16_sanitize_0[4];
    {
        uint4 _p16_packet_2 = *reinterpret_cast<const uint4*>(routed + (thread_offset));
        uint32_t _p16_word_2_0 = _p16_packet_2.x;
        if ((_p16_word_2_0 & 0x0000ffffu) == 0x00008000u) _p16_word_2_0 &= 0xffff0000u;
        if ((_p16_word_2_0 & 0xffff0000u) == 0x80000000u) _p16_word_2_0 &= 0x0000ffffu;
        _packed16_sanitize_0[0] = _p16_word_2_0;
        uint32_t _p16_word_2_1 = _p16_packet_2.y;
        if ((_p16_word_2_1 & 0x0000ffffu) == 0x00008000u) _p16_word_2_1 &= 0xffff0000u;
        if ((_p16_word_2_1 & 0xffff0000u) == 0x80000000u) _p16_word_2_1 &= 0x0000ffffu;
        _packed16_sanitize_0[1] = _p16_word_2_1;
        uint32_t _p16_word_2_2 = _p16_packet_2.z;
        if ((_p16_word_2_2 & 0x0000ffffu) == 0x00008000u) _p16_word_2_2 &= 0xffff0000u;
        if ((_p16_word_2_2 & 0xffff0000u) == 0x80000000u) _p16_word_2_2 &= 0x0000ffffu;
        _packed16_sanitize_0[2] = _p16_word_2_2;
        uint32_t _p16_word_2_3 = _p16_packet_2.w;
        if ((_p16_word_2_3 & 0x0000ffffu) == 0x00008000u) _p16_word_2_3 &= 0xffff0000u;
        if ((_p16_word_2_3 & 0xffff0000u) == 0x80000000u) _p16_word_2_3 &= 0x0000ffffu;
        _packed16_sanitize_0[3] = _p16_word_2_3;
    }
    unsigned long long scatter_index = ((unsigned long long)dest_token * 12 + (unsigned long long)rank) * 3584 + (unsigned long long)element;
    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[dest_rank]) + (current_epoch_elements + scatter_index))[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    {
        unsigned char* _mlc_base_3 = reinterpret_cast<unsigned char*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
        uint32_t _mlc_dirty_3 = static_cast<uint32_t>(_vec_load_0[1]);
        uint32_t _mlc_stage_bytes_3 = static_cast<uint32_t>(dirty_stage_bytes);
        uint32_t _mlc_stages_3 = static_cast<uint32_t>(dirty_num_stages);
        uint32_t _mlc_clear_3[4] = {static_cast<uint32_t>(_vec_load_1[0]), static_cast<uint32_t>(_vec_load_1[1]), static_cast<uint32_t>(_vec_load_1[2]), static_cast<uint32_t>(_vec_load_1[3])};
        uint32_t _mlc_global_cta_3 = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t _mlc_global_tid_3 = _mlc_global_cta_3 * blockDim.x + threadIdx.x;
        uint32_t _mlc_num_threads_3 = gridDim.x * gridDim.y * blockDim.x;
        uint4 _mlc_init_3 = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};
        if (0u < _mlc_stages_3) {
            size_t _mlc_stage_offset = static_cast<size_t>((static_cast<uint8_t>(_mlc_dirty_3) * _mlc_stages_3 + static_cast<uint8_t>(0u)) * static_cast<size_t>(_mlc_stage_bytes_3));
            uint32_t _mlc_boundary = (_mlc_clear_3[0u] + 15u) / 16u;
            for (uint32_t _mlc_packed = _mlc_global_tid_3; _mlc_packed < _mlc_boundary; _mlc_packed += _mlc_num_threads_3) {
                reinterpret_cast<uint4*>(_mlc_base_3 + _mlc_stage_offset)[_mlc_packed] = _mlc_init_3;
            }
        }
    }
    bool is_owner = dest_rank == rank;
    unsigned int reduced_words[4];
    float thread_sum = 0.0f;
    if (is_owner) {
        uint32_t _mnnvl_twoshot_reduce_0[4];
        {
            float2 _mtr_accum_4[4];
            bool _mtr_valid_4;
            do {
                _mtr_valid_4 = true;
                #pragma unroll
                for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) _mtr_accum_4[_mtr_pair] = make_float2(0.0f, 0.0f);
                uint32_t _mtr_rank_4_0[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_0[0]), "=r"(_mtr_rank_4_0[1]), "=r"(_mtr_rank_4_0[2]), "=r"(_mtr_rank_4_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + (unsigned long long)dest_token * 12 * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_1[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_1[0]), "=r"(_mtr_rank_4_1[1]), "=r"(_mtr_rank_4_1[2]), "=r"(_mtr_rank_4_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 1) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_2[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_2[0]), "=r"(_mtr_rank_4_2[1]), "=r"(_mtr_rank_4_2[2]), "=r"(_mtr_rank_4_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 2) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_3[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_3[0]), "=r"(_mtr_rank_4_3[1]), "=r"(_mtr_rank_4_3[2]), "=r"(_mtr_rank_4_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 3) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_4[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_4[0]), "=r"(_mtr_rank_4_4[1]), "=r"(_mtr_rank_4_4[2]), "=r"(_mtr_rank_4_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 4) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_5[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_5[0]), "=r"(_mtr_rank_4_5[1]), "=r"(_mtr_rank_4_5[2]), "=r"(_mtr_rank_4_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 5) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_6[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_6[0]), "=r"(_mtr_rank_4_6[1]), "=r"(_mtr_rank_4_6[2]), "=r"(_mtr_rank_4_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 6) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_7[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_7[0]), "=r"(_mtr_rank_4_7[1]), "=r"(_mtr_rank_4_7[2]), "=r"(_mtr_rank_4_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 7) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_8[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_8[0]), "=r"(_mtr_rank_4_8[1]), "=r"(_mtr_rank_4_8[2]), "=r"(_mtr_rank_4_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 8) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_9[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_9[0]), "=r"(_mtr_rank_4_9[1]), "=r"(_mtr_rank_4_9[2]), "=r"(_mtr_rank_4_9[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 9) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_10[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_10[0]), "=r"(_mtr_rank_4_10[1]), "=r"(_mtr_rank_4_10[2]), "=r"(_mtr_rank_4_10[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 10) * 3584 + (unsigned long long)element)) : "memory");
                uint32_t _mtr_rank_4_11[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_11[0]), "=r"(_mtr_rank_4_11[1]), "=r"(_mtr_rank_4_11[2]), "=r"(_mtr_rank_4_11[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)dest_token * 12 + 11) * 3584 + (unsigned long long)element)) : "memory");
                bool _mtr_dirty_4_0 = false;
                _mtr_dirty_4_0 |= (_mtr_rank_4_0[0] == 0x80000000u);
                _mtr_dirty_4_0 |= (_mtr_rank_4_0[1] == 0x80000000u);
                _mtr_dirty_4_0 |= (_mtr_rank_4_0[2] == 0x80000000u);
                _mtr_dirty_4_0 |= (_mtr_rank_4_0[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_0;
                bool _mtr_dirty_4_1 = false;
                _mtr_dirty_4_1 |= (_mtr_rank_4_1[0] == 0x80000000u);
                _mtr_dirty_4_1 |= (_mtr_rank_4_1[1] == 0x80000000u);
                _mtr_dirty_4_1 |= (_mtr_rank_4_1[2] == 0x80000000u);
                _mtr_dirty_4_1 |= (_mtr_rank_4_1[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_1;
                bool _mtr_dirty_4_2 = false;
                _mtr_dirty_4_2 |= (_mtr_rank_4_2[0] == 0x80000000u);
                _mtr_dirty_4_2 |= (_mtr_rank_4_2[1] == 0x80000000u);
                _mtr_dirty_4_2 |= (_mtr_rank_4_2[2] == 0x80000000u);
                _mtr_dirty_4_2 |= (_mtr_rank_4_2[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_2;
                bool _mtr_dirty_4_3 = false;
                _mtr_dirty_4_3 |= (_mtr_rank_4_3[0] == 0x80000000u);
                _mtr_dirty_4_3 |= (_mtr_rank_4_3[1] == 0x80000000u);
                _mtr_dirty_4_3 |= (_mtr_rank_4_3[2] == 0x80000000u);
                _mtr_dirty_4_3 |= (_mtr_rank_4_3[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_3;
                bool _mtr_dirty_4_4 = false;
                _mtr_dirty_4_4 |= (_mtr_rank_4_4[0] == 0x80000000u);
                _mtr_dirty_4_4 |= (_mtr_rank_4_4[1] == 0x80000000u);
                _mtr_dirty_4_4 |= (_mtr_rank_4_4[2] == 0x80000000u);
                _mtr_dirty_4_4 |= (_mtr_rank_4_4[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_4;
                bool _mtr_dirty_4_5 = false;
                _mtr_dirty_4_5 |= (_mtr_rank_4_5[0] == 0x80000000u);
                _mtr_dirty_4_5 |= (_mtr_rank_4_5[1] == 0x80000000u);
                _mtr_dirty_4_5 |= (_mtr_rank_4_5[2] == 0x80000000u);
                _mtr_dirty_4_5 |= (_mtr_rank_4_5[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_5;
                bool _mtr_dirty_4_6 = false;
                _mtr_dirty_4_6 |= (_mtr_rank_4_6[0] == 0x80000000u);
                _mtr_dirty_4_6 |= (_mtr_rank_4_6[1] == 0x80000000u);
                _mtr_dirty_4_6 |= (_mtr_rank_4_6[2] == 0x80000000u);
                _mtr_dirty_4_6 |= (_mtr_rank_4_6[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_6;
                bool _mtr_dirty_4_7 = false;
                _mtr_dirty_4_7 |= (_mtr_rank_4_7[0] == 0x80000000u);
                _mtr_dirty_4_7 |= (_mtr_rank_4_7[1] == 0x80000000u);
                _mtr_dirty_4_7 |= (_mtr_rank_4_7[2] == 0x80000000u);
                _mtr_dirty_4_7 |= (_mtr_rank_4_7[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_7;
                bool _mtr_dirty_4_8 = false;
                _mtr_dirty_4_8 |= (_mtr_rank_4_8[0] == 0x80000000u);
                _mtr_dirty_4_8 |= (_mtr_rank_4_8[1] == 0x80000000u);
                _mtr_dirty_4_8 |= (_mtr_rank_4_8[2] == 0x80000000u);
                _mtr_dirty_4_8 |= (_mtr_rank_4_8[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_8;
                bool _mtr_dirty_4_9 = false;
                _mtr_dirty_4_9 |= (_mtr_rank_4_9[0] == 0x80000000u);
                _mtr_dirty_4_9 |= (_mtr_rank_4_9[1] == 0x80000000u);
                _mtr_dirty_4_9 |= (_mtr_rank_4_9[2] == 0x80000000u);
                _mtr_dirty_4_9 |= (_mtr_rank_4_9[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_9;
                bool _mtr_dirty_4_10 = false;
                _mtr_dirty_4_10 |= (_mtr_rank_4_10[0] == 0x80000000u);
                _mtr_dirty_4_10 |= (_mtr_rank_4_10[1] == 0x80000000u);
                _mtr_dirty_4_10 |= (_mtr_rank_4_10[2] == 0x80000000u);
                _mtr_dirty_4_10 |= (_mtr_rank_4_10[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_10;
                bool _mtr_dirty_4_11 = false;
                _mtr_dirty_4_11 |= (_mtr_rank_4_11[0] == 0x80000000u);
                _mtr_dirty_4_11 |= (_mtr_rank_4_11[1] == 0x80000000u);
                _mtr_dirty_4_11 |= (_mtr_rank_4_11[2] == 0x80000000u);
                _mtr_dirty_4_11 |= (_mtr_rank_4_11[3] == 0x80000000u);
                _mtr_valid_4 &= !_mtr_dirty_4_11;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_0[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_0[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_0[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_0[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_1[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_1[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_1[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_1[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_2[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_2[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_2[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_2[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_3[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_3[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_3[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_3[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_4[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_4[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_4[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_4[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_5[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_5[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_5[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_5[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_6[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_6[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_6[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_6[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_7[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_7[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_7[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_7[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_8[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_8[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_8[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_8[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_9[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_9[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_9[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_9[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_10[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_10[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_10[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_10[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[0]).x), "+f"((_mtr_accum_4[0]).y) : "r"(_mtr_rank_4_11[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[1]).x), "+f"((_mtr_accum_4[1]).y) : "r"(_mtr_rank_4_11[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[2]).x), "+f"((_mtr_accum_4[2]).y) : "r"(_mtr_rank_4_11[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mtr_accum_4[3]).x), "+f"((_mtr_accum_4[3]).y) : "r"(_mtr_rank_4_11[3]));
            } while (!_mtr_valid_4);
            #pragma unroll
            for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) {
                __nv_bfloat162 _mtr_out = __float22bfloat162_rn(_mtr_accum_4[_mtr_pair]);
                _mnnvl_twoshot_reduce_0[_mtr_pair] = *reinterpret_cast<uint32_t*>(&_mtr_out);
            }
        }
        for (int word = 0; word < 4; word++) {
            reduced_words[word] = _mnnvl_twoshot_reduce_0[word];
        }
        float reduced_words_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&reduced_words_f32[_pair * 2])[0]), "=f"((&reduced_words_f32[_pair * 2])[1])
                : "r"(reduced_words[_pair]));
        }
        for (int lane_1 = 0; lane_1 < 8; lane_1++) {
            thread_sum = thread_sum + reduced_words_f32[lane_1] * reduced_words_f32[lane_1];
        }
    }
    // BlockReduceF32: threads=448, warps=14, all-thread broadcast
    float _block_reduce_f32_5_accum = thread_sum;
    const int _block_reduce_f32_5_lane = threadIdx.x & 31;
    const int _block_reduce_f32_5_warp = threadIdx.x >> 5;
    // Full CTA schedule: one barrier; every warp performs level two.
    #pragma unroll
    for (int _block_reduce_f32_5_offset_level1 = 16; _block_reduce_f32_5_offset_level1 > 0; _block_reduce_f32_5_offset_level1 >>= 1) {
        _block_reduce_f32_5_accum += __shfl_xor_sync(0xffffffffu, _block_reduce_f32_5_accum, _block_reduce_f32_5_offset_level1);
    }
    if (_block_reduce_f32_5_lane == 0) { reduce_scratch[_block_reduce_f32_5_warp] = _block_reduce_f32_5_accum; }
    __syncthreads();
    float _block_reduce_f32_5_level2 = (_block_reduce_f32_5_lane < 14) ? reduce_scratch[_block_reduce_f32_5_lane] : 0.0f;
    #pragma unroll
    for (int _block_reduce_f32_5_offset_level2 = 16; _block_reduce_f32_5_offset_level2 > 0; _block_reduce_f32_5_offset_level2 >>= 1) {
        _block_reduce_f32_5_level2 += __shfl_xor_sync(0xffffffffu, _block_reduce_f32_5_level2, _block_reduce_f32_5_offset_level2);
    }
    float _block_reduce_f32_0 = _block_reduce_f32_5_level2;
    if (is_owner) {
        float _rsqrt_0 = rsqrtf(_block_reduce_f32_0 / 3584.0f + epsilon);
        float rcp_rms = _rsqrt_0;
        float reduced_words_f32_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&reduced_words_f32_1[_pair * 2])[0]), "=f"((&reduced_words_f32_1[_pair * 2])[1])
                : "r"(reduced_words[_pair]));
        }
        float xn_f32[8];
        for (int lane_2 = 0; lane_2 < 8; lane_2++) {
            xn_f32[lane_2] = reduced_words_f32_1[lane_2] * rcp_rms;
        }
        uint32_t xn_f32_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xn_f32[_lp*2 + 0], xn_f32[_lp*2+1 + 0]));
            xn_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        float xn_f32_bf16_f32[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&xn_f32_bf16_f32[_pair * 2])[0]), "=f"((&xn_f32_bf16_f32[_pair * 2])[1])
                : "r"(xn_f32_bf16[_pair]));
        }
        float _vec_load_2[8];
        {
            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gamma + element + 0);
            uint4 _vld_6[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_6[_blk] = _vptr_6[_blk];
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
        }
        float y_f32[8];
        for (int lane_3 = 0; lane_3 < 8; lane_3++) {
            y_f32[lane_3] = xn_f32_bf16_f32[lane_3] * _vec_load_2[lane_3];
        }
        uint32_t y_f32_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_f32[_lp*2 + 0], y_f32[_lp*2+1 + 0]));
            y_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _packed16_sanitize_1[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_7 = static_cast<uint32_t>(y_f32_bf16[_p16s_word]);
            uint32_t _p16s_lo_7 = _p16s_carrier_7 & 0x0000ffffu;
            uint32_t _p16s_hi_7 = (_p16s_carrier_7 >> 16) & 0x0000ffffu;
            if (_p16s_lo_7 == 0x00008000u) _p16s_lo_7 = 0u;
            if (_p16s_hi_7 == 0x00008000u) _p16s_hi_7 = 0u;
            _packed16_sanitize_1[_p16s_word] = _p16s_lo_7 | (_p16s_hi_7 << 16);
        }
        reinterpret_cast<int4*>(mcast_ptr + (current_epoch_elements + stage_elements + (unsigned long long)thread_offset))[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
    }
    {
        unsigned char* _mlc_base_8 = reinterpret_cast<unsigned char*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
        uint32_t _mlc_dirty_8 = static_cast<uint32_t>(_vec_load_0[1]);
        uint32_t _mlc_stage_bytes_8 = static_cast<uint32_t>(dirty_stage_bytes);
        uint32_t _mlc_stages_8 = static_cast<uint32_t>(dirty_num_stages);
        uint32_t _mlc_clear_8[4] = {static_cast<uint32_t>(_vec_load_1[0]), static_cast<uint32_t>(_vec_load_1[1]), static_cast<uint32_t>(_vec_load_1[2]), static_cast<uint32_t>(_vec_load_1[3])};
        uint32_t _mlc_global_cta_8 = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t _mlc_global_tid_8 = _mlc_global_cta_8 * blockDim.x + threadIdx.x;
        uint32_t _mlc_num_threads_8 = gridDim.x * gridDim.y * blockDim.x;
        uint4 _mlc_init_8 = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};
        if (1u < _mlc_stages_8) {
            size_t _mlc_stage_offset = static_cast<size_t>((static_cast<uint8_t>(_mlc_dirty_8) * _mlc_stages_8 + static_cast<uint8_t>(1u)) * static_cast<size_t>(_mlc_stage_bytes_8));
            uint32_t _mlc_boundary = (_mlc_clear_8[1u] + 15u) / 16u;
            for (uint32_t _mlc_packed = _mlc_global_tid_8; _mlc_packed < _mlc_boundary; _mlc_packed += _mlc_num_threads_8) {
                reinterpret_cast<uint4*>(_mlc_base_8 + _mlc_stage_offset)[_mlc_packed] = _mlc_init_8;
            }
        }
    }
    if (warp == 0) {
        asm volatile("barrier.sync 1, 448;" ::: "memory");
    } else {
        asm volatile("barrier.arrive 1, 448;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    uint32_t _sysv_poll_group_0[4];
    do {
        asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + stage_elements + (unsigned long long)thread_offset)) : "memory");
    } while ((_sysv_poll_group_0[0] == 0x80000000u) || (_sysv_poll_group_0[1] == 0x80000000u) || (_sysv_poll_group_0[2] == 0x80000000u) || (_sysv_poll_group_0[3] == 0x80000000u));
    reinterpret_cast<int4*>(y_out + thread_offset)[0] = reinterpret_cast<int4*>(_sysv_poll_group_0)[0];
    if (token == 0) {
        if (tid == 0) {
            int rounded_tokens = (num_tokens + 12 - 1) / 12 * 12;
            {
                unsigned int* _mlf_flags_9 = reinterpret_cast<unsigned int*>(buffer_flags);
                volatile unsigned int* _mlf_access_9 = _mlf_flags_9 + 8;
                while (*_mlf_access_9 < static_cast<unsigned int>(gridDim.x * gridDim.y * gridDim.z)) {}
                uint4* _mlf_vectors_9 = reinterpret_cast<uint4*>(_mlf_flags_9);
                _mlf_vectors_9[0] = {
                    (static_cast<unsigned int>(_vec_load_0[0]) + 1u) % 3u,
                    static_cast<unsigned int>(_vec_load_0[0]),
                    static_cast<unsigned int>(_vec_load_0[2]),
                    static_cast<unsigned int>(2)
                };
                _mlf_vectors_9[1] = {
                    static_cast<unsigned int>(rounded_tokens * 3584 * 2),
                    static_cast<unsigned int>(num_tokens * 3584 * 2),
                    static_cast<unsigned int>(0),
                    static_cast<unsigned int>(0)
                };
                _mlf_flags_9[8] = 0u;
            }
        }
    }
}

} // extern "C"
