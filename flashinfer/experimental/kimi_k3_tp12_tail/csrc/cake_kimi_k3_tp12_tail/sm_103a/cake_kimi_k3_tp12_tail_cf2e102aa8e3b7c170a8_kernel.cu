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
#define SMEM_STAGE0_OFF 0
#define SMEM_STAGE0_STAGE_BYTES 7168
#define SMEM_STAGE0_STRIDE 7168
#define SMEM_TOTAL 14336
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_cf2e102aa8e3b7c170a8(__nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ gemm_slice, __nv_bfloat16* __restrict__ out, long long* __restrict__ peer_ptrs, __nv_bfloat16* __restrict__ mcast_ptr, unsigned int* __restrict__ buffer_flags, int num_tokens, int rank, int my_col_begin, int my_cols, int gemm_plane_stride, int num_gemm_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* stage0 = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int stage0_addr = smem + 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int first_token = blockIdx.x;
    int stride = gridDim.x;
    int col = blockIdx.y * 3584 + tid * 8;
    int owner = 0;
    int owner_col_begin = 0;
    if (col >= 640) {
        owner = owner + 1;
        owner_col_begin = 640;
    }
    if (col >= 1280) {
        owner = owner + 1;
        owner_col_begin = 1280;
    }
    if (col >= 1920) {
        owner = owner + 1;
        owner_col_begin = 1920;
    }
    if (col >= 2560) {
        owner = owner + 1;
        owner_col_begin = 2560;
    }
    if (col >= 3200) {
        owner = owner + 1;
        owner_col_begin = 3200;
    }
    if (col >= 3840) {
        owner = owner + 1;
        owner_col_begin = 3840;
    }
    if (col >= 4480) {
        owner = owner + 1;
        owner_col_begin = 4480;
    }
    if (col >= 5120) {
        owner = owner + 1;
        owner_col_begin = 5120;
    }
    if (col >= 5632) {
        owner = owner + 1;
        owner_col_begin = 5632;
    }
    if (col >= 6144) {
        owner = owner + 1;
        owner_col_begin = 6144;
    }
    if (col >= 6656) {
        owner = owner + 1;
        owner_col_begin = 6656;
    }
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
    unsigned long long slot_col = (unsigned long long)(col - owner_col_begin);
    int local_col = col - my_col_begin;
    int my_iters = 0;
    if (first_token < num_tokens) {
        my_iters = (num_tokens - first_token + stride - 1) / stride;
    }
    int total_iters = my_iters + 2;
    #pragma unroll 1
    for (int k = 0; k < total_iters; k++) {
        int token_a = first_token + k * stride;
        if (token_a < num_tokens) {
            uint32_t _packed16_sanitize_0[4];
            {
                uint4 _p16_packet_2 = *reinterpret_cast<const uint4*>(shared + (token_a * 7168 + col));
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
            unsigned long long scatter_index = ((unsigned long long)token_a * 12 + (unsigned long long)rank) * 640 + slot_col;
            unsigned int stage_addr = stage0_addr + (unsigned int)(k % 2) * 7168;
            if (tid < 12) {
                asm volatile("cp.async.bulk.wait_group.read 1;");
            }
            asm volatile("barrier.sync 2, 448;" ::: "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(stage_addr + (unsigned int)(tid * 16)), "r"(*reinterpret_cast<uint32_t*>(&_packed16_sanitize_0[0])), "r"(*reinterpret_cast<uint32_t*>(&_packed16_sanitize_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&_packed16_sanitize_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&_packed16_sanitize_0[(0) + 3])));
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 2, 448;" ::: "memory");
            if (tid < 12) {
                int piece_lo = 0;
                int piece_hi = 0;
                int piece_begin = 0;
                if (tid == 0) {
                    piece_lo = 0;
                    piece_hi = 640;
                    piece_begin = 0;
                }
                if (tid == 1) {
                    piece_lo = 640;
                    piece_hi = 1280;
                    piece_begin = 640;
                }
                if (tid == 2) {
                    piece_lo = 1280;
                    piece_hi = 1920;
                    piece_begin = 1280;
                }
                if (tid == 3) {
                    piece_lo = 1920;
                    piece_hi = 2560;
                    piece_begin = 1920;
                }
                if (tid == 4) {
                    piece_lo = 2560;
                    piece_hi = 3200;
                    piece_begin = 2560;
                }
                if (tid == 5) {
                    piece_lo = 3200;
                    piece_hi = 3840;
                    piece_begin = 3200;
                }
                if (tid == 6) {
                    piece_lo = 3840;
                    piece_hi = 4480;
                    piece_begin = 3840;
                }
                if (tid == 7) {
                    piece_lo = 4480;
                    piece_hi = 5120;
                    piece_begin = 4480;
                }
                if (tid == 8) {
                    piece_lo = 5120;
                    piece_hi = 5632;
                    piece_begin = 5120;
                }
                if (tid == 9) {
                    piece_lo = 5632;
                    piece_hi = 6144;
                    piece_begin = 5632;
                }
                if (tid == 10) {
                    piece_lo = 6144;
                    piece_hi = 6656;
                    piece_begin = 6144;
                }
                if (tid == 11) {
                    piece_lo = 6656;
                    piece_hi = 7168;
                    piece_begin = 6656;
                }
                int half_lo = blockIdx.y * 3584;
                int half_hi = half_lo + 3584;
                if (piece_lo < half_lo) {
                    piece_lo = half_lo;
                }
                if (piece_hi > half_hi) {
                    piece_hi = half_hi;
                }
                if (piece_hi > piece_lo) {
                    unsigned long long slot_o = ((unsigned long long)token_a * 12 + (unsigned long long)rank) * 640 + (unsigned long long)(piece_lo - piece_begin);
                    {
                        void* _cpbulk_dst_3 = reinterpret_cast<void*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[tid]) + (current_epoch_elements + slot_o));
                        asm volatile(
                            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                            :: "l"(_cpbulk_dst_3), "r"(stage_addr + (unsigned int)((piece_lo - half_lo) * 2)), "r"((uint32_t)((piece_hi - piece_lo) * 2))
                            : "memory");
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
        if (k == 0) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            {
                unsigned char* _mlc_base_4 = reinterpret_cast<unsigned char*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
                uint32_t _mlc_dirty_4 = static_cast<uint32_t>(_vec_load_0[1]);
                uint32_t _mlc_stage_bytes_4 = static_cast<uint32_t>(dirty_stage_bytes);
                uint32_t _mlc_stages_4 = static_cast<uint32_t>(dirty_num_stages);
                uint32_t _mlc_clear_4[4] = {static_cast<uint32_t>(_vec_load_1[0]), static_cast<uint32_t>(_vec_load_1[1]), static_cast<uint32_t>(_vec_load_1[2]), static_cast<uint32_t>(_vec_load_1[3])};
                uint32_t _mlc_global_cta_4 = blockIdx.x * gridDim.y + blockIdx.y;
                uint32_t _mlc_global_tid_4 = _mlc_global_cta_4 * blockDim.x + threadIdx.x;
                uint32_t _mlc_num_threads_4 = gridDim.x * gridDim.y * blockDim.x;
                uint4 _mlc_init_4 = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};
                if (0u < _mlc_stages_4) {
                    size_t _mlc_stage_offset = static_cast<size_t>((static_cast<uint8_t>(_mlc_dirty_4) * _mlc_stages_4 + static_cast<uint8_t>(0u)) * static_cast<size_t>(_mlc_stage_bytes_4));
                    uint32_t _mlc_boundary = (_mlc_clear_4[0u] + 15u) / 16u;
                    for (uint32_t _mlc_packed = _mlc_global_tid_4; _mlc_packed < _mlc_boundary; _mlc_packed += _mlc_num_threads_4) {
                        reinterpret_cast<uint4*>(_mlc_base_4 + _mlc_stage_offset)[_mlc_packed] = _mlc_init_4;
                    }
                }
            }
        }
        int token_b = token_a - stride;
        if (token_b >= 0) {
            if (token_b < num_tokens) {
                if (owner == rank) {
                    uint32_t _mnnvl_twoshot_reduce_0[4];
                    {
                        float2 _mtr_accum_5[4];
                        bool _mtr_valid_5;
                        do {
                            _mtr_valid_5 = true;
                            #pragma unroll
                            for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) _mtr_accum_5[_mtr_pair] = make_float2(0.0f, 0.0f);
                            uint32_t _mtr_rank_5_0[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_0[0]), "=r"(_mtr_rank_5_0[1]), "=r"(_mtr_rank_5_0[2]), "=r"(_mtr_rank_5_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + (unsigned long long)token_b * 12 * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_0 = false;
                            _mtr_dirty_5_0 |= (_mtr_rank_5_0[0] == 0x80000000u);
                            _mtr_dirty_5_0 |= (_mtr_rank_5_0[1] == 0x80000000u);
                            _mtr_dirty_5_0 |= (_mtr_rank_5_0[2] == 0x80000000u);
                            _mtr_dirty_5_0 |= (_mtr_rank_5_0[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_0;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_0[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_0[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_0[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_0[3]));
                            uint32_t _mtr_rank_5_1[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_1[0]), "=r"(_mtr_rank_5_1[1]), "=r"(_mtr_rank_5_1[2]), "=r"(_mtr_rank_5_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 1) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_1 = false;
                            _mtr_dirty_5_1 |= (_mtr_rank_5_1[0] == 0x80000000u);
                            _mtr_dirty_5_1 |= (_mtr_rank_5_1[1] == 0x80000000u);
                            _mtr_dirty_5_1 |= (_mtr_rank_5_1[2] == 0x80000000u);
                            _mtr_dirty_5_1 |= (_mtr_rank_5_1[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_1;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_1[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_1[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_1[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_1[3]));
                            uint32_t _mtr_rank_5_2[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_2[0]), "=r"(_mtr_rank_5_2[1]), "=r"(_mtr_rank_5_2[2]), "=r"(_mtr_rank_5_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 2) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_2 = false;
                            _mtr_dirty_5_2 |= (_mtr_rank_5_2[0] == 0x80000000u);
                            _mtr_dirty_5_2 |= (_mtr_rank_5_2[1] == 0x80000000u);
                            _mtr_dirty_5_2 |= (_mtr_rank_5_2[2] == 0x80000000u);
                            _mtr_dirty_5_2 |= (_mtr_rank_5_2[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_2;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_2[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_2[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_2[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_2[3]));
                            uint32_t _mtr_rank_5_3[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_3[0]), "=r"(_mtr_rank_5_3[1]), "=r"(_mtr_rank_5_3[2]), "=r"(_mtr_rank_5_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 3) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_3 = false;
                            _mtr_dirty_5_3 |= (_mtr_rank_5_3[0] == 0x80000000u);
                            _mtr_dirty_5_3 |= (_mtr_rank_5_3[1] == 0x80000000u);
                            _mtr_dirty_5_3 |= (_mtr_rank_5_3[2] == 0x80000000u);
                            _mtr_dirty_5_3 |= (_mtr_rank_5_3[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_3;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_3[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_3[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_3[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_3[3]));
                            uint32_t _mtr_rank_5_4[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_4[0]), "=r"(_mtr_rank_5_4[1]), "=r"(_mtr_rank_5_4[2]), "=r"(_mtr_rank_5_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 4) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_4 = false;
                            _mtr_dirty_5_4 |= (_mtr_rank_5_4[0] == 0x80000000u);
                            _mtr_dirty_5_4 |= (_mtr_rank_5_4[1] == 0x80000000u);
                            _mtr_dirty_5_4 |= (_mtr_rank_5_4[2] == 0x80000000u);
                            _mtr_dirty_5_4 |= (_mtr_rank_5_4[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_4;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_4[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_4[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_4[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_4[3]));
                            uint32_t _mtr_rank_5_5[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_5[0]), "=r"(_mtr_rank_5_5[1]), "=r"(_mtr_rank_5_5[2]), "=r"(_mtr_rank_5_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 5) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_5 = false;
                            _mtr_dirty_5_5 |= (_mtr_rank_5_5[0] == 0x80000000u);
                            _mtr_dirty_5_5 |= (_mtr_rank_5_5[1] == 0x80000000u);
                            _mtr_dirty_5_5 |= (_mtr_rank_5_5[2] == 0x80000000u);
                            _mtr_dirty_5_5 |= (_mtr_rank_5_5[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_5;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_5[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_5[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_5[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_5[3]));
                            uint32_t _mtr_rank_5_6[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_6[0]), "=r"(_mtr_rank_5_6[1]), "=r"(_mtr_rank_5_6[2]), "=r"(_mtr_rank_5_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 6) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_6 = false;
                            _mtr_dirty_5_6 |= (_mtr_rank_5_6[0] == 0x80000000u);
                            _mtr_dirty_5_6 |= (_mtr_rank_5_6[1] == 0x80000000u);
                            _mtr_dirty_5_6 |= (_mtr_rank_5_6[2] == 0x80000000u);
                            _mtr_dirty_5_6 |= (_mtr_rank_5_6[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_6;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_6[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_6[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_6[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_6[3]));
                            uint32_t _mtr_rank_5_7[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_7[0]), "=r"(_mtr_rank_5_7[1]), "=r"(_mtr_rank_5_7[2]), "=r"(_mtr_rank_5_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 7) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_7 = false;
                            _mtr_dirty_5_7 |= (_mtr_rank_5_7[0] == 0x80000000u);
                            _mtr_dirty_5_7 |= (_mtr_rank_5_7[1] == 0x80000000u);
                            _mtr_dirty_5_7 |= (_mtr_rank_5_7[2] == 0x80000000u);
                            _mtr_dirty_5_7 |= (_mtr_rank_5_7[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_7;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_7[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_7[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_7[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_7[3]));
                            uint32_t _mtr_rank_5_8[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_8[0]), "=r"(_mtr_rank_5_8[1]), "=r"(_mtr_rank_5_8[2]), "=r"(_mtr_rank_5_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 8) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_8 = false;
                            _mtr_dirty_5_8 |= (_mtr_rank_5_8[0] == 0x80000000u);
                            _mtr_dirty_5_8 |= (_mtr_rank_5_8[1] == 0x80000000u);
                            _mtr_dirty_5_8 |= (_mtr_rank_5_8[2] == 0x80000000u);
                            _mtr_dirty_5_8 |= (_mtr_rank_5_8[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_8;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_8[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_8[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_8[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_8[3]));
                            uint32_t _mtr_rank_5_9[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_9[0]), "=r"(_mtr_rank_5_9[1]), "=r"(_mtr_rank_5_9[2]), "=r"(_mtr_rank_5_9[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 9) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_9 = false;
                            _mtr_dirty_5_9 |= (_mtr_rank_5_9[0] == 0x80000000u);
                            _mtr_dirty_5_9 |= (_mtr_rank_5_9[1] == 0x80000000u);
                            _mtr_dirty_5_9 |= (_mtr_rank_5_9[2] == 0x80000000u);
                            _mtr_dirty_5_9 |= (_mtr_rank_5_9[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_9;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_9[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_9[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_9[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_9[3]));
                            uint32_t _mtr_rank_5_10[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_10[0]), "=r"(_mtr_rank_5_10[1]), "=r"(_mtr_rank_5_10[2]), "=r"(_mtr_rank_5_10[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 10) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_10 = false;
                            _mtr_dirty_5_10 |= (_mtr_rank_5_10[0] == 0x80000000u);
                            _mtr_dirty_5_10 |= (_mtr_rank_5_10[1] == 0x80000000u);
                            _mtr_dirty_5_10 |= (_mtr_rank_5_10[2] == 0x80000000u);
                            _mtr_dirty_5_10 |= (_mtr_rank_5_10[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_10;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_10[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_10[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_10[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_10[3]));
                            uint32_t _mtr_rank_5_11[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_5_11[0]), "=r"(_mtr_rank_5_11[1]), "=r"(_mtr_rank_5_11[2]), "=r"(_mtr_rank_5_11[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 11) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_5_11 = false;
                            _mtr_dirty_5_11 |= (_mtr_rank_5_11[0] == 0x80000000u);
                            _mtr_dirty_5_11 |= (_mtr_rank_5_11[1] == 0x80000000u);
                            _mtr_dirty_5_11 |= (_mtr_rank_5_11[2] == 0x80000000u);
                            _mtr_dirty_5_11 |= (_mtr_rank_5_11[3] == 0x80000000u);
                            _mtr_valid_5 &= !_mtr_dirty_5_11;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[0]).x), "+f"((_mtr_accum_5[0]).y) : "r"(_mtr_rank_5_11[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[1]).x), "+f"((_mtr_accum_5[1]).y) : "r"(_mtr_rank_5_11[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[2]).x), "+f"((_mtr_accum_5[2]).y) : "r"(_mtr_rank_5_11[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_5[3]).x), "+f"((_mtr_accum_5[3]).y) : "r"(_mtr_rank_5_11[3]));
                        } while (!_mtr_valid_5);
                        #pragma unroll
                        for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) {
                            __nv_bfloat162 _mtr_out = __float22bfloat162_rn(_mtr_accum_5[_mtr_pair]);
                            _mnnvl_twoshot_reduce_0[_mtr_pair] = *reinterpret_cast<uint32_t*>(&_mtr_out);
                        }
                    }
                    float _mnnvl_twoshot_reduce_0_f32[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_mnnvl_twoshot_reduce_0_f32[_pair * 2])[0]), "=f"((&_mnnvl_twoshot_reduce_0_f32[_pair * 2])[1])
                            : "r"(_mnnvl_twoshot_reduce_0[_pair]));
                    }
                    int gemm_index = token_b * my_cols + local_col;
                    float out_f32[8];
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(gemm_slice + gemm_index + 0);
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
                    for (int lane_1 = 0; lane_1 < 8; lane_1++) {
                        out_f32[lane_1] = _mnnvl_twoshot_reduce_0_f32[lane_1] + _vec_load_2[lane_1];
                    }
                    uint32_t out_f32_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f32[_lp*2 + 0], out_f32[_lp*2+1 + 0]));
                        out_f32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _packed16_sanitize_1[4];
                    #pragma unroll
                    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                        uint32_t _p16s_carrier_7 = static_cast<uint32_t>(out_f32_bf16[_p16s_word]);
                        uint32_t _p16s_lo_7 = _p16s_carrier_7 & 0x0000ffffu;
                        uint32_t _p16s_hi_7 = (_p16s_carrier_7 >> 16) & 0x0000ffffu;
                        if (_p16s_lo_7 == 0x00008000u) _p16s_lo_7 = 0u;
                        if (_p16s_hi_7 == 0x00008000u) _p16s_hi_7 = 0u;
                        _packed16_sanitize_1[_p16s_word] = _p16s_lo_7 | (_p16s_hi_7 << 16);
                    }
                    reinterpret_cast<int4*>(mcast_ptr + (current_epoch_elements + stage_elements + (unsigned long long)(token_b * 7168 + col)))[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
                }
            }
        }
        int token_c = token_b - stride;
        if (token_c >= 0) {
            int row_c = token_c * 7168 + col;
            uint32_t _sysv_poll_group_0[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + stage_elements + (unsigned long long)row_c)) : "memory");
            } while ((_sysv_poll_group_0[0] == 0x80000000u) || (_sysv_poll_group_0[1] == 0x80000000u) || (_sysv_poll_group_0[2] == 0x80000000u) || (_sysv_poll_group_0[3] == 0x80000000u));
            reinterpret_cast<int4*>(out + row_c)[0] = reinterpret_cast<int4*>(_sysv_poll_group_0)[0];
        }
    }
    if (tid < 12) {
        asm volatile("cp.async.bulk.wait_group.read 0;");
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
    if (first_token == 0) {
        if (blockIdx.y == 0) {
            if (tid == 0) {
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
                        static_cast<unsigned int>(num_tokens * 12 * 640 * 2),
                        static_cast<unsigned int>(num_tokens * 7168 * 2),
                        static_cast<unsigned int>(0),
                        static_cast<unsigned int>(0)
                    };
                    _mlf_flags_9[8] = 0u;
                }
            }
        }
    }
}

} // extern "C"
