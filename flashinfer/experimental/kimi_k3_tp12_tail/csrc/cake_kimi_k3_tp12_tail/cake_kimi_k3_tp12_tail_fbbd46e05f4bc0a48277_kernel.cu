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
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_fbbd46e05f4bc0a48277(__nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ gemm_slice, __nv_bfloat16* __restrict__ out, long long* __restrict__ peer_ptrs, __nv_bfloat16* __restrict__ mcast_ptr, unsigned int* __restrict__ buffer_flags, int num_tokens, int rank, int my_col_begin, int my_cols, int gemm_plane_stride, int num_gemm_splits)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
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
        if (k == 0) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            uint8_t* bytes_base = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
            unsigned int dirty = (unsigned int)_vec_load_0[1];
            unsigned int stage_bytes = (unsigned int)dirty_stage_bytes;
            unsigned int num_stages = (unsigned int)dirty_num_stages;
            unsigned int clear[4];
            clear[0] = (unsigned int)_vec_load_1[0];
            clear[1] = (unsigned int)_vec_load_1[1];
            clear[2] = (unsigned int)_vec_load_1[2];
            clear[3] = (unsigned int)_vec_load_1[3];
            unsigned int block = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
            unsigned int thread = block * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
            unsigned int threads = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
            unsigned int sentinel[4] = {2147483648, 2147483648, 2147483648, 2147483648};
            if (num_stages > 0) {
                unsigned long long stage_offset = (unsigned long long)((unsigned int)(uint8_t)dirty * num_stages + (unsigned int)0) * (unsigned long long)stage_bytes;
                unsigned int boundary = (clear[0] + 15) / 16;
                for (unsigned int packed = thread; packed < boundary; packed += threads) {
                    {
                        auto _gst_2 = reinterpret_cast<unsigned __int128*>(bytes_base + stage_offset) + packed;
                        __builtin_assume(__isGlobal(_gst_2));
                        reinterpret_cast<int4*>(_gst_2)[0] = reinterpret_cast<int4*>(sentinel)[0];
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
        }
        int token_b = token_a;
        if (token_b >= 0) {
            if (token_b < num_tokens) {
                if (owner == rank) {
                    uint32_t _mnnvl_twoshot_reduce_0[4];
                    {
                        float2 _mtr_accum_3[4];
                        bool _mtr_valid_3;
                        do {
                            _mtr_valid_3 = true;
                            #pragma unroll
                            for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) _mtr_accum_3[_mtr_pair] = make_float2(0.0f, 0.0f);
                            uint32_t _mtr_rank_3_0[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_0[0]), "=r"(_mtr_rank_3_0[1]), "=r"(_mtr_rank_3_0[2]), "=r"(_mtr_rank_3_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + (unsigned long long)token_b * 12 * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_1[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_1[0]), "=r"(_mtr_rank_3_1[1]), "=r"(_mtr_rank_3_1[2]), "=r"(_mtr_rank_3_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 1) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_2[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_2[0]), "=r"(_mtr_rank_3_2[1]), "=r"(_mtr_rank_3_2[2]), "=r"(_mtr_rank_3_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 2) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_3[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_3[0]), "=r"(_mtr_rank_3_3[1]), "=r"(_mtr_rank_3_3[2]), "=r"(_mtr_rank_3_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 3) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_4[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_4[0]), "=r"(_mtr_rank_3_4[1]), "=r"(_mtr_rank_3_4[2]), "=r"(_mtr_rank_3_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 4) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_5[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_5[0]), "=r"(_mtr_rank_3_5[1]), "=r"(_mtr_rank_3_5[2]), "=r"(_mtr_rank_3_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 5) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_6[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_6[0]), "=r"(_mtr_rank_3_6[1]), "=r"(_mtr_rank_3_6[2]), "=r"(_mtr_rank_3_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 6) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_7[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_7[0]), "=r"(_mtr_rank_3_7[1]), "=r"(_mtr_rank_3_7[2]), "=r"(_mtr_rank_3_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 7) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_8[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_8[0]), "=r"(_mtr_rank_3_8[1]), "=r"(_mtr_rank_3_8[2]), "=r"(_mtr_rank_3_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 8) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_9[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_9[0]), "=r"(_mtr_rank_3_9[1]), "=r"(_mtr_rank_3_9[2]), "=r"(_mtr_rank_3_9[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 9) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_10[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_10[0]), "=r"(_mtr_rank_3_10[1]), "=r"(_mtr_rank_3_10[2]), "=r"(_mtr_rank_3_10[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 10) * 640 + (unsigned long long)local_col)) : "memory");
                            uint32_t _mtr_rank_3_11[4];
                            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_3_11[0]), "=r"(_mtr_rank_3_11[1]), "=r"(_mtr_rank_3_11[2]), "=r"(_mtr_rank_3_11[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)token_b * 12 + 11) * 640 + (unsigned long long)local_col)) : "memory");
                            bool _mtr_dirty_3_0 = false;
                            _mtr_dirty_3_0 |= (_mtr_rank_3_0[0] == 0x80000000u);
                            _mtr_dirty_3_0 |= (_mtr_rank_3_0[1] == 0x80000000u);
                            _mtr_dirty_3_0 |= (_mtr_rank_3_0[2] == 0x80000000u);
                            _mtr_dirty_3_0 |= (_mtr_rank_3_0[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_0;
                            bool _mtr_dirty_3_1 = false;
                            _mtr_dirty_3_1 |= (_mtr_rank_3_1[0] == 0x80000000u);
                            _mtr_dirty_3_1 |= (_mtr_rank_3_1[1] == 0x80000000u);
                            _mtr_dirty_3_1 |= (_mtr_rank_3_1[2] == 0x80000000u);
                            _mtr_dirty_3_1 |= (_mtr_rank_3_1[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_1;
                            bool _mtr_dirty_3_2 = false;
                            _mtr_dirty_3_2 |= (_mtr_rank_3_2[0] == 0x80000000u);
                            _mtr_dirty_3_2 |= (_mtr_rank_3_2[1] == 0x80000000u);
                            _mtr_dirty_3_2 |= (_mtr_rank_3_2[2] == 0x80000000u);
                            _mtr_dirty_3_2 |= (_mtr_rank_3_2[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_2;
                            bool _mtr_dirty_3_3 = false;
                            _mtr_dirty_3_3 |= (_mtr_rank_3_3[0] == 0x80000000u);
                            _mtr_dirty_3_3 |= (_mtr_rank_3_3[1] == 0x80000000u);
                            _mtr_dirty_3_3 |= (_mtr_rank_3_3[2] == 0x80000000u);
                            _mtr_dirty_3_3 |= (_mtr_rank_3_3[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_3;
                            bool _mtr_dirty_3_4 = false;
                            _mtr_dirty_3_4 |= (_mtr_rank_3_4[0] == 0x80000000u);
                            _mtr_dirty_3_4 |= (_mtr_rank_3_4[1] == 0x80000000u);
                            _mtr_dirty_3_4 |= (_mtr_rank_3_4[2] == 0x80000000u);
                            _mtr_dirty_3_4 |= (_mtr_rank_3_4[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_4;
                            bool _mtr_dirty_3_5 = false;
                            _mtr_dirty_3_5 |= (_mtr_rank_3_5[0] == 0x80000000u);
                            _mtr_dirty_3_5 |= (_mtr_rank_3_5[1] == 0x80000000u);
                            _mtr_dirty_3_5 |= (_mtr_rank_3_5[2] == 0x80000000u);
                            _mtr_dirty_3_5 |= (_mtr_rank_3_5[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_5;
                            bool _mtr_dirty_3_6 = false;
                            _mtr_dirty_3_6 |= (_mtr_rank_3_6[0] == 0x80000000u);
                            _mtr_dirty_3_6 |= (_mtr_rank_3_6[1] == 0x80000000u);
                            _mtr_dirty_3_6 |= (_mtr_rank_3_6[2] == 0x80000000u);
                            _mtr_dirty_3_6 |= (_mtr_rank_3_6[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_6;
                            bool _mtr_dirty_3_7 = false;
                            _mtr_dirty_3_7 |= (_mtr_rank_3_7[0] == 0x80000000u);
                            _mtr_dirty_3_7 |= (_mtr_rank_3_7[1] == 0x80000000u);
                            _mtr_dirty_3_7 |= (_mtr_rank_3_7[2] == 0x80000000u);
                            _mtr_dirty_3_7 |= (_mtr_rank_3_7[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_7;
                            bool _mtr_dirty_3_8 = false;
                            _mtr_dirty_3_8 |= (_mtr_rank_3_8[0] == 0x80000000u);
                            _mtr_dirty_3_8 |= (_mtr_rank_3_8[1] == 0x80000000u);
                            _mtr_dirty_3_8 |= (_mtr_rank_3_8[2] == 0x80000000u);
                            _mtr_dirty_3_8 |= (_mtr_rank_3_8[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_8;
                            bool _mtr_dirty_3_9 = false;
                            _mtr_dirty_3_9 |= (_mtr_rank_3_9[0] == 0x80000000u);
                            _mtr_dirty_3_9 |= (_mtr_rank_3_9[1] == 0x80000000u);
                            _mtr_dirty_3_9 |= (_mtr_rank_3_9[2] == 0x80000000u);
                            _mtr_dirty_3_9 |= (_mtr_rank_3_9[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_9;
                            bool _mtr_dirty_3_10 = false;
                            _mtr_dirty_3_10 |= (_mtr_rank_3_10[0] == 0x80000000u);
                            _mtr_dirty_3_10 |= (_mtr_rank_3_10[1] == 0x80000000u);
                            _mtr_dirty_3_10 |= (_mtr_rank_3_10[2] == 0x80000000u);
                            _mtr_dirty_3_10 |= (_mtr_rank_3_10[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_10;
                            bool _mtr_dirty_3_11 = false;
                            _mtr_dirty_3_11 |= (_mtr_rank_3_11[0] == 0x80000000u);
                            _mtr_dirty_3_11 |= (_mtr_rank_3_11[1] == 0x80000000u);
                            _mtr_dirty_3_11 |= (_mtr_rank_3_11[2] == 0x80000000u);
                            _mtr_dirty_3_11 |= (_mtr_rank_3_11[3] == 0x80000000u);
                            _mtr_valid_3 &= !_mtr_dirty_3_11;
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_0[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_0[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_0[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_0[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_1[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_1[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_1[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_1[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_2[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_2[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_2[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_2[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_3[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_3[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_3[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_3[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_4[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_4[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_4[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_4[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_5[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_5[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_5[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_5[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_6[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_6[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_6[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_6[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_7[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_7[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_7[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_7[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_8[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_8[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_8[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_8[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_9[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_9[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_9[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_9[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_10[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_10[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_10[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_10[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[0]).x), "+f"((_mtr_accum_3[0]).y) : "r"(_mtr_rank_3_11[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[1]).x), "+f"((_mtr_accum_3[1]).y) : "r"(_mtr_rank_3_11[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[2]).x), "+f"((_mtr_accum_3[2]).y) : "r"(_mtr_rank_3_11[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 lo, hi;\n\t"
                                "mov.b32 {lo, hi}, %2;\n\t"
                                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                                "}\n"
                                : "+f"((_mtr_accum_3[3]).x), "+f"((_mtr_accum_3[3]).y) : "r"(_mtr_rank_3_11[3]));
                        } while (!_mtr_valid_3);
                        #pragma unroll
                        for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) {
                            __nv_bfloat162 _mtr_out = __float22bfloat162_rn(_mtr_accum_3[_mtr_pair]);
                            _mnnvl_twoshot_reduce_0[_mtr_pair] = *reinterpret_cast<uint32_t*>(&_mtr_out);
                        }
                    }
                    float _unpack_bf16_0[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_mnnvl_twoshot_reduce_0[_pair]) << 16);
                        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_mnnvl_twoshot_reduce_0[_pair]) & 0xffff0000u);
                    }
                    int gemm_index = token_b * my_cols + local_col;
                    float out_f32[8];
                    float _vec_load_2[8];
                    {
                        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(gemm_slice + gemm_index + 0);
                        uint4 _vld_4[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_4[_blk] = _vptr_4[_blk];
                            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                    for (int lane_1 = 0; lane_1 < 8; lane_1++) {
                        out_f32[lane_1] = _unpack_bf16_0[lane_1] + _vec_load_2[lane_1];
                    }
                    unsigned int out_f32_bf16[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f32[_lp*2 + 0], out_f32[_lp*2+1 + 0]));
                        out_f32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _packed16_sanitize_0[4];
                    #pragma unroll
                    for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                        uint32_t _p16s_carrier_5 = static_cast<uint32_t>(out_f32_bf16[_p16s_word]);
                        uint32_t _p16s_lo_5 = _p16s_carrier_5 & 0x0000ffffu;
                        uint32_t _p16s_hi_5 = (_p16s_carrier_5 >> 16) & 0x0000ffffu;
                        if (_p16s_lo_5 == 0x00008000u) _p16s_lo_5 = 0u;
                        if (_p16s_hi_5 == 0x00008000u) _p16s_hi_5 = 0u;
                        _packed16_sanitize_0[_p16s_word] = _p16s_lo_5 | (_p16s_hi_5 << 16);
                    }
                    reinterpret_cast<int4*>(mcast_ptr + (current_epoch_elements + stage_elements + (unsigned long long)(token_b * 7168 + col)))[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
                }
            }
        }
        int token_c = token_b - 2 * stride;
        if (token_c >= 0) {
            int row_c = token_c * 7168 + col;
            uint32_t _sysv_poll_group_0[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + stage_elements + (unsigned long long)row_c)) : "memory");
            } while (
                (_sysv_poll_group_0[0] == 0x80000000u) || (_sysv_poll_group_0[1] == 0x80000000u) || (_sysv_poll_group_0[2] == 0x80000000u) || (_sysv_poll_group_0[3] == 0x80000000u)
            );
            reinterpret_cast<int4*>(out + row_c)[0] = reinterpret_cast<int4*>(_sysv_poll_group_0)[0];
        }
    }
    uint8_t* bytes_base_1 = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
    unsigned int dirty_1 = (unsigned int)_vec_load_0[1];
    unsigned int stage_bytes_1 = (unsigned int)dirty_stage_bytes;
    unsigned int num_stages_1 = (unsigned int)dirty_num_stages;
    unsigned int clear_1[4];
    clear_1[0] = (unsigned int)_vec_load_1[0];
    clear_1[1] = (unsigned int)_vec_load_1[1];
    clear_1[2] = (unsigned int)_vec_load_1[2];
    clear_1[3] = (unsigned int)_vec_load_1[3];
    unsigned int block_1 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
    unsigned int thread_1 = block_1 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
    unsigned int threads_1 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
    unsigned int sentinel_1[4] = {2147483648, 2147483648, 2147483648, 2147483648};
    if (num_stages_1 > 1) {
        unsigned long long stage_offset_1 = (unsigned long long)((unsigned int)(uint8_t)dirty_1 * num_stages_1 + (unsigned int)1) * (unsigned long long)stage_bytes_1;
        unsigned int boundary_1 = (clear_1[1] + 15) / 16;
        for (unsigned int packed_1 = thread_1; packed_1 < boundary_1; packed_1 += threads_1) {
            {
                auto _gst_6 = reinterpret_cast<unsigned __int128*>(bytes_base_1 + stage_offset_1) + packed_1;
                __builtin_assume(__isGlobal(_gst_6));
                reinterpret_cast<int4*>(_gst_6)[0] = reinterpret_cast<int4*>(sentinel_1)[0];
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
                    volatile unsigned int* _sysv_wait_p_7 = reinterpret_cast<volatile unsigned int*>(buffer_flags) + (8);
                    while (*_sysv_wait_p_7 < static_cast<unsigned int>(gridDim.x * gridDim.y * gridDim.z)) {}
                }
                unsigned int current = (unsigned int)_vec_load_0[0];
                unsigned int head[4];
                head[0] = (current + 1) % 3;
                head[1] = current;
                head[2] = (unsigned int)_vec_load_0[2];
                head[3] = 2;
                unsigned int clear_0[4];
                clear_0[0] = (unsigned int)(num_tokens * 12 * 640 * 2);
                clear_0[1] = (unsigned int)(num_tokens * 7168 * 2);
                clear_0[2] = 0;
                clear_0[3] = 0;
                reinterpret_cast<int4*>(buffer_flags)[0] = reinterpret_cast<int4*>(head)[0];
                reinterpret_cast<int4*>(buffer_flags + 4)[0] = reinterpret_cast<int4*>(clear_0)[0];
                buffer_flags[8] = 0;
            }
        }
    }
}

} // extern "C"
