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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 1792
#define SMEM_PARTIALS_STRIDE 1792
#define SMEM_GEMM_SM_OFF 1792
#define SMEM_GEMM_SM_STAGE_BYTES 128
#define SMEM_GEMM_SM_STRIDE 128
#define SMEM_TOTAL 1920
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_4940004b2ec7c345f8f1(__nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ w_slice, __nv_bfloat16* __restrict__ out, long long* __restrict__ peer_ptrs, __nv_bfloat16* __restrict__ mcast_ptr, unsigned int* __restrict__ buffer_flags, int num_tokens, int rank, int my_col_begin)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;
    float* gemm_sm = reinterpret_cast<float*>(smem_raw + 1792);
    const int gemm_sm_addr = smem + 1792;

    // === Task calls (dependency order) ===
    int row0 = blockIdx.x * 8;
    int k0 = tid * 8;
    int col = my_col_begin + row0;
    unsigned int wv[32];
    for (int r = 0; r < 8; r++) {
        unsigned int _vec_load_0[4];
        {
            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(w_slice) + (((row0 + r) * 3584 + k0) / 2) + 0);
            _vec_load_0[0 + 0] = _uv4_0.x;
            _vec_load_0[0 + 1] = _uv4_0.y;
            _vec_load_0[0 + 2] = _uv4_0.z;
            _vec_load_0[0 + 3] = _uv4_0.w;
        }
        for (int q = 0; q < 4; q++) {
            wv[r * 4 + q] = _vec_load_0[q];
        }
    }
    unsigned int _vec_load_1[4];
    {
        uint4 _uv4_1 = *reinterpret_cast<const uint4*>(buffer_flags + 0);
        _vec_load_1[0 + 0] = _uv4_1.x;
        _vec_load_1[0 + 1] = _uv4_1.y;
        _vec_load_1[0 + 2] = _uv4_1.z;
        _vec_load_1[0 + 3] = _uv4_1.w;
    }
    unsigned int _vec_load_2[4];
    {
        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(buffer_flags + 4);
        _vec_load_2[0 + 0] = _uv4_2.x;
        _vec_load_2[0 + 1] = _uv4_2.y;
        _vec_load_2[0 + 2] = _uv4_2.z;
        _vec_load_2[0 + 3] = _uv4_2.w;
    }
    unsigned int dirty_num_stages = _vec_load_1[3];
    unsigned int dirty_stage_bytes = 0;
    if (dirty_num_stages > 0) {
        dirty_stage_bytes = _vec_load_1[2] / dirty_num_stages;
    }
    unsigned long long current_epoch_elements = (unsigned long long)_vec_load_1[0] * (unsigned long long)_vec_load_1[2] / 2;
    unsigned long long stage_elements = (unsigned long long)_vec_load_1[2] / 2 / 2;
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    {
        unsigned char* _mlc_base_3 = reinterpret_cast<unsigned char*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
        uint32_t _mlc_dirty_3 = static_cast<uint32_t>(_vec_load_1[1]);
        uint32_t _mlc_stage_bytes_3 = static_cast<uint32_t>(dirty_stage_bytes);
        uint32_t _mlc_stages_3 = static_cast<uint32_t>(dirty_num_stages);
        uint32_t _mlc_clear_3[4] = {static_cast<uint32_t>(_vec_load_2[0]), static_cast<uint32_t>(_vec_load_2[1]), static_cast<uint32_t>(_vec_load_2[2]), static_cast<uint32_t>(_vec_load_2[3])};
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
    float shared_sum_f32[8];
    for (int lane_1 = 0; lane_1 < 8; lane_1++) {
        shared_sum_f32[lane_1] = 0.0f;
    }
    if (tid < num_tokens) {
        uint32_t _mnnvl_twoshot_reduce_0[4];
        {
            float2 _mtr_accum_4[4];
            bool _mtr_valid_4;
            do {
                _mtr_valid_4 = true;
                #pragma unroll
                for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) _mtr_accum_4[_mtr_pair] = make_float2(0.0f, 0.0f);
                uint32_t _mtr_rank_4_0[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_0[0]), "=r"(_mtr_rank_4_0[1]), "=r"(_mtr_rank_4_0[2]), "=r"(_mtr_rank_4_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + (unsigned long long)tid * 12 * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_1[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_1[0]), "=r"(_mtr_rank_4_1[1]), "=r"(_mtr_rank_4_1[2]), "=r"(_mtr_rank_4_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 1) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_2[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_2[0]), "=r"(_mtr_rank_4_2[1]), "=r"(_mtr_rank_4_2[2]), "=r"(_mtr_rank_4_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 2) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_3[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_3[0]), "=r"(_mtr_rank_4_3[1]), "=r"(_mtr_rank_4_3[2]), "=r"(_mtr_rank_4_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 3) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_4[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_4[0]), "=r"(_mtr_rank_4_4[1]), "=r"(_mtr_rank_4_4[2]), "=r"(_mtr_rank_4_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 4) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_5[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_5[0]), "=r"(_mtr_rank_4_5[1]), "=r"(_mtr_rank_4_5[2]), "=r"(_mtr_rank_4_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 5) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_6[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_6[0]), "=r"(_mtr_rank_4_6[1]), "=r"(_mtr_rank_4_6[2]), "=r"(_mtr_rank_4_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 6) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_7[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_7[0]), "=r"(_mtr_rank_4_7[1]), "=r"(_mtr_rank_4_7[2]), "=r"(_mtr_rank_4_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 7) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_8[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_8[0]), "=r"(_mtr_rank_4_8[1]), "=r"(_mtr_rank_4_8[2]), "=r"(_mtr_rank_4_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 8) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_9[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_9[0]), "=r"(_mtr_rank_4_9[1]), "=r"(_mtr_rank_4_9[2]), "=r"(_mtr_rank_4_9[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 9) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_10[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_10[0]), "=r"(_mtr_rank_4_10[1]), "=r"(_mtr_rank_4_10[2]), "=r"(_mtr_rank_4_10[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 10) * 640 + (unsigned long long)row0)) : "memory");
                uint32_t _mtr_rank_4_11[4];
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_4_11[0]), "=r"(_mtr_rank_4_11[1]), "=r"(_mtr_rank_4_11[2]), "=r"(_mtr_rank_4_11[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + ((unsigned long long)tid * 12 + 11) * 640 + (unsigned long long)row0)) : "memory");
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
        for (int lane_2 = 0; lane_2 < 8; lane_2++) {
            shared_sum_f32[lane_2] = _mnnvl_twoshot_reduce_0_f32[lane_2];
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float acc[32];
    for (int i = 0; i < 32; i++) {
        acc[i] = 0.0f;
    }
    for (int m = 0; m < 4; m++) {
        if (m < num_tokens) {
            unsigned int _vec_load_3[4];
            {
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(y) + ((m * 3584 + k0) / 2) + 0);
                _vec_load_3[0 + 0] = _uv4_5.x;
                _vec_load_3[0 + 1] = _uv4_5.y;
                _vec_load_3[0 + 2] = _uv4_5.z;
                _vec_load_3[0 + 3] = _uv4_5.w;
            }
            for (int r_1 = 0; r_1 < 8; r_1++) {
                float _bf16x2_dot_f32_0;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_0) : "r"(wv[r_1 * 4]), "r"(_vec_load_3[0]), "f"(0.0f));
                float _bf16x2_dot_f32_1;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_1) : "r"(wv[r_1 * 4 + 1]), "r"(_vec_load_3[1]), "f"(_bf16x2_dot_f32_0));
                float _bf16x2_dot_f32_2;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_2) : "r"(wv[r_1 * 4 + 2]), "r"(_vec_load_3[2]), "f"(_bf16x2_dot_f32_1));
                float _bf16x2_dot_f32_3;
                asm volatile(
                    "{\n\t"
                    ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                    "mov.b32 {a_lo, a_hi}, %1;\n\t"
                    "mov.b32 {b_lo, b_hi}, %2;\n\t"
                    "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                    "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                    "}\n"
                    : "=f"(_bf16x2_dot_f32_3) : "r"(wv[r_1 * 4 + 3]), "r"(_vec_load_3[3]), "f"(_bf16x2_dot_f32_2));
                acc[r_1 * 4 + m] = _bf16x2_dot_f32_3;
            }
        }
    }
    for (int m_1 = 0; m_1 < 4; m_1++) {
        if (m_1 < num_tokens) {
            for (int r_2 = 0; r_2 < 8; r_2++) {
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 4 + m_1], 16);
                acc[r_2 * 4 + m_1] = acc[r_2 * 4 + m_1] + _shfl_xor_0;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 4 + m_1], 8);
                acc[r_2 * 4 + m_1] = acc[r_2 * 4 + m_1] + _shfl_xor_1;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 4 + m_1], 4);
                acc[r_2 * 4 + m_1] = acc[r_2 * 4 + m_1] + _shfl_xor_2;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 4 + m_1], 2);
                acc[r_2 * 4 + m_1] = acc[r_2 * 4 + m_1] + _shfl_xor_3;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, acc[r_2 * 4 + m_1], 1);
                acc[r_2 * 4 + m_1] = acc[r_2 * 4 + m_1] + _shfl_xor_4;
            }
            if (lane == 0) {
                for (int r_3 = 0; r_3 < 8; r_3++) {
                    partials[warp * 32 + r_3 * 4 + m_1] = acc[r_3 * 4 + m_1];
                }
            }
        }
    }
    asm volatile("barrier.sync 1, 448;" ::: "memory");
    if (tid < 32) {
        int r_out = tid / 4;
        int m_out = tid % 4;
        if (m_out < num_tokens) {
            float total = 0.0f;
            for (int wgt = 0; wgt < 14; wgt++) {
                total = total + partials[wgt * 32 + tid];
            }
            gemm_sm[m_out * 8 + r_out] = total;
        }
    }
    asm volatile("barrier.sync 1, 448;" ::: "memory");
    if (tid < num_tokens) {
        float out_f32[8];
        for (int lane_3 = 0; lane_3 < 8; lane_3++) {
            out_f32[lane_3] = shared_sum_f32[lane_3] + gemm_sm[tid * 8 + lane_3];
        }
        uint32_t out_f32_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f32[_lp*2 + 0], out_f32[_lp*2+1 + 0]));
            out_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _packed16_sanitize_0[4];
        #pragma unroll
        for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
            uint32_t _p16s_carrier_6 = static_cast<uint32_t>(out_f32_bf16[_p16s_word]);
            uint32_t _p16s_lo_6 = _p16s_carrier_6 & 0x0000ffffu;
            uint32_t _p16s_hi_6 = (_p16s_carrier_6 >> 16) & 0x0000ffffu;
            if (_p16s_lo_6 == 0x00008000u) _p16s_lo_6 = 0u;
            if (_p16s_hi_6 == 0x00008000u) _p16s_hi_6 = 0u;
            _packed16_sanitize_0[_p16s_word] = _p16s_lo_6 | (_p16s_hi_6 << 16);
        }
        reinterpret_cast<int4*>(mcast_ptr + (current_epoch_elements + stage_elements + (unsigned long long)(tid * 7168 + col)))[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
    }
    {
        unsigned char* _mlc_base_7 = reinterpret_cast<unsigned char*>(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]));
        uint32_t _mlc_dirty_7 = static_cast<uint32_t>(_vec_load_1[1]);
        uint32_t _mlc_stage_bytes_7 = static_cast<uint32_t>(dirty_stage_bytes);
        uint32_t _mlc_stages_7 = static_cast<uint32_t>(dirty_num_stages);
        uint32_t _mlc_clear_7[4] = {static_cast<uint32_t>(_vec_load_2[0]), static_cast<uint32_t>(_vec_load_2[1]), static_cast<uint32_t>(_vec_load_2[2]), static_cast<uint32_t>(_vec_load_2[3])};
        uint32_t _mlc_global_cta_7 = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t _mlc_global_tid_7 = _mlc_global_cta_7 * blockDim.x + threadIdx.x;
        uint32_t _mlc_num_threads_7 = gridDim.x * gridDim.y * blockDim.x;
        uint4 _mlc_init_7 = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};
        if (1u < _mlc_stages_7) {
            size_t _mlc_stage_offset = static_cast<size_t>((static_cast<uint8_t>(_mlc_dirty_7) * _mlc_stages_7 + static_cast<uint8_t>(1u)) * static_cast<size_t>(_mlc_stage_bytes_7));
            uint32_t _mlc_boundary = (_mlc_clear_7[1u] + 15u) / 16u;
            for (uint32_t _mlc_packed = _mlc_global_tid_7; _mlc_packed < _mlc_boundary; _mlc_packed += _mlc_num_threads_7) {
                reinterpret_cast<uint4*>(_mlc_base_7 + _mlc_stage_offset)[_mlc_packed] = _mlc_init_7;
            }
        }
    }
    if (warp == 0) {
        asm volatile("barrier.sync 2, 448;" ::: "memory");
    } else {
        asm volatile("barrier.arrive 2, 448;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    int g = blockIdx.x * 448 + tid;
    if (g < num_tokens * 896) {
        int tok = g / 896;
        int pk = g - tok * 896;
        int row_index = tok * 7168 + pk * 8;
        uint32_t _sysv_poll_group_0[4];
        do {
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(peer_ptrs[rank]) + (current_epoch_elements + stage_elements + (unsigned long long)row_index)) : "memory");
        } while ((_sysv_poll_group_0[0] == 0x80000000u) || (_sysv_poll_group_0[1] == 0x80000000u) || (_sysv_poll_group_0[2] == 0x80000000u) || (_sysv_poll_group_0[3] == 0x80000000u));
        reinterpret_cast<int4*>(out + row_index)[0] = reinterpret_cast<int4*>(_sysv_poll_group_0)[0];
    }
    if (blockIdx.x == 0) {
        if (tid == 0) {
            {
                unsigned int* _mlf_flags_8 = reinterpret_cast<unsigned int*>(buffer_flags);
                volatile unsigned int* _mlf_access_8 = _mlf_flags_8 + 8;
                while (*_mlf_access_8 < static_cast<unsigned int>(gridDim.x * gridDim.y * gridDim.z)) {}
                uint4* _mlf_vectors_8 = reinterpret_cast<uint4*>(_mlf_flags_8);
                _mlf_vectors_8[0] = {
                    (static_cast<unsigned int>(_vec_load_1[0]) + 1u) % 3u,
                    static_cast<unsigned int>(_vec_load_1[0]),
                    static_cast<unsigned int>(_vec_load_1[2]),
                    static_cast<unsigned int>(2)
                };
                _mlf_vectors_8[1] = {
                    static_cast<unsigned int>(num_tokens * 12 * 640 * 2),
                    static_cast<unsigned int>(num_tokens * 7168 * 2),
                    static_cast<unsigned int>(0),
                    static_cast<unsigned int>(0)
                };
                _mlf_flags_8[8] = 0u;
            }
        }
    }
}

} // extern "C"
