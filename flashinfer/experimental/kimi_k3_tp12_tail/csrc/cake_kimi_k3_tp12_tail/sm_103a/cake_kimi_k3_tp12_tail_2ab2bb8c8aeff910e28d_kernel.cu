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
#define SMEM_REDUCE_SCRATCH_OFF 0
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 128
#define SMEM_REDUCE_SCRATCH_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_2ab2bb8c8aeff910e28d(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ y_out, __nv_bfloat16* __restrict__ gamma, __nv_bfloat16* __restrict__ mcast_ptr, __nv_bfloat16* __restrict__ local_unicast_ptr, unsigned int* __restrict__ buffer_flags, long long* __restrict__ k3_peer_ptrs, __nv_bfloat16* __restrict__ k3_mcast_ptr, unsigned int* __restrict__ k3_flags, int num_tokens, float epsilon)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

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
    int input_index = token * 3584 + element;
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
    unsigned int _vec_load_2[4];
    {
        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(k3_flags + 0);
        _vec_load_2[0 + 0] = _uv4_2.x;
        _vec_load_2[0 + 1] = _uv4_2.y;
        _vec_load_2[0 + 2] = _uv4_2.z;
        _vec_load_2[0 + 3] = _uv4_2.w;
    }
    unsigned int dirty_num_stages = _vec_load_0[3];
    unsigned int dirty_stage_bytes = 0;
    if (dirty_num_stages > 0) {
        dirty_stage_bytes = _vec_load_0[2] / dirty_num_stages;
    }
    unsigned long long buffer_base = (unsigned long long)_vec_load_0[0] * (unsigned long long)_vec_load_0[2] / 2;
    unsigned long long k3_epoch_elements = (unsigned long long)_vec_load_2[0] * (unsigned long long)_vec_load_2[2] / 2;
    uint32_t _packed16_sanitize_0[4];
    {
        uint4 _p16_packet_3 = *reinterpret_cast<const uint4*>(routed + (input_index));
        uint32_t _p16_word_3_0 = _p16_packet_3.x;
        if ((_p16_word_3_0 & 0x0000ffffu) == 0x00008000u) _p16_word_3_0 &= 0xffff0000u;
        if ((_p16_word_3_0 & 0xffff0000u) == 0x80000000u) _p16_word_3_0 &= 0x0000ffffu;
        _packed16_sanitize_0[0] = _p16_word_3_0;
        uint32_t _p16_word_3_1 = _p16_packet_3.y;
        if ((_p16_word_3_1 & 0x0000ffffu) == 0x00008000u) _p16_word_3_1 &= 0xffff0000u;
        if ((_p16_word_3_1 & 0xffff0000u) == 0x80000000u) _p16_word_3_1 &= 0x0000ffffu;
        _packed16_sanitize_0[1] = _p16_word_3_1;
        uint32_t _p16_word_3_2 = _p16_packet_3.z;
        if ((_p16_word_3_2 & 0x0000ffffu) == 0x00008000u) _p16_word_3_2 &= 0xffff0000u;
        if ((_p16_word_3_2 & 0xffff0000u) == 0x80000000u) _p16_word_3_2 &= 0x0000ffffu;
        _packed16_sanitize_0[2] = _p16_word_3_2;
        uint32_t _p16_word_3_3 = _p16_packet_3.w;
        if ((_p16_word_3_3 & 0x0000ffffu) == 0x00008000u) _p16_word_3_3 &= 0xffff0000u;
        if ((_p16_word_3_3 & 0xffff0000u) == 0x80000000u) _p16_word_3_3 &= 0x0000ffffu;
        _packed16_sanitize_0[3] = _p16_word_3_3;
    }
    int col_a = element;
    int col_b = 3584 + element;
    int shared_row = token * 7168;
    uint32_t _packed16_sanitize_1[4];
    {
        uint4 _p16_packet_4 = *reinterpret_cast<const uint4*>(shared + (shared_row + col_a));
        uint32_t _p16_word_4_0 = _p16_packet_4.x;
        if ((_p16_word_4_0 & 0x0000ffffu) == 0x00008000u) _p16_word_4_0 &= 0xffff0000u;
        if ((_p16_word_4_0 & 0xffff0000u) == 0x80000000u) _p16_word_4_0 &= 0x0000ffffu;
        _packed16_sanitize_1[0] = _p16_word_4_0;
        uint32_t _p16_word_4_1 = _p16_packet_4.y;
        if ((_p16_word_4_1 & 0x0000ffffu) == 0x00008000u) _p16_word_4_1 &= 0xffff0000u;
        if ((_p16_word_4_1 & 0xffff0000u) == 0x80000000u) _p16_word_4_1 &= 0x0000ffffu;
        _packed16_sanitize_1[1] = _p16_word_4_1;
        uint32_t _p16_word_4_2 = _p16_packet_4.z;
        if ((_p16_word_4_2 & 0x0000ffffu) == 0x00008000u) _p16_word_4_2 &= 0xffff0000u;
        if ((_p16_word_4_2 & 0xffff0000u) == 0x80000000u) _p16_word_4_2 &= 0x0000ffffu;
        _packed16_sanitize_1[2] = _p16_word_4_2;
        uint32_t _p16_word_4_3 = _p16_packet_4.w;
        if ((_p16_word_4_3 & 0x0000ffffu) == 0x00008000u) _p16_word_4_3 &= 0xffff0000u;
        if ((_p16_word_4_3 & 0xffff0000u) == 0x80000000u) _p16_word_4_3 &= 0x0000ffffu;
        _packed16_sanitize_1[3] = _p16_word_4_3;
    }
    uint32_t _packed16_sanitize_2[4];
    {
        uint4 _p16_packet_5 = *reinterpret_cast<const uint4*>(shared + (shared_row + col_b));
        uint32_t _p16_word_5_0 = _p16_packet_5.x;
        if ((_p16_word_5_0 & 0x0000ffffu) == 0x00008000u) _p16_word_5_0 &= 0xffff0000u;
        if ((_p16_word_5_0 & 0xffff0000u) == 0x80000000u) _p16_word_5_0 &= 0x0000ffffu;
        _packed16_sanitize_2[0] = _p16_word_5_0;
        uint32_t _p16_word_5_1 = _p16_packet_5.y;
        if ((_p16_word_5_1 & 0x0000ffffu) == 0x00008000u) _p16_word_5_1 &= 0xffff0000u;
        if ((_p16_word_5_1 & 0xffff0000u) == 0x80000000u) _p16_word_5_1 &= 0x0000ffffu;
        _packed16_sanitize_2[1] = _p16_word_5_1;
        uint32_t _p16_word_5_2 = _p16_packet_5.z;
        if ((_p16_word_5_2 & 0x0000ffffu) == 0x00008000u) _p16_word_5_2 &= 0xffff0000u;
        if ((_p16_word_5_2 & 0xffff0000u) == 0x80000000u) _p16_word_5_2 &= 0x0000ffffu;
        _packed16_sanitize_2[2] = _p16_word_5_2;
        uint32_t _p16_word_5_3 = _p16_packet_5.w;
        if ((_p16_word_5_3 & 0x0000ffffu) == 0x00008000u) _p16_word_5_3 &= 0xffff0000u;
        if ((_p16_word_5_3 & 0xffff0000u) == 0x80000000u) _p16_word_5_3 &= 0x0000ffffu;
        _packed16_sanitize_2[3] = _p16_word_5_3;
    }
    unsigned long long publish_index = buffer_base + ((unsigned long long)token * 12 + 8) * 3584 + (unsigned long long)element;
    reinterpret_cast<int4*>(mcast_ptr + publish_index)[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
    int owner_a = 0;
    int owner_a_begin = 0;
    if (col_a >= 640) {
        owner_a = owner_a + 1;
        owner_a_begin = 640;
    }
    if (col_a >= 1280) {
        owner_a = owner_a + 1;
        owner_a_begin = 1280;
    }
    if (col_a >= 1920) {
        owner_a = owner_a + 1;
        owner_a_begin = 1920;
    }
    if (col_a >= 2560) {
        owner_a = owner_a + 1;
        owner_a_begin = 2560;
    }
    if (col_a >= 3200) {
        owner_a = owner_a + 1;
        owner_a_begin = 3200;
    }
    if (col_a >= 3840) {
        owner_a = owner_a + 1;
        owner_a_begin = 3840;
    }
    if (col_a >= 4480) {
        owner_a = owner_a + 1;
        owner_a_begin = 4480;
    }
    if (col_a >= 5120) {
        owner_a = owner_a + 1;
        owner_a_begin = 5120;
    }
    if (col_a >= 5632) {
        owner_a = owner_a + 1;
        owner_a_begin = 5632;
    }
    if (col_a >= 6144) {
        owner_a = owner_a + 1;
        owner_a_begin = 6144;
    }
    if (col_a >= 6656) {
        owner_a = owner_a + 1;
        owner_a_begin = 6656;
    }
    unsigned long long scatter_a = ((unsigned long long)token * 12 + 8) * 640 + (unsigned long long)(col_a - owner_a_begin);
    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[owner_a]) + (k3_epoch_elements + scatter_a))[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
    int owner_b = 0;
    int owner_b_begin = 0;
    if (col_b >= 640) {
        owner_b = owner_b + 1;
        owner_b_begin = 640;
    }
    if (col_b >= 1280) {
        owner_b = owner_b + 1;
        owner_b_begin = 1280;
    }
    if (col_b >= 1920) {
        owner_b = owner_b + 1;
        owner_b_begin = 1920;
    }
    if (col_b >= 2560) {
        owner_b = owner_b + 1;
        owner_b_begin = 2560;
    }
    if (col_b >= 3200) {
        owner_b = owner_b + 1;
        owner_b_begin = 3200;
    }
    if (col_b >= 3840) {
        owner_b = owner_b + 1;
        owner_b_begin = 3840;
    }
    if (col_b >= 4480) {
        owner_b = owner_b + 1;
        owner_b_begin = 4480;
    }
    if (col_b >= 5120) {
        owner_b = owner_b + 1;
        owner_b_begin = 5120;
    }
    if (col_b >= 5632) {
        owner_b = owner_b + 1;
        owner_b_begin = 5632;
    }
    if (col_b >= 6144) {
        owner_b = owner_b + 1;
        owner_b_begin = 6144;
    }
    if (col_b >= 6656) {
        owner_b = owner_b + 1;
        owner_b_begin = 6656;
    }
    unsigned long long scatter_b = ((unsigned long long)token * 12 + 8) * 640 + (unsigned long long)(col_b - owner_b_begin);
    reinterpret_cast<int4*>(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[owner_b]) + (k3_epoch_elements + scatter_b))[0] = reinterpret_cast<int4*>(_packed16_sanitize_2)[0];
    if (warp == 0) {
        asm volatile("barrier.sync 1, 448;" ::: "memory");
    } else {
        asm volatile("barrier.arrive 1, 448;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    {
        unsigned char* _mlc_base_6 = reinterpret_cast<unsigned char*>(local_unicast_ptr);
        uint32_t _mlc_dirty_6 = static_cast<uint32_t>(_vec_load_0[1]);
        uint32_t _mlc_stage_bytes_6 = static_cast<uint32_t>(dirty_stage_bytes);
        uint32_t _mlc_stages_6 = static_cast<uint32_t>(dirty_num_stages);
        uint32_t _mlc_clear_6[4] = {static_cast<uint32_t>(_vec_load_1[0]), static_cast<uint32_t>(_vec_load_1[1]), static_cast<uint32_t>(_vec_load_1[2]), static_cast<uint32_t>(_vec_load_1[3])};
        uint32_t _mlc_global_cta_6 = blockIdx.x * gridDim.y + blockIdx.y;
        uint32_t _mlc_global_tid_6 = _mlc_global_cta_6 * blockDim.x + threadIdx.x;
        uint32_t _mlc_num_threads_6 = gridDim.x * gridDim.y * blockDim.x;
        uint4 _mlc_init_6 = {0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u};
        for (uint32_t _mlc_stage = 0; _mlc_stage < _mlc_stages_6; ++_mlc_stage) {
            size_t _mlc_stage_offset = static_cast<size_t>((static_cast<uint8_t>(_mlc_dirty_6) * _mlc_stages_6 + static_cast<uint8_t>(_mlc_stage)) * static_cast<size_t>(_mlc_stage_bytes_6));
            uint32_t _mlc_boundary = (_mlc_clear_6[_mlc_stage] + 15u) / 16u;
            for (uint32_t _mlc_packed = _mlc_global_tid_6; _mlc_packed < _mlc_boundary; _mlc_packed += _mlc_num_threads_6) {
                reinterpret_cast<uint4*>(_mlc_base_6 + _mlc_stage_offset)[_mlc_packed] = _mlc_init_6;
            }
        }
    }
    uint32_t _mnnvl_oneshot_reduce_0[4];
    {
        uint32_t _mor_rank_7_0[4];
        uint32_t _mor_rank_7_1[4];
        uint32_t _mor_rank_7_2[4];
        uint32_t _mor_rank_7_3[4];
        uint32_t _mor_rank_7_4[4];
        uint32_t _mor_rank_7_5[4];
        uint32_t _mor_rank_7_6[4];
        uint32_t _mor_rank_7_7[4];
        uint32_t _mor_rank_7_9[4];
        uint32_t _mor_rank_7_10[4];
        uint32_t _mor_rank_7_11[4];
        bool _mor_valid_7;
        do {
            uint32_t _mor_retry_rank_7_0[4];
            uint32_t _mor_retry_rank_7_1[4];
            uint32_t _mor_retry_rank_7_2[4];
            uint32_t _mor_retry_rank_7_3[4];
            uint32_t _mor_retry_rank_7_4[4];
            uint32_t _mor_retry_rank_7_5[4];
            uint32_t _mor_retry_rank_7_6[4];
            uint32_t _mor_retry_rank_7_7[4];
            uint32_t _mor_retry_rank_7_9[4];
            uint32_t _mor_retry_rank_7_10[4];
            uint32_t _mor_retry_rank_7_11[4];
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_0[0]), "=r"(_mor_rank_7_0[1]), "=r"(_mor_rank_7_0[2]), "=r"(_mor_rank_7_0[3]) : "l"(local_unicast_ptr + (buffer_base + (unsigned long long)token * 12 * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_1[0]), "=r"(_mor_rank_7_1[1]), "=r"(_mor_rank_7_1[2]), "=r"(_mor_rank_7_1[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 1) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_2[0]), "=r"(_mor_rank_7_2[1]), "=r"(_mor_rank_7_2[2]), "=r"(_mor_rank_7_2[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 2) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_3[0]), "=r"(_mor_rank_7_3[1]), "=r"(_mor_rank_7_3[2]), "=r"(_mor_rank_7_3[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 3) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_4[0]), "=r"(_mor_rank_7_4[1]), "=r"(_mor_rank_7_4[2]), "=r"(_mor_rank_7_4[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 4) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_5[0]), "=r"(_mor_rank_7_5[1]), "=r"(_mor_rank_7_5[2]), "=r"(_mor_rank_7_5[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 5) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_6[0]), "=r"(_mor_rank_7_6[1]), "=r"(_mor_rank_7_6[2]), "=r"(_mor_rank_7_6[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 6) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_7[0]), "=r"(_mor_rank_7_7[1]), "=r"(_mor_rank_7_7[2]), "=r"(_mor_rank_7_7[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 7) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_9[0]), "=r"(_mor_rank_7_9[1]), "=r"(_mor_rank_7_9[2]), "=r"(_mor_rank_7_9[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 9) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_10[0]), "=r"(_mor_rank_7_10[1]), "=r"(_mor_rank_7_10[2]), "=r"(_mor_rank_7_10[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 10) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_7_11[0]), "=r"(_mor_rank_7_11[1]), "=r"(_mor_rank_7_11[2]), "=r"(_mor_rank_7_11[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 11) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_0[0]), "=r"(_mor_retry_rank_7_0[1]), "=r"(_mor_retry_rank_7_0[2]), "=r"(_mor_retry_rank_7_0[3]) : "l"(local_unicast_ptr + (buffer_base + (unsigned long long)token * 12 * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_1[0]), "=r"(_mor_retry_rank_7_1[1]), "=r"(_mor_retry_rank_7_1[2]), "=r"(_mor_retry_rank_7_1[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 1) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_2[0]), "=r"(_mor_retry_rank_7_2[1]), "=r"(_mor_retry_rank_7_2[2]), "=r"(_mor_retry_rank_7_2[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 2) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_3[0]), "=r"(_mor_retry_rank_7_3[1]), "=r"(_mor_retry_rank_7_3[2]), "=r"(_mor_retry_rank_7_3[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 3) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_4[0]), "=r"(_mor_retry_rank_7_4[1]), "=r"(_mor_retry_rank_7_4[2]), "=r"(_mor_retry_rank_7_4[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 4) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_5[0]), "=r"(_mor_retry_rank_7_5[1]), "=r"(_mor_retry_rank_7_5[2]), "=r"(_mor_retry_rank_7_5[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 5) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_6[0]), "=r"(_mor_retry_rank_7_6[1]), "=r"(_mor_retry_rank_7_6[2]), "=r"(_mor_retry_rank_7_6[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 6) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_7[0]), "=r"(_mor_retry_rank_7_7[1]), "=r"(_mor_retry_rank_7_7[2]), "=r"(_mor_retry_rank_7_7[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 7) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_9[0]), "=r"(_mor_retry_rank_7_9[1]), "=r"(_mor_retry_rank_7_9[2]), "=r"(_mor_retry_rank_7_9[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 9) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_10[0]), "=r"(_mor_retry_rank_7_10[1]), "=r"(_mor_retry_rank_7_10[2]), "=r"(_mor_retry_rank_7_10[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 10) * 3584 + (unsigned long long)element)) : "memory");
            asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_7_11[0]), "=r"(_mor_retry_rank_7_11[1]), "=r"(_mor_retry_rank_7_11[2]), "=r"(_mor_retry_rank_7_11[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 11) * 3584 + (unsigned long long)element)) : "memory");
            bool _mor_dirty_7 = false;
            bool _mor_retry_dirty_7 = false;
            _mor_dirty_7 |= (_mor_rank_7_0[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_0[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_0[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_0[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_0[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_0[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_0[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_0[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_1[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_1[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_1[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_1[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_1[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_1[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_1[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_1[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_2[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_2[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_2[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_2[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_2[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_2[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_2[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_2[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_3[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_3[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_3[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_3[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_3[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_3[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_3[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_3[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_4[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_4[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_4[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_4[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_4[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_4[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_4[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_4[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_5[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_5[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_5[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_5[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_5[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_5[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_5[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_5[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_6[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_6[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_6[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_6[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_6[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_6[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_6[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_6[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_7[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_7[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_7[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_7[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_7[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_7[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_7[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_7[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_9[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_9[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_9[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_9[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_9[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_9[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_9[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_9[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_10[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_10[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_10[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_10[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_10[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_10[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_10[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_10[3] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_11[0] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_11[0] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_11[1] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_11[1] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_11[2] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_11[2] == 0x80000000u);
            _mor_dirty_7 |= (_mor_rank_7_11[3] == 0x80000000u);
            _mor_retry_dirty_7 |= (_mor_retry_rank_7_11[3] == 0x80000000u);
            _mor_valid_7 = !_mor_dirty_7;
            if (!_mor_valid_7 && !_mor_retry_dirty_7) {
                _mor_rank_7_0[0] = _mor_retry_rank_7_0[0];
                _mor_rank_7_0[1] = _mor_retry_rank_7_0[1];
                _mor_rank_7_0[2] = _mor_retry_rank_7_0[2];
                _mor_rank_7_0[3] = _mor_retry_rank_7_0[3];
                _mor_rank_7_1[0] = _mor_retry_rank_7_1[0];
                _mor_rank_7_1[1] = _mor_retry_rank_7_1[1];
                _mor_rank_7_1[2] = _mor_retry_rank_7_1[2];
                _mor_rank_7_1[3] = _mor_retry_rank_7_1[3];
                _mor_rank_7_2[0] = _mor_retry_rank_7_2[0];
                _mor_rank_7_2[1] = _mor_retry_rank_7_2[1];
                _mor_rank_7_2[2] = _mor_retry_rank_7_2[2];
                _mor_rank_7_2[3] = _mor_retry_rank_7_2[3];
                _mor_rank_7_3[0] = _mor_retry_rank_7_3[0];
                _mor_rank_7_3[1] = _mor_retry_rank_7_3[1];
                _mor_rank_7_3[2] = _mor_retry_rank_7_3[2];
                _mor_rank_7_3[3] = _mor_retry_rank_7_3[3];
                _mor_rank_7_4[0] = _mor_retry_rank_7_4[0];
                _mor_rank_7_4[1] = _mor_retry_rank_7_4[1];
                _mor_rank_7_4[2] = _mor_retry_rank_7_4[2];
                _mor_rank_7_4[3] = _mor_retry_rank_7_4[3];
                _mor_rank_7_5[0] = _mor_retry_rank_7_5[0];
                _mor_rank_7_5[1] = _mor_retry_rank_7_5[1];
                _mor_rank_7_5[2] = _mor_retry_rank_7_5[2];
                _mor_rank_7_5[3] = _mor_retry_rank_7_5[3];
                _mor_rank_7_6[0] = _mor_retry_rank_7_6[0];
                _mor_rank_7_6[1] = _mor_retry_rank_7_6[1];
                _mor_rank_7_6[2] = _mor_retry_rank_7_6[2];
                _mor_rank_7_6[3] = _mor_retry_rank_7_6[3];
                _mor_rank_7_7[0] = _mor_retry_rank_7_7[0];
                _mor_rank_7_7[1] = _mor_retry_rank_7_7[1];
                _mor_rank_7_7[2] = _mor_retry_rank_7_7[2];
                _mor_rank_7_7[3] = _mor_retry_rank_7_7[3];
                _mor_rank_7_9[0] = _mor_retry_rank_7_9[0];
                _mor_rank_7_9[1] = _mor_retry_rank_7_9[1];
                _mor_rank_7_9[2] = _mor_retry_rank_7_9[2];
                _mor_rank_7_9[3] = _mor_retry_rank_7_9[3];
                _mor_rank_7_10[0] = _mor_retry_rank_7_10[0];
                _mor_rank_7_10[1] = _mor_retry_rank_7_10[1];
                _mor_rank_7_10[2] = _mor_retry_rank_7_10[2];
                _mor_rank_7_10[3] = _mor_retry_rank_7_10[3];
                _mor_rank_7_11[0] = _mor_retry_rank_7_11[0];
                _mor_rank_7_11[1] = _mor_retry_rank_7_11[1];
                _mor_rank_7_11[2] = _mor_retry_rank_7_11[2];
                _mor_rank_7_11[3] = _mor_retry_rank_7_11[3];
                _mor_valid_7 = true;
            }
        } while (!_mor_valid_7);
        {
            float2 _mor_accum_7_0 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_0[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_1[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_2[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_3[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_4[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_5[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_6[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_7[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_packed16_sanitize_0[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_9[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_10[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_0).x), "+f"((_mor_accum_7_0).y) : "r"(_mor_rank_7_11[0]));
            __nv_bfloat162 _mor_out_7_0 = __float22bfloat162_rn(_mor_accum_7_0);
            _mnnvl_oneshot_reduce_0[0] = *reinterpret_cast<uint32_t*>(&_mor_out_7_0);
        }
        {
            float2 _mor_accum_7_1 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_0[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_1[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_2[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_3[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_4[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_5[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_6[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_7[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_packed16_sanitize_0[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_9[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_10[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_1).x), "+f"((_mor_accum_7_1).y) : "r"(_mor_rank_7_11[1]));
            __nv_bfloat162 _mor_out_7_1 = __float22bfloat162_rn(_mor_accum_7_1);
            _mnnvl_oneshot_reduce_0[1] = *reinterpret_cast<uint32_t*>(&_mor_out_7_1);
        }
        {
            float2 _mor_accum_7_2 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_0[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_1[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_2[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_3[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_4[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_5[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_6[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_7[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_packed16_sanitize_0[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_9[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_10[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_2).x), "+f"((_mor_accum_7_2).y) : "r"(_mor_rank_7_11[2]));
            __nv_bfloat162 _mor_out_7_2 = __float22bfloat162_rn(_mor_accum_7_2);
            _mnnvl_oneshot_reduce_0[2] = *reinterpret_cast<uint32_t*>(&_mor_out_7_2);
        }
        {
            float2 _mor_accum_7_3 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_0[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_1[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_2[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_3[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_4[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_5[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_6[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_7[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_packed16_sanitize_0[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_9[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_10[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_7_3).x), "+f"((_mor_accum_7_3).y) : "r"(_mor_rank_7_11[3]));
            __nv_bfloat162 _mor_out_7_3 = __float22bfloat162_rn(_mor_accum_7_3);
            _mnnvl_oneshot_reduce_0[3] = *reinterpret_cast<uint32_t*>(&_mor_out_7_3);
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float _mnnvl_oneshot_reduce_0_f32[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        asm volatile(
            "{\n\t"
            "shl.b32 %0, %2, 16;\n\t"
            "and.b32 %1, %2, 0xffff0000;\n\t"
            "}\n"
            : "=f"((&_mnnvl_oneshot_reduce_0_f32[_pair * 2])[0]), "=f"((&_mnnvl_oneshot_reduce_0_f32[_pair * 2])[1])
            : "r"(_mnnvl_oneshot_reduce_0[_pair]));
    }
    float thread_sum = 0.0f;
    for (int lane_1 = 0; lane_1 < 8; lane_1++) {
        thread_sum = thread_sum + _mnnvl_oneshot_reduce_0_f32[lane_1] * _mnnvl_oneshot_reduce_0_f32[lane_1];
    }
    // BlockReduceF32: threads=448, warps=14, all-thread broadcast
    float _block_reduce_f32_8_accum = thread_sum;
    const int _block_reduce_f32_8_lane = threadIdx.x & 31;
    const int _block_reduce_f32_8_warp = threadIdx.x >> 5;
    // Full CTA schedule: one barrier; every warp performs level two.
    #pragma unroll
    for (int _block_reduce_f32_8_offset_level1 = 16; _block_reduce_f32_8_offset_level1 > 0; _block_reduce_f32_8_offset_level1 >>= 1) {
        _block_reduce_f32_8_accum += __shfl_xor_sync(0xffffffffu, _block_reduce_f32_8_accum, _block_reduce_f32_8_offset_level1);
    }
    if (_block_reduce_f32_8_lane == 0) { reduce_scratch[_block_reduce_f32_8_warp] = _block_reduce_f32_8_accum; }
    __syncthreads();
    float _block_reduce_f32_8_level2 = (_block_reduce_f32_8_lane < 14) ? reduce_scratch[_block_reduce_f32_8_lane] : 0.0f;
    #pragma unroll
    for (int _block_reduce_f32_8_offset_level2 = 16; _block_reduce_f32_8_offset_level2 > 0; _block_reduce_f32_8_offset_level2 >>= 1) {
        _block_reduce_f32_8_level2 += __shfl_xor_sync(0xffffffffu, _block_reduce_f32_8_level2, _block_reduce_f32_8_offset_level2);
    }
    float _block_reduce_f32_0 = _block_reduce_f32_8_level2;
    float _rsqrt_0 = rsqrtf(_block_reduce_f32_0 / 3584.0f + epsilon);
    float rcp_rms = _rsqrt_0;
    float xn_f32[8];
    for (int lane_2 = 0; lane_2 < 8; lane_2++) {
        xn_f32[lane_2] = _mnnvl_oneshot_reduce_0_f32[lane_2] * rcp_rms;
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
    float _vec_load_3[8];
    {
        const uint4* _vptr_9 = reinterpret_cast<const uint4*>(gamma + element + 0);
        uint4 _vld_9[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_9[_blk] = _vptr_9[_blk];
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                    : "r"(_vpairs_9[_pair]));
            }
        }
    }
    float y_f32[8];
    for (int lane_3 = 0; lane_3 < 8; lane_3++) {
        y_f32[lane_3] = xn_f32_bf16_f32[lane_3] * _vec_load_3[lane_3];
    }
    uint32_t y_f32_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_f32[_lp*2 + 0], y_f32[_lp*2+1 + 0]));
        y_f32_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(y_out + input_index)[0] = reinterpret_cast<int4*>(y_f32_bf16)[0];
    if (token == 0) {
        if (tid == 0) {
            {
                unsigned int* _mlf_flags_10 = reinterpret_cast<unsigned int*>(buffer_flags);
                volatile unsigned int* _mlf_access_10 = _mlf_flags_10 + 8;
                while (*_mlf_access_10 < static_cast<unsigned int>(num_tokens)) {}
                uint4* _mlf_vectors_10 = reinterpret_cast<uint4*>(_mlf_flags_10);
                _mlf_vectors_10[0] = {
                    (static_cast<unsigned int>(_vec_load_0[0]) + 1u) % 3u,
                    static_cast<unsigned int>(_vec_load_0[0]),
                    static_cast<unsigned int>(_vec_load_0[2]),
                    static_cast<unsigned int>(1)
                };
                _mlf_vectors_10[1] = {
                    static_cast<unsigned int>(num_tokens * 3584 * 12 * 2),
                    static_cast<unsigned int>(0),
                    static_cast<unsigned int>(0),
                    static_cast<unsigned int>(0)
                };
                _mlf_flags_10[8] = 0u;
            }
        }
    }
}

} // extern "C"
