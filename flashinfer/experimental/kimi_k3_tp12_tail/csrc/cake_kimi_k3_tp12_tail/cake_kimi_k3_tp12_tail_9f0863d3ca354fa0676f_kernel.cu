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
#define SMEM_REDUCE_SCRATCH_OFF 0
#define SMEM_REDUCE_SCRATCH_STAGE_BYTES 128
#define SMEM_REDUCE_SCRATCH_STRIDE 128
#define SMEM_TOTAL 128
#define THREADS 448

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_9f0863d3ca354fa0676f(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ y_out, __nv_bfloat16* __restrict__ gamma, __nv_bfloat16* __restrict__ mcast_ptr, __nv_bfloat16* __restrict__ local_unicast_ptr, unsigned int* __restrict__ buffer_flags, long long* __restrict__ k3_peer_ptrs, __nv_bfloat16* __restrict__ k3_mcast_ptr, unsigned int* __restrict__ k3_flags, int num_tokens, int rank, float epsilon)
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

    // === Task calls (declaration order) ===
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
    unsigned int _vec_load_3[4];
    {
        uint4 _uv4_3 = *reinterpret_cast<const uint4*>(routed + input_index + 0);
        _vec_load_3[0 + 0] = _uv4_3.x;
        _vec_load_3[0 + 1] = _uv4_3.y;
        _vec_load_3[0 + 2] = _uv4_3.z;
        _vec_load_3[0 + 3] = _uv4_3.w;
    }
    uint32_t _packed16_sanitize_0[4];
    {
        uint32_t _p16_word_4_0 = _vec_load_3[0];
        if ((_p16_word_4_0 & 0x0000ffffu) == 0x00008000u) _p16_word_4_0 &= 0xffff0000u;
        if ((_p16_word_4_0 & 0xffff0000u) == 0x80000000u) _p16_word_4_0 &= 0x0000ffffu;
        _packed16_sanitize_0[0] = _p16_word_4_0;
        uint32_t _p16_word_4_1 = _vec_load_3[1];
        if ((_p16_word_4_1 & 0x0000ffffu) == 0x00008000u) _p16_word_4_1 &= 0xffff0000u;
        if ((_p16_word_4_1 & 0xffff0000u) == 0x80000000u) _p16_word_4_1 &= 0x0000ffffu;
        _packed16_sanitize_0[1] = _p16_word_4_1;
        uint32_t _p16_word_4_2 = _vec_load_3[2];
        if ((_p16_word_4_2 & 0x0000ffffu) == 0x00008000u) _p16_word_4_2 &= 0xffff0000u;
        if ((_p16_word_4_2 & 0xffff0000u) == 0x80000000u) _p16_word_4_2 &= 0x0000ffffu;
        _packed16_sanitize_0[2] = _p16_word_4_2;
        uint32_t _p16_word_4_3 = _vec_load_3[3];
        if ((_p16_word_4_3 & 0x0000ffffu) == 0x00008000u) _p16_word_4_3 &= 0xffff0000u;
        if ((_p16_word_4_3 & 0xffff0000u) == 0x80000000u) _p16_word_4_3 &= 0x0000ffffu;
        _packed16_sanitize_0[3] = _p16_word_4_3;
    }
    unsigned int _vec_load_4[4];
    {
        uint4 _uv4_5 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(gamma) + (element / 2) + 0);
        _vec_load_4[0 + 0] = _uv4_5.x;
        _vec_load_4[0 + 1] = _uv4_5.y;
        _vec_load_4[0 + 2] = _uv4_5.z;
        _vec_load_4[0 + 3] = _uv4_5.w;
    }
    int col_a = element;
    int col_b = 3584 + element;
    int shared_row = token * 7168;
    unsigned int _vec_load_5[4];
    {
        uint4 _uv4_6 = *reinterpret_cast<const uint4*>(shared + (shared_row + col_a) + 0);
        _vec_load_5[0 + 0] = _uv4_6.x;
        _vec_load_5[0 + 1] = _uv4_6.y;
        _vec_load_5[0 + 2] = _uv4_6.z;
        _vec_load_5[0 + 3] = _uv4_6.w;
    }
    uint32_t _packed16_sanitize_1[4];
    {
        uint32_t _p16_word_7_0 = _vec_load_5[0];
        if ((_p16_word_7_0 & 0x0000ffffu) == 0x00008000u) _p16_word_7_0 &= 0xffff0000u;
        if ((_p16_word_7_0 & 0xffff0000u) == 0x80000000u) _p16_word_7_0 &= 0x0000ffffu;
        _packed16_sanitize_1[0] = _p16_word_7_0;
        uint32_t _p16_word_7_1 = _vec_load_5[1];
        if ((_p16_word_7_1 & 0x0000ffffu) == 0x00008000u) _p16_word_7_1 &= 0xffff0000u;
        if ((_p16_word_7_1 & 0xffff0000u) == 0x80000000u) _p16_word_7_1 &= 0x0000ffffu;
        _packed16_sanitize_1[1] = _p16_word_7_1;
        uint32_t _p16_word_7_2 = _vec_load_5[2];
        if ((_p16_word_7_2 & 0x0000ffffu) == 0x00008000u) _p16_word_7_2 &= 0xffff0000u;
        if ((_p16_word_7_2 & 0xffff0000u) == 0x80000000u) _p16_word_7_2 &= 0x0000ffffu;
        _packed16_sanitize_1[2] = _p16_word_7_2;
        uint32_t _p16_word_7_3 = _vec_load_5[3];
        if ((_p16_word_7_3 & 0x0000ffffu) == 0x00008000u) _p16_word_7_3 &= 0xffff0000u;
        if ((_p16_word_7_3 & 0xffff0000u) == 0x80000000u) _p16_word_7_3 &= 0x0000ffffu;
        _packed16_sanitize_1[3] = _p16_word_7_3;
    }
    unsigned int _vec_load_6[4];
    {
        uint4 _uv4_8 = *reinterpret_cast<const uint4*>(shared + (shared_row + col_b) + 0);
        _vec_load_6[0 + 0] = _uv4_8.x;
        _vec_load_6[0 + 1] = _uv4_8.y;
        _vec_load_6[0 + 2] = _uv4_8.z;
        _vec_load_6[0 + 3] = _uv4_8.w;
    }
    uint32_t _packed16_sanitize_2[4];
    {
        uint32_t _p16_word_9_0 = _vec_load_6[0];
        if ((_p16_word_9_0 & 0x0000ffffu) == 0x00008000u) _p16_word_9_0 &= 0xffff0000u;
        if ((_p16_word_9_0 & 0xffff0000u) == 0x80000000u) _p16_word_9_0 &= 0x0000ffffu;
        _packed16_sanitize_2[0] = _p16_word_9_0;
        uint32_t _p16_word_9_1 = _vec_load_6[1];
        if ((_p16_word_9_1 & 0x0000ffffu) == 0x00008000u) _p16_word_9_1 &= 0xffff0000u;
        if ((_p16_word_9_1 & 0xffff0000u) == 0x80000000u) _p16_word_9_1 &= 0x0000ffffu;
        _packed16_sanitize_2[1] = _p16_word_9_1;
        uint32_t _p16_word_9_2 = _vec_load_6[2];
        if ((_p16_word_9_2 & 0x0000ffffu) == 0x00008000u) _p16_word_9_2 &= 0xffff0000u;
        if ((_p16_word_9_2 & 0xffff0000u) == 0x80000000u) _p16_word_9_2 &= 0x0000ffffu;
        _packed16_sanitize_2[2] = _p16_word_9_2;
        uint32_t _p16_word_9_3 = _vec_load_6[3];
        if ((_p16_word_9_3 & 0x0000ffffu) == 0x00008000u) _p16_word_9_3 &= 0xffff0000u;
        if ((_p16_word_9_3 & 0xffff0000u) == 0x80000000u) _p16_word_9_3 &= 0x0000ffffu;
        _packed16_sanitize_2[3] = _p16_word_9_3;
    }
    unsigned long long rank_u64 = (unsigned long long)rank;
    unsigned long long publish_index = buffer_base + ((unsigned long long)token * 12 + rank_u64) * 3584 + (unsigned long long)element;
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
    unsigned long long scatter_a = ((unsigned long long)token * 12 + rank_u64) * 640 + (unsigned long long)(col_a - owner_a_begin);
    {
        auto _gst_10 = reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[owner_a]) + (k3_epoch_elements + scatter_a);
        __builtin_assume(__isGlobal(_gst_10));
        reinterpret_cast<int4*>(_gst_10)[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
    }
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
    unsigned long long scatter_b = ((unsigned long long)token * 12 + rank_u64) * 640 + (unsigned long long)(col_b - owner_b_begin);
    {
        auto _gst_11 = reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[owner_b]) + (k3_epoch_elements + scatter_b);
        __builtin_assume(__isGlobal(_gst_11));
        reinterpret_cast<int4*>(_gst_11)[0] = reinterpret_cast<int4*>(_packed16_sanitize_2)[0];
    }
    if (warp == 0) {
        asm volatile("barrier.sync 1, 448;" ::: "memory");
    } else {
        asm volatile("barrier.arrive 1, 448;" ::: "memory");
    }
    if (tid == 0) {
        asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
    }
    uint8_t* bytes_base = reinterpret_cast<uint8_t*>(local_unicast_ptr);
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
    for (unsigned int stage = 0; stage < num_stages; stage++) {
        unsigned long long stage_offset = (unsigned long long)((unsigned int)(uint8_t)dirty * num_stages + (unsigned int)(uint8_t)stage) * (unsigned long long)stage_bytes;
        unsigned int boundary = (clear[stage] + 15) / 16;
        for (unsigned int packed = thread; packed < boundary; packed += threads) {
            {
                auto _gst_12 = reinterpret_cast<unsigned __int128*>(bytes_base + stage_offset) + packed;
                __builtin_assume(__isGlobal(_gst_12));
                reinterpret_cast<int4*>(_gst_12)[0] = reinterpret_cast<int4*>(sentinel)[0];
            }
        }
    }
    uint32_t _mnnvl_oneshot_reduce_0[4];
    {
        uint32_t _mor_rank_13_0[4];
        uint32_t _mor_rank_13_1[4];
        uint32_t _mor_rank_13_2[4];
        uint32_t _mor_rank_13_3[4];
        uint32_t _mor_rank_13_4[4];
        uint32_t _mor_rank_13_5[4];
        uint32_t _mor_rank_13_6[4];
        uint32_t _mor_rank_13_7[4];
        uint32_t _mor_rank_13_8[4];
        uint32_t _mor_rank_13_9[4];
        uint32_t _mor_rank_13_10[4];
        uint32_t _mor_rank_13_11[4];
        bool _mor_valid_13;
        do {
            uint32_t _mor_retry_rank_13_0[4];
            uint32_t _mor_retry_rank_13_1[4];
            uint32_t _mor_retry_rank_13_2[4];
            uint32_t _mor_retry_rank_13_3[4];
            uint32_t _mor_retry_rank_13_4[4];
            uint32_t _mor_retry_rank_13_5[4];
            uint32_t _mor_retry_rank_13_6[4];
            uint32_t _mor_retry_rank_13_7[4];
            uint32_t _mor_retry_rank_13_8[4];
            uint32_t _mor_retry_rank_13_9[4];
            uint32_t _mor_retry_rank_13_10[4];
            uint32_t _mor_retry_rank_13_11[4];
            if (rank == 0) {
                _mor_rank_13_0[0] = _packed16_sanitize_0[0];
                _mor_rank_13_0[1] = _packed16_sanitize_0[1];
                _mor_rank_13_0[2] = _packed16_sanitize_0[2];
                _mor_rank_13_0[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_0[0]), "=r"(_mor_rank_13_0[1]), "=r"(_mor_rank_13_0[2]), "=r"(_mor_rank_13_0[3]) : "l"(local_unicast_ptr + (buffer_base + (unsigned long long)token * 12 * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 1) {
                _mor_rank_13_1[0] = _packed16_sanitize_0[0];
                _mor_rank_13_1[1] = _packed16_sanitize_0[1];
                _mor_rank_13_1[2] = _packed16_sanitize_0[2];
                _mor_rank_13_1[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_1[0]), "=r"(_mor_rank_13_1[1]), "=r"(_mor_rank_13_1[2]), "=r"(_mor_rank_13_1[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 1) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 2) {
                _mor_rank_13_2[0] = _packed16_sanitize_0[0];
                _mor_rank_13_2[1] = _packed16_sanitize_0[1];
                _mor_rank_13_2[2] = _packed16_sanitize_0[2];
                _mor_rank_13_2[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_2[0]), "=r"(_mor_rank_13_2[1]), "=r"(_mor_rank_13_2[2]), "=r"(_mor_rank_13_2[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 2) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 3) {
                _mor_rank_13_3[0] = _packed16_sanitize_0[0];
                _mor_rank_13_3[1] = _packed16_sanitize_0[1];
                _mor_rank_13_3[2] = _packed16_sanitize_0[2];
                _mor_rank_13_3[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_3[0]), "=r"(_mor_rank_13_3[1]), "=r"(_mor_rank_13_3[2]), "=r"(_mor_rank_13_3[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 3) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 4) {
                _mor_rank_13_4[0] = _packed16_sanitize_0[0];
                _mor_rank_13_4[1] = _packed16_sanitize_0[1];
                _mor_rank_13_4[2] = _packed16_sanitize_0[2];
                _mor_rank_13_4[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_4[0]), "=r"(_mor_rank_13_4[1]), "=r"(_mor_rank_13_4[2]), "=r"(_mor_rank_13_4[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 4) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 5) {
                _mor_rank_13_5[0] = _packed16_sanitize_0[0];
                _mor_rank_13_5[1] = _packed16_sanitize_0[1];
                _mor_rank_13_5[2] = _packed16_sanitize_0[2];
                _mor_rank_13_5[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_5[0]), "=r"(_mor_rank_13_5[1]), "=r"(_mor_rank_13_5[2]), "=r"(_mor_rank_13_5[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 5) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 6) {
                _mor_rank_13_6[0] = _packed16_sanitize_0[0];
                _mor_rank_13_6[1] = _packed16_sanitize_0[1];
                _mor_rank_13_6[2] = _packed16_sanitize_0[2];
                _mor_rank_13_6[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_6[0]), "=r"(_mor_rank_13_6[1]), "=r"(_mor_rank_13_6[2]), "=r"(_mor_rank_13_6[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 6) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 7) {
                _mor_rank_13_7[0] = _packed16_sanitize_0[0];
                _mor_rank_13_7[1] = _packed16_sanitize_0[1];
                _mor_rank_13_7[2] = _packed16_sanitize_0[2];
                _mor_rank_13_7[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_7[0]), "=r"(_mor_rank_13_7[1]), "=r"(_mor_rank_13_7[2]), "=r"(_mor_rank_13_7[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 7) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 8) {
                _mor_rank_13_8[0] = _packed16_sanitize_0[0];
                _mor_rank_13_8[1] = _packed16_sanitize_0[1];
                _mor_rank_13_8[2] = _packed16_sanitize_0[2];
                _mor_rank_13_8[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_8[0]), "=r"(_mor_rank_13_8[1]), "=r"(_mor_rank_13_8[2]), "=r"(_mor_rank_13_8[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 8) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 9) {
                _mor_rank_13_9[0] = _packed16_sanitize_0[0];
                _mor_rank_13_9[1] = _packed16_sanitize_0[1];
                _mor_rank_13_9[2] = _packed16_sanitize_0[2];
                _mor_rank_13_9[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_9[0]), "=r"(_mor_rank_13_9[1]), "=r"(_mor_rank_13_9[2]), "=r"(_mor_rank_13_9[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 9) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 10) {
                _mor_rank_13_10[0] = _packed16_sanitize_0[0];
                _mor_rank_13_10[1] = _packed16_sanitize_0[1];
                _mor_rank_13_10[2] = _packed16_sanitize_0[2];
                _mor_rank_13_10[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_10[0]), "=r"(_mor_rank_13_10[1]), "=r"(_mor_rank_13_10[2]), "=r"(_mor_rank_13_10[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 10) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 11) {
                _mor_rank_13_11[0] = _packed16_sanitize_0[0];
                _mor_rank_13_11[1] = _packed16_sanitize_0[1];
                _mor_rank_13_11[2] = _packed16_sanitize_0[2];
                _mor_rank_13_11[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_13_11[0]), "=r"(_mor_rank_13_11[1]), "=r"(_mor_rank_13_11[2]), "=r"(_mor_rank_13_11[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 11) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 0) {
                _mor_retry_rank_13_0[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_0[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_0[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_0[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_0[0]), "=r"(_mor_retry_rank_13_0[1]), "=r"(_mor_retry_rank_13_0[2]), "=r"(_mor_retry_rank_13_0[3]) : "l"(local_unicast_ptr + (buffer_base + (unsigned long long)token * 12 * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 1) {
                _mor_retry_rank_13_1[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_1[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_1[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_1[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_1[0]), "=r"(_mor_retry_rank_13_1[1]), "=r"(_mor_retry_rank_13_1[2]), "=r"(_mor_retry_rank_13_1[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 1) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 2) {
                _mor_retry_rank_13_2[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_2[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_2[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_2[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_2[0]), "=r"(_mor_retry_rank_13_2[1]), "=r"(_mor_retry_rank_13_2[2]), "=r"(_mor_retry_rank_13_2[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 2) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 3) {
                _mor_retry_rank_13_3[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_3[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_3[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_3[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_3[0]), "=r"(_mor_retry_rank_13_3[1]), "=r"(_mor_retry_rank_13_3[2]), "=r"(_mor_retry_rank_13_3[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 3) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 4) {
                _mor_retry_rank_13_4[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_4[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_4[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_4[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_4[0]), "=r"(_mor_retry_rank_13_4[1]), "=r"(_mor_retry_rank_13_4[2]), "=r"(_mor_retry_rank_13_4[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 4) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 5) {
                _mor_retry_rank_13_5[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_5[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_5[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_5[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_5[0]), "=r"(_mor_retry_rank_13_5[1]), "=r"(_mor_retry_rank_13_5[2]), "=r"(_mor_retry_rank_13_5[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 5) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 6) {
                _mor_retry_rank_13_6[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_6[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_6[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_6[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_6[0]), "=r"(_mor_retry_rank_13_6[1]), "=r"(_mor_retry_rank_13_6[2]), "=r"(_mor_retry_rank_13_6[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 6) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 7) {
                _mor_retry_rank_13_7[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_7[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_7[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_7[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_7[0]), "=r"(_mor_retry_rank_13_7[1]), "=r"(_mor_retry_rank_13_7[2]), "=r"(_mor_retry_rank_13_7[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 7) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 8) {
                _mor_retry_rank_13_8[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_8[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_8[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_8[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_8[0]), "=r"(_mor_retry_rank_13_8[1]), "=r"(_mor_retry_rank_13_8[2]), "=r"(_mor_retry_rank_13_8[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 8) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 9) {
                _mor_retry_rank_13_9[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_9[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_9[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_9[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_9[0]), "=r"(_mor_retry_rank_13_9[1]), "=r"(_mor_retry_rank_13_9[2]), "=r"(_mor_retry_rank_13_9[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 9) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 10) {
                _mor_retry_rank_13_10[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_10[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_10[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_10[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_10[0]), "=r"(_mor_retry_rank_13_10[1]), "=r"(_mor_retry_rank_13_10[2]), "=r"(_mor_retry_rank_13_10[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 10) * 3584 + (unsigned long long)element)) : "memory");
            }
            if (rank == 11) {
                _mor_retry_rank_13_11[0] = _packed16_sanitize_0[0];
                _mor_retry_rank_13_11[1] = _packed16_sanitize_0[1];
                _mor_retry_rank_13_11[2] = _packed16_sanitize_0[2];
                _mor_retry_rank_13_11[3] = _packed16_sanitize_0[3];
            } else {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_13_11[0]), "=r"(_mor_retry_rank_13_11[1]), "=r"(_mor_retry_rank_13_11[2]), "=r"(_mor_retry_rank_13_11[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 11) * 3584 + (unsigned long long)element)) : "memory");
            }
            bool _mor_dirty_13 = false;
            bool _mor_retry_dirty_13 = false;
            _mor_dirty_13 |= (_mor_rank_13_0[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_0[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_0[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_0[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_0[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_0[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_0[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_0[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_1[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_1[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_1[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_1[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_1[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_1[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_1[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_1[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_2[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_2[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_2[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_2[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_2[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_2[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_2[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_2[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_3[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_3[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_3[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_3[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_3[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_3[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_3[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_3[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_4[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_4[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_4[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_4[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_4[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_4[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_4[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_4[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_5[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_5[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_5[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_5[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_5[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_5[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_5[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_5[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_6[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_6[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_6[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_6[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_6[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_6[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_6[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_6[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_7[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_7[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_7[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_7[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_7[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_7[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_7[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_7[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_8[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_8[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_8[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_8[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_8[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_8[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_8[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_8[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_9[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_9[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_9[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_9[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_9[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_9[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_9[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_9[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_10[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_10[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_10[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_10[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_10[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_10[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_10[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_10[3] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_11[0] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_11[0] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_11[1] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_11[1] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_11[2] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_11[2] == 0x80000000u);
            _mor_dirty_13 |= (_mor_rank_13_11[3] == 0x80000000u);
            _mor_retry_dirty_13 |= (_mor_retry_rank_13_11[3] == 0x80000000u);
            _mor_valid_13 = !_mor_dirty_13;
            if (!_mor_valid_13 && !_mor_retry_dirty_13) {
                _mor_rank_13_0[0] = _mor_retry_rank_13_0[0];
                _mor_rank_13_0[1] = _mor_retry_rank_13_0[1];
                _mor_rank_13_0[2] = _mor_retry_rank_13_0[2];
                _mor_rank_13_0[3] = _mor_retry_rank_13_0[3];
                _mor_rank_13_1[0] = _mor_retry_rank_13_1[0];
                _mor_rank_13_1[1] = _mor_retry_rank_13_1[1];
                _mor_rank_13_1[2] = _mor_retry_rank_13_1[2];
                _mor_rank_13_1[3] = _mor_retry_rank_13_1[3];
                _mor_rank_13_2[0] = _mor_retry_rank_13_2[0];
                _mor_rank_13_2[1] = _mor_retry_rank_13_2[1];
                _mor_rank_13_2[2] = _mor_retry_rank_13_2[2];
                _mor_rank_13_2[3] = _mor_retry_rank_13_2[3];
                _mor_rank_13_3[0] = _mor_retry_rank_13_3[0];
                _mor_rank_13_3[1] = _mor_retry_rank_13_3[1];
                _mor_rank_13_3[2] = _mor_retry_rank_13_3[2];
                _mor_rank_13_3[3] = _mor_retry_rank_13_3[3];
                _mor_rank_13_4[0] = _mor_retry_rank_13_4[0];
                _mor_rank_13_4[1] = _mor_retry_rank_13_4[1];
                _mor_rank_13_4[2] = _mor_retry_rank_13_4[2];
                _mor_rank_13_4[3] = _mor_retry_rank_13_4[3];
                _mor_rank_13_5[0] = _mor_retry_rank_13_5[0];
                _mor_rank_13_5[1] = _mor_retry_rank_13_5[1];
                _mor_rank_13_5[2] = _mor_retry_rank_13_5[2];
                _mor_rank_13_5[3] = _mor_retry_rank_13_5[3];
                _mor_rank_13_6[0] = _mor_retry_rank_13_6[0];
                _mor_rank_13_6[1] = _mor_retry_rank_13_6[1];
                _mor_rank_13_6[2] = _mor_retry_rank_13_6[2];
                _mor_rank_13_6[3] = _mor_retry_rank_13_6[3];
                _mor_rank_13_7[0] = _mor_retry_rank_13_7[0];
                _mor_rank_13_7[1] = _mor_retry_rank_13_7[1];
                _mor_rank_13_7[2] = _mor_retry_rank_13_7[2];
                _mor_rank_13_7[3] = _mor_retry_rank_13_7[3];
                _mor_rank_13_8[0] = _mor_retry_rank_13_8[0];
                _mor_rank_13_8[1] = _mor_retry_rank_13_8[1];
                _mor_rank_13_8[2] = _mor_retry_rank_13_8[2];
                _mor_rank_13_8[3] = _mor_retry_rank_13_8[3];
                _mor_rank_13_9[0] = _mor_retry_rank_13_9[0];
                _mor_rank_13_9[1] = _mor_retry_rank_13_9[1];
                _mor_rank_13_9[2] = _mor_retry_rank_13_9[2];
                _mor_rank_13_9[3] = _mor_retry_rank_13_9[3];
                _mor_rank_13_10[0] = _mor_retry_rank_13_10[0];
                _mor_rank_13_10[1] = _mor_retry_rank_13_10[1];
                _mor_rank_13_10[2] = _mor_retry_rank_13_10[2];
                _mor_rank_13_10[3] = _mor_retry_rank_13_10[3];
                _mor_rank_13_11[0] = _mor_retry_rank_13_11[0];
                _mor_rank_13_11[1] = _mor_retry_rank_13_11[1];
                _mor_rank_13_11[2] = _mor_retry_rank_13_11[2];
                _mor_rank_13_11[3] = _mor_retry_rank_13_11[3];
                _mor_valid_13 = true;
            }
        } while (!_mor_valid_13);
        {
            float2 _mor_accum_13_0 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_0[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_1[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_2[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_3[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_4[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_5[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_6[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_7[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_8[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_9[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_10[0]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_0).x), "+f"((_mor_accum_13_0).y) : "r"(_mor_rank_13_11[0]));
            __nv_bfloat162 _mor_out_13_0 = __float22bfloat162_rn(_mor_accum_13_0);
            _mnnvl_oneshot_reduce_0[0] = *reinterpret_cast<uint32_t*>(&_mor_out_13_0);
        }
        {
            float2 _mor_accum_13_1 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_0[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_1[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_2[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_3[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_4[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_5[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_6[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_7[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_8[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_9[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_10[1]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_1).x), "+f"((_mor_accum_13_1).y) : "r"(_mor_rank_13_11[1]));
            __nv_bfloat162 _mor_out_13_1 = __float22bfloat162_rn(_mor_accum_13_1);
            _mnnvl_oneshot_reduce_0[1] = *reinterpret_cast<uint32_t*>(&_mor_out_13_1);
        }
        {
            float2 _mor_accum_13_2 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_0[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_1[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_2[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_3[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_4[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_5[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_6[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_7[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_8[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_9[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_10[2]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_2).x), "+f"((_mor_accum_13_2).y) : "r"(_mor_rank_13_11[2]));
            __nv_bfloat162 _mor_out_13_2 = __float22bfloat162_rn(_mor_accum_13_2);
            _mnnvl_oneshot_reduce_0[2] = *reinterpret_cast<uint32_t*>(&_mor_out_13_2);
        }
        {
            float2 _mor_accum_13_3 = make_float2(0.0f, 0.0f);
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_0[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_1[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_2[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_3[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_4[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_5[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_6[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_7[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_8[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_9[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_10[3]));
            asm volatile(
                "{\n\t"
                ".reg .b16 lo, hi;\n\t"
                "mov.b32 {lo, hi}, %2;\n\t"
                "add.rn.f32.bf16 %0, lo, %0;\n\t"
                "add.rn.f32.bf16 %1, hi, %1;\n\t"
                "}\n"
                : "+f"((_mor_accum_13_3).x), "+f"((_mor_accum_13_3).y) : "r"(_mor_rank_13_11[3]));
            __nv_bfloat162 _mor_out_13_3 = __float22bfloat162_rn(_mor_accum_13_3);
            _mnnvl_oneshot_reduce_0[3] = *reinterpret_cast<uint32_t*>(&_mor_out_13_3);
        }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    float _unpack_bf16_0[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_mnnvl_oneshot_reduce_0[_pair]) << 16);
        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_mnnvl_oneshot_reduce_0[_pair]) & 0xffff0000u);
    }
    float thread_sum = 0.0f;
    for (int lane_1 = 0; lane_1 < 8; lane_1++) {
        thread_sum = __fmaf_rn(_unpack_bf16_0[lane_1], _unpack_bf16_0[lane_1], thread_sum);
    }
    // BlockReduce: threads=448, warps=14, all-thread broadcast
    float _block_reduce_15_accum = thread_sum;
    const int _block_reduce_15_lane = threadIdx.x & 31;
    const int _block_reduce_15_warp = threadIdx.x >> 5;
    // Full CTA schedule: one barrier; every warp performs level two.
    #pragma unroll
    for (int _block_reduce_15_offset_level1 = 16; _block_reduce_15_offset_level1 > 0; _block_reduce_15_offset_level1 >>= 1) {
        _block_reduce_15_accum += __shfl_xor_sync(0xffffffffu, _block_reduce_15_accum, _block_reduce_15_offset_level1);
    }
    if (_block_reduce_15_lane == 0) { reduce_scratch[_block_reduce_15_warp] = _block_reduce_15_accum; }
    __syncthreads();
    float _block_reduce_15_level2 = (_block_reduce_15_lane < 14) ? reduce_scratch[_block_reduce_15_lane] : 0.0f;
    #pragma unroll
    for (int _block_reduce_15_offset_level2 = 16; _block_reduce_15_offset_level2 > 0; _block_reduce_15_offset_level2 >>= 1) {
        _block_reduce_15_level2 += __shfl_xor_sync(0xffffffffu, _block_reduce_15_level2, _block_reduce_15_offset_level2);
    }
    float _block_reduce_14 = _block_reduce_15_level2;
    float rcp_rms = rsqrtf(_block_reduce_14 / 3584.0f + epsilon);
    float xn_f32[8];
    for (int lane_2 = 0; lane_2 < 8; lane_2++) {
        xn_f32[lane_2] = _unpack_bf16_0[lane_2] * rcp_rms;
    }
    unsigned int xn_f32_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xn_f32[_lp*2 + 0], xn_f32[_lp*2+1 + 0]));
        xn_f32_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    float _unpack_bf16_1[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(xn_f32_bf16[_pair]) << 16);
        (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(xn_f32_bf16[_pair]) & 0xffff0000u);
    }
    float _unpack_bf16_2[8];
    #pragma unroll
    for (int _pair = 0; _pair < 4; _pair++) {
        (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_4[_pair]) << 16);
        (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_4[_pair]) & 0xffff0000u);
    }
    float y_f32[8];
    for (int lane_3 = 0; lane_3 < 8; lane_3++) {
        y_f32[lane_3] = _unpack_bf16_1[lane_3] * _unpack_bf16_2[lane_3];
    }
    unsigned int y_f32_bf16[4];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_f32[_lp*2 + 0], y_f32[_lp*2+1 + 0]));
        y_f32_bf16[_lp] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int4*>(y_out + input_index)[0] = reinterpret_cast<int4*>(y_f32_bf16)[0];
    if (token == 0) {
        if (tid == 0) {
            {
                volatile unsigned int* _sysv_wait_p_16 = reinterpret_cast<volatile unsigned int*>(buffer_flags) + (8);
                while (*_sysv_wait_p_16 < static_cast<unsigned int>(num_tokens)) {}
            }
            unsigned int current = (unsigned int)_vec_load_0[0];
            unsigned int head[4];
            head[0] = (current + 1) % 3;
            head[1] = current;
            head[2] = (unsigned int)_vec_load_0[2];
            head[3] = 1;
            unsigned int clear_0[4];
            clear_0[0] = (unsigned int)(num_tokens * 3584 * 12 * 2);
            clear_0[1] = 0;
            clear_0[2] = 0;
            clear_0[3] = 0;
            reinterpret_cast<int4*>(buffer_flags)[0] = reinterpret_cast<int4*>(head)[0];
            reinterpret_cast<int4*>(buffer_flags + 4)[0] = reinterpret_cast<int4*>(clear_0)[0];
            buffer_flags[8] = 0;
        }
    }
}

} // extern "C"
