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
#define SMEM_PARTIALS_OFF 0
#define SMEM_PARTIALS_STAGE_BYTES 3840
#define SMEM_PARTIALS_STRIDE 3840
#define SMEM_TOTAL 3840
#define THREADS 448
#define K23_CAPACITY 8

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(448) void
kernel_cake_kimi_k3_tp12_tail_5b5ceb1d68849a0d1565(__nv_bfloat16* __restrict__ routed, __nv_bfloat16* __restrict__ shared, __nv_bfloat16* y, __nv_bfloat16* __restrict__ gamma, __nv_bfloat16* __restrict__ w_slice, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ mcast_ptr, __nv_bfloat16* __restrict__ local_unicast_ptr, unsigned int* __restrict__ buffer_flags, long long* __restrict__ k3_peer_ptrs, __nv_bfloat16* __restrict__ k3_mcast_ptr, unsigned int* __restrict__ k3_flags, int num_tokens, int rank, int my_col_begin, int k23_ctas, float epsilon)
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
    float* partials = reinterpret_cast<float*>(smem_raw + 0);
    const int partials_addr = smem + 0;

    // === Task calls (declaration order) ===
    int kb = blockIdx.x - num_tokens;
    int row0 = kb * 8;
    int k0 = tid * 8;
    int lane_row = (int)(lane >> 2 & 7);
    unsigned int wv[32];
    if (blockIdx.x >= num_tokens) {
        for (int r = 0; r < 8; r++) {
            unsigned int _vec_load_0[4];
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(w_slice) + (((row0 + (r ^ lane_row)) * 3584 + k0) / 2) + 0);
                _vec_load_0[0 + 0] = _uv4_0.x;
                _vec_load_0[0 + 1] = _uv4_0.y;
                _vec_load_0[0 + 2] = _uv4_0.z;
                _vec_load_0[0 + 3] = _uv4_0.w;
            }
            for (int q = 0; q < 4; q++) {
                wv[r * 4 + q] = _vec_load_0[q];
            }
        }
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
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
    unsigned int _vec_load_3[4];
    {
        uint4 _uv4_3 = *reinterpret_cast<const uint4*>(k3_flags + 0);
        _vec_load_3[0 + 0] = _uv4_3.x;
        _vec_load_3[0 + 1] = _uv4_3.y;
        _vec_load_3[0 + 2] = _uv4_3.z;
        _vec_load_3[0 + 3] = _uv4_3.w;
    }
    unsigned int _vec_load_4[4];
    {
        uint4 _uv4_4 = *reinterpret_cast<const uint4*>(k3_flags + 4);
        _vec_load_4[0 + 0] = _uv4_4.x;
        _vec_load_4[0 + 1] = _uv4_4.y;
        _vec_load_4[0 + 2] = _uv4_4.z;
        _vec_load_4[0 + 3] = _uv4_4.w;
    }
    unsigned long long buffer_base = (unsigned long long)_vec_load_1[0] * (unsigned long long)_vec_load_1[2] / 2;
    unsigned int k3_num_stages = _vec_load_3[3];
    unsigned int k3_stage_bytes = 0;
    if (k3_num_stages > 0) {
        k3_stage_bytes = _vec_load_3[2] / k3_num_stages;
    }
    unsigned long long k3_epoch_elements = (unsigned long long)_vec_load_3[0] * (unsigned long long)_vec_load_3[2] / 2;
    unsigned long long k3_stage_elements = (unsigned long long)_vec_load_3[2] / 2 / 2;
    unsigned long long rank_u64 = (unsigned long long)rank;
    if (blockIdx.x < num_tokens) {
        int token = blockIdx.x;
        int input_index = token * 3584 + k0;
        unsigned int _vec_load_5[4];
        {
            uint4 _uv4_5 = *reinterpret_cast<const uint4*>(routed + input_index + 0);
            _vec_load_5[0 + 0] = _uv4_5.x;
            _vec_load_5[0 + 1] = _uv4_5.y;
            _vec_load_5[0 + 2] = _uv4_5.z;
            _vec_load_5[0 + 3] = _uv4_5.w;
        }
        uint32_t _packed16_sanitize_0[4];
        {
            uint32_t _p16_word_6_0 = _vec_load_5[0];
            if ((_p16_word_6_0 & 0x0000ffffu) == 0x00008000u) _p16_word_6_0 &= 0xffff0000u;
            if ((_p16_word_6_0 & 0xffff0000u) == 0x80000000u) _p16_word_6_0 &= 0x0000ffffu;
            _packed16_sanitize_0[0] = _p16_word_6_0;
            uint32_t _p16_word_6_1 = _vec_load_5[1];
            if ((_p16_word_6_1 & 0x0000ffffu) == 0x00008000u) _p16_word_6_1 &= 0xffff0000u;
            if ((_p16_word_6_1 & 0xffff0000u) == 0x80000000u) _p16_word_6_1 &= 0x0000ffffu;
            _packed16_sanitize_0[1] = _p16_word_6_1;
            uint32_t _p16_word_6_2 = _vec_load_5[2];
            if ((_p16_word_6_2 & 0x0000ffffu) == 0x00008000u) _p16_word_6_2 &= 0xffff0000u;
            if ((_p16_word_6_2 & 0xffff0000u) == 0x80000000u) _p16_word_6_2 &= 0x0000ffffu;
            _packed16_sanitize_0[2] = _p16_word_6_2;
            uint32_t _p16_word_6_3 = _vec_load_5[3];
            if ((_p16_word_6_3 & 0x0000ffffu) == 0x00008000u) _p16_word_6_3 &= 0xffff0000u;
            if ((_p16_word_6_3 & 0xffff0000u) == 0x80000000u) _p16_word_6_3 &= 0x0000ffffu;
            _packed16_sanitize_0[3] = _p16_word_6_3;
        }
        unsigned int _vec_load_6[4];
        {
            uint4 _uv4_7 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(gamma) + (k0 / 2) + 0);
            _vec_load_6[0 + 0] = _uv4_7.x;
            _vec_load_6[0 + 1] = _uv4_7.y;
            _vec_load_6[0 + 2] = _uv4_7.z;
            _vec_load_6[0 + 3] = _uv4_7.w;
        }
        unsigned long long publish_index = buffer_base + ((unsigned long long)token * 12 + rank_u64) * 3584 + (unsigned long long)k0;
        reinterpret_cast<int4*>(mcast_ptr + publish_index)[0] = reinterpret_cast<int4*>(_packed16_sanitize_0)[0];
        if (warp == 0) {
            asm volatile("barrier.sync 3, 448;" ::: "memory");
        } else {
            asm volatile("barrier.arrive 3, 448;" ::: "memory");
        }
        if (tid == 0) {
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
        uint8_t* bytes_base = reinterpret_cast<uint8_t*>(local_unicast_ptr);
        unsigned int dirty = (unsigned int)_vec_load_1[1];
        unsigned int stage_bytes = (unsigned int)dirty_stage_bytes;
        unsigned int num_stages = (unsigned int)dirty_num_stages;
        unsigned int clear[4];
        clear[0] = (unsigned int)_vec_load_2[0];
        clear[1] = (unsigned int)_vec_load_2[1];
        clear[2] = (unsigned int)_vec_load_2[2];
        clear[3] = (unsigned int)_vec_load_2[3];
        unsigned int block = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
        unsigned int thread = block * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
        unsigned int threads = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
        unsigned int sentinel[4] = {2147483648, 2147483648, 2147483648, 2147483648};
        for (unsigned int stage = 0; stage < num_stages; stage++) {
            unsigned long long stage_offset = (unsigned long long)((unsigned int)(uint8_t)dirty * num_stages + (unsigned int)(uint8_t)stage) * (unsigned long long)stage_bytes;
            unsigned int boundary = (clear[stage] + 15) / 16;
            for (unsigned int packed = thread; packed < boundary; packed += threads) {
                {
                    auto _gst_8 = reinterpret_cast<unsigned __int128*>(bytes_base + stage_offset) + packed;
                    __builtin_assume(__isGlobal(_gst_8));
                    reinterpret_cast<int4*>(_gst_8)[0] = reinterpret_cast<int4*>(sentinel)[0];
                }
            }
        }
        uint32_t _mnnvl_oneshot_reduce_0[4];
        {
            uint32_t _mor_rank_9_0[4];
            uint32_t _mor_rank_9_1[4];
            uint32_t _mor_rank_9_2[4];
            uint32_t _mor_rank_9_3[4];
            uint32_t _mor_rank_9_4[4];
            uint32_t _mor_rank_9_5[4];
            uint32_t _mor_rank_9_6[4];
            uint32_t _mor_rank_9_7[4];
            uint32_t _mor_rank_9_8[4];
            uint32_t _mor_rank_9_9[4];
            uint32_t _mor_rank_9_10[4];
            uint32_t _mor_rank_9_11[4];
            bool _mor_valid_9;
            do {
                uint32_t _mor_retry_rank_9_0[4];
                uint32_t _mor_retry_rank_9_1[4];
                uint32_t _mor_retry_rank_9_2[4];
                uint32_t _mor_retry_rank_9_3[4];
                uint32_t _mor_retry_rank_9_4[4];
                uint32_t _mor_retry_rank_9_5[4];
                uint32_t _mor_retry_rank_9_6[4];
                uint32_t _mor_retry_rank_9_7[4];
                uint32_t _mor_retry_rank_9_8[4];
                uint32_t _mor_retry_rank_9_9[4];
                uint32_t _mor_retry_rank_9_10[4];
                uint32_t _mor_retry_rank_9_11[4];
                if (rank == 0) {
                    _mor_rank_9_0[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_0[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_0[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_0[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_0[0]), "=r"(_mor_rank_9_0[1]), "=r"(_mor_rank_9_0[2]), "=r"(_mor_rank_9_0[3]) : "l"(local_unicast_ptr + (buffer_base + (unsigned long long)token * 12 * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 1) {
                    _mor_rank_9_1[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_1[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_1[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_1[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_1[0]), "=r"(_mor_rank_9_1[1]), "=r"(_mor_rank_9_1[2]), "=r"(_mor_rank_9_1[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 1) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 2) {
                    _mor_rank_9_2[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_2[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_2[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_2[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_2[0]), "=r"(_mor_rank_9_2[1]), "=r"(_mor_rank_9_2[2]), "=r"(_mor_rank_9_2[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 2) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 3) {
                    _mor_rank_9_3[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_3[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_3[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_3[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_3[0]), "=r"(_mor_rank_9_3[1]), "=r"(_mor_rank_9_3[2]), "=r"(_mor_rank_9_3[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 3) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 4) {
                    _mor_rank_9_4[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_4[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_4[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_4[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_4[0]), "=r"(_mor_rank_9_4[1]), "=r"(_mor_rank_9_4[2]), "=r"(_mor_rank_9_4[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 4) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 5) {
                    _mor_rank_9_5[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_5[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_5[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_5[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_5[0]), "=r"(_mor_rank_9_5[1]), "=r"(_mor_rank_9_5[2]), "=r"(_mor_rank_9_5[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 5) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 6) {
                    _mor_rank_9_6[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_6[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_6[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_6[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_6[0]), "=r"(_mor_rank_9_6[1]), "=r"(_mor_rank_9_6[2]), "=r"(_mor_rank_9_6[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 6) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 7) {
                    _mor_rank_9_7[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_7[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_7[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_7[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_7[0]), "=r"(_mor_rank_9_7[1]), "=r"(_mor_rank_9_7[2]), "=r"(_mor_rank_9_7[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 7) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 8) {
                    _mor_rank_9_8[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_8[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_8[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_8[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_8[0]), "=r"(_mor_rank_9_8[1]), "=r"(_mor_rank_9_8[2]), "=r"(_mor_rank_9_8[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 8) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 9) {
                    _mor_rank_9_9[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_9[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_9[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_9[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_9[0]), "=r"(_mor_rank_9_9[1]), "=r"(_mor_rank_9_9[2]), "=r"(_mor_rank_9_9[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 9) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 10) {
                    _mor_rank_9_10[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_10[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_10[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_10[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_10[0]), "=r"(_mor_rank_9_10[1]), "=r"(_mor_rank_9_10[2]), "=r"(_mor_rank_9_10[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 10) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 11) {
                    _mor_rank_9_11[0] = _packed16_sanitize_0[0];
                    _mor_rank_9_11[1] = _packed16_sanitize_0[1];
                    _mor_rank_9_11[2] = _packed16_sanitize_0[2];
                    _mor_rank_9_11[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_rank_9_11[0]), "=r"(_mor_rank_9_11[1]), "=r"(_mor_rank_9_11[2]), "=r"(_mor_rank_9_11[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 11) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 0) {
                    _mor_retry_rank_9_0[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_0[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_0[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_0[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_0[0]), "=r"(_mor_retry_rank_9_0[1]), "=r"(_mor_retry_rank_9_0[2]), "=r"(_mor_retry_rank_9_0[3]) : "l"(local_unicast_ptr + (buffer_base + (unsigned long long)token * 12 * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 1) {
                    _mor_retry_rank_9_1[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_1[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_1[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_1[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_1[0]), "=r"(_mor_retry_rank_9_1[1]), "=r"(_mor_retry_rank_9_1[2]), "=r"(_mor_retry_rank_9_1[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 1) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 2) {
                    _mor_retry_rank_9_2[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_2[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_2[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_2[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_2[0]), "=r"(_mor_retry_rank_9_2[1]), "=r"(_mor_retry_rank_9_2[2]), "=r"(_mor_retry_rank_9_2[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 2) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 3) {
                    _mor_retry_rank_9_3[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_3[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_3[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_3[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_3[0]), "=r"(_mor_retry_rank_9_3[1]), "=r"(_mor_retry_rank_9_3[2]), "=r"(_mor_retry_rank_9_3[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 3) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 4) {
                    _mor_retry_rank_9_4[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_4[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_4[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_4[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_4[0]), "=r"(_mor_retry_rank_9_4[1]), "=r"(_mor_retry_rank_9_4[2]), "=r"(_mor_retry_rank_9_4[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 4) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 5) {
                    _mor_retry_rank_9_5[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_5[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_5[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_5[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_5[0]), "=r"(_mor_retry_rank_9_5[1]), "=r"(_mor_retry_rank_9_5[2]), "=r"(_mor_retry_rank_9_5[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 5) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 6) {
                    _mor_retry_rank_9_6[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_6[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_6[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_6[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_6[0]), "=r"(_mor_retry_rank_9_6[1]), "=r"(_mor_retry_rank_9_6[2]), "=r"(_mor_retry_rank_9_6[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 6) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 7) {
                    _mor_retry_rank_9_7[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_7[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_7[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_7[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_7[0]), "=r"(_mor_retry_rank_9_7[1]), "=r"(_mor_retry_rank_9_7[2]), "=r"(_mor_retry_rank_9_7[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 7) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 8) {
                    _mor_retry_rank_9_8[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_8[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_8[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_8[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_8[0]), "=r"(_mor_retry_rank_9_8[1]), "=r"(_mor_retry_rank_9_8[2]), "=r"(_mor_retry_rank_9_8[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 8) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 9) {
                    _mor_retry_rank_9_9[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_9[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_9[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_9[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_9[0]), "=r"(_mor_retry_rank_9_9[1]), "=r"(_mor_retry_rank_9_9[2]), "=r"(_mor_retry_rank_9_9[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 9) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 10) {
                    _mor_retry_rank_9_10[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_10[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_10[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_10[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_10[0]), "=r"(_mor_retry_rank_9_10[1]), "=r"(_mor_retry_rank_9_10[2]), "=r"(_mor_retry_rank_9_10[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 10) * 3584 + (unsigned long long)k0)) : "memory");
                }
                if (rank == 11) {
                    _mor_retry_rank_9_11[0] = _packed16_sanitize_0[0];
                    _mor_retry_rank_9_11[1] = _packed16_sanitize_0[1];
                    _mor_retry_rank_9_11[2] = _packed16_sanitize_0[2];
                    _mor_retry_rank_9_11[3] = _packed16_sanitize_0[3];
                } else {
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mor_retry_rank_9_11[0]), "=r"(_mor_retry_rank_9_11[1]), "=r"(_mor_retry_rank_9_11[2]), "=r"(_mor_retry_rank_9_11[3]) : "l"(local_unicast_ptr + (buffer_base + ((unsigned long long)token * 12 + 11) * 3584 + (unsigned long long)k0)) : "memory");
                }
                bool _mor_dirty_9 = false;
                bool _mor_retry_dirty_9 = false;
                _mor_dirty_9 |= (_mor_rank_9_0[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_0[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_0[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_0[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_0[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_0[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_0[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_0[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_1[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_1[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_1[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_1[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_1[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_1[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_1[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_1[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_2[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_2[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_2[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_2[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_2[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_2[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_2[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_2[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_3[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_3[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_3[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_3[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_3[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_3[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_3[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_3[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_4[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_4[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_4[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_4[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_4[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_4[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_4[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_4[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_5[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_5[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_5[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_5[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_5[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_5[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_5[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_5[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_6[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_6[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_6[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_6[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_6[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_6[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_6[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_6[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_7[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_7[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_7[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_7[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_7[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_7[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_7[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_7[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_8[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_8[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_8[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_8[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_8[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_8[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_8[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_8[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_9[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_9[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_9[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_9[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_9[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_9[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_9[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_9[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_10[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_10[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_10[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_10[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_10[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_10[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_10[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_10[3] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_11[0] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_11[0] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_11[1] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_11[1] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_11[2] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_11[2] == 0x80000000u);
                _mor_dirty_9 |= (_mor_rank_9_11[3] == 0x80000000u);
                _mor_retry_dirty_9 |= (_mor_retry_rank_9_11[3] == 0x80000000u);
                _mor_valid_9 = !_mor_dirty_9;
                if (!_mor_valid_9 && !_mor_retry_dirty_9) {
                    _mor_rank_9_0[0] = _mor_retry_rank_9_0[0];
                    _mor_rank_9_0[1] = _mor_retry_rank_9_0[1];
                    _mor_rank_9_0[2] = _mor_retry_rank_9_0[2];
                    _mor_rank_9_0[3] = _mor_retry_rank_9_0[3];
                    _mor_rank_9_1[0] = _mor_retry_rank_9_1[0];
                    _mor_rank_9_1[1] = _mor_retry_rank_9_1[1];
                    _mor_rank_9_1[2] = _mor_retry_rank_9_1[2];
                    _mor_rank_9_1[3] = _mor_retry_rank_9_1[3];
                    _mor_rank_9_2[0] = _mor_retry_rank_9_2[0];
                    _mor_rank_9_2[1] = _mor_retry_rank_9_2[1];
                    _mor_rank_9_2[2] = _mor_retry_rank_9_2[2];
                    _mor_rank_9_2[3] = _mor_retry_rank_9_2[3];
                    _mor_rank_9_3[0] = _mor_retry_rank_9_3[0];
                    _mor_rank_9_3[1] = _mor_retry_rank_9_3[1];
                    _mor_rank_9_3[2] = _mor_retry_rank_9_3[2];
                    _mor_rank_9_3[3] = _mor_retry_rank_9_3[3];
                    _mor_rank_9_4[0] = _mor_retry_rank_9_4[0];
                    _mor_rank_9_4[1] = _mor_retry_rank_9_4[1];
                    _mor_rank_9_4[2] = _mor_retry_rank_9_4[2];
                    _mor_rank_9_4[3] = _mor_retry_rank_9_4[3];
                    _mor_rank_9_5[0] = _mor_retry_rank_9_5[0];
                    _mor_rank_9_5[1] = _mor_retry_rank_9_5[1];
                    _mor_rank_9_5[2] = _mor_retry_rank_9_5[2];
                    _mor_rank_9_5[3] = _mor_retry_rank_9_5[3];
                    _mor_rank_9_6[0] = _mor_retry_rank_9_6[0];
                    _mor_rank_9_6[1] = _mor_retry_rank_9_6[1];
                    _mor_rank_9_6[2] = _mor_retry_rank_9_6[2];
                    _mor_rank_9_6[3] = _mor_retry_rank_9_6[3];
                    _mor_rank_9_7[0] = _mor_retry_rank_9_7[0];
                    _mor_rank_9_7[1] = _mor_retry_rank_9_7[1];
                    _mor_rank_9_7[2] = _mor_retry_rank_9_7[2];
                    _mor_rank_9_7[3] = _mor_retry_rank_9_7[3];
                    _mor_rank_9_8[0] = _mor_retry_rank_9_8[0];
                    _mor_rank_9_8[1] = _mor_retry_rank_9_8[1];
                    _mor_rank_9_8[2] = _mor_retry_rank_9_8[2];
                    _mor_rank_9_8[3] = _mor_retry_rank_9_8[3];
                    _mor_rank_9_9[0] = _mor_retry_rank_9_9[0];
                    _mor_rank_9_9[1] = _mor_retry_rank_9_9[1];
                    _mor_rank_9_9[2] = _mor_retry_rank_9_9[2];
                    _mor_rank_9_9[3] = _mor_retry_rank_9_9[3];
                    _mor_rank_9_10[0] = _mor_retry_rank_9_10[0];
                    _mor_rank_9_10[1] = _mor_retry_rank_9_10[1];
                    _mor_rank_9_10[2] = _mor_retry_rank_9_10[2];
                    _mor_rank_9_10[3] = _mor_retry_rank_9_10[3];
                    _mor_rank_9_11[0] = _mor_retry_rank_9_11[0];
                    _mor_rank_9_11[1] = _mor_retry_rank_9_11[1];
                    _mor_rank_9_11[2] = _mor_retry_rank_9_11[2];
                    _mor_rank_9_11[3] = _mor_retry_rank_9_11[3];
                    _mor_valid_9 = true;
                }
            } while (!_mor_valid_9);
            {
                float2 _mor_accum_9_0 = make_float2(0.0f, 0.0f);
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_0[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_1[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_2[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_3[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_4[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_5[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_6[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_7[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_8[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_9[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_10[0]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_0).x), "+f"((_mor_accum_9_0).y) : "r"(_mor_rank_9_11[0]));
                __nv_bfloat162 _mor_out_9_0 = __float22bfloat162_rn(_mor_accum_9_0);
                _mnnvl_oneshot_reduce_0[0] = *reinterpret_cast<uint32_t*>(&_mor_out_9_0);
            }
            {
                float2 _mor_accum_9_1 = make_float2(0.0f, 0.0f);
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_0[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_1[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_2[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_3[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_4[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_5[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_6[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_7[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_8[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_9[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_10[1]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_1).x), "+f"((_mor_accum_9_1).y) : "r"(_mor_rank_9_11[1]));
                __nv_bfloat162 _mor_out_9_1 = __float22bfloat162_rn(_mor_accum_9_1);
                _mnnvl_oneshot_reduce_0[1] = *reinterpret_cast<uint32_t*>(&_mor_out_9_1);
            }
            {
                float2 _mor_accum_9_2 = make_float2(0.0f, 0.0f);
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_0[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_1[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_2[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_3[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_4[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_5[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_6[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_7[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_8[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_9[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_10[2]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_2).x), "+f"((_mor_accum_9_2).y) : "r"(_mor_rank_9_11[2]));
                __nv_bfloat162 _mor_out_9_2 = __float22bfloat162_rn(_mor_accum_9_2);
                _mnnvl_oneshot_reduce_0[2] = *reinterpret_cast<uint32_t*>(&_mor_out_9_2);
            }
            {
                float2 _mor_accum_9_3 = make_float2(0.0f, 0.0f);
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_0[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_1[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_2[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_3[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_4[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_5[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_6[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_7[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_8[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_9[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_10[3]));
                asm volatile(
                    "{\n\t"
                    ".reg .b16 lo, hi;\n\t"
                    "mov.b32 {lo, hi}, %2;\n\t"
                    "add.rn.f32.bf16 %0, lo, %0;\n\t"
                    "add.rn.f32.bf16 %1, hi, %1;\n\t"
                    "}\n"
                    : "+f"((_mor_accum_9_3).x), "+f"((_mor_accum_9_3).y) : "r"(_mor_rank_9_11[3]));
                __nv_bfloat162 _mor_out_9_3 = __float22bfloat162_rn(_mor_accum_9_3);
                _mnnvl_oneshot_reduce_0[3] = *reinterpret_cast<uint32_t*>(&_mor_out_9_3);
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
        float _block_reduce_11_accum = thread_sum;
        const int _block_reduce_11_lane = threadIdx.x & 31;
        const int _block_reduce_11_warp = threadIdx.x >> 5;
        // Full CTA schedule: one barrier; every warp performs level two.
        #pragma unroll
        for (int _block_reduce_11_offset_level1 = 16; _block_reduce_11_offset_level1 > 0; _block_reduce_11_offset_level1 >>= 1) {
            _block_reduce_11_accum += __shfl_xor_sync(0xffffffffu, _block_reduce_11_accum, _block_reduce_11_offset_level1);
        }
        if (_block_reduce_11_lane == 0) { reduce_scratch[_block_reduce_11_warp] = _block_reduce_11_accum; }
        __syncthreads();
        float _block_reduce_11_level2 = (_block_reduce_11_lane < 14) ? reduce_scratch[_block_reduce_11_lane] : 0.0f;
        #pragma unroll
        for (int _block_reduce_11_offset_level2 = 16; _block_reduce_11_offset_level2 > 0; _block_reduce_11_offset_level2 >>= 1) {
            _block_reduce_11_level2 += __shfl_xor_sync(0xffffffffu, _block_reduce_11_level2, _block_reduce_11_offset_level2);
        }
        float _block_reduce_10 = _block_reduce_11_level2;
        float rcp_rms = rsqrtf(_block_reduce_10 / 3584.0f + epsilon);
        float xn_f32[8];
        for (int lane_1_1 = 0; lane_1_1 < 8; lane_1_1++) {
            xn_f32[lane_1_1] = _unpack_bf16_0[lane_1_1] * rcp_rms;
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
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vec_load_6[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vec_load_6[_pair]) & 0xffff0000u);
        }
        float y_f32[8];
        for (int lane_1_2 = 0; lane_1_2 < 8; lane_1_2++) {
            y_f32[lane_1_2] = _unpack_bf16_1[lane_1_2] * _unpack_bf16_2[lane_1_2];
        }
        unsigned int y_f32_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_f32[_lp*2 + 0], y_f32[_lp*2+1 + 0]));
            y_f32_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(y + input_index)[0] = reinterpret_cast<int4*>(y_f32_bf16)[0];
        if (warp == 0) {
            asm volatile("barrier.sync 2, 448;" ::: "memory");
        } else {
            asm volatile("barrier.arrive 2, 448;" ::: "memory");
        }
        if (tid == 0) {
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(k3_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
        uint8_t* bytes_base_0 = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]));
        unsigned int dirty_1 = (unsigned int)_vec_load_3[1];
        unsigned int stage_bytes_2 = (unsigned int)k3_stage_bytes;
        unsigned int num_stages_3 = (unsigned int)k3_num_stages;
        unsigned int clear_4[4];
        clear_4[0] = (unsigned int)_vec_load_4[0];
        clear_4[1] = (unsigned int)_vec_load_4[1];
        clear_4[2] = (unsigned int)_vec_load_4[2];
        clear_4[3] = (unsigned int)_vec_load_4[3];
        unsigned int block_5 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
        unsigned int thread_6 = block_5 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
        unsigned int threads_7 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
        unsigned int sentinel_8[4] = {2147483648, 2147483648, 2147483648, 2147483648};
        if (num_stages_3 > 0) {
            unsigned long long stage_offset_0 = (unsigned long long)((unsigned int)(uint8_t)dirty_1 * num_stages_3 + (unsigned int)0) * (unsigned long long)stage_bytes_2;
            unsigned int boundary_1 = (clear_4[0] + 15) / 16;
            for (unsigned int packed_1 = thread_6; packed_1 < boundary_1; packed_1 += threads_7) {
                {
                    auto _gst_12 = reinterpret_cast<unsigned __int128*>(bytes_base_0 + stage_offset_0) + packed_1;
                    __builtin_assume(__isGlobal(_gst_12));
                    reinterpret_cast<int4*>(_gst_12)[0] = reinterpret_cast<int4*>(sentinel_8)[0];
                }
            }
        }
        uint8_t* bytes_base_9 = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]));
        unsigned int dirty_10 = (unsigned int)_vec_load_3[1];
        unsigned int stage_bytes_11 = (unsigned int)k3_stage_bytes;
        unsigned int num_stages_12 = (unsigned int)k3_num_stages;
        unsigned int clear_13[4];
        clear_13[0] = (unsigned int)_vec_load_4[0];
        clear_13[1] = (unsigned int)_vec_load_4[1];
        clear_13[2] = (unsigned int)_vec_load_4[2];
        clear_13[3] = (unsigned int)_vec_load_4[3];
        unsigned int block_14 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
        unsigned int thread_15 = block_14 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
        unsigned int threads_16 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
        unsigned int sentinel_17[4] = {2147483648, 2147483648, 2147483648, 2147483648};
        if (num_stages_12 > 1) {
            unsigned long long stage_offset_0_1 = (unsigned long long)((unsigned int)(uint8_t)dirty_10 * num_stages_12 + (unsigned int)1) * (unsigned long long)stage_bytes_11;
            unsigned int boundary_1_1 = (clear_13[1] + 15) / 16;
            for (unsigned int packed_1_1 = thread_15; packed_1_1 < boundary_1_1; packed_1_1 += threads_16) {
                {
                    auto _gst_13 = reinterpret_cast<unsigned __int128*>(bytes_base_9 + stage_offset_0_1) + packed_1_1;
                    __builtin_assume(__isGlobal(_gst_13));
                    reinterpret_cast<int4*>(_gst_13)[0] = reinterpret_cast<int4*>(sentinel_17)[0];
                }
            }
        }
        if (token == 0) {
            if (tid == 0) {
                {
                    volatile unsigned int* _sysv_wait_p_14 = reinterpret_cast<volatile unsigned int*>(buffer_flags) + (8);
                    while (*_sysv_wait_p_14 < static_cast<unsigned int>(num_tokens + k23_ctas)) {}
                }
                unsigned int current = (unsigned int)_vec_load_1[0];
                unsigned int head[4];
                head[0] = (current + 1) % 3;
                head[1] = current;
                head[2] = (unsigned int)_vec_load_1[2];
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
    } else {
        int col = my_col_begin + row0;
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        if (warp == 0) {
            asm volatile("barrier.sync 3, 448;" ::: "memory");
        } else {
            asm volatile("barrier.arrive 3, 448;" ::: "memory");
        }
        if (tid == 0) {
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(buffer_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
        uint8_t* bytes_base_1 = reinterpret_cast<uint8_t*>(local_unicast_ptr);
        unsigned int dirty_2 = (unsigned int)_vec_load_1[1];
        unsigned int stage_bytes_1 = (unsigned int)dirty_stage_bytes;
        unsigned int num_stages_1 = (unsigned int)dirty_num_stages;
        unsigned int clear_1[4];
        clear_1[0] = (unsigned int)_vec_load_2[0];
        clear_1[1] = (unsigned int)_vec_load_2[1];
        clear_1[2] = (unsigned int)_vec_load_2[2];
        clear_1[3] = (unsigned int)_vec_load_2[3];
        unsigned int block_1 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
        unsigned int thread_1 = block_1 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
        unsigned int threads_1 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
        unsigned int sentinel_1[4] = {2147483648, 2147483648, 2147483648, 2147483648};
        for (unsigned int stage_1 = 0; stage_1 < num_stages_1; stage_1++) {
            unsigned long long stage_offset_1 = (unsigned long long)((unsigned int)(uint8_t)dirty_2 * num_stages_1 + (unsigned int)(uint8_t)stage_1) * (unsigned long long)stage_bytes_1;
            unsigned int boundary_2 = (clear_1[stage_1] + 15) / 16;
            for (unsigned int packed_2 = thread_1; packed_2 < boundary_2; packed_2 += threads_1) {
                {
                    auto _gst_15 = reinterpret_cast<unsigned __int128*>(bytes_base_1 + stage_offset_1) + packed_2;
                    __builtin_assume(__isGlobal(_gst_15));
                    reinterpret_cast<int4*>(_gst_15)[0] = reinterpret_cast<int4*>(sentinel_1)[0];
                }
            }
        }
        int p = kb * 448 + tid;
        if (p < num_tokens * 896) {
            int s_token = p / 896;
            int s_col = (p - s_token * 896) * 8;
            unsigned int _vec_load_7[4];
            {
                uint4 _uv4_16 = *reinterpret_cast<const uint4*>(shared + (s_token * 7168 + s_col) + 0);
                _vec_load_7[0 + 0] = _uv4_16.x;
                _vec_load_7[0 + 1] = _uv4_16.y;
                _vec_load_7[0 + 2] = _uv4_16.z;
                _vec_load_7[0 + 3] = _uv4_16.w;
            }
            uint32_t _packed16_sanitize_1[4];
            {
                uint32_t _p16_word_17_0 = _vec_load_7[0];
                if ((_p16_word_17_0 & 0x0000ffffu) == 0x00008000u) _p16_word_17_0 &= 0xffff0000u;
                if ((_p16_word_17_0 & 0xffff0000u) == 0x80000000u) _p16_word_17_0 &= 0x0000ffffu;
                _packed16_sanitize_1[0] = _p16_word_17_0;
                uint32_t _p16_word_17_1 = _vec_load_7[1];
                if ((_p16_word_17_1 & 0x0000ffffu) == 0x00008000u) _p16_word_17_1 &= 0xffff0000u;
                if ((_p16_word_17_1 & 0xffff0000u) == 0x80000000u) _p16_word_17_1 &= 0x0000ffffu;
                _packed16_sanitize_1[1] = _p16_word_17_1;
                uint32_t _p16_word_17_2 = _vec_load_7[2];
                if ((_p16_word_17_2 & 0x0000ffffu) == 0x00008000u) _p16_word_17_2 &= 0xffff0000u;
                if ((_p16_word_17_2 & 0xffff0000u) == 0x80000000u) _p16_word_17_2 &= 0x0000ffffu;
                _packed16_sanitize_1[2] = _p16_word_17_2;
                uint32_t _p16_word_17_3 = _vec_load_7[3];
                if ((_p16_word_17_3 & 0x0000ffffu) == 0x00008000u) _p16_word_17_3 &= 0xffff0000u;
                if ((_p16_word_17_3 & 0xffff0000u) == 0x80000000u) _p16_word_17_3 &= 0x0000ffffu;
                _packed16_sanitize_1[3] = _p16_word_17_3;
            }
            int s_owner = 0;
            int s_owner_begin = 0;
            if (s_col >= 640) {
                s_owner = s_owner + 1;
                s_owner_begin = 640;
            }
            if (s_col >= 1280) {
                s_owner = s_owner + 1;
                s_owner_begin = 1280;
            }
            if (s_col >= 1920) {
                s_owner = s_owner + 1;
                s_owner_begin = 1920;
            }
            if (s_col >= 2560) {
                s_owner = s_owner + 1;
                s_owner_begin = 2560;
            }
            if (s_col >= 3200) {
                s_owner = s_owner + 1;
                s_owner_begin = 3200;
            }
            if (s_col >= 3840) {
                s_owner = s_owner + 1;
                s_owner_begin = 3840;
            }
            if (s_col >= 4480) {
                s_owner = s_owner + 1;
                s_owner_begin = 4480;
            }
            if (s_col >= 5120) {
                s_owner = s_owner + 1;
                s_owner_begin = 5120;
            }
            if (s_col >= 5632) {
                s_owner = s_owner + 1;
                s_owner_begin = 5632;
            }
            if (s_col >= 6144) {
                s_owner = s_owner + 1;
                s_owner_begin = 6144;
            }
            if (s_col >= 6656) {
                s_owner = s_owner + 1;
                s_owner_begin = 6656;
            }
            unsigned long long s_index = ((unsigned long long)s_token * 12 + rank_u64) * 640 + (unsigned long long)(s_col - s_owner_begin);
            {
                auto _gst_18 = reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[s_owner]) + (k3_epoch_elements + s_index);
                __builtin_assume(__isGlobal(_gst_18));
                reinterpret_cast<int4*>(_gst_18)[0] = reinterpret_cast<int4*>(_packed16_sanitize_1)[0];
            }
        }
        uint8_t* bytes_base_0_1 = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]));
        unsigned int dirty_1_1 = (unsigned int)_vec_load_3[1];
        unsigned int stage_bytes_2_1 = (unsigned int)k3_stage_bytes;
        unsigned int num_stages_3_1 = (unsigned int)k3_num_stages;
        unsigned int clear_4_1[4];
        clear_4_1[0] = (unsigned int)_vec_load_4[0];
        clear_4_1[1] = (unsigned int)_vec_load_4[1];
        clear_4_1[2] = (unsigned int)_vec_load_4[2];
        clear_4_1[3] = (unsigned int)_vec_load_4[3];
        unsigned int block_5_1 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
        unsigned int thread_6_1 = block_5_1 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
        unsigned int threads_7_1 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
        unsigned int sentinel_8_1[4] = {2147483648, 2147483648, 2147483648, 2147483648};
        if (num_stages_3_1 > 0) {
            unsigned long long stage_offset_0_2 = (unsigned long long)((unsigned int)(uint8_t)dirty_1_1 * num_stages_3_1 + (unsigned int)0) * (unsigned long long)stage_bytes_2_1;
            unsigned int boundary_1_2 = (clear_4_1[0] + 15) / 16;
            for (unsigned int packed_1_2 = thread_6_1; packed_1_2 < boundary_1_2; packed_1_2 += threads_7_1) {
                {
                    auto _gst_19 = reinterpret_cast<unsigned __int128*>(bytes_base_0_1 + stage_offset_0_2) + packed_1_2;
                    __builtin_assume(__isGlobal(_gst_19));
                    reinterpret_cast<int4*>(_gst_19)[0] = reinterpret_cast<int4*>(sentinel_8_1)[0];
                }
            }
        }
        float shared_sum_f32[8];
        for (int lane_1_3 = 0; lane_1_3 < 8; lane_1_3++) {
            shared_sum_f32[lane_1_3] = 0.0f;
        }
        if (tid < num_tokens) {
            uint32_t _mnnvl_twoshot_reduce_0[4];
            {
                float2 _mtr_accum_20[4];
                bool _mtr_valid_20;
                do {
                    _mtr_valid_20 = true;
                    #pragma unroll
                    for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) _mtr_accum_20[_mtr_pair] = make_float2(0.0f, 0.0f);
                    uint32_t _mtr_rank_20_0[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_0[0]), "=r"(_mtr_rank_20_0[1]), "=r"(_mtr_rank_20_0[2]), "=r"(_mtr_rank_20_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + (unsigned long long)tid * 12 * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_0 = false;
                    _mtr_dirty_20_0 |= (_mtr_rank_20_0[0] == 0x80000000u);
                    _mtr_dirty_20_0 |= (_mtr_rank_20_0[1] == 0x80000000u);
                    _mtr_dirty_20_0 |= (_mtr_rank_20_0[2] == 0x80000000u);
                    _mtr_dirty_20_0 |= (_mtr_rank_20_0[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_0[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_0[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_0[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_0[3]));
                    uint32_t _mtr_rank_20_1[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_1[0]), "=r"(_mtr_rank_20_1[1]), "=r"(_mtr_rank_20_1[2]), "=r"(_mtr_rank_20_1[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 1) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_1 = false;
                    _mtr_dirty_20_1 |= (_mtr_rank_20_1[0] == 0x80000000u);
                    _mtr_dirty_20_1 |= (_mtr_rank_20_1[1] == 0x80000000u);
                    _mtr_dirty_20_1 |= (_mtr_rank_20_1[2] == 0x80000000u);
                    _mtr_dirty_20_1 |= (_mtr_rank_20_1[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_1;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_1[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_1[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_1[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_1[3]));
                    uint32_t _mtr_rank_20_2[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_2[0]), "=r"(_mtr_rank_20_2[1]), "=r"(_mtr_rank_20_2[2]), "=r"(_mtr_rank_20_2[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 2) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_2 = false;
                    _mtr_dirty_20_2 |= (_mtr_rank_20_2[0] == 0x80000000u);
                    _mtr_dirty_20_2 |= (_mtr_rank_20_2[1] == 0x80000000u);
                    _mtr_dirty_20_2 |= (_mtr_rank_20_2[2] == 0x80000000u);
                    _mtr_dirty_20_2 |= (_mtr_rank_20_2[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_2;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_2[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_2[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_2[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_2[3]));
                    uint32_t _mtr_rank_20_3[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_3[0]), "=r"(_mtr_rank_20_3[1]), "=r"(_mtr_rank_20_3[2]), "=r"(_mtr_rank_20_3[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 3) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_3 = false;
                    _mtr_dirty_20_3 |= (_mtr_rank_20_3[0] == 0x80000000u);
                    _mtr_dirty_20_3 |= (_mtr_rank_20_3[1] == 0x80000000u);
                    _mtr_dirty_20_3 |= (_mtr_rank_20_3[2] == 0x80000000u);
                    _mtr_dirty_20_3 |= (_mtr_rank_20_3[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_3;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_3[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_3[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_3[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_3[3]));
                    uint32_t _mtr_rank_20_4[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_4[0]), "=r"(_mtr_rank_20_4[1]), "=r"(_mtr_rank_20_4[2]), "=r"(_mtr_rank_20_4[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 4) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_4 = false;
                    _mtr_dirty_20_4 |= (_mtr_rank_20_4[0] == 0x80000000u);
                    _mtr_dirty_20_4 |= (_mtr_rank_20_4[1] == 0x80000000u);
                    _mtr_dirty_20_4 |= (_mtr_rank_20_4[2] == 0x80000000u);
                    _mtr_dirty_20_4 |= (_mtr_rank_20_4[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_4;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_4[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_4[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_4[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_4[3]));
                    uint32_t _mtr_rank_20_5[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_5[0]), "=r"(_mtr_rank_20_5[1]), "=r"(_mtr_rank_20_5[2]), "=r"(_mtr_rank_20_5[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 5) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_5 = false;
                    _mtr_dirty_20_5 |= (_mtr_rank_20_5[0] == 0x80000000u);
                    _mtr_dirty_20_5 |= (_mtr_rank_20_5[1] == 0x80000000u);
                    _mtr_dirty_20_5 |= (_mtr_rank_20_5[2] == 0x80000000u);
                    _mtr_dirty_20_5 |= (_mtr_rank_20_5[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_5;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_5[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_5[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_5[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_5[3]));
                    uint32_t _mtr_rank_20_6[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_6[0]), "=r"(_mtr_rank_20_6[1]), "=r"(_mtr_rank_20_6[2]), "=r"(_mtr_rank_20_6[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 6) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_6 = false;
                    _mtr_dirty_20_6 |= (_mtr_rank_20_6[0] == 0x80000000u);
                    _mtr_dirty_20_6 |= (_mtr_rank_20_6[1] == 0x80000000u);
                    _mtr_dirty_20_6 |= (_mtr_rank_20_6[2] == 0x80000000u);
                    _mtr_dirty_20_6 |= (_mtr_rank_20_6[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_6;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_6[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_6[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_6[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_6[3]));
                    uint32_t _mtr_rank_20_7[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_7[0]), "=r"(_mtr_rank_20_7[1]), "=r"(_mtr_rank_20_7[2]), "=r"(_mtr_rank_20_7[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 7) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_7 = false;
                    _mtr_dirty_20_7 |= (_mtr_rank_20_7[0] == 0x80000000u);
                    _mtr_dirty_20_7 |= (_mtr_rank_20_7[1] == 0x80000000u);
                    _mtr_dirty_20_7 |= (_mtr_rank_20_7[2] == 0x80000000u);
                    _mtr_dirty_20_7 |= (_mtr_rank_20_7[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_7;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_7[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_7[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_7[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_7[3]));
                    uint32_t _mtr_rank_20_8[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_8[0]), "=r"(_mtr_rank_20_8[1]), "=r"(_mtr_rank_20_8[2]), "=r"(_mtr_rank_20_8[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 8) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_8 = false;
                    _mtr_dirty_20_8 |= (_mtr_rank_20_8[0] == 0x80000000u);
                    _mtr_dirty_20_8 |= (_mtr_rank_20_8[1] == 0x80000000u);
                    _mtr_dirty_20_8 |= (_mtr_rank_20_8[2] == 0x80000000u);
                    _mtr_dirty_20_8 |= (_mtr_rank_20_8[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_8;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_8[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_8[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_8[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_8[3]));
                    uint32_t _mtr_rank_20_9[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_9[0]), "=r"(_mtr_rank_20_9[1]), "=r"(_mtr_rank_20_9[2]), "=r"(_mtr_rank_20_9[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 9) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_9 = false;
                    _mtr_dirty_20_9 |= (_mtr_rank_20_9[0] == 0x80000000u);
                    _mtr_dirty_20_9 |= (_mtr_rank_20_9[1] == 0x80000000u);
                    _mtr_dirty_20_9 |= (_mtr_rank_20_9[2] == 0x80000000u);
                    _mtr_dirty_20_9 |= (_mtr_rank_20_9[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_9;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_9[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_9[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_9[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_9[3]));
                    uint32_t _mtr_rank_20_10[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_10[0]), "=r"(_mtr_rank_20_10[1]), "=r"(_mtr_rank_20_10[2]), "=r"(_mtr_rank_20_10[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 10) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_10 = false;
                    _mtr_dirty_20_10 |= (_mtr_rank_20_10[0] == 0x80000000u);
                    _mtr_dirty_20_10 |= (_mtr_rank_20_10[1] == 0x80000000u);
                    _mtr_dirty_20_10 |= (_mtr_rank_20_10[2] == 0x80000000u);
                    _mtr_dirty_20_10 |= (_mtr_rank_20_10[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_10;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_10[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_10[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_10[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_10[3]));
                    uint32_t _mtr_rank_20_11[4];
                    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_mtr_rank_20_11[0]), "=r"(_mtr_rank_20_11[1]), "=r"(_mtr_rank_20_11[2]), "=r"(_mtr_rank_20_11[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + ((unsigned long long)tid * 12 + 11) * 640 + (unsigned long long)row0)) : "memory");
                    bool _mtr_dirty_20_11 = false;
                    _mtr_dirty_20_11 |= (_mtr_rank_20_11[0] == 0x80000000u);
                    _mtr_dirty_20_11 |= (_mtr_rank_20_11[1] == 0x80000000u);
                    _mtr_dirty_20_11 |= (_mtr_rank_20_11[2] == 0x80000000u);
                    _mtr_dirty_20_11 |= (_mtr_rank_20_11[3] == 0x80000000u);
                    _mtr_valid_20 &= !_mtr_dirty_20_11;
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[0]).x), "+f"((_mtr_accum_20[0]).y) : "r"(_mtr_rank_20_11[0]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[1]).x), "+f"((_mtr_accum_20[1]).y) : "r"(_mtr_rank_20_11[1]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[2]).x), "+f"((_mtr_accum_20[2]).y) : "r"(_mtr_rank_20_11[2]));
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 lo, hi;\n\t"
                        "mov.b32 {lo, hi}, %2;\n\t"
                        "add.rn.f32.bf16 %0, lo, %0;\n\t"
                        "add.rn.f32.bf16 %1, hi, %1;\n\t"
                        "}\n"
                        : "+f"((_mtr_accum_20[3]).x), "+f"((_mtr_accum_20[3]).y) : "r"(_mtr_rank_20_11[3]));
                } while (!_mtr_valid_20);
                #pragma unroll
                for (int _mtr_pair = 0; _mtr_pair < 4; ++_mtr_pair) {
                    __nv_bfloat162 _mtr_out = __float22bfloat162_rn(_mtr_accum_20[_mtr_pair]);
                    _mnnvl_twoshot_reduce_0[_mtr_pair] = *reinterpret_cast<uint32_t*>(&_mtr_out);
                }
            }
            float _unpack_bf16_3[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_mnnvl_twoshot_reduce_0[_pair]) << 16);
                (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_mnnvl_twoshot_reduce_0[_pair]) & 0xffff0000u);
            }
            for (int lane_1_4 = 0; lane_1_4 < 8; lane_1_4++) {
                shared_sum_f32[lane_1_4] = _unpack_bf16_3[lane_1_4];
            }
        }
        if (tid == 0) {
            {
            unsigned int _acquire_observed;
            do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(k3_flags) + (8))) : "memory");
            } while (_acquire_observed < static_cast<unsigned int>(num_tokens));
            }
        }
        asm volatile("barrier.sync 1, 448;" ::: "memory");
        __syncwarp();
        float acc[8];
        unsigned int ycache[32];
        for (int m = 0; m < K23_CAPACITY; m++) {
            if (m < num_tokens) {
                unsigned int _vec_load_8[4];
                {
                    uint4 _uv4_21 = *reinterpret_cast<const uint4*>(reinterpret_cast<unsigned int*>(y) + ((m * 3584 + k0) / 2) + 0);
                    _vec_load_8[0 + 0] = _uv4_21.x;
                    _vec_load_8[0 + 1] = _uv4_21.y;
                    _vec_load_8[0 + 2] = _uv4_21.z;
                    _vec_load_8[0 + 3] = _uv4_21.w;
                }
                for (int q_1 = 0; q_1 < 4; q_1++) {
                    ycache[m * 4 + q_1] = _vec_load_8[q_1];
                }
            }
        }
        float accg[32];
        for (int g = 0; g < 2; g++) {
            int mq = g * 4;
            if (mq + 4 <= num_tokens) {
                for (int j = 0; j < 4; j++) {
                    for (int r_1 = 0; r_1 < 8; r_1++) {
                        float _dot_25;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_25) : "r"(wv[r_1 * 4]), "r"(ycache[(mq + j) * 4]), "f"(0.0f));
                        float _dot_24;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_24) : "r"(wv[r_1 * 4 + 1]), "r"(ycache[(mq + j) * 4 + 1]), "f"(_dot_25));
                        float _dot_23;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_23) : "r"(wv[r_1 * 4 + 2]), "r"(ycache[(mq + j) * 4 + 2]), "f"(_dot_24));
                        float _dot_22;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_22) : "r"(wv[r_1 * 4 + 3]), "r"(ycache[(mq + j) * 4 + 3]), "f"(_dot_23));
                        accg[j * 8 + r_1] = _dot_22;
                    }
                }
                for (int j_1 = 0; j_1 < 4; j_1++) {
                    for (int r_2 = 0; r_2 < 4; r_2++) {
                        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, accg[j_1 * 8 + r_2 + 4], 16);
                        accg[j_1 * 8 + r_2] = accg[j_1 * 8 + r_2] + _shfl_xor_sync_26;
                    }
                }
                for (int j_2 = 0; j_2 < 4; j_2++) {
                    for (int r_3 = 0; r_3 < 2; r_3++) {
                        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, accg[j_2 * 8 + r_3 + 2], 8);
                        accg[j_2 * 8 + r_3] = accg[j_2 * 8 + r_3] + _shfl_xor_sync_27;
                    }
                }
                for (int j_3 = 0; j_3 < 4; j_3++) {
                    float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, accg[j_3 * 8 + 1], 4);
                    accg[j_3 * 8] = accg[j_3 * 8] + _shfl_xor_sync_28;
                }
                for (int j_4 = 0; j_4 < 4; j_4++) {
                    float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, accg[j_4 * 8], 2);
                    accg[j_4 * 8] = accg[j_4 * 8] + _shfl_xor_sync_29;
                }
                for (int j_5 = 0; j_5 < 4; j_5++) {
                    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, accg[j_5 * 8], 1);
                    accg[j_5 * 8] = accg[j_5 * 8] + _shfl_xor_sync_30;
                }
                if ((lane & 3) == 0) {
                    for (int j_6 = 0; j_6 < 4; j_6++) {
                        partials[warp * (8 * K23_CAPACITY) + lane_row * K23_CAPACITY + mq + j_6] = accg[j_6 * 8];
                    }
                }
            } else {
                for (int j_7 = 0; j_7 < 4; j_7++) {
                    if (mq + j_7 < num_tokens) {
                        unsigned int yv2[4];
                        for (int q_2 = 0; q_2 < 4; q_2++) {
                            yv2[q_2] = ycache[(mq + j_7) * 4 + q_2];
                        }
                        float _dot_34;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_34) : "r"(wv[0]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_33;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_33) : "r"(wv[1]), "r"(yv2[1]), "f"(_dot_34));
                        float _dot_32;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_32) : "r"(wv[2]), "r"(yv2[2]), "f"(_dot_33));
                        float _dot_31;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_31) : "r"(wv[3]), "r"(yv2[3]), "f"(_dot_32));
                        acc[0] = _dot_31;
                        float _dot_38;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_38) : "r"(wv[4]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_37;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_37) : "r"(wv[5]), "r"(yv2[1]), "f"(_dot_38));
                        float _dot_36;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_36) : "r"(wv[6]), "r"(yv2[2]), "f"(_dot_37));
                        float _dot_35;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_35) : "r"(wv[7]), "r"(yv2[3]), "f"(_dot_36));
                        acc[1] = _dot_35;
                        float _dot_42;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_42) : "r"(wv[8]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_41;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_41) : "r"(wv[9]), "r"(yv2[1]), "f"(_dot_42));
                        float _dot_40;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_40) : "r"(wv[10]), "r"(yv2[2]), "f"(_dot_41));
                        float _dot_39;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_39) : "r"(wv[11]), "r"(yv2[3]), "f"(_dot_40));
                        acc[2] = _dot_39;
                        float _dot_46;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_46) : "r"(wv[12]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_45;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_45) : "r"(wv[13]), "r"(yv2[1]), "f"(_dot_46));
                        float _dot_44;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_44) : "r"(wv[14]), "r"(yv2[2]), "f"(_dot_45));
                        float _dot_43;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_43) : "r"(wv[15]), "r"(yv2[3]), "f"(_dot_44));
                        acc[3] = _dot_43;
                        float _dot_50;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_50) : "r"(wv[16]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_49;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_49) : "r"(wv[17]), "r"(yv2[1]), "f"(_dot_50));
                        float _dot_48;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_48) : "r"(wv[18]), "r"(yv2[2]), "f"(_dot_49));
                        float _dot_47;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_47) : "r"(wv[19]), "r"(yv2[3]), "f"(_dot_48));
                        acc[4] = _dot_47;
                        float _dot_54;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_54) : "r"(wv[20]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_53;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_53) : "r"(wv[21]), "r"(yv2[1]), "f"(_dot_54));
                        float _dot_52;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_52) : "r"(wv[22]), "r"(yv2[2]), "f"(_dot_53));
                        float _dot_51;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_51) : "r"(wv[23]), "r"(yv2[3]), "f"(_dot_52));
                        acc[5] = _dot_51;
                        float _dot_58;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_58) : "r"(wv[24]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_57;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_57) : "r"(wv[25]), "r"(yv2[1]), "f"(_dot_58));
                        float _dot_56;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_56) : "r"(wv[26]), "r"(yv2[2]), "f"(_dot_57));
                        float _dot_55;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_55) : "r"(wv[27]), "r"(yv2[3]), "f"(_dot_56));
                        acc[6] = _dot_55;
                        float _dot_62;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_62) : "r"(wv[28]), "r"(yv2[0]), "f"(0.0f));
                        float _dot_61;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_61) : "r"(wv[29]), "r"(yv2[1]), "f"(_dot_62));
                        float _dot_60;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_60) : "r"(wv[30]), "r"(yv2[2]), "f"(_dot_61));
                        float _dot_59;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_59) : "r"(wv[31]), "r"(yv2[3]), "f"(_dot_60));
                        acc[7] = _dot_59;
                        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, acc[4], 16);
                        acc[0] = acc[0] + _shfl_xor_sync_63;
                        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, acc[5], 16);
                        acc[1] = acc[1] + _shfl_xor_sync_64;
                        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, acc[6], 16);
                        acc[2] = acc[2] + _shfl_xor_sync_65;
                        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, acc[7], 16);
                        acc[3] = acc[3] + _shfl_xor_sync_66;
                        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, acc[2], 8);
                        acc[0] = acc[0] + _shfl_xor_sync_67;
                        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, acc[3], 8);
                        acc[1] = acc[1] + _shfl_xor_sync_68;
                        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, acc[1], 4);
                        acc[0] = acc[0] + _shfl_xor_sync_69;
                        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, acc[0], 2);
                        acc[0] = acc[0] + _shfl_xor_sync_70;
                        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, acc[0], 1);
                        acc[0] = acc[0] + _shfl_xor_sync_71;
                        if ((lane & 3) == 0) {
                            partials[warp * (8 * K23_CAPACITY) + lane_row * K23_CAPACITY + (mq + j_7)] = acc[0];
                        }
                    }
                }
            }
        }
        asm volatile("barrier.sync 1, 448;" ::: "memory");
        float* gemm_sm = partials + (112 * K23_CAPACITY);
        if (tid < 8 * K23_CAPACITY) {
            int r_out = tid / K23_CAPACITY;
            int m_out = tid % K23_CAPACITY;
            if (m_out < num_tokens) {
                float total = 0.0f;
                for (int wgt = 0; wgt < 14; wgt++) {
                    total = total + partials[wgt * (8 * K23_CAPACITY) + tid];
                }
                gemm_sm[m_out * 8 + r_out] = total;
            }
        }
        asm volatile("barrier.sync 1, 448;" ::: "memory");
        if (tid < num_tokens) {
            float out_f32[8];
            for (int lane_1_5 = 0; lane_1_5 < 8; lane_1_5++) {
                out_f32[lane_1_5] = shared_sum_f32[lane_1_5] + gemm_sm[tid * 8 + lane_1_5];
            }
            unsigned int out_f32_bf16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_f32[_lp*2 + 0], out_f32[_lp*2+1 + 0]));
                out_f32_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            uint32_t _packed16_sanitize_2[4];
            #pragma unroll
            for (int _p16s_word = 0; _p16s_word < 4; ++_p16s_word) {
                uint32_t _p16s_carrier_72 = static_cast<uint32_t>(out_f32_bf16[_p16s_word]);
                uint32_t _p16s_lo_72 = _p16s_carrier_72 & 0x0000ffffu;
                uint32_t _p16s_hi_72 = (_p16s_carrier_72 >> 16) & 0x0000ffffu;
                if (_p16s_lo_72 == 0x00008000u) _p16s_lo_72 = 0u;
                if (_p16s_hi_72 == 0x00008000u) _p16s_hi_72 = 0u;
                _packed16_sanitize_2[_p16s_word] = _p16s_lo_72 | (_p16s_hi_72 << 16);
            }
            reinterpret_cast<int4*>(k3_mcast_ptr + (k3_epoch_elements + k3_stage_elements + (unsigned long long)(tid * 7168 + col)))[0] = reinterpret_cast<int4*>(_packed16_sanitize_2)[0];
        }
        uint8_t* bytes_base_9_1 = reinterpret_cast<uint8_t*>(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]));
        unsigned int dirty_10_1 = (unsigned int)_vec_load_3[1];
        unsigned int stage_bytes_11_1 = (unsigned int)k3_stage_bytes;
        unsigned int num_stages_12_1 = (unsigned int)k3_num_stages;
        unsigned int clear_13_1[4];
        clear_13_1[0] = (unsigned int)_vec_load_4[0];
        clear_13_1[1] = (unsigned int)_vec_load_4[1];
        clear_13_1[2] = (unsigned int)_vec_load_4[2];
        clear_13_1[3] = (unsigned int)_vec_load_4[3];
        unsigned int block_14_1 = (unsigned int)blockIdx.x * (unsigned int)gridDim.y + (unsigned int)blockIdx.y;
        unsigned int thread_15_1 = block_14_1 * (unsigned int)blockDim.x + (unsigned int)threadIdx.x;
        unsigned int threads_16_1 = (unsigned int)gridDim.x * (unsigned int)gridDim.y * (unsigned int)blockDim.x;
        unsigned int sentinel_17_1[4] = {2147483648, 2147483648, 2147483648, 2147483648};
        if (num_stages_12_1 > 1) {
            unsigned long long stage_offset_0_3 = (unsigned long long)((unsigned int)(uint8_t)dirty_10_1 * num_stages_12_1 + (unsigned int)1) * (unsigned long long)stage_bytes_11_1;
            unsigned int boundary_1_3 = (clear_13_1[1] + 15) / 16;
            for (unsigned int packed_1_3 = thread_15_1; packed_1_3 < boundary_1_3; packed_1_3 += threads_16_1) {
                {
                    auto _gst_73 = reinterpret_cast<unsigned __int128*>(bytes_base_9_1 + stage_offset_0_3) + packed_1_3;
                    __builtin_assume(__isGlobal(_gst_73));
                    reinterpret_cast<int4*>(_gst_73)[0] = reinterpret_cast<int4*>(sentinel_17_1)[0];
                }
            }
        }
        if (warp == 0) {
            asm volatile("barrier.sync 2, 448;" ::: "memory");
        } else {
            asm volatile("barrier.arrive 2, 448;" ::: "memory");
        }
        if (tid == 0) {
            asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(k3_flags) + (8))), "r"(static_cast<unsigned int>(1)) : "memory");
        }
        int g_1 = kb * 448 + tid;
        if (g_1 < num_tokens * 896) {
            int tok = g_1 / 896;
            int pk = g_1 - tok * 896;
            int row_index = tok * 7168 + pk * 8;
            uint32_t _sysv_poll_group_0[4];
            do {
                asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(_sysv_poll_group_0[0]), "=r"(_sysv_poll_group_0[1]), "=r"(_sysv_poll_group_0[2]), "=r"(_sysv_poll_group_0[3]) : "l"(reinterpret_cast<__nv_bfloat16*>(k3_peer_ptrs[rank]) + (k3_epoch_elements + k3_stage_elements + (unsigned long long)row_index)) : "memory");
            } while (
                (_sysv_poll_group_0[0] == 0x80000000u) || (_sysv_poll_group_0[1] == 0x80000000u) || (_sysv_poll_group_0[2] == 0x80000000u) || (_sysv_poll_group_0[3] == 0x80000000u)
            );
            reinterpret_cast<int4*>(out + row_index)[0] = reinterpret_cast<int4*>(_sysv_poll_group_0)[0];
        }
        if (kb == 0) {
            if (tid == 0) {
                {
                    volatile unsigned int* _sysv_wait_p_74 = reinterpret_cast<volatile unsigned int*>(k3_flags) + (8);
                    while (*_sysv_wait_p_74 < static_cast<unsigned int>(num_tokens + k23_ctas)) {}
                }
                unsigned int current_1 = (unsigned int)_vec_load_3[0];
                unsigned int head_1[4];
                head_1[0] = (current_1 + 1) % 3;
                head_1[1] = current_1;
                head_1[2] = (unsigned int)_vec_load_3[2];
                head_1[3] = 2;
                unsigned int clear_0_1[4];
                clear_0_1[0] = (unsigned int)(num_tokens * 12 * 640 * 2);
                clear_0_1[1] = (unsigned int)(num_tokens * 7168 * 2);
                clear_0_1[2] = 0;
                clear_0_1[3] = 0;
                reinterpret_cast<int4*>(k3_flags)[0] = reinterpret_cast<int4*>(head_1)[0];
                reinterpret_cast<int4*>(k3_flags + 4)[0] = reinterpret_cast<int4*>(clear_0_1)[0];
                k3_flags[8] = 0;
            }
        }
    }
}

} // extern "C"
