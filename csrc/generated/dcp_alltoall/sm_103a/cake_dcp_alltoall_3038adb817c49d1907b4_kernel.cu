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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_PACKET_OFF 0
#define SMEM_PACKET_STAGE_BYTES 1024
#define SMEM_PACKET_STRIDE 1024
#define SMEM_TOTAL 1088
#define THREADS 64

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ void
kernel_cake_dcp_alltoall_3038adb817c49d1907b4(__nv_bfloat16* __restrict__ partial_o, float* __restrict__ softmax_stats, __nv_bfloat16* __restrict__ partial_o_out, float* __restrict__ softmax_stats_out, uint8_t* __restrict__ workspace, unsigned long long workspace_stride_in_u64, int cp_rank, int entry_count, int max_channel_count)
{
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int warp = threadIdx.y;
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 1024;
    #define ready_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    unsigned long long* packet = reinterpret_cast<unsigned long long*>(smem_raw + 0);
    const int packet_addr = smem + 0;

    // === Task calls (dependency order) ===
    int _uniform_0 = make_warp_uniform(warp);
    int group = _uniform_0;
    int lane_1 = threadIdx.x;
    int peer = group;
    int channel = blockIdx.y;
    int run_channels = gridDim.y;
    int is_sender = blockIdx.z == 0;
    int is_direct = peer == cp_rank;
    int sender_rank = ((is_sender != 0) ? cp_rank : peer);
    int receiver_rank = ((is_sender != 0) ? peer : cp_rank);
    unsigned long long rank_stride_bytes = workspace_stride_in_u64 * 8;
    unsigned long long fifo_region_bytes = 1048576 * (unsigned long long)max_channel_count;
    unsigned long long info_plane_bytes = 512 * (unsigned long long)max_channel_count;
    unsigned long long fifo_rank_offset = (unsigned long long)receiver_rank * rank_stride_bytes;
    unsigned long long fifo_offset = fifo_rank_offset + (unsigned long long)sender_rank * (unsigned long long)max_channel_count * 524288 + (unsigned long long)channel * 524288;
    unsigned long long sender_info_offset = (unsigned long long)sender_rank * rank_stride_bytes + fifo_region_bytes + (unsigned long long)receiver_rank * (unsigned long long)max_channel_count * 256 + (unsigned long long)channel * 256;
    unsigned long long receiver_info_offset = (unsigned long long)receiver_rank * rank_stride_bytes + fifo_region_bytes + info_plane_bytes + (unsigned long long)sender_rank * (unsigned long long)max_channel_count * 256 + (unsigned long long)channel * 256;
    unsigned int group_smem_addr = packet_addr + (unsigned int)(group * 512);
    int group_word_base = group * 64;
    unsigned int phase = 0;
    if (elect_sync()) {
        mbarrier_init(ready_addr + (group) * 8, 32);
    }
    __syncwarp();
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (is_direct != 0) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        #pragma unroll 1
        for (int entry = channel; entry < entry_count * is_sender; entry += run_channels) {
            long long data_index = (long long)entry * 2 + (long long)cp_rank;
            if (lane_1 < 16) {
                float _vec_load_0[8];
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(partial_o + (data_index * 128 + (long long)(lane_1 * 8)) + 0);
                    uint4 _vld_0[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_0[_blk] = _vptr_0[_blk];
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                }
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_o_out + (data_index * 128 + (long long)(lane_1 * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
            if (lane_1 == 0) {
                float _vec_load_1[2];
                {
                    float2 _v2_1 = *reinterpret_cast<const float2*>(softmax_stats + (data_index * 2) + 0);
                    _vec_load_1[0] = _v2_1.x;
                    _vec_load_1[0 + 1] = _v2_1.y;
                }
                {
                    float2 _v2 = make_float2(_vec_load_1[0 + 0], _vec_load_1[0 + 1]);
                    *reinterpret_cast<float2*>(softmax_stats_out + (data_index * 2) + 0) = _v2;
                }
            }
        }
    } else if (is_sender != 0) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        long long head = reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[0];
        long long observed_tail = reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[1];
        int fifo_entry_block_base = 1024;
        int fifo_index = -1;
        #pragma unroll 1
        for (int entry_1 = channel; entry_1 < entry_count; entry_1 += run_channels) {
            long long data_index_1 = (long long)entry_1 * 2 + (long long)peer;
            if (elect_sync()) {
                cp_async_bulk_gmem2smem(group_smem_addr, partial_o + (data_index_1 * 128), 256, ready_addr + (group) * 8);
            }
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.ca.shared::cta.global [%1], [%2], 8;\n\t"
                "}"
                :: "r"((lane_1 == 0) ? 1 : 0), "r"(group_smem_addr + 256), "l"(softmax_stats + (data_index_1 * 2)));
            asm volatile("cp.async.commit_group;");
            mbarrier_arrive_expect_tx(ready_addr + (group) * 8, ((lane_1 == 0) ? 256 : 0));
            if (fifo_entry_block_base + 4 > 1024) {
                if (fifo_index >= 0) {
                    head += 1;
                    __syncwarp();
                    reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[0] = head;
                }
                fifo_index = (int)(head % 4);
                fifo_entry_block_base = 0;
                while (head >= observed_tail + 4) {
                    observed_tail = reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[1];
                }
                __syncwarp();
            }
            int fifo_base_block = fifo_index * 1024 + fifo_entry_block_base;
            asm volatile("cp.async.wait_group 0;");
            unsigned int ready_token = 0;
            while (ready_token == 0) {
                uint32_t _mbar_token_0 = mbarrier_try_wait_plain(ready_addr + (group) * 8, phase);
                ready_token = _mbar_token_0;
            }
            phase ^= 1;
            int half_lane = lane_1 % 16;
            int half_index = lane_1 / 16;
            int pack_base = half_index * 15;
            if (half_index == 0) {
                unsigned long long tail_value = (unsigned long long)head;
                int tail_flag_word = fifo_base_block + 3 + half_index & 15;
                int tail_data_word = half_lane;
                if (half_lane < 15 && pack_base + half_lane < 3) {
                    int data_block = pack_base + half_lane;
                    int flag_word = fifo_base_block + data_block & 15;
                    tail_value = packet[group_word_base + data_block * 16 + flag_word];
                    packet[group_word_base + data_block * 16 + flag_word] = (unsigned long long)head;
                }
                if (tail_data_word >= tail_flag_word) {
                    tail_data_word += 1;
                }
                if (half_lane == 15) {
                    tail_data_word = tail_flag_word;
                }
                packet[group_word_base + 48 + tail_data_word] = tail_value;
            }
            __syncwarp();
            int fifo_vec[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&fifo_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&fifo_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&fifo_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&fifo_vec[(0) + 3]))
                : "r"(group_smem_addr + (unsigned int)(lane_1 * 16)));
            {
                int4 _iv4 = make_int4(fifo_vec[0 + 0], fifo_vec[0 + 1], fifo_vec[0 + 2], fifo_vec[0 + 3]);
                *reinterpret_cast<int4*>(workspace + (fifo_offset + (unsigned long long)fifo_index * 131072 + (unsigned long long)fifo_entry_block_base * 128 + (unsigned long long)lane_1 * 16) + 0) = _iv4;
            }
            __syncwarp();
            fifo_entry_block_base += 4;
        }
        if (fifo_entry_block_base > 0) {
            head += 1;
            reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[0] = head;
        }
    } else {
        long long tail = reinterpret_cast<volatile long long*>(workspace + receiver_info_offset)[1];
        int fifo_entry_block_base_1 = 1024;
        int fifo_index_1 = -1;
        int need_release = 0;
        #pragma unroll 1
        for (int entry_2 = channel; entry_2 < entry_count; entry_2 += run_channels) {
            if (entry_2 + run_channels >= entry_count) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (fifo_entry_block_base_1 + 4 > 1024) {
                if (fifo_index_1 >= 0) {
                    tail += 1;
                    need_release = 1;
                }
                fifo_index_1 = (int)(tail % 4);
                fifo_entry_block_base_1 = 0;
                __syncwarp();
            }
            int fifo_base_block_1 = fifo_index_1 * 1024 + fifo_entry_block_base_1;
            int loaded_blocks = 0;
            while (loaded_blocks < 4) {
                int remaining_blocks = 4 - loaded_blocks;
                if (elect_sync()) {
                    cp_async_bulk_gmem2smem(group_smem_addr + (unsigned int)(loaded_blocks * 128), workspace + (fifo_offset + (unsigned long long)fifo_index_1 * 131072 + (unsigned long long)fifo_entry_block_base_1 * 128 + (unsigned long long)loaded_blocks * 128), remaining_blocks * 128, ready_addr + (group) * 8);
                }
                mbarrier_arrive_expect_tx(ready_addr + (group) * 8, ((lane_1 == 0) ? remaining_blocks * 128 : 0));
                if (need_release != 0) {
                    reinterpret_cast<volatile long long*>(workspace + receiver_info_offset)[1] = tail;
                    reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[1] = tail;
                    need_release = 0;
                }
                unsigned int ready_token_1 = 0;
                while (ready_token_1 == 0) {
                    uint32_t _mbar_token_1 = mbarrier_try_wait_plain(ready_addr + (group) * 8, phase);
                    ready_token_1 = _mbar_token_1;
                }
                phase ^= 1;
                int matched = 0;
                if (lane_1 < remaining_blocks) {
                    int probe_block = loaded_blocks + lane_1;
                    int probe_flag_word = (fifo_base_block_1 + probe_block) % 16;
                    unsigned long long probe_flag = packet[group_word_base + probe_block * 16 + probe_flag_word];
                    matched = probe_flag == (unsigned long long)tail;
                }
                __syncwarp();
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, matched != 0);
                unsigned int valid_mask = _vote_0;
                int _ffs_0 = __ffs(~valid_mask);
                int valid_blocks = _ffs_0 - 1;
                loaded_blocks += valid_blocks;
            }
            int half_lane_1 = lane_1 % 16;
            int half_index_1 = lane_1 / 16;
            int unpack_base = half_index_1 * 15;
            if (half_lane_1 < 15 && unpack_base + half_lane_1 < 3) {
                int data_block_1 = unpack_base + half_lane_1;
                int flag_word_1 = fifo_base_block_1 + data_block_1 & 15;
                int tail_flag_word_1 = fifo_base_block_1 + 3 + half_index_1 & 15;
                int tail_data_word_1 = half_lane_1;
                if (tail_data_word_1 >= tail_flag_word_1) {
                    tail_data_word_1 += 1;
                }
                unsigned long long carried = packet[group_word_base + 48 + tail_data_word_1];
                packet[group_word_base + data_block_1 * 16 + flag_word_1] = carried;
            }
            __syncwarp();
            long long data_index_2 = (long long)entry_2 * 2 + (long long)peer;
            if (elect_sync()) {
                {
                    void* _cpbulk_dst_2 = reinterpret_cast<void*>(partial_o_out + (data_index_2 * 128));
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                        :: "l"(_cpbulk_dst_2), "r"(group_smem_addr), "r"((uint32_t)(256))
                        : "memory");
                }
                float stats[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&stats[0])), "=r"(*reinterpret_cast<uint32_t*>(&stats[(0) + 1]))
                    : "r"(group_smem_addr + 256));
                {
                    float2 _v2 = make_float2(stats[0 + 0], stats[0 + 1]);
                    *reinterpret_cast<float2*>(softmax_stats_out + (data_index_2 * 2) + 0) = _v2;
                }
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group.read 0;");
            fifo_entry_block_base_1 += 4;
        }
        if (fifo_entry_block_base_1 > 0) {
            tail += 1;
            reinterpret_cast<volatile long long*>(workspace + receiver_info_offset)[1] = tail;
            reinterpret_cast<volatile long long*>(workspace + sender_info_offset)[1] = tail;
        }
    }
}

} // extern "C"
