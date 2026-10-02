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
 *
 * Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
 * DeepGEMM portions are licensed under MIT; see DEEPGEMM_NOTICE.txt.
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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_LOGICAL_OFF 0
#define SMEM_LOGICAL_STAGE_BYTES 16384
#define SMEM_LOGICAL_STRIDE 16384
#define SMEM_PACKED_BLOCKS_OFF 16384
#define SMEM_PACKED_BLOCKS_STAGE_BYTES 16384
#define SMEM_PACKED_BLOCKS_STRIDE 16384
#define SMEM_WARP_SUMS_OFF 32768
#define SMEM_WARP_SUMS_STAGE_BYTES 32
#define SMEM_WARP_SUMS_STRIDE 32
#define SMEM_HIST_OFF 32800
#define SMEM_HIST_STAGE_BYTES 288
#define SMEM_HIST_STRIDE 288
#define SMEM_STATE_OFF 33088
#define SMEM_STATE_STAGE_BYTES 32
#define SMEM_STATE_STRIDE 32
#define SMEM_TOTAL 33152
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, 4) void
kernel_cake_deepgemm_sparse_mqa_6a8db9d32c43b0fa39ee(unsigned int* __restrict__ Starts, unsigned int* __restrict__ Ends, unsigned int* __restrict__ Context, unsigned int* __restrict__ BlockTable, unsigned int* __restrict__ Requests, unsigned int* __restrict__ Sparse, unsigned int* __restrict__ Metadata, unsigned int* __restrict__ Workspace, unsigned int num_q_tokens, unsigned int num_kv_tokens, unsigned int block_table_stride, unsigned int num_ctas, unsigned int num_sms, CakeFastDivmod sms_divmod, unsigned int num_max_sparse_blocks, unsigned int blocks_per_split, CakeFastDivmod split_divmod, unsigned int sparse_block_kv, unsigned int block_shift, unsigned int blocks_per_page, unsigned int page_shift)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* logical = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int logical_addr = smem + 0;
    unsigned int* packed_blocks = reinterpret_cast<unsigned int*>(smem_raw + 16384);
    const int packed_blocks_addr = smem + 16384;
    unsigned int* warp_sums = reinterpret_cast<unsigned int*>(smem_raw + 32768);
    const int warp_sums_addr = smem + 32768;
    unsigned int* hist = reinterpret_cast<unsigned int*>(smem_raw + 32800);
    const int hist_addr = smem + 32800;
    unsigned int* state = reinterpret_cast<unsigned int*>(smem_raw + 33088);
    const int state_addr = smem + 33088;

    // === Task calls (dependency order) ===
    unsigned int qidx = bid * 2;
    unsigned int rec_base = 96 + num_q_tokens * 2;
    #pragma unroll 1
    for (unsigned int claim_round = 0; claim_round < num_q_tokens + 1; claim_round++) {
        if (((0) ? warp == 0 : tid == 0)) {
            state[1] = 0;
            #pragma unroll 1
            for (unsigned int candidate = 0; candidate < num_q_tokens + 1; candidate++) {
                if (qidx >= num_q_tokens) {
                    break;
                }
                unsigned int nq = ((num_q_tokens - qidx >= 2) ? 2 : 1);
                unsigned int valid_q = 1;
                if (valid_q != 0) {
                    state[0] = qidx;
                    state[1] = nq;
                    #pragma unroll
                    for (int qi = 0; qi < 2; qi++) {
                        unsigned int nblocks = 0;
                        if (nq > (unsigned int)qi) {
                            unsigned int _min_0 = ((Starts[qidx + (unsigned int)qi]) < (num_kv_tokens) ? (Starts[qidx + (unsigned int)qi]) : (num_kv_tokens));
                            unsigned int begin = _min_0;
                            unsigned int _min_1 = ((Ends[qidx + (unsigned int)qi]) < (num_kv_tokens) ? (Ends[qidx + (unsigned int)qi]) : (num_kv_tokens));
                            unsigned int _max_0 = ((begin) > (_min_1) ? (begin) : (_min_1));
                            unsigned int end = _max_0;
                            unsigned int _min_2 = ((num_max_sparse_blocks) < (end - begin + sparse_block_kv - 1 >> block_shift) ? (num_max_sparse_blocks) : (end - begin + sparse_block_kv - 1 >> block_shift));
                            nblocks = _min_2;
                        }
                        state[2 + qi] = nblocks;
                    }
                    break;
                }
            }
        }
        __syncthreads();
        unsigned int nq_1 = state[1];
        if (nq_1 == 0) {
            break;
        }
        unsigned int qbase = state[0];
        unsigned int n0 = state[2];
        unsigned int n1 = state[3];
        #pragma unroll
        for (int qi_1 = 0; qi_1 < 2; qi_1++) {
            unsigned int num = ((qi_1 == 0) ? n0 : n1);
            #pragma unroll 1
            for (unsigned int slot = tid * 4; slot < num; slot += 1024) {
                asm volatile("cp.async.cg.shared::cta.global.L2::256B [%0], [%1], 16;"
                    :: "r"(logical_addr + ((unsigned int)(qi_1 * 2048) + slot) * 4), "l"(Sparse + ((qbase + (unsigned int)qi_1) * num_max_sparse_blocks + slot)));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        unsigned int inputs = n0 + n1;
        unsigned int merge_begin = (unsigned int)tid * inputs / 256;
        unsigned int merge_end = (unsigned int)(tid + 1) * inputs / 256;
        unsigned int _max_1 = ((merge_begin) > (n1) ? (merge_begin) : (n1));
        unsigned int lo = _max_1 - n1;
        unsigned int _min_3 = ((merge_begin) < (n0) ? (merge_begin) : (n0));
        unsigned int hi = _min_3;
        #pragma unroll 1
        for (unsigned int search = 0; search < n0 + 1; search++) {
            if (lo >= hi) {
                break;
            }
            unsigned int i0 = (lo + hi) / 2;
            unsigned int i1 = merge_begin - i0;
            unsigned int go_right = 0;
            if (i1 > 0 && i0 < n0) {
                unsigned int preceding_q1 = logical[2048 + i1 - 1];
                unsigned int current_q0 = logical[i0];
                if (preceding_q1 >= current_q0) {
                    go_right = 1;
                }
            }
            if (go_right != 0) {
                lo = i0 + 1;
            } else {
                hi = i0;
            }
        }
        unsigned int i0_1 = lo;
        unsigned int i1_1 = merge_begin - lo;
        unsigned int remaining = merge_end - merge_begin;
        unsigned int merged_count = 0;
        if (remaining > 0 && i0_1 > 0 && i1_1 < n1) {
            if (logical[i0_1 - 1] == logical[2048 + i1_1]) {
                i1_1 += 1;
                remaining -= 1;
            }
        }
        unsigned int packed_local[16];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            if (remaining != 0) {
                unsigned int v0 = 4294967295;
                unsigned int v1 = 4294967295;
                if (i0_1 < n0) {
                    v0 = logical[i0_1];
                }
                if (i1_1 < n1) {
                    v1 = logical[2048 + i1_1];
                }
                unsigned int in0 = ((v0 <= v1) ? 1 : 0);
                unsigned int in1 = ((v1 <= v0) ? 1 : 0);
                unsigned int consume1 = ((in1 != 0 && remaining > in0) ? 1 : 0);
                packed_local[j] = i0_1 | in0 << 15 | (i1_1 | in1 << 15) << 16;
                merged_count += 1;
                i0_1 += in0;
                i1_1 += consume1;
                remaining -= in0 + consume1;
            }
        }
        unsigned int lane_sum = merged_count;
        unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 1, 32);
        unsigned int synced = _shfl_up_0;
        if (lane >= 1) {
            lane_sum += synced;
        }
        unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 2, 32);
        unsigned int synced_0 = _shfl_up_1;
        if (lane >= 2) {
            lane_sum += synced_0;
        }
        unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 4, 32);
        unsigned int synced_1 = _shfl_up_2;
        if (lane >= 4) {
            lane_sum += synced_1;
        }
        unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 8, 32);
        unsigned int synced_2 = _shfl_up_3;
        if (lane >= 8) {
            lane_sum += synced_2;
        }
        unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, lane_sum, 16, 32);
        unsigned int synced_3 = _shfl_up_4;
        if (lane >= 16) {
            lane_sum += synced_3;
        }
        if (lane == 31) {
            warp_sums[warp] = lane_sum;
        }
        __syncthreads();
        unsigned int warp_total = 0;
        if (lane < 8) {
            warp_total = warp_sums[lane];
        }
        unsigned int warp_sum = warp_total;
        unsigned int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 1, 32);
        unsigned int synced_4 = _shfl_up_5;
        if (lane >= 1) {
            warp_sum += synced_4;
        }
        unsigned int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 2, 32);
        unsigned int synced_5 = _shfl_up_6;
        if (lane >= 2) {
            warp_sum += synced_5;
        }
        unsigned int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 4, 32);
        unsigned int synced_6 = _shfl_up_7;
        if (lane >= 4) {
            warp_sum += synced_6;
        }
        unsigned int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 8, 32);
        unsigned int synced_7 = _shfl_up_8;
        if (lane >= 8) {
            warp_sum += synced_7;
        }
        unsigned int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, warp_sum, 16, 32);
        unsigned int synced_8 = _shfl_up_9;
        if (lane >= 16) {
            warp_sum += synced_8;
        }
        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, warp_sum, 7);
        unsigned int total = _shfl_0;
        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, warp_sum - warp_total, warp);
        unsigned int preceding = _shfl_1;
        uint32_t _fast_div_q_0 = (split_divmod.divisor != 1) ? (__umulhi((uint32_t)(total + blocks_per_split - 1), split_divmod.multiplier) >> split_divmod.shift_right) : (uint32_t)(total + blocks_per_split - 1);
        uint32_t _fast_div_r_0 = (uint32_t)(total + blocks_per_split - 1) - _fast_div_q_0 * (uint32_t)(split_divmod.divisor);
        if (tid == 0) {
            state[4] = 0;
            if (_fast_div_q_0 != 0) {
                unsigned int _atomic_old_0 = atomicAdd(&Workspace[0], _fast_div_q_0);
                state[4] = _atomic_old_0;
            }
            Workspace[96 + qbase * 2] = state[4];
            Workspace[96 + qbase * 2 + 1] = _fast_div_q_0;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 16; j_1++) {
            if (merged_count > (unsigned int)j_1) {
                unsigned int merged_idx = lane_sum - merged_count + preceding + (unsigned int)j_1;
                packed_blocks[merged_idx] = packed_local[j_1];
            }
        }
        __syncthreads();
        unsigned int split_base = state[4];
        unsigned int range_end = split_base + _fast_div_q_0;
        #pragma unroll 1
        for (unsigned int owned = tid; owned < _fast_div_q_0; owned += 256) {
            unsigned long long owner_record = (unsigned long long)qbase | (unsigned long long)range_end << 32;
            *(reinterpret_cast<unsigned long long*>(Workspace + (rec_base + (split_base + owned) * 2)) + (0)) = owner_record;
        }
        unsigned int block_scale = sparse_block_kv;
        #pragma unroll 1
        for (unsigned int pair_idx = tid; pair_idx < (total + 1) / 2; pair_idx += 256) {
            unsigned int first = pair_idx * 2;
            uint32_t _fast_div_q_1 = (split_divmod.divisor != 1) ? (__umulhi((uint32_t)(first), split_divmod.multiplier) >> split_divmod.shift_right) : (uint32_t)(first);
            uint32_t _fast_div_r_1 = (uint32_t)(first) - _fast_div_q_1 * (uint32_t)(split_divmod.divisor);
            unsigned int split = split_base + _fast_div_q_1;
            unsigned int bases = packed_blocks[first - _fast_div_r_1];
            unsigned int b0 = bases & 32767;
            unsigned int b1 = bases >> 16 & 32767;
            unsigned int packed_value = packed_blocks[first];
            unsigned int s0 = packed_value & 32767;
            unsigned int s1 = packed_value >> 16 & 32767;
            unsigned int in0_1 = packed_value & 32768;
            unsigned int in1_1 = packed_value & 2147483648u;
            unsigned int logical_block = ((in0_1 != 0) ? logical[s0] : logical[2048 + s1]);
            unsigned int o0 = ((in0_1 != 0) ? s0 - b0 : (unsigned int)65535);
            unsigned int o1 = ((in1_1 != 0) ? s1 - b1 : (unsigned int)65535);
            unsigned int pair[4];
            pair[0] = logical_block * block_scale;
            pair[1] = o0 | o1 << 16;
            unsigned int second_physical = 0;
            unsigned int second_offsets = (unsigned int)4294967295;
            if (total > first + 1) {
                unsigned int next_value = packed_blocks[first + 1];
                unsigned int t0 = next_value & 32767;
                unsigned int t1 = next_value >> 16 & 32767;
                unsigned int next_in0 = next_value & 32768;
                unsigned int next_in1 = next_value & 2147483648u;
                unsigned int next_block = ((next_in0 != 0) ? logical[t0] : logical[2048 + t1]);
                second_physical = next_block * block_scale;
                unsigned int p0 = ((next_in0 != 0) ? t0 - b0 : (unsigned int)65535);
                unsigned int p1 = ((next_in1 != 0) ? t1 - b1 : (unsigned int)65535);
                second_offsets = p0 | p1 << 16;
            }
            pair[2] = second_physical;
            pair[3] = second_offsets;
            unsigned int dst = 4 + split * (4 + blocks_per_split * 2);
            if (_fast_div_r_1 == 0) {
                unsigned int _min_4 = ((blocks_per_split) < (total - first) ? (blocks_per_split) : (total - first));
                unsigned int num_1 = _min_4;
                unsigned int contiguous = 0;
                unsigned int last = packed_blocks[first + num_1 - 1];
                unsigned int last0 = last & 32768;
                unsigned int last_slot = ((last0 != 0) ? last & 32767 : last >> 16 & 32767);
                unsigned int last_block = ((last0 != 0) ? logical[last_slot] : logical[2048 + last_slot]);
                if (num_1 == blocks_per_split && last_block == logical_block + num_1 - 1) {
                    contiguous = 2147483648;
                }
                unsigned int split_header[4];
                split_header[0] = qbase;
                split_header[1] = num_1 | contiguous;
                split_header[2] = b0;
                split_header[3] = ((nq_1 == 2) ? b1 : (unsigned int)4294967295);
                reinterpret_cast<int4*>(Metadata + dst)[0] = reinterpret_cast<int4*>(split_header)[0];
            }
            reinterpret_cast<int4*>(Metadata + (dst + 4 + _fast_div_r_1 * 2))[0] = reinterpret_cast<int4*>(pair)[0];
        }
        #pragma unroll 1
        for (unsigned int padded = (total + 1) / 2 * 2 + (unsigned int)(tid * 2); padded < _fast_div_q_0 * blocks_per_split; padded += 512) {
            uint32_t _fast_div_q_2 = (split_divmod.divisor != 1) ? (__umulhi((uint32_t)(padded), split_divmod.multiplier) >> split_divmod.shift_right) : (uint32_t)(padded);
            uint32_t _fast_div_r_2 = (uint32_t)(padded) - _fast_div_q_2 * (uint32_t)(split_divmod.divisor);
            unsigned int dst_1 = 4 + (split_base + _fast_div_q_2) * (4 + blocks_per_split * 2) + 4 + _fast_div_r_2 * 2;
            unsigned int pad_pair[4];
            pad_pair[0] = 0;
            pad_pair[1] = (unsigned int)4294967295;
            pad_pair[2] = 0;
            pad_pair[3] = (unsigned int)4294967295;
            reinterpret_cast<int4*>(Metadata + dst_1)[0] = reinterpret_cast<int4*>(pad_pair)[0];
        }
        if (((0) ? warp == 0 : tid == 0)) {
            qidx += num_ctas * 2;
        }
    }
    if (tid == 0) {
        unsigned int _atomic_old_1;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(_atomic_old_1) : "l"(&Workspace[64]), "r"(static_cast<uint32_t>(1)) : "memory");
        state[5] = ((_atomic_old_1 + 1 == num_ctas) ? 1 : 0);
        if (state[5] != 0) {
            unsigned int _load_acquire_0;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_0) : "l"((reinterpret_cast<unsigned int*>(Workspace) + (64))) : "memory");
        }
    }
    __syncthreads();
    if (state[5] != 0) {
        unsigned int total_1 = Workspace[0];
        unsigned int sched = 4 + total_1 * (4 + blocks_per_split * 2);
        if (tid == 0) {
            state[6] = 1;
        }
        __syncthreads();
        unsigned int nentries = 0;
        if ((unsigned int)tid < num_sms) {
            uint32_t _fast_div_q_3 = (sms_divmod.divisor != 1) ? (__umulhi((uint32_t)(total_1 * (unsigned int)tid + num_sms - 1), sms_divmod.multiplier) >> sms_divmod.shift_right) : (uint32_t)(total_1 * (unsigned int)tid + num_sms - 1);
            uint32_t _fast_div_r_3 = (uint32_t)(total_1 * (unsigned int)tid + num_sms - 1) - _fast_div_q_3 * (uint32_t)(sms_divmod.divisor);
            uint32_t _fast_div_q_4 = (sms_divmod.divisor != 1) ? (__umulhi((uint32_t)(total_1 * (unsigned int)(tid + 1) + num_sms - 1), sms_divmod.multiplier) >> sms_divmod.shift_right) : (uint32_t)(total_1 * (unsigned int)(tid + 1) + num_sms - 1);
            uint32_t _fast_div_r_4 = (uint32_t)(total_1 * (unsigned int)(tid + 1) + num_sms - 1) - _fast_div_q_4 * (uint32_t)(sms_divmod.divisor);
            unsigned int split_1 = _fast_div_q_3;
            unsigned int end_1 = _fast_div_q_4;
            #pragma unroll 1
            for (unsigned int schedule_entry = 0; schedule_entry < end_1; schedule_entry++) {
                if (split_1 >= end_1) {
                    break;
                }
                unsigned long long _vec_load_0[1];
                {
                    _vec_load_0[0] = *reinterpret_cast<const unsigned long long*>(Workspace + (rec_base + split_1 * 2));
                }
                unsigned int qbase_1 = (unsigned int)_vec_load_0[0];
                unsigned int _min_5 = ((end_1) < ((unsigned int)(_vec_load_0[0] >> 32)) ? (end_1) : ((unsigned int)(_vec_load_0[0] >> 32)));
                unsigned int entry_end = _min_5;
                unsigned int dst_2 = sched + (nentries * num_sms + (unsigned int)tid) * 4;
                unsigned int schedule_entry_0[4];
                schedule_entry_0[0] = split_1;
                schedule_entry_0[1] = entry_end;
                schedule_entry_0[2] = qbase_1;
                int _min_6 = ((2) < (num_q_tokens - qbase_1) ? (2) : (num_q_tokens - qbase_1));
                schedule_entry_0[3] = _min_6;
                reinterpret_cast<int4*>(Metadata + dst_2)[0] = reinterpret_cast<int4*>(schedule_entry_0)[0];
                split_1 = entry_end;
                nentries += 1;
            }
            atomicMax(&state[6], nentries);
        }
        __syncthreads();
        unsigned int waves = state[6];
        if ((unsigned int)tid < num_sms) {
            #pragma unroll 1
            for (unsigned int wave = nentries; wave < waves; wave++) {
                unsigned int dst_3 = sched + (wave * num_sms + (unsigned int)tid) * 4;
                unsigned int empty_entry[4];
                #pragma unroll
                for (int j_2 = 0; j_2 < 4; j_2++) {
                    empty_entry[j_2] = 0;
                }
                reinterpret_cast<int4*>(Metadata + dst_3)[0] = reinterpret_cast<int4*>(empty_entry)[0];
            }
        }
        if (tid == 0) {
            Metadata[0] = total_1;
            Metadata[1] = waves;
            Metadata[2] = 0;
            Workspace[0] = 0;
            Workspace[32] = 0;
            Workspace[64] = 0;
        }
    }
}

} // extern "C"
