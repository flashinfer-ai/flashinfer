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
#define NUM_SMS 148

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

extern "C" {

__global__ __launch_bounds__(256, 4) void
kernel_cake_deepgemm_sparse_mqa_eddc11afc3a31783d4ab(unsigned int* __restrict__ Starts, unsigned int* __restrict__ Ends, unsigned int* __restrict__ Context, unsigned int* __restrict__ BlockTable, unsigned int* __restrict__ Requests, unsigned int* __restrict__ Sparse, unsigned int* __restrict__ Metadata, unsigned int* __restrict__ Workspace, unsigned int num_q_tokens, unsigned int num_kv_tokens, unsigned int block_table_stride, unsigned int num_ctas, unsigned int num_sms, CakeFastDivmod sms_divmod, unsigned int num_max_sparse_blocks, unsigned int blocks_per_split, CakeFastDivmod split_divmod, unsigned int sparse_block_kv, unsigned int block_shift, unsigned int blocks_per_page, unsigned int page_shift)
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
    unsigned int qidx = bid;
    #pragma unroll 1
    for (unsigned int claim_round = 0; claim_round < num_q_tokens + 1; claim_round++) {
        if (((1) ? warp == 0 : tid == 0)) {
            state[1] = 0;
            #pragma unroll 1
            for (unsigned int candidate = 0; candidate < num_q_tokens + 1; candidate++) {
                if (qidx >= num_q_tokens) {
                    break;
                }
                unsigned int nq = ((num_q_tokens - qidx >= 2) ? 2 : 1);
                unsigned int valid_q = 1;
                unsigned int request = Requests[qidx];
                unsigned int probe_pos = 32;
                if (qidx > lane) {
                    if (Requests[qidx - 1 - lane] != request) {
                        probe_pos = lane;
                    }
                }
                unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, probe_pos, 16);
                unsigned int probe_other = _shfl_xor_0;
                unsigned int _min_0 = ((probe_pos) < (probe_other) ? (probe_pos) : (probe_other));
                probe_pos = _min_0;
                unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, probe_pos, 8);
                unsigned int probe_other_0 = _shfl_xor_1;
                unsigned int _min_1 = ((probe_pos) < (probe_other_0) ? (probe_pos) : (probe_other_0));
                probe_pos = _min_1;
                unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, probe_pos, 4);
                unsigned int probe_other_1 = _shfl_xor_2;
                unsigned int _min_2 = ((probe_pos) < (probe_other_1) ? (probe_pos) : (probe_other_1));
                probe_pos = _min_2;
                unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, probe_pos, 2);
                unsigned int probe_other_2 = _shfl_xor_3;
                unsigned int _min_3 = ((probe_pos) < (probe_other_2) ? (probe_pos) : (probe_other_2));
                probe_pos = _min_3;
                unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, probe_pos, 1);
                unsigned int probe_other_3 = _shfl_xor_4;
                unsigned int _min_4 = ((probe_pos) < (probe_other_3) ? (probe_pos) : (probe_other_3));
                probe_pos = _min_4;
                unsigned int _min_5 = ((qidx) < ((unsigned int)32) ? (qidx) : ((unsigned int)32));
                unsigned int _min_6 = ((probe_pos) < (_min_5) ? (probe_pos) : (_min_5));
                unsigned int request_begin = qidx - _min_6;
                if (probe_pos == 32) {
                    #pragma unroll 1
                    for (unsigned int previous = 0; previous < request_begin; previous++) {
                        if (Requests[request_begin - 1] != request) {
                            break;
                        }
                        request_begin -= 1;
                    }
                }
                if ((qidx - request_begin) % 2 != 0) {
                    if (num_ctas >= num_q_tokens) {
                        qidx = num_q_tokens;
                    } else {
                        unsigned int claim = 0;
                        if (lane == 0) {
                            unsigned int _atomic_old_0 = atomicAdd(&Workspace[32], 1);
                            claim = _atomic_old_0;
                        }
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, claim, 0);
                        claim = _shfl_0;
                        qidx = num_ctas + claim;
                    }
                    valid_q = 0;
                } else {
                    nq = 1;
                    if (qidx + 1 < num_q_tokens) {
                        if (Requests[qidx + 1] == request) {
                            nq = 2;
                        }
                    }
                }
                if (valid_q != 0) {
                    state[0] = qidx;
                    state[1] = nq;
                    #pragma unroll
                    for (int qi = 0; qi < 2; qi++) {
                        unsigned int nblocks = 0;
                        if (nq > (unsigned int)qi) {
                            unsigned int begin = 0;
                            unsigned int end = Context[qidx + (unsigned int)qi];
                            int _min_7 = ((2048) < (end - begin + 8 - 1 >> 3) ? (2048) : (end - begin + 8 - 1 >> 3));
                            nblocks = _min_7;
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
                    :: "r"(logical_addr + ((unsigned int)(qi_1 * 2048) + slot) * 4), "l"(Sparse + ((qbase + (unsigned int)qi_1) * 2048 + slot)));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        unsigned int inputs = n0 + n1;
        unsigned int merge_begin = (unsigned int)tid * inputs / 256;
        unsigned int merge_end = (unsigned int)(tid + 1) * inputs / 256;
        unsigned int _max_0 = ((merge_begin) > (n1) ? (merge_begin) : (n1));
        unsigned int lo = _max_0 - n1;
        unsigned int _min_8 = ((merge_begin) < (n0) ? (merge_begin) : (n0));
        unsigned int hi = _min_8;
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
        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, warp_sum, 7);
        unsigned int total = _shfl_1;
        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, warp_sum - warp_total, warp);
        unsigned int preceding = _shfl_2;
        uint32_t _fast_div_q_0 = (split_divmod.divisor != 1) ? (__umulhi((uint32_t)(total + blocks_per_split - 1), split_divmod.multiplier) >> split_divmod.shift_right) : (uint32_t)(total + blocks_per_split - 1);
        uint32_t _fast_div_r_0 = (uint32_t)(total + blocks_per_split - 1) - _fast_div_q_0 * (uint32_t)(split_divmod.divisor);
        unsigned int nsplits = _fast_div_q_0;
        if (tid == 0) {
            state[4] = 0;
            if (nsplits != 0) {
                unsigned int _atomic_old_1 = atomicAdd(&Workspace[0], nsplits);
                state[4] = _atomic_old_1;
            }
            unsigned long long qinfo_record = (unsigned long long)state[4] | (unsigned long long)nsplits << 32;
            *(reinterpret_cast<unsigned long long*>(Workspace + (96 + qbase * 2)) + (0)) = qinfo_record;
            if (nq_1 == 2) {
                *(reinterpret_cast<unsigned long long*>(Workspace + (96 + (qbase + 1) * 2)) + (0)) = (unsigned long long)0;
            }
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
        unsigned int npairs = (total + 1) / 2;
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            unsigned int pr_phys0[2];
            unsigned int pr_phys1[2];
            #pragma unroll
            for (int kk = 0; kk < 2; kk++) {
                unsigned int pair_idx = tid + (kc * 2 + kk) * 256;
                if (pair_idx < npairs) {
                    unsigned int page_mask = 7;
                    unsigned int first = pair_idx * 2;
                    unsigned int packed_value = packed_blocks[first];
                    unsigned int s0 = packed_value & 32767;
                    unsigned int s1 = packed_value >> 16 & 32767;
                    unsigned int in0_1 = packed_value & 32768;
                    unsigned int logical_block = ((in0_1 != 0) ? logical[s0] : logical[2048 + s1]);
                    unsigned int logical_page = logical_block >> 3;
                    unsigned int in_page = logical_block & page_mask;
                    unsigned int physical0 = (BlockTable[(unsigned long long)qbase * (unsigned long long)block_table_stride + (unsigned long long)logical_page] << 3) + in_page;
                    unsigned int physical1 = 0;
                    if (total > first + 1) {
                        unsigned int next_value = packed_blocks[first + 1];
                        unsigned int t0 = next_value & 32767;
                        unsigned int t1 = next_value >> 16 & 32767;
                        unsigned int next_in0 = next_value & 32768;
                        unsigned int next_block = ((next_in0 != 0) ? logical[t0] : logical[2048 + t1]);
                        unsigned int next_page = next_block >> 3;
                        unsigned int next_in_page = next_block & page_mask;
                        physical1 = (BlockTable[(unsigned long long)qbase * (unsigned long long)block_table_stride + (unsigned long long)next_page] << 3) + next_in_page;
                    }
                    pr_phys0[kk] = physical0;
                    pr_phys1[kk] = physical1;
                }
            }
            #pragma unroll
            for (int kk_1 = 0; kk_1 < 2; kk_1++) {
                unsigned int pair_idx_1 = tid + (kc * 2 + kk_1) * 256;
                if (pair_idx_1 < npairs) {
                    unsigned int first_1 = pair_idx_1 * 2;
                    uint32_t _fast_div_q_1 = (split_divmod.divisor != 1) ? (__umulhi((uint32_t)(first_1), split_divmod.multiplier) >> split_divmod.shift_right) : (uint32_t)(first_1);
                    uint32_t _fast_div_r_1 = (uint32_t)(first_1) - _fast_div_q_1 * (uint32_t)(split_divmod.divisor);
                    unsigned int split_off = _fast_div_q_1;
                    unsigned int block = _fast_div_r_1;
                    unsigned int split = split_base + split_off;
                    unsigned int bases = packed_blocks[first_1 - block];
                    unsigned int b0 = bases & 32767;
                    unsigned int b1 = bases >> 16 & 32767;
                    unsigned int packed_value_1 = packed_blocks[first_1];
                    unsigned int s0_1 = packed_value_1 & 32767;
                    unsigned int s1_1 = packed_value_1 >> 16 & 32767;
                    unsigned int in0_2 = packed_value_1 & 32768;
                    unsigned int in1_1 = packed_value_1 & 2147483648u;
                    unsigned int logical_block_1 = ((in0_2 != 0) ? logical[s0_1] : logical[2048 + s1_1]);
                    unsigned int o0 = ((in0_2 != 0) ? s0_1 - b0 : (unsigned int)65535);
                    unsigned int o1 = ((in1_1 != 0) ? s1_1 - b1 : (unsigned int)65535);
                    unsigned int pair[4];
                    pair[0] = pr_phys0[kk_1];
                    pair[1] = o0 | o1 << 16;
                    unsigned int second_physical = 0;
                    unsigned int second_offsets = (unsigned int)4294967295;
                    if (total > first_1 + 1) {
                        unsigned int next_value_1 = packed_blocks[first_1 + 1];
                        unsigned int t0_1 = next_value_1 & 32767;
                        unsigned int t1_1 = next_value_1 >> 16 & 32767;
                        unsigned int next_in0_1 = next_value_1 & 32768;
                        unsigned int next_in1 = next_value_1 & 2147483648u;
                        unsigned int next_block_1 = ((next_in0_1 != 0) ? logical[t0_1] : logical[2048 + t1_1]);
                        second_physical = pr_phys1[kk_1];
                        unsigned int p0 = ((next_in0_1 != 0) ? t0_1 - b0 : (unsigned int)65535);
                        unsigned int p1 = ((next_in1 != 0) ? t1_1 - b1 : (unsigned int)65535);
                        second_offsets = p0 | p1 << 16;
                    }
                    pair[2] = second_physical;
                    pair[3] = second_offsets;
                    unsigned int dst = 4 + split * 4 + split * (blocks_per_split * 2);
                    if (block == 0) {
                        unsigned int _min_9 = ((blocks_per_split) < (total - first_1) ? (blocks_per_split) : (total - first_1));
                        unsigned int num_1 = _min_9;
                        unsigned int contiguous = 0;
                        unsigned int split_header[4];
                        split_header[0] = qbase;
                        split_header[1] = num_1 | contiguous;
                        split_header[2] = b0;
                        split_header[3] = ((nq_1 == 2) ? b1 : (unsigned int)4294967295);
                        reinterpret_cast<int4*>(Metadata + dst)[0] = reinterpret_cast<int4*>(split_header)[0];
                    }
                    reinterpret_cast<int4*>(Metadata + (dst + 4 + block * 2))[0] = reinterpret_cast<int4*>(pair)[0];
                }
            }
        }
        #pragma unroll 1
        for (unsigned int padded = (total + 1) / 2 * 2 + (unsigned int)(tid * 2); padded < nsplits * blocks_per_split; padded += 512) {
            uint32_t _fast_div_q_2 = (split_divmod.divisor != 1) ? (__umulhi((uint32_t)(padded), split_divmod.multiplier) >> split_divmod.shift_right) : (uint32_t)(padded);
            uint32_t _fast_div_r_2 = (uint32_t)(padded) - _fast_div_q_2 * (uint32_t)(split_divmod.divisor);
            unsigned int pad_split = split_base + _fast_div_q_2;
            unsigned int pad_block = _fast_div_r_2;
            unsigned int dst_1 = 4 + pad_split * 4 + pad_split * (blocks_per_split * 2) + 4 + pad_block * 2;
            unsigned int pad_pair[4];
            pad_pair[0] = 0;
            pad_pair[1] = (unsigned int)4294967295;
            pad_pair[2] = 0;
            pad_pair[3] = (unsigned int)4294967295;
            reinterpret_cast<int4*>(Metadata + dst_1)[0] = reinterpret_cast<int4*>(pad_pair)[0];
        }
        if (((1) ? warp == 0 : tid == 0)) {
            if (num_ctas >= num_q_tokens) {
                qidx = num_q_tokens;
            } else {
                unsigned int claim_t = 0;
                if (lane == 0) {
                    unsigned int _atomic_old_2 = atomicAdd(&Workspace[32], 1);
                    claim_t = _atomic_old_2;
                }
                unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, claim_t, 0);
                claim_t = _shfl_3;
                qidx = num_ctas + claim_t;
            }
        }
    }
    if (tid == 0) {
        unsigned int _atomic_old_3;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
            : "=r"(_atomic_old_3) : "l"(&Workspace[64]), "r"(static_cast<uint32_t>(1)) : "memory");
        state[5] = ((_atomic_old_3 + 1 == num_ctas) ? 1 : 0);
        if (state[5] != 0) {
            unsigned int _load_acquire_0;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_0) : "l"((reinterpret_cast<unsigned int*>(Workspace) + (64))) : "memory");
        }
    }
    __syncthreads();
    if (state[5] != 0) {
        unsigned int total_1 = Workspace[0];
        unsigned int sched = 4 + total_1 * 4 + total_1 * (blocks_per_split * 2);
        unsigned int nentries = 0;
        #pragma unroll 1
        for (unsigned int qbatch = 0; qbatch < num_q_tokens; qbatch += 512) {
            unsigned int qi_2 = qbatch + (unsigned int)tid;
            unsigned int qj = qbatch + 256 + (unsigned int)tid;
            unsigned int qsplit = 0;
            unsigned int qcount = 0;
            unsigned int qsplit2 = 0;
            unsigned int qcount2 = 0;
            if (qi_2 < num_q_tokens) {
                unsigned long long _vec_load_0[1];
                {
                    _vec_load_0[0] = *reinterpret_cast<const unsigned long long*>(Workspace + (96 + qi_2 * 2));
                }
                qsplit = (unsigned int)_vec_load_0[0];
                qcount = (unsigned int)(_vec_load_0[0] >> 32);
            }
            if (qj < num_q_tokens) {
                unsigned long long _vec_load_1[1];
                {
                    _vec_load_1[0] = *reinterpret_cast<const unsigned long long*>(Workspace + (96 + qj * 2));
                }
                qsplit2 = (unsigned int)_vec_load_1[0];
                qcount2 = (unsigned int)(_vec_load_1[0] >> 32);
            }
            unsigned int nq_2 = 1;
            if (qi_2 + 1 < num_q_tokens) {
                if (Requests[qi_2 + 1] == Requests[qi_2]) {
                    nq_2 = 2;
                }
            }
            unsigned int nq2 = 1;
            if (qj + 1 < num_q_tokens) {
                if (Requests[qj + 1] == Requests[qj]) {
                    nq2 = 2;
                }
            }
            unsigned int nqentries = (qcount + 7) / 8;
            unsigned int nqentries2 = (qcount2 + 7) / 8;
            unsigned int lane_sum_1 = nqentries;
            unsigned int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, lane_sum_1, 1, 32);
            unsigned int synced_9 = _shfl_up_10;
            if (lane >= 1) {
                lane_sum_1 += synced_9;
            }
            unsigned int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, lane_sum_1, 2, 32);
            unsigned int synced_0_1 = _shfl_up_11;
            if (lane >= 2) {
                lane_sum_1 += synced_0_1;
            }
            unsigned int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, lane_sum_1, 4, 32);
            unsigned int synced_1_1 = _shfl_up_12;
            if (lane >= 4) {
                lane_sum_1 += synced_1_1;
            }
            unsigned int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, lane_sum_1, 8, 32);
            unsigned int synced_2_1 = _shfl_up_13;
            if (lane >= 8) {
                lane_sum_1 += synced_2_1;
            }
            unsigned int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, lane_sum_1, 16, 32);
            unsigned int synced_3_1 = _shfl_up_14;
            if (lane >= 16) {
                lane_sum_1 += synced_3_1;
            }
            if (lane == 31) {
                warp_sums[warp] = lane_sum_1;
            }
            __syncthreads();
            unsigned int warp_total_1 = 0;
            if (lane < 8) {
                warp_total_1 = warp_sums[lane];
            }
            unsigned int warp_sum_1 = warp_total_1;
            unsigned int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, warp_sum_1, 1, 32);
            unsigned int synced_4_1 = _shfl_up_15;
            if (lane >= 1) {
                warp_sum_1 += synced_4_1;
            }
            unsigned int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, warp_sum_1, 2, 32);
            unsigned int synced_5_1 = _shfl_up_16;
            if (lane >= 2) {
                warp_sum_1 += synced_5_1;
            }
            unsigned int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, warp_sum_1, 4, 32);
            unsigned int synced_6_1 = _shfl_up_17;
            if (lane >= 4) {
                warp_sum_1 += synced_6_1;
            }
            unsigned int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, warp_sum_1, 8, 32);
            unsigned int synced_7_1 = _shfl_up_18;
            if (lane >= 8) {
                warp_sum_1 += synced_7_1;
            }
            unsigned int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, warp_sum_1, 16, 32);
            unsigned int synced_8_1 = _shfl_up_19;
            if (lane >= 16) {
                warp_sum_1 += synced_8_1;
            }
            unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, warp_sum_1, 7);
            unsigned int total_9 = _shfl_4;
            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, warp_sum_1 - warp_total_1, warp);
            unsigned int preceding_1 = _shfl_5;
            unsigned int per_entry = ((nqentries != 0) ? qcount / nqentries : (unsigned int)0);
            unsigned int larger = ((nqentries != 0) ? qcount % nqentries : (unsigned int)0);
            unsigned int cursor = qsplit;
            #pragma unroll 1
            for (unsigned int entry = 0; entry < nqentries; entry++) {
                unsigned int entry_end = cursor + per_entry + (unsigned int)(((larger > entry) ? 1 : 0));
                unsigned int dst_2 = sched + (nentries + (lane_sum_1 - nqentries + preceding_1) + entry) * 4;
                unsigned int schedule_entry[4];
                schedule_entry[0] = cursor;
                schedule_entry[1] = entry_end;
                schedule_entry[2] = qi_2;
                schedule_entry[3] = nq_2;
                reinterpret_cast<int4*>(Metadata + dst_2)[0] = reinterpret_cast<int4*>(schedule_entry)[0];
                cursor = entry_end;
            }
            nentries += total_9;
            __syncthreads();
            unsigned int lane_sum_10 = nqentries2;
            unsigned int _shfl_up_20 = __shfl_up_sync(0xFFFFFFFF, lane_sum_10, 1, 32);
            unsigned int synced_11 = _shfl_up_20;
            if (lane >= 1) {
                lane_sum_10 += synced_11;
            }
            unsigned int _shfl_up_21 = __shfl_up_sync(0xFFFFFFFF, lane_sum_10, 2, 32);
            unsigned int synced_12 = _shfl_up_21;
            if (lane >= 2) {
                lane_sum_10 += synced_12;
            }
            unsigned int _shfl_up_22 = __shfl_up_sync(0xFFFFFFFF, lane_sum_10, 4, 32);
            unsigned int synced_13 = _shfl_up_22;
            if (lane >= 4) {
                lane_sum_10 += synced_13;
            }
            unsigned int _shfl_up_23 = __shfl_up_sync(0xFFFFFFFF, lane_sum_10, 8, 32);
            unsigned int synced_14 = _shfl_up_23;
            if (lane >= 8) {
                lane_sum_10 += synced_14;
            }
            unsigned int _shfl_up_24 = __shfl_up_sync(0xFFFFFFFF, lane_sum_10, 16, 32);
            unsigned int synced_15 = _shfl_up_24;
            if (lane >= 16) {
                lane_sum_10 += synced_15;
            }
            if (lane == 31) {
                warp_sums[warp] = lane_sum_10;
            }
            __syncthreads();
            unsigned int warp_total_16 = 0;
            if (lane < 8) {
                warp_total_16 = warp_sums[lane];
            }
            unsigned int warp_sum_17 = warp_total_16;
            unsigned int _shfl_up_25 = __shfl_up_sync(0xFFFFFFFF, warp_sum_17, 1, 32);
            unsigned int synced_18 = _shfl_up_25;
            if (lane >= 1) {
                warp_sum_17 += synced_18;
            }
            unsigned int _shfl_up_26 = __shfl_up_sync(0xFFFFFFFF, warp_sum_17, 2, 32);
            unsigned int synced_19 = _shfl_up_26;
            if (lane >= 2) {
                warp_sum_17 += synced_19;
            }
            unsigned int _shfl_up_27 = __shfl_up_sync(0xFFFFFFFF, warp_sum_17, 4, 32);
            unsigned int synced_20 = _shfl_up_27;
            if (lane >= 4) {
                warp_sum_17 += synced_20;
            }
            unsigned int _shfl_up_28 = __shfl_up_sync(0xFFFFFFFF, warp_sum_17, 8, 32);
            unsigned int synced_21 = _shfl_up_28;
            if (lane >= 8) {
                warp_sum_17 += synced_21;
            }
            unsigned int _shfl_up_29 = __shfl_up_sync(0xFFFFFFFF, warp_sum_17, 16, 32);
            unsigned int synced_22 = _shfl_up_29;
            if (lane >= 16) {
                warp_sum_17 += synced_22;
            }
            unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, warp_sum_17, 7);
            unsigned int total_23 = _shfl_6;
            unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, warp_sum_17 - warp_total_16, warp);
            unsigned int preceding_24 = _shfl_7;
            unsigned int per_entry_25 = ((nqentries2 != 0) ? qcount2 / nqentries2 : (unsigned int)0);
            unsigned int larger_26 = ((nqentries2 != 0) ? qcount2 % nqentries2 : (unsigned int)0);
            unsigned int cursor_27 = qsplit2;
            #pragma unroll 1
            for (unsigned int entry_1 = 0; entry_1 < nqentries2; entry_1++) {
                unsigned int entry_end_1 = cursor_27 + per_entry_25 + (unsigned int)(((larger_26 > entry_1) ? 1 : 0));
                unsigned int dst_3 = sched + (nentries + (lane_sum_10 - nqentries2 + preceding_24) + entry_1) * 4;
                unsigned int schedule_entry_1[4];
                schedule_entry_1[0] = cursor_27;
                schedule_entry_1[1] = entry_end_1;
                schedule_entry_1[2] = qj;
                schedule_entry_1[3] = nq2;
                reinterpret_cast<int4*>(Metadata + dst_3)[0] = reinterpret_cast<int4*>(schedule_entry_1)[0];
                cursor_27 = entry_end_1;
            }
            nentries += total_23;
            __syncthreads();
        }
        int _max_1 = ((1) > ((nentries + (unsigned int)NUM_SMS - 1) / (unsigned int)NUM_SMS) ? (1) : ((nentries + (unsigned int)NUM_SMS - 1) / (unsigned int)NUM_SMS));
        unsigned int waves = _max_1;
        #pragma unroll 1
        for (unsigned int entry_2 = nentries + (unsigned int)tid; entry_2 < waves * (unsigned int)NUM_SMS; entry_2 += 256) {
            unsigned int empty_entry[4];
            #pragma unroll
            for (int j_2 = 0; j_2 < 4; j_2++) {
                empty_entry[j_2] = 0;
            }
            reinterpret_cast<int4*>(Metadata + (sched + entry_2 * 4))[0] = reinterpret_cast<int4*>(empty_entry)[0];
        }
        __syncthreads();
        if (waves != 1) {
            unsigned int entry_regs[20];
            unsigned int ranks[5];
            #pragma unroll 1
            for (unsigned int wave = warp; wave < waves; wave += 8) {
                #pragma unroll
                for (int ei = 0; ei < 5; ei++) {
                    unsigned int sm = lane + (unsigned int)(ei * 32);
                    if (sm < (unsigned int)NUM_SMS) {
                        unsigned int _vec_load_2[4];
                        {
                            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(Metadata + sched + (wave * (unsigned int)NUM_SMS + sm) * 4);
                            _vec_load_2[0 + 0] = _uv4_0.x;
                            _vec_load_2[0 + 1] = _uv4_0.y;
                            _vec_load_2[0 + 2] = _uv4_0.z;
                            _vec_load_2[0 + 3] = _uv4_0.w;
                        }
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 4; j_3++) {
                            entry_regs[ei * 4 + j_3] = _vec_load_2[j_3];
                        }
                    }
                }
                #pragma unroll 1
                for (unsigned int count = lane; count < 9; count += 32) {
                    hist[warp * 9 + count] = 0;
                }
                __syncwarp();
                #pragma unroll
                for (int ei_1 = 0; ei_1 < 5; ei_1++) {
                    if (lane + (unsigned int)(ei_1 * 32) < (unsigned int)NUM_SMS) {
                        unsigned int count_1 = entry_regs[ei_1 * 4 + 1] - entry_regs[ei_1 * 4];
                        uint32_t _shared_atomic_old_0;
                        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(_shared_atomic_old_0) : "r"(static_cast<uint32_t>((hist_addr + 4 * (warp * 9 + count_1)))), "r"(static_cast<uint32_t>(1)) : "memory");
                        ranks[ei_1] = _shared_atomic_old_0;
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    unsigned int rank_begin = 0;
                    #pragma unroll
                    for (int count_2 = 0; count_2 < 9; count_2++) {
                        unsigned int freq = hist[warp * 9 + (unsigned int)count_2];
                        hist[warp * 9 + (unsigned int)count_2] = rank_begin;
                        rank_begin += freq;
                    }
                }
                __syncwarp();
                #pragma unroll
                for (int ei_2 = 0; ei_2 < 5; ei_2++) {
                    if (lane + (unsigned int)(ei_2 * 32) < (unsigned int)NUM_SMS) {
                        unsigned int count_3 = entry_regs[ei_2 * 4 + 1] - entry_regs[ei_2 * 4];
                        unsigned int rank = hist[warp * 9 + count_3] + ranks[ei_2];
                        unsigned int dst_sm = ((wave % 2 == 0) ? rank : (unsigned int)(NUM_SMS - 1) - rank);
                        unsigned int entry_store[4];
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 4; j_4++) {
                            entry_store[j_4] = entry_regs[ei_2 * 4 + j_4];
                        }
                        reinterpret_cast<int4*>(Metadata + (sched + (wave * (unsigned int)NUM_SMS + dst_sm) * 4))[0] = reinterpret_cast<int4*>(entry_store)[0];
                    }
                }
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
