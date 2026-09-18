/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_RECIPROCAL_MAGICS_OFF 0
#define SMEM_RECIPROCAL_MAGICS_STAGE_BYTES 8
#define SMEM_RECIPROCAL_MAGICS_STRIDE 8
#define SMEM_TOTAL 128
#define THREADS 512

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_cake_ulysses_a2a_b4d01efa1606195e00e8(__nv_bfloat16* __restrict__ inp, CakePeerPointerTable<__nv_bfloat16> staging_peers, CakePeerPointerTable<unsigned int> signals_signals, unsigned int* __restrict__ signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    unsigned int* reciprocal_magics = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int reciprocal_magics_addr = smem + 0;

    // === Task calls (dependency order) ===
    unsigned int sequences = (unsigned int)S_local;
    unsigned int local_heads = (unsigned int)H_local;
    unsigned int head_dim = (unsigned int)D;
    unsigned int block_len = local_heads * head_dim;
    unsigned int units_per_row = block_len / 8;
    if (tid == 0) {
        if (units_per_row == 160 && sequences == 8190) {
            reciprocal_magics[0] = (unsigned int)26843545;
            reciprocal_magics[1] = (unsigned int)524416;
        } else {
            if (block_len % 8 == 0) {
                reciprocal_magics[0] = (unsigned int)4294967295 / units_per_row;
            }
            reciprocal_magics[1] = (unsigned int)4294967295 / sequences;
        }
    }
    // block_signal_sync phase=entry parity=counter_lsb layout=self[36][8]+peer[2][36][8]
    {
        const int __signal_world = 4;
        const int __signal_rank = pg_rank;
        const int __signal_bid = bid;
        const int __signal_grid = num_bids;
        (void)__signal_grid;  // validated against max_blocks by the launch preflight
        if ((int)threadIdx.x < __signal_world) {
            unsigned* __signal_self = signals_self;
            const int __signal_self_idx = __signal_bid * 8 + (int)threadIdx.x;
            const unsigned __signal_val = __signal_self[__signal_self_idx] + 1u;
            __signal_self[__signal_self_idx] = __signal_val;
            const unsigned __signal_parity = __signal_val & 1u;
            const int __signal_peer_idx = 288 + ((int)(__signal_parity * 36u) + __signal_bid) * 8 + __signal_rank;
            const int __signal_local_idx = 288 + ((int)(__signal_parity * 36u) + __signal_bid) * 8 + (int)threadIdx.x;
            unsigned* __signal_peer = (signals_signals).ptrs[(int)threadIdx.x] + __signal_peer_idx;
            unsigned* __signal_local = __signal_self + __signal_local_idx;
            asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(__signal_val), "l"(__signal_peer) : "memory");
            unsigned __signal_seen;
            do {
                asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(__signal_seen) : "l"(__signal_local) : "memory");
            } while (__signal_seen != __signal_val);
        }
        __syncthreads();
    }
    unsigned int batches = (unsigned int)B;
    unsigned int rank = (unsigned int)pg_rank;
    unsigned int grid = (unsigned int)num_bids;
    unsigned int block = (unsigned int)bid;
    unsigned int num_rows = batches * 4 * sequences;
    unsigned int rows_per_block = (unsigned int)1;
    if (grid == 36) {
        rows_per_block = (num_rows + 36 - 1) / 36;
    }
    unsigned int row_lo = block * rows_per_block;
    unsigned int row_hi = row_lo + rows_per_block;
    if (row_hi > num_rows) {
        row_hi = num_rows;
    }
    if (row_lo < row_hi) {
        int vector_aligned = (int)(block_len % 8 == 0 && (unsigned long long)inp % 16 == 0);
        if (vector_aligned != 0) {
            unsigned int units_magic = reciprocal_magics[0];
            unsigned int sequences_magic = reciprocal_magics[1];
            int w4_bf16_performance_shape = (int)(batches == 1 && sequences == 8190 && units_per_row == 160 && grid == 36);
            unsigned int w4_block_peer = block / 9;
            unsigned int w4_peer = w4_block_peer % 4;
            unsigned int w4_block_in_peer = block - w4_block_peer * 9;
            unsigned int total_units = (row_hi - row_lo) * units_per_row;
            #pragma unroll 4
            for (unsigned int unit_index = (unsigned int)tid; unit_index < total_units; unit_index += 512) {
                unsigned int local_row = (unsigned int)0;
                unsigned int unit_in_row = (unsigned int)0;
                unsigned int row = (unsigned int)0;
                unsigned int row_quotient = (unsigned int)0;
                unsigned int sequence = (unsigned int)0;
                unsigned int peer = (unsigned int)0;
                if (w4_bf16_performance_shape != 0) {
                    local_row = unit_index / 160;
                    unit_in_row = unit_index - local_row * 160;
                    row = row_lo + local_row;
                    row_quotient = w4_block_peer;
                    peer = w4_peer;
                    sequence = w4_block_in_peer * 910 + local_row;
                } else {
                    unsigned int local_row_guess = (unsigned int)((unsigned long long)unit_index * (unsigned long long)units_magic >> 32);
                    unsigned int local_row_raw_remainder = unit_index - local_row_guess * units_per_row;
                    unsigned int local_row_correction = (unsigned int)(local_row_raw_remainder >= units_per_row);
                    local_row = local_row_guess + local_row_correction;
                    unit_in_row = local_row_raw_remainder - ((local_row_correction != 0) ? units_per_row : (unsigned int)0);
                    row = row_lo + local_row;
                    unsigned int row_quotient_guess = (unsigned int)((unsigned long long)row * (unsigned long long)sequences_magic >> 32);
                    unsigned int sequence_raw_remainder = row - row_quotient_guess * sequences;
                    unsigned int sequence_correction = (unsigned int)(sequence_raw_remainder >= sequences);
                    row_quotient = row_quotient_guess + sequence_correction;
                    sequence = sequence_raw_remainder - ((sequence_correction != 0) ? sequences : (unsigned int)0);
                    peer = row_quotient % 4;
                }
                unsigned int slab_offset = row * block_len + unit_in_row * 8;
                unsigned int peer_span_offset = peer * sequences * block_len;
                unsigned int src_offset = slab_offset;
                unsigned int dst_offset = slab_offset + (sequence * 3 + rank) * block_len - peer_span_offset;
                *reinterpret_cast<uint4*>((staging_peers.ptrs[peer] + dst_offset)) = *reinterpret_cast<const uint4*>((inp + src_offset));
            }
        } else {
            unsigned int sequences_magic_1 = reciprocal_magics[1] + 1;
            #pragma unroll 1
            for (unsigned int row_1 = row_lo; row_1 < row_hi; row_1++) {
                unsigned int row_quotient_guess_1 = (unsigned int)((unsigned long long)row_1 * (unsigned long long)sequences_magic_1 >> 32);
                unsigned int row_quotient_1 = ((sequences == 1) ? row_1 : row_quotient_guess_1 - (unsigned int)(((row_quotient_guess_1 * sequences > row_1) ? 1 : 0)));
                unsigned int sequence_1 = row_1 - row_quotient_1 * sequences;
                unsigned int peer_1 = row_quotient_1 % 4;
                unsigned int batch = row_quotient_1 / 4;
                unsigned int src_offset_1 = (batch * (sequences * 4) + peer_1 * sequences + sequence_1) * block_len;
                unsigned int dst_offset_1 = ((batch * sequences + sequence_1) * (local_heads * 4) + rank * local_heads) * head_dim;
                #pragma unroll 1
                for (unsigned int element = (unsigned int)tid; element < block_len; element += 512) {
                    staging_peers.ptrs[peer_1][dst_offset_1 + element] = inp[src_offset_1 + element];
                }
            }
        }
    }
    // block_signal_sync phase=exit parity=counter_lsb layout=self[36][8]+peer[2][36][8]
    {
        const int __signal_world = 4;
        const int __signal_rank = pg_rank;
        const int __signal_bid = bid;
        const int __signal_grid = num_bids;
        (void)__signal_grid;  // validated against max_blocks by the launch preflight
        __syncthreads();
        if ((int)threadIdx.x < __signal_world) {
            unsigned* __signal_self = signals_self;
            const int __signal_self_idx = __signal_bid * 8 + (int)threadIdx.x;
            const unsigned __signal_val = __signal_self[__signal_self_idx] + 1u;
            __signal_self[__signal_self_idx] = __signal_val;
            const unsigned __signal_parity = __signal_val & 1u;
            const int __signal_peer_idx = 288 + ((int)(__signal_parity * 36u) + __signal_bid) * 8 + __signal_rank;
            const int __signal_local_idx = 288 + ((int)(__signal_parity * 36u) + __signal_bid) * 8 + (int)threadIdx.x;
            unsigned* __signal_peer = (signals_signals).ptrs[(int)threadIdx.x] + __signal_peer_idx;
            unsigned* __signal_local = __signal_self + __signal_local_idx;
            asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(__signal_val), "l"(__signal_peer) : "memory");
            unsigned __signal_seen;
            do {
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(__signal_seen) : "l"(__signal_local) : "memory");
            } while (__signal_seen != __signal_val);
        }
        __syncthreads();
    }
}

} // extern "C"
