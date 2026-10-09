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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_dsa_indexer_topk_device_common.cuh"
#include <cub/block/block_radix_sort.cuh>

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SCRATCH_OFF 0
#define SMEM_SCRATCH_STAGE_BYTES 40960
#define SMEM_SCRATCH_STRIDE 40960
#define SMEM_TOTAL 40960
#define THREADS 512

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsa_indexer_topk_4c792ffa06c1b93d77fa(int* __restrict__ Indices, float* __restrict__ Scores, int top_k, int key_bits)
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
    unsigned int* scratch = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SCRATCH_OFF);
    const int scratch_addr = smem + SMEM_SCRATCH_OFF;

    // === Task calls (dependency order) ===
    long long row_base = (long long)bid * (long long)top_k;
    unsigned int keys[8];
    unsigned int vals[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int pos = tid * 8 + i;
        keys[i] = 0;
        vals[i] = 4286578688;
        if (pos < top_k) {
            long long slot = row_base + (long long)pos;
            int kid = Indices[slot];
            float score = Scores[slot];
            unsigned int score_bits = 0;
            score_bits = reinterpret_cast<unsigned int*>(&score)[0];
            keys[i] = ~(unsigned int)kid;
            vals[i] = score_bits;
        }
    }
    {
        constexpr int _block_radix_sort_0_items = static_cast<int>(sizeof(keys) / sizeof(uint32_t));
        static_assert(sizeof(keys) == sizeof(uint32_t) * _block_radix_sort_0_items && sizeof(vals) == sizeof(keys), "BlockRadixSortDescending key and value arrays must hold the same whole number of uint32_t items");
        using _block_radix_sort_0 = cub::BlockRadixSort<uint32_t, THREADS, _block_radix_sort_0_items, uint32_t, 4, true, cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte>;
        static_assert(sizeof(typename _block_radix_sort_0::TempStorage) <= SMEM_SCRATCH_STAGE_BYTES, "BlockRadixSortDescending scratch is too small for this CUB build");
        static_assert(1024 % alignof(typename _block_radix_sort_0::TempStorage) == 0 && 0 % alignof(typename _block_radix_sort_0::TempStorage) == 0, "BlockRadixSortDescending scratch is misaligned");
        _block_radix_sort_0(*reinterpret_cast<typename _block_radix_sort_0::TempStorage*>(__cvta_shared_to_generic(scratch_addr))).SortDescending(keys, vals, 0, key_bits);
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        int pos_1 = tid * 8 + i_1;
        if (pos_1 < top_k) {
            long long slot_1 = row_base + (long long)pos_1;
            unsigned int out_bits = vals[i_1];
            float out_score = 0.0f;
            out_score = reinterpret_cast<float*>(&out_bits)[0];
            Indices[slot_1] = (int)~keys[i_1];
            Scores[slot_1] = out_score;
        }
    }
}

} // extern "C"
