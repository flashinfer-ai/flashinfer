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

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_trtllm_mla_blackwell_70c928af75813dc1cb03(int* __restrict__ seq_lens, int* __restrict__ work_order, int q_len, int total_rows)
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

    // Prologue
    unsigned int* keys = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int keys_addr = smem + 0;
    unsigned int* row_lens = reinterpret_cast<unsigned int*>(smem_raw + 4096);
    const int row_lens_addr = smem + 4096;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int tid_1 = tid;
    int padded = (total_rows + 128 - 1) / 128 * 128;
    int reqs[2];
    int offs[2];
    #pragma unroll
    for (int s = 0; s < 2; s++) {
        int div_row = tid_1 + s * 512;
        int b = (int)((unsigned int)div_row / (unsigned int)q_len);
        offs[s] = div_row - b * q_len;
        if (div_row >= total_rows) {
            b = 0;
        }
        reqs[s] = b;
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int lens[2];
    #pragma unroll
    for (int s_1 = 0; s_1 < 2; s_1++) {
        lens[s_1] = seq_lens[reqs[s_1]] - q_len + offs[s_1] + 1;
    }
    #pragma unroll
    for (int s_2 = 0; s_2 < 2; s_2++) {
        int key_row = tid_1 + s_2 * 512;
        if (key_row < padded) {
            unsigned int k = 4294967295u;
            int kv_len = 0;
            if (key_row < total_rows) {
                int _max_0 = ((lens[s_2]) > (0) ? (lens[s_2]) : (0));
                kv_len = _max_0;
                int tiles = kv_len >> 7;
                if ((kv_len & 127) != 0) {
                    tiles = tiles + 1;
                }
                if (tiles > 4194303) {
                    tiles = 4194303;
                }
                k = (unsigned int)(4194303 - tiles) << 10 | (unsigned int)key_row;
            }
            keys[key_row] = k;
            row_lens[key_row] = (unsigned int)kv_len;
        }
    }
    __syncthreads();
    int row0 = blockIdx.x * 64 + warp * 4;
    if (row0 < total_rows) {
        int lane_1 = lane;
        unsigned int mine[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mine[0])), "=r"(*reinterpret_cast<uint32_t*>(&mine[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mine[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mine[(0) + 3]))
            : "r"(keys_addr + (unsigned int)(row0 * 4)) : "memory");
        unsigned int mine_lens[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mine_lens[0])), "=r"(*reinterpret_cast<uint32_t*>(&mine_lens[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mine_lens[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mine_lens[(0) + 3]))
            : "r"(row_lens_addr + (unsigned int)(row0 * 4)) : "memory");
        unsigned int below[4] = {0};
        unsigned int quad[4];
        #pragma unroll 2
        for (int j = 0; j < padded; j += 128) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                : "r"(keys_addr + (unsigned int)((j + lane_1 * 4) * 4)) : "memory");
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                #pragma unroll
                for (int m = 0; m < 4; m++) {
                    if (quad[q] < mine[m]) {
                        below[m] = below[m] + 1;
                    }
                }
            }
        }
        #pragma unroll
        for (int m_1 = 0; m_1 < 4; m_1++) {
            unsigned int rank;
            asm volatile("redux.sync.add.u32 %0, %1, 0xFFFFFFFF;" : "=r"(rank) : "r"(below[m_1]));
            if (lane_1 == m_1) {
                if (row0 + m_1 < total_rows) {
                    work_order[(int)rank] = row0 + m_1;
                    work_order[1024 + (int)rank] = (int)mine_lens[m_1];
                }
            }
        }
    }
}

} // extern "C"
