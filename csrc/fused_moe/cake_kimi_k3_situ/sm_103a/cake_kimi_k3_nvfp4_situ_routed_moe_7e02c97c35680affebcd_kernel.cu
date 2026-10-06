/*
 * Copyright (c) 2023 by FlashInfer team.
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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_STAGING_OFF 0
#define SMEM_STAGING_STAGE_BYTES 8192
#define SMEM_STAGING_STRIDE 8192
#define SMEM_TOTAL 8192
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_7e02c97c35680affebcd(uint8_t* __restrict__ SFB, int* __restrict__ route_map, int* __restrict__ tile_mn_limit, int* __restrict__ total_tiles, uint8_t* __restrict__ SFBS, int K, int K_tiles, int grid_n)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    int* staging = reinterpret_cast<int*>(smem_raw + 0);
    const int staging_addr = smem + 0;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int n_tile = blockIdx.x;
    if (n_tile < total_tiles[0] && n_tile < grid_n) {
        int valid_rows = tile_mn_limit[n_tile] - n_tile * 128;
        int row = tid;
        int sf_stride = K / 16;
        int routed = 0;
        if (row < valid_rows) {
            routed = route_map[n_tile * 128 + row];
        }
        int src_row = routed * sf_stride;
        int dst_word = row % 32 * 4 + row / 32;
        int sf_words[8] = {0};
        #pragma unroll 1
        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
            int buf = iter_k % 2 * 1024;
            if (row < valid_rows) {
                {
                    const int4* _ivptr_0 = reinterpret_cast<const int4*>(SFB + (src_row + iter_k * 32) + 0);
                    int4 _ivld_0;
                    _ivld_0 = *_ivptr_0;
                    sf_words[0 + 0] = _ivld_0.x;
                    sf_words[0 + 1] = _ivld_0.y;
                    sf_words[0 + 2] = _ivld_0.z;
                    sf_words[0 + 3] = _ivld_0.w;
                }
                {
                    const int4* _ivptr_1 = reinterpret_cast<const int4*>(SFB + (src_row + iter_k * 32 + 16) + 0);
                    int4 _ivld_1;
                    _ivld_1 = *_ivptr_1;
                    sf_words[4 + 0] = _ivld_1.x;
                    sf_words[4 + 1] = _ivld_1.y;
                    sf_words[4 + 2] = _ivld_1.z;
                    sf_words[4 + 3] = _ivld_1.w;
                }
            }
            for (int j = 0; j < 8; j++) {
                staging[buf + j * 128 + dst_word] = sf_words[j];
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            int _staging_reg_0[8];
            {
                const int* _smem_ptr = reinterpret_cast<const int*>(staging);
                #pragma unroll
                for (int _lr = 0; _lr < 8; _lr++)
                    _staging_reg_0[_lr] = _smem_ptr[(buf + row * 8) + _lr];
            }
            int slab = (n_tile * K_tiles + iter_k) * 4096 + row * 32;
            {
                int4 _iv4 = make_int4(_staging_reg_0[0 + 0], _staging_reg_0[0 + 1], _staging_reg_0[0 + 2], _staging_reg_0[0 + 3]);
                *reinterpret_cast<int4*>(SFBS + slab + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(_staging_reg_0[4 + 0], _staging_reg_0[4 + 1], _staging_reg_0[4 + 2], _staging_reg_0[4 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab + 16) + 0) = _iv4;
            }
        }
    }
}

} // extern "C"
