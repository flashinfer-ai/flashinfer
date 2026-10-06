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
#define THREADS 32

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(32) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_d76afbe143434c6757bd(uint8_t* __restrict__ SFB, int* __restrict__ route_map, int* __restrict__ tile_mn_limit, int* __restrict__ total_tiles, uint8_t* __restrict__ SFBS, int K, int K_tiles, int grid_n)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int n_tile = blockIdx.x;
    if (n_tile < total_tiles[0] && n_tile < grid_n) {
        int valid_rows = tile_mn_limit[n_tile] - n_tile * 128;
        int lane_1 = tid;
        int sf_stride = K / 16;
        int src_row[4] = {0};
        if (valid_rows > lane_1) {
            src_row[0] = route_map[n_tile * 128 + lane_1] * sf_stride;
        }
        if (valid_rows > lane_1 + 32) {
            src_row[1] = route_map[n_tile * 128 + lane_1 + 32] * sf_stride;
        }
        if (valid_rows > lane_1 + 64) {
            src_row[2] = route_map[n_tile * 128 + lane_1 + 64] * sf_stride;
        }
        if (valid_rows > lane_1 + 96) {
            src_row[3] = route_map[n_tile * 128 + lane_1 + 96] * sf_stride;
        }
        int sf_out[32] = {0};
        int sf_words_a[32] = {0};
        int sf_words_b[32] = {0};
        if (valid_rows > lane_1) {
            {
                const int4* _ivptr_0 = reinterpret_cast<const int4*>(SFB + src_row[0] + 0);
                int4 _ivld_0;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_0.x), "=r"(_ivld_0.y), "=r"(_ivld_0.z), "=r"(_ivld_0.w) : "l"((const void*)(_ivptr_0)) : "memory");
                sf_words_a[0 + 0] = _ivld_0.x;
                sf_words_a[0 + 1] = _ivld_0.y;
                sf_words_a[0 + 2] = _ivld_0.z;
                sf_words_a[0 + 3] = _ivld_0.w;
            }
            {
                const int4* _ivptr_1 = reinterpret_cast<const int4*>(SFB + (src_row[0] + 16) + 0);
                int4 _ivld_1;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_1.x), "=r"(_ivld_1.y), "=r"(_ivld_1.z), "=r"(_ivld_1.w) : "l"((const void*)(_ivptr_1)) : "memory");
                sf_words_a[4 + 0] = _ivld_1.x;
                sf_words_a[4 + 1] = _ivld_1.y;
                sf_words_a[4 + 2] = _ivld_1.z;
                sf_words_a[4 + 3] = _ivld_1.w;
            }
        }
        if (valid_rows > lane_1 + 32) {
            {
                const int4* _ivptr_2 = reinterpret_cast<const int4*>(SFB + src_row[1] + 0);
                int4 _ivld_2;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_2.x), "=r"(_ivld_2.y), "=r"(_ivld_2.z), "=r"(_ivld_2.w) : "l"((const void*)(_ivptr_2)) : "memory");
                sf_words_a[8 + 0] = _ivld_2.x;
                sf_words_a[8 + 1] = _ivld_2.y;
                sf_words_a[8 + 2] = _ivld_2.z;
                sf_words_a[8 + 3] = _ivld_2.w;
            }
            {
                const int4* _ivptr_3 = reinterpret_cast<const int4*>(SFB + (src_row[1] + 16) + 0);
                int4 _ivld_3;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_3.x), "=r"(_ivld_3.y), "=r"(_ivld_3.z), "=r"(_ivld_3.w) : "l"((const void*)(_ivptr_3)) : "memory");
                sf_words_a[12 + 0] = _ivld_3.x;
                sf_words_a[12 + 1] = _ivld_3.y;
                sf_words_a[12 + 2] = _ivld_3.z;
                sf_words_a[12 + 3] = _ivld_3.w;
            }
        }
        if (valid_rows > lane_1 + 64) {
            {
                const int4* _ivptr_4 = reinterpret_cast<const int4*>(SFB + src_row[2] + 0);
                int4 _ivld_4;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_4.x), "=r"(_ivld_4.y), "=r"(_ivld_4.z), "=r"(_ivld_4.w) : "l"((const void*)(_ivptr_4)) : "memory");
                sf_words_a[16 + 0] = _ivld_4.x;
                sf_words_a[16 + 1] = _ivld_4.y;
                sf_words_a[16 + 2] = _ivld_4.z;
                sf_words_a[16 + 3] = _ivld_4.w;
            }
            {
                const int4* _ivptr_5 = reinterpret_cast<const int4*>(SFB + (src_row[2] + 16) + 0);
                int4 _ivld_5;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_5.x), "=r"(_ivld_5.y), "=r"(_ivld_5.z), "=r"(_ivld_5.w) : "l"((const void*)(_ivptr_5)) : "memory");
                sf_words_a[20 + 0] = _ivld_5.x;
                sf_words_a[20 + 1] = _ivld_5.y;
                sf_words_a[20 + 2] = _ivld_5.z;
                sf_words_a[20 + 3] = _ivld_5.w;
            }
        }
        if (valid_rows > lane_1 + 96) {
            {
                const int4* _ivptr_6 = reinterpret_cast<const int4*>(SFB + src_row[3] + 0);
                int4 _ivld_6;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_6.x), "=r"(_ivld_6.y), "=r"(_ivld_6.z), "=r"(_ivld_6.w) : "l"((const void*)(_ivptr_6)) : "memory");
                sf_words_a[24 + 0] = _ivld_6.x;
                sf_words_a[24 + 1] = _ivld_6.y;
                sf_words_a[24 + 2] = _ivld_6.z;
                sf_words_a[24 + 3] = _ivld_6.w;
            }
            {
                const int4* _ivptr_7 = reinterpret_cast<const int4*>(SFB + (src_row[3] + 16) + 0);
                int4 _ivld_7;
                asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_ivld_7.x), "=r"(_ivld_7.y), "=r"(_ivld_7.z), "=r"(_ivld_7.w) : "l"((const void*)(_ivptr_7)) : "memory");
                sf_words_a[28 + 0] = _ivld_7.x;
                sf_words_a[28 + 1] = _ivld_7.y;
                sf_words_a[28 + 2] = _ivld_7.z;
                sf_words_a[28 + 3] = _ivld_7.w;
            }
        }
        #pragma unroll 1
        for (int iter_k = 0; iter_k < K_tiles; iter_k += 2) {
            if (iter_k + 1 < K_tiles) {
                if (valid_rows > lane_1) {
                    {
                        const int4* _ivptr_8 = reinterpret_cast<const int4*>(SFB + (src_row[0] + (iter_k + 1) * 32) + 0);
                        int4 _ivld_8;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_8.x), "=r"(_ivld_8.y), "=r"(_ivld_8.z), "=r"(_ivld_8.w) : "l"((const void*)(_ivptr_8)) : "memory");
                        sf_words_b[0 + 0] = _ivld_8.x;
                        sf_words_b[0 + 1] = _ivld_8.y;
                        sf_words_b[0 + 2] = _ivld_8.z;
                        sf_words_b[0 + 3] = _ivld_8.w;
                    }
                    {
                        const int4* _ivptr_9 = reinterpret_cast<const int4*>(SFB + (src_row[0] + (iter_k + 1) * 32 + 16) + 0);
                        int4 _ivld_9;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_9.x), "=r"(_ivld_9.y), "=r"(_ivld_9.z), "=r"(_ivld_9.w) : "l"((const void*)(_ivptr_9)) : "memory");
                        sf_words_b[4 + 0] = _ivld_9.x;
                        sf_words_b[4 + 1] = _ivld_9.y;
                        sf_words_b[4 + 2] = _ivld_9.z;
                        sf_words_b[4 + 3] = _ivld_9.w;
                    }
                }
                if (valid_rows > lane_1 + 32) {
                    {
                        const int4* _ivptr_10 = reinterpret_cast<const int4*>(SFB + (src_row[1] + (iter_k + 1) * 32) + 0);
                        int4 _ivld_10;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_10.x), "=r"(_ivld_10.y), "=r"(_ivld_10.z), "=r"(_ivld_10.w) : "l"((const void*)(_ivptr_10)) : "memory");
                        sf_words_b[8 + 0] = _ivld_10.x;
                        sf_words_b[8 + 1] = _ivld_10.y;
                        sf_words_b[8 + 2] = _ivld_10.z;
                        sf_words_b[8 + 3] = _ivld_10.w;
                    }
                    {
                        const int4* _ivptr_11 = reinterpret_cast<const int4*>(SFB + (src_row[1] + (iter_k + 1) * 32 + 16) + 0);
                        int4 _ivld_11;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_11.x), "=r"(_ivld_11.y), "=r"(_ivld_11.z), "=r"(_ivld_11.w) : "l"((const void*)(_ivptr_11)) : "memory");
                        sf_words_b[12 + 0] = _ivld_11.x;
                        sf_words_b[12 + 1] = _ivld_11.y;
                        sf_words_b[12 + 2] = _ivld_11.z;
                        sf_words_b[12 + 3] = _ivld_11.w;
                    }
                }
                if (valid_rows > lane_1 + 64) {
                    {
                        const int4* _ivptr_12 = reinterpret_cast<const int4*>(SFB + (src_row[2] + (iter_k + 1) * 32) + 0);
                        int4 _ivld_12;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_12.x), "=r"(_ivld_12.y), "=r"(_ivld_12.z), "=r"(_ivld_12.w) : "l"((const void*)(_ivptr_12)) : "memory");
                        sf_words_b[16 + 0] = _ivld_12.x;
                        sf_words_b[16 + 1] = _ivld_12.y;
                        sf_words_b[16 + 2] = _ivld_12.z;
                        sf_words_b[16 + 3] = _ivld_12.w;
                    }
                    {
                        const int4* _ivptr_13 = reinterpret_cast<const int4*>(SFB + (src_row[2] + (iter_k + 1) * 32 + 16) + 0);
                        int4 _ivld_13;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_13.x), "=r"(_ivld_13.y), "=r"(_ivld_13.z), "=r"(_ivld_13.w) : "l"((const void*)(_ivptr_13)) : "memory");
                        sf_words_b[20 + 0] = _ivld_13.x;
                        sf_words_b[20 + 1] = _ivld_13.y;
                        sf_words_b[20 + 2] = _ivld_13.z;
                        sf_words_b[20 + 3] = _ivld_13.w;
                    }
                }
                if (valid_rows > lane_1 + 96) {
                    {
                        const int4* _ivptr_14 = reinterpret_cast<const int4*>(SFB + (src_row[3] + (iter_k + 1) * 32) + 0);
                        int4 _ivld_14;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_14.x), "=r"(_ivld_14.y), "=r"(_ivld_14.z), "=r"(_ivld_14.w) : "l"((const void*)(_ivptr_14)) : "memory");
                        sf_words_b[24 + 0] = _ivld_14.x;
                        sf_words_b[24 + 1] = _ivld_14.y;
                        sf_words_b[24 + 2] = _ivld_14.z;
                        sf_words_b[24 + 3] = _ivld_14.w;
                    }
                    {
                        const int4* _ivptr_15 = reinterpret_cast<const int4*>(SFB + (src_row[3] + (iter_k + 1) * 32 + 16) + 0);
                        int4 _ivld_15;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_15.x), "=r"(_ivld_15.y), "=r"(_ivld_15.z), "=r"(_ivld_15.w) : "l"((const void*)(_ivptr_15)) : "memory");
                        sf_words_b[28 + 0] = _ivld_15.x;
                        sf_words_b[28 + 1] = _ivld_15.y;
                        sf_words_b[28 + 2] = _ivld_15.z;
                        sf_words_b[28 + 3] = _ivld_15.w;
                    }
                }
            }
            sf_out[0] = sf_words_a[0];
            sf_out[1] = sf_words_a[8];
            sf_out[2] = sf_words_a[16];
            sf_out[3] = sf_words_a[24];
            sf_out[4] = sf_words_a[1];
            sf_out[5] = sf_words_a[9];
            sf_out[6] = sf_words_a[17];
            sf_out[7] = sf_words_a[25];
            sf_out[8] = sf_words_a[2];
            sf_out[9] = sf_words_a[10];
            sf_out[10] = sf_words_a[18];
            sf_out[11] = sf_words_a[26];
            sf_out[12] = sf_words_a[3];
            sf_out[13] = sf_words_a[11];
            sf_out[14] = sf_words_a[19];
            sf_out[15] = sf_words_a[27];
            sf_out[16] = sf_words_a[4];
            sf_out[17] = sf_words_a[12];
            sf_out[18] = sf_words_a[20];
            sf_out[19] = sf_words_a[28];
            sf_out[20] = sf_words_a[5];
            sf_out[21] = sf_words_a[13];
            sf_out[22] = sf_words_a[21];
            sf_out[23] = sf_words_a[29];
            sf_out[24] = sf_words_a[6];
            sf_out[25] = sf_words_a[14];
            sf_out[26] = sf_words_a[22];
            sf_out[27] = sf_words_a[30];
            sf_out[28] = sf_words_a[7];
            sf_out[29] = sf_words_a[15];
            sf_out[30] = sf_words_a[23];
            sf_out[31] = sf_words_a[31];
            int slab_a = (n_tile * K_tiles + iter_k) * 4096 + lane_1 * 16;
            {
                int4 _iv4 = make_int4(sf_out[0 + 0], sf_out[0 + 1], sf_out[0 + 2], sf_out[0 + 3]);
                *reinterpret_cast<int4*>(SFBS + slab_a + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[4 + 0], sf_out[4 + 1], sf_out[4 + 2], sf_out[4 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 512) + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[8 + 0], sf_out[8 + 1], sf_out[8 + 2], sf_out[8 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 1024) + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[12 + 0], sf_out[12 + 1], sf_out[12 + 2], sf_out[12 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 1536) + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[16 + 0], sf_out[16 + 1], sf_out[16 + 2], sf_out[16 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 2048) + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[20 + 0], sf_out[20 + 1], sf_out[20 + 2], sf_out[20 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 2560) + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[24 + 0], sf_out[24 + 1], sf_out[24 + 2], sf_out[24 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 3072) + 0) = _iv4;
            }
            {
                int4 _iv4 = make_int4(sf_out[28 + 0], sf_out[28 + 1], sf_out[28 + 2], sf_out[28 + 3]);
                *reinterpret_cast<int4*>(SFBS + (slab_a + 3584) + 0) = _iv4;
            }
            if (iter_k + 2 < K_tiles) {
                if (valid_rows > lane_1) {
                    {
                        const int4* _ivptr_16 = reinterpret_cast<const int4*>(SFB + (src_row[0] + (iter_k + 2) * 32) + 0);
                        int4 _ivld_16;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_16.x), "=r"(_ivld_16.y), "=r"(_ivld_16.z), "=r"(_ivld_16.w) : "l"((const void*)(_ivptr_16)) : "memory");
                        sf_words_a[0 + 0] = _ivld_16.x;
                        sf_words_a[0 + 1] = _ivld_16.y;
                        sf_words_a[0 + 2] = _ivld_16.z;
                        sf_words_a[0 + 3] = _ivld_16.w;
                    }
                    {
                        const int4* _ivptr_17 = reinterpret_cast<const int4*>(SFB + (src_row[0] + (iter_k + 2) * 32 + 16) + 0);
                        int4 _ivld_17;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_17.x), "=r"(_ivld_17.y), "=r"(_ivld_17.z), "=r"(_ivld_17.w) : "l"((const void*)(_ivptr_17)) : "memory");
                        sf_words_a[4 + 0] = _ivld_17.x;
                        sf_words_a[4 + 1] = _ivld_17.y;
                        sf_words_a[4 + 2] = _ivld_17.z;
                        sf_words_a[4 + 3] = _ivld_17.w;
                    }
                }
                if (valid_rows > lane_1 + 32) {
                    {
                        const int4* _ivptr_18 = reinterpret_cast<const int4*>(SFB + (src_row[1] + (iter_k + 2) * 32) + 0);
                        int4 _ivld_18;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_18.x), "=r"(_ivld_18.y), "=r"(_ivld_18.z), "=r"(_ivld_18.w) : "l"((const void*)(_ivptr_18)) : "memory");
                        sf_words_a[8 + 0] = _ivld_18.x;
                        sf_words_a[8 + 1] = _ivld_18.y;
                        sf_words_a[8 + 2] = _ivld_18.z;
                        sf_words_a[8 + 3] = _ivld_18.w;
                    }
                    {
                        const int4* _ivptr_19 = reinterpret_cast<const int4*>(SFB + (src_row[1] + (iter_k + 2) * 32 + 16) + 0);
                        int4 _ivld_19;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_19.x), "=r"(_ivld_19.y), "=r"(_ivld_19.z), "=r"(_ivld_19.w) : "l"((const void*)(_ivptr_19)) : "memory");
                        sf_words_a[12 + 0] = _ivld_19.x;
                        sf_words_a[12 + 1] = _ivld_19.y;
                        sf_words_a[12 + 2] = _ivld_19.z;
                        sf_words_a[12 + 3] = _ivld_19.w;
                    }
                }
                if (valid_rows > lane_1 + 64) {
                    {
                        const int4* _ivptr_20 = reinterpret_cast<const int4*>(SFB + (src_row[2] + (iter_k + 2) * 32) + 0);
                        int4 _ivld_20;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_20.x), "=r"(_ivld_20.y), "=r"(_ivld_20.z), "=r"(_ivld_20.w) : "l"((const void*)(_ivptr_20)) : "memory");
                        sf_words_a[16 + 0] = _ivld_20.x;
                        sf_words_a[16 + 1] = _ivld_20.y;
                        sf_words_a[16 + 2] = _ivld_20.z;
                        sf_words_a[16 + 3] = _ivld_20.w;
                    }
                    {
                        const int4* _ivptr_21 = reinterpret_cast<const int4*>(SFB + (src_row[2] + (iter_k + 2) * 32 + 16) + 0);
                        int4 _ivld_21;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_21.x), "=r"(_ivld_21.y), "=r"(_ivld_21.z), "=r"(_ivld_21.w) : "l"((const void*)(_ivptr_21)) : "memory");
                        sf_words_a[20 + 0] = _ivld_21.x;
                        sf_words_a[20 + 1] = _ivld_21.y;
                        sf_words_a[20 + 2] = _ivld_21.z;
                        sf_words_a[20 + 3] = _ivld_21.w;
                    }
                }
                if (valid_rows > lane_1 + 96) {
                    {
                        const int4* _ivptr_22 = reinterpret_cast<const int4*>(SFB + (src_row[3] + (iter_k + 2) * 32) + 0);
                        int4 _ivld_22;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_22.x), "=r"(_ivld_22.y), "=r"(_ivld_22.z), "=r"(_ivld_22.w) : "l"((const void*)(_ivptr_22)) : "memory");
                        sf_words_a[24 + 0] = _ivld_22.x;
                        sf_words_a[24 + 1] = _ivld_22.y;
                        sf_words_a[24 + 2] = _ivld_22.z;
                        sf_words_a[24 + 3] = _ivld_22.w;
                    }
                    {
                        const int4* _ivptr_23 = reinterpret_cast<const int4*>(SFB + (src_row[3] + (iter_k + 2) * 32 + 16) + 0);
                        int4 _ivld_23;
                        asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                            : "=r"(_ivld_23.x), "=r"(_ivld_23.y), "=r"(_ivld_23.z), "=r"(_ivld_23.w) : "l"((const void*)(_ivptr_23)) : "memory");
                        sf_words_a[28 + 0] = _ivld_23.x;
                        sf_words_a[28 + 1] = _ivld_23.y;
                        sf_words_a[28 + 2] = _ivld_23.z;
                        sf_words_a[28 + 3] = _ivld_23.w;
                    }
                }
            }
            if (iter_k + 1 < K_tiles) {
                sf_out[0] = sf_words_b[0];
                sf_out[1] = sf_words_b[8];
                sf_out[2] = sf_words_b[16];
                sf_out[3] = sf_words_b[24];
                sf_out[4] = sf_words_b[1];
                sf_out[5] = sf_words_b[9];
                sf_out[6] = sf_words_b[17];
                sf_out[7] = sf_words_b[25];
                sf_out[8] = sf_words_b[2];
                sf_out[9] = sf_words_b[10];
                sf_out[10] = sf_words_b[18];
                sf_out[11] = sf_words_b[26];
                sf_out[12] = sf_words_b[3];
                sf_out[13] = sf_words_b[11];
                sf_out[14] = sf_words_b[19];
                sf_out[15] = sf_words_b[27];
                sf_out[16] = sf_words_b[4];
                sf_out[17] = sf_words_b[12];
                sf_out[18] = sf_words_b[20];
                sf_out[19] = sf_words_b[28];
                sf_out[20] = sf_words_b[5];
                sf_out[21] = sf_words_b[13];
                sf_out[22] = sf_words_b[21];
                sf_out[23] = sf_words_b[29];
                sf_out[24] = sf_words_b[6];
                sf_out[25] = sf_words_b[14];
                sf_out[26] = sf_words_b[22];
                sf_out[27] = sf_words_b[30];
                sf_out[28] = sf_words_b[7];
                sf_out[29] = sf_words_b[15];
                sf_out[30] = sf_words_b[23];
                sf_out[31] = sf_words_b[31];
                int slab_b = (n_tile * K_tiles + iter_k + 1) * 4096 + lane_1 * 16;
                {
                    int4 _iv4 = make_int4(sf_out[0 + 0], sf_out[0 + 1], sf_out[0 + 2], sf_out[0 + 3]);
                    *reinterpret_cast<int4*>(SFBS + slab_b + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[4 + 0], sf_out[4 + 1], sf_out[4 + 2], sf_out[4 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 512) + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[8 + 0], sf_out[8 + 1], sf_out[8 + 2], sf_out[8 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 1024) + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[12 + 0], sf_out[12 + 1], sf_out[12 + 2], sf_out[12 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 1536) + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[16 + 0], sf_out[16 + 1], sf_out[16 + 2], sf_out[16 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 2048) + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[20 + 0], sf_out[20 + 1], sf_out[20 + 2], sf_out[20 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 2560) + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[24 + 0], sf_out[24 + 1], sf_out[24 + 2], sf_out[24 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 3072) + 0) = _iv4;
                }
                {
                    int4 _iv4 = make_int4(sf_out[28 + 0], sf_out[28 + 1], sf_out[28 + 2], sf_out[28 + 3]);
                    *reinterpret_cast<int4*>(SFBS + (slab_b + 3584) + 0) = _iv4;
                }
            }
        }
    }
}

} // extern "C"
