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
template <typename T, int Capacity = 8>
struct __align__(16) CakePeerPointerTable { T* ptrs[Capacity]; };
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
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_ulysses_a2a_e56fbba5639d18797849(__half* __restrict__ inp, __half* __restrict__ out, CakePeerPointerTable<__half> staging_peers, CakePeerPointerTable<unsigned int> signals_peers, unsigned int* signals_self, int32_t pg_rank, int B, int S_local, int H_local, int D)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    unsigned int rank = (unsigned int)pg_rank;
    unsigned int tid_0 = (unsigned int)tid;
    unsigned int bid_1 = (unsigned int)bid;
    unsigned int warp_1 = tid_0 / 32;
    unsigned int sequences = (unsigned int)S_local;
    unsigned int block_len = (unsigned int)H_local * (unsigned int)D;
    unsigned int rows = (unsigned int)B * sequences;
    uint32_t _relaxed_ld_0;
    asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(_relaxed_ld_0) : "l"(signals_self) : "memory");
    unsigned int epoch = _relaxed_ld_0 + 1;
    int vector = (int)(block_len % 8 == 0 && (unsigned long long)inp % 16 == 0 && (unsigned long long)out % 16 == 0);
    if ((unsigned long long)staging_peers.ptrs[0] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[1] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[2] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[3] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[4] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[5] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[6] % 16 != 0) {
        vector = 0;
    }
    if ((unsigned long long)staging_peers.ptrs[7] % 16 != 0) {
        vector = 0;
    }
    unsigned int block_units = block_len;
    if (vector != 0) {
        block_units = block_len / 8;
    }
    unsigned int units = rows * block_units;
    unsigned int pitch_units = block_units * 8;
    unsigned int units_magic = (unsigned int)4294967295 / block_units;
    unsigned int sequences_magic = (unsigned int)4294967295 / sequences;
    unsigned int part_quotient = units / 96;
    unsigned int part_remainder = units % 96;
    unsigned int* inp_words = reinterpret_cast<unsigned int*>(inp);
    unsigned int* out_words = reinterpret_cast<unsigned int*>(out);
    if (bid_1 < 96) {
        unsigned int lo = bid_1 * part_quotient + bid_1 * part_remainder / 96;
        unsigned int hi = (bid_1 + 1) * part_quotient + (bid_1 + 1) * part_remainder / 96;
        if (bid_1 == 0) {
            if (tid_0 < 8) {
                if (tid_0 != rank) {
                    asm volatile("st.relaxed.sys.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(signals_peers.ptrs[tid_0]) + (8 + rank))), "r"(static_cast<unsigned int>(epoch)) : "memory");
                }
            }
        }
        int entered = 0;
        #pragma unroll 1
        for (unsigned int k = 0; k < 8; k++) {
            unsigned int destination = (rank + 1 + bid_1 + k) % 8;
            if (destination == rank) {
                if (warp_1 < 15) {
                    if (vector != 0) {
                        unsigned int last = hi - 1;
                        unsigned int base = lo + tid_0;
                        while (base < hi) {
                            unsigned int unit = base;
                            unsigned int clamped = ((unit < hi) ? unit : last);
                            unsigned int quotient = (unsigned int)((unsigned long long)clamped * (unsigned long long)units_magic >> 32);
                            unsigned int remainder = clamped - quotient * block_units;
                            unsigned int correction = (unsigned int)(remainder >= block_units);
                            quotient = quotient + correction;
                            remainder = remainder - correction * block_units;
                            unsigned int quotient_0 = (unsigned int)((unsigned long long)quotient * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_1 = quotient - quotient_0 * sequences;
                            unsigned int correction_2 = (unsigned int)(remainder_1 >= sequences);
                            quotient_0 = quotient_0 + correction_2;
                            remainder_1 = remainder_1 - correction_2 * sequences;
                            unsigned int src = (quotient + (quotient_0 * 7 + destination) * sequences) * block_units + remainder;
                            unsigned int dst = quotient * pitch_units + rank * block_units + remainder;
                            unsigned int _vec_load_0[4];
                            {
                                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(inp_words + (src * 4) + 0);
                                _vec_load_0[0 + 0] = _uv4_0.x;
                                _vec_load_0[0 + 1] = _uv4_0.y;
                                _vec_load_0[0 + 2] = _uv4_0.z;
                                _vec_load_0[0 + 3] = _uv4_0.w;
                            }
                            unsigned int unit_3 = base + 480;
                            unsigned int clamped_4 = ((unit_3 < hi) ? unit_3 : last);
                            unsigned int quotient_5 = (unsigned int)((unsigned long long)clamped_4 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_6 = clamped_4 - quotient_5 * block_units;
                            unsigned int correction_7 = (unsigned int)(remainder_6 >= block_units);
                            quotient_5 = quotient_5 + correction_7;
                            remainder_6 = remainder_6 - correction_7 * block_units;
                            unsigned int quotient_8 = (unsigned int)((unsigned long long)quotient_5 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_9 = quotient_5 - quotient_8 * sequences;
                            unsigned int correction_10 = (unsigned int)(remainder_9 >= sequences);
                            quotient_8 = quotient_8 + correction_10;
                            remainder_9 = remainder_9 - correction_10 * sequences;
                            unsigned int src_11 = (quotient_5 + (quotient_8 * 7 + destination) * sequences) * block_units + remainder_6;
                            unsigned int dst_12 = quotient_5 * pitch_units + rank * block_units + remainder_6;
                            unsigned int _vec_load_1[4];
                            {
                                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(inp_words + (src_11 * 4) + 0);
                                _vec_load_1[0 + 0] = _uv4_1.x;
                                _vec_load_1[0 + 1] = _uv4_1.y;
                                _vec_load_1[0 + 2] = _uv4_1.z;
                                _vec_load_1[0 + 3] = _uv4_1.w;
                            }
                            unsigned int unit_13 = base + 960;
                            unsigned int clamped_14 = ((unit_13 < hi) ? unit_13 : last);
                            unsigned int quotient_15 = (unsigned int)((unsigned long long)clamped_14 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_16 = clamped_14 - quotient_15 * block_units;
                            unsigned int correction_17 = (unsigned int)(remainder_16 >= block_units);
                            quotient_15 = quotient_15 + correction_17;
                            remainder_16 = remainder_16 - correction_17 * block_units;
                            unsigned int quotient_18 = (unsigned int)((unsigned long long)quotient_15 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_19 = quotient_15 - quotient_18 * sequences;
                            unsigned int correction_20 = (unsigned int)(remainder_19 >= sequences);
                            quotient_18 = quotient_18 + correction_20;
                            remainder_19 = remainder_19 - correction_20 * sequences;
                            unsigned int src_21 = (quotient_15 + (quotient_18 * 7 + destination) * sequences) * block_units + remainder_16;
                            unsigned int dst_22 = quotient_15 * pitch_units + rank * block_units + remainder_16;
                            unsigned int _vec_load_2[4];
                            {
                                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(inp_words + (src_21 * 4) + 0);
                                _vec_load_2[0 + 0] = _uv4_2.x;
                                _vec_load_2[0 + 1] = _uv4_2.y;
                                _vec_load_2[0 + 2] = _uv4_2.z;
                                _vec_load_2[0 + 3] = _uv4_2.w;
                            }
                            unsigned int unit_23 = base + 1440;
                            unsigned int clamped_24 = ((unit_23 < hi) ? unit_23 : last);
                            unsigned int quotient_25 = (unsigned int)((unsigned long long)clamped_24 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_26 = clamped_24 - quotient_25 * block_units;
                            unsigned int correction_27 = (unsigned int)(remainder_26 >= block_units);
                            quotient_25 = quotient_25 + correction_27;
                            remainder_26 = remainder_26 - correction_27 * block_units;
                            unsigned int quotient_28 = (unsigned int)((unsigned long long)quotient_25 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_29 = quotient_25 - quotient_28 * sequences;
                            unsigned int correction_30 = (unsigned int)(remainder_29 >= sequences);
                            quotient_28 = quotient_28 + correction_30;
                            remainder_29 = remainder_29 - correction_30 * sequences;
                            unsigned int src_31 = (quotient_25 + (quotient_28 * 7 + destination) * sequences) * block_units + remainder_26;
                            unsigned int dst_32 = quotient_25 * pitch_units + rank * block_units + remainder_26;
                            unsigned int _vec_load_3[4];
                            {
                                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(inp_words + (src_31 * 4) + 0);
                                _vec_load_3[0 + 0] = _uv4_3.x;
                                _vec_load_3[0 + 1] = _uv4_3.y;
                                _vec_load_3[0 + 2] = _uv4_3.z;
                                _vec_load_3[0 + 3] = _uv4_3.w;
                            }
                            unsigned int unit_33 = base + 1920;
                            unsigned int clamped_34 = ((unit_33 < hi) ? unit_33 : last);
                            unsigned int quotient_35 = (unsigned int)((unsigned long long)clamped_34 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_36 = clamped_34 - quotient_35 * block_units;
                            unsigned int correction_37 = (unsigned int)(remainder_36 >= block_units);
                            quotient_35 = quotient_35 + correction_37;
                            remainder_36 = remainder_36 - correction_37 * block_units;
                            unsigned int quotient_38 = (unsigned int)((unsigned long long)quotient_35 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_39 = quotient_35 - quotient_38 * sequences;
                            unsigned int correction_40 = (unsigned int)(remainder_39 >= sequences);
                            quotient_38 = quotient_38 + correction_40;
                            remainder_39 = remainder_39 - correction_40 * sequences;
                            unsigned int src_41 = (quotient_35 + (quotient_38 * 7 + destination) * sequences) * block_units + remainder_36;
                            unsigned int dst_42 = quotient_35 * pitch_units + rank * block_units + remainder_36;
                            unsigned int _vec_load_4[4];
                            {
                                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(inp_words + (src_41 * 4) + 0);
                                _vec_load_4[0 + 0] = _uv4_4.x;
                                _vec_load_4[0 + 1] = _uv4_4.y;
                                _vec_load_4[0 + 2] = _uv4_4.z;
                                _vec_load_4[0 + 3] = _uv4_4.w;
                            }
                            unsigned int unit_43 = base + 2400;
                            unsigned int clamped_44 = ((unit_43 < hi) ? unit_43 : last);
                            unsigned int quotient_45 = (unsigned int)((unsigned long long)clamped_44 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_46 = clamped_44 - quotient_45 * block_units;
                            unsigned int correction_47 = (unsigned int)(remainder_46 >= block_units);
                            quotient_45 = quotient_45 + correction_47;
                            remainder_46 = remainder_46 - correction_47 * block_units;
                            unsigned int quotient_48 = (unsigned int)((unsigned long long)quotient_45 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_49 = quotient_45 - quotient_48 * sequences;
                            unsigned int correction_50 = (unsigned int)(remainder_49 >= sequences);
                            quotient_48 = quotient_48 + correction_50;
                            remainder_49 = remainder_49 - correction_50 * sequences;
                            unsigned int src_51 = (quotient_45 + (quotient_48 * 7 + destination) * sequences) * block_units + remainder_46;
                            unsigned int dst_52 = quotient_45 * pitch_units + rank * block_units + remainder_46;
                            unsigned int _vec_load_5[4];
                            {
                                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(inp_words + (src_51 * 4) + 0);
                                _vec_load_5[0 + 0] = _uv4_5.x;
                                _vec_load_5[0 + 1] = _uv4_5.y;
                                _vec_load_5[0 + 2] = _uv4_5.z;
                                _vec_load_5[0 + 3] = _uv4_5.w;
                            }
                            unsigned int unit_53 = base + 2880;
                            unsigned int clamped_54 = ((unit_53 < hi) ? unit_53 : last);
                            unsigned int quotient_55 = (unsigned int)((unsigned long long)clamped_54 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_56 = clamped_54 - quotient_55 * block_units;
                            unsigned int correction_57 = (unsigned int)(remainder_56 >= block_units);
                            quotient_55 = quotient_55 + correction_57;
                            remainder_56 = remainder_56 - correction_57 * block_units;
                            unsigned int quotient_58 = (unsigned int)((unsigned long long)quotient_55 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_59 = quotient_55 - quotient_58 * sequences;
                            unsigned int correction_60 = (unsigned int)(remainder_59 >= sequences);
                            quotient_58 = quotient_58 + correction_60;
                            remainder_59 = remainder_59 - correction_60 * sequences;
                            unsigned int src_61 = (quotient_55 + (quotient_58 * 7 + destination) * sequences) * block_units + remainder_56;
                            unsigned int dst_62 = quotient_55 * pitch_units + rank * block_units + remainder_56;
                            unsigned int _vec_load_6[4];
                            {
                                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(inp_words + (src_61 * 4) + 0);
                                _vec_load_6[0 + 0] = _uv4_6.x;
                                _vec_load_6[0 + 1] = _uv4_6.y;
                                _vec_load_6[0 + 2] = _uv4_6.z;
                                _vec_load_6[0 + 3] = _uv4_6.w;
                            }
                            unsigned int unit_63 = base + 3360;
                            unsigned int clamped_64 = ((unit_63 < hi) ? unit_63 : last);
                            unsigned int quotient_65 = (unsigned int)((unsigned long long)clamped_64 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_66 = clamped_64 - quotient_65 * block_units;
                            unsigned int correction_67 = (unsigned int)(remainder_66 >= block_units);
                            quotient_65 = quotient_65 + correction_67;
                            remainder_66 = remainder_66 - correction_67 * block_units;
                            unsigned int quotient_68 = (unsigned int)((unsigned long long)quotient_65 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_69 = quotient_65 - quotient_68 * sequences;
                            unsigned int correction_70 = (unsigned int)(remainder_69 >= sequences);
                            quotient_68 = quotient_68 + correction_70;
                            remainder_69 = remainder_69 - correction_70 * sequences;
                            unsigned int src_71 = (quotient_65 + (quotient_68 * 7 + destination) * sequences) * block_units + remainder_66;
                            unsigned int dst_72 = quotient_65 * pitch_units + rank * block_units + remainder_66;
                            unsigned int _vec_load_7[4];
                            {
                                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(inp_words + (src_71 * 4) + 0);
                                _vec_load_7[0 + 0] = _uv4_7.x;
                                _vec_load_7[0 + 1] = _uv4_7.y;
                                _vec_load_7[0 + 2] = _uv4_7.z;
                                _vec_load_7[0 + 3] = _uv4_7.w;
                            }
                            if (hi > base) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst * 4) + (0)), "r"(_vec_load_0[(0) + 0]), "r"(_vec_load_0[(0) + 1]), "r"(_vec_load_0[(0) + 2]), "r"(_vec_load_0[(0) + 3]) : "memory");
                            }
                            if (hi > base + 480) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_12 * 4) + (0)), "r"(_vec_load_1[(0) + 0]), "r"(_vec_load_1[(0) + 1]), "r"(_vec_load_1[(0) + 2]), "r"(_vec_load_1[(0) + 3]) : "memory");
                            }
                            if (hi > base + 960) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_22 * 4) + (0)), "r"(_vec_load_2[(0) + 0]), "r"(_vec_load_2[(0) + 1]), "r"(_vec_load_2[(0) + 2]), "r"(_vec_load_2[(0) + 3]) : "memory");
                            }
                            if (hi > base + 1440) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_32 * 4) + (0)), "r"(_vec_load_3[(0) + 0]), "r"(_vec_load_3[(0) + 1]), "r"(_vec_load_3[(0) + 2]), "r"(_vec_load_3[(0) + 3]) : "memory");
                            }
                            if (hi > base + 1920) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_42 * 4) + (0)), "r"(_vec_load_4[(0) + 0]), "r"(_vec_load_4[(0) + 1]), "r"(_vec_load_4[(0) + 2]), "r"(_vec_load_4[(0) + 3]) : "memory");
                            }
                            if (hi > base + 2400) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_52 * 4) + (0)), "r"(_vec_load_5[(0) + 0]), "r"(_vec_load_5[(0) + 1]), "r"(_vec_load_5[(0) + 2]), "r"(_vec_load_5[(0) + 3]) : "memory");
                            }
                            if (hi > base + 2880) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_62 * 4) + (0)), "r"(_vec_load_6[(0) + 0]), "r"(_vec_load_6[(0) + 1]), "r"(_vec_load_6[(0) + 2]), "r"(_vec_load_6[(0) + 3]) : "memory");
                            }
                            if (hi > base + 3360) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_72 * 4) + (0)), "r"(_vec_load_7[(0) + 0]), "r"(_vec_load_7[(0) + 1]), "r"(_vec_load_7[(0) + 2]), "r"(_vec_load_7[(0) + 3]) : "memory");
                            }
                            base = base + 3840;
                        }
                    } else {
                        #pragma unroll 1
                        for (unsigned int u = lo + tid_0; u < hi; u += 480) {
                            unsigned int quotient_1 = (unsigned int)((unsigned long long)u * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_2 = u - quotient_1 * block_units;
                            unsigned int correction_1 = (unsigned int)(remainder_2 >= block_units);
                            quotient_1 = quotient_1 + correction_1;
                            remainder_2 = remainder_2 - correction_1 * block_units;
                            unsigned int quotient_0_1 = (unsigned int)((unsigned long long)quotient_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_1_1 = quotient_1 - quotient_0_1 * sequences;
                            unsigned int correction_2_1 = (unsigned int)(remainder_1_1 >= sequences);
                            quotient_0_1 = quotient_0_1 + correction_2_1;
                            remainder_1_1 = remainder_1_1 - correction_2_1 * sequences;
                            unsigned int src_1 = (quotient_1 + (quotient_0_1 * 7 + destination) * sequences) * block_units + remainder_2;
                            unsigned int dst_1 = quotient_1 * pitch_units + rank * block_units + remainder_2;
                            out[dst_1] = inp[src_1];
                        }
                    }
                }
            } else {
                if (entered == 0) {
                    if (tid_0 < 8) {
                        if (tid_0 != rank) {
                            {
                            unsigned int _acquire_observed;
                            do {
                            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(signals_self) + (8 + tid_0))) : "memory");
                            } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(epoch)) >= static_cast<unsigned int>(1));
                            }
                        }
                    }
                    __syncthreads();
                    entered = 1;
                }
                if (warp_1 < 15) {
                    if (vector != 0) {
                        unsigned int* peer_words = reinterpret_cast<unsigned int*>(staging_peers.ptrs[destination]);
                        unsigned int last_1 = hi - 1;
                        unsigned int base_1 = lo + tid_0;
                        while (base_1 < hi) {
                            unsigned int unit_1 = base_1;
                            unsigned int clamped_1 = ((unit_1 < hi) ? unit_1 : last_1);
                            unsigned int quotient_2 = (unsigned int)((unsigned long long)clamped_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_3 = clamped_1 - quotient_2 * block_units;
                            unsigned int correction_3 = (unsigned int)(remainder_3 >= block_units);
                            quotient_2 = quotient_2 + correction_3;
                            remainder_3 = remainder_3 - correction_3 * block_units;
                            unsigned int quotient_0_2 = (unsigned int)((unsigned long long)quotient_2 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_1_2 = quotient_2 - quotient_0_2 * sequences;
                            unsigned int correction_2_2 = (unsigned int)(remainder_1_2 >= sequences);
                            quotient_0_2 = quotient_0_2 + correction_2_2;
                            remainder_1_2 = remainder_1_2 - correction_2_2 * sequences;
                            unsigned int src_2 = (quotient_2 + (quotient_0_2 * 7 + destination) * sequences) * block_units + remainder_3;
                            unsigned int dst_2 = quotient_2 * pitch_units + rank * block_units + remainder_3;
                            unsigned int _vec_load_8[4];
                            {
                                uint4 _uv4_8 = *reinterpret_cast<const uint4*>(inp_words + (src_2 * 4) + 0);
                                _vec_load_8[0 + 0] = _uv4_8.x;
                                _vec_load_8[0 + 1] = _uv4_8.y;
                                _vec_load_8[0 + 2] = _uv4_8.z;
                                _vec_load_8[0 + 3] = _uv4_8.w;
                            }
                            unsigned int unit_3_1 = base_1 + 480;
                            unsigned int clamped_4_1 = ((unit_3_1 < hi) ? unit_3_1 : last_1);
                            unsigned int quotient_5_1 = (unsigned int)((unsigned long long)clamped_4_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_6_1 = clamped_4_1 - quotient_5_1 * block_units;
                            unsigned int correction_7_1 = (unsigned int)(remainder_6_1 >= block_units);
                            quotient_5_1 = quotient_5_1 + correction_7_1;
                            remainder_6_1 = remainder_6_1 - correction_7_1 * block_units;
                            unsigned int quotient_8_1 = (unsigned int)((unsigned long long)quotient_5_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_9_1 = quotient_5_1 - quotient_8_1 * sequences;
                            unsigned int correction_10_1 = (unsigned int)(remainder_9_1 >= sequences);
                            quotient_8_1 = quotient_8_1 + correction_10_1;
                            remainder_9_1 = remainder_9_1 - correction_10_1 * sequences;
                            unsigned int src_11_1 = (quotient_5_1 + (quotient_8_1 * 7 + destination) * sequences) * block_units + remainder_6_1;
                            unsigned int dst_12_1 = quotient_5_1 * pitch_units + rank * block_units + remainder_6_1;
                            unsigned int _vec_load_9[4];
                            {
                                uint4 _uv4_9 = *reinterpret_cast<const uint4*>(inp_words + (src_11_1 * 4) + 0);
                                _vec_load_9[0 + 0] = _uv4_9.x;
                                _vec_load_9[0 + 1] = _uv4_9.y;
                                _vec_load_9[0 + 2] = _uv4_9.z;
                                _vec_load_9[0 + 3] = _uv4_9.w;
                            }
                            unsigned int unit_13_1 = base_1 + 960;
                            unsigned int clamped_14_1 = ((unit_13_1 < hi) ? unit_13_1 : last_1);
                            unsigned int quotient_15_1 = (unsigned int)((unsigned long long)clamped_14_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_16_1 = clamped_14_1 - quotient_15_1 * block_units;
                            unsigned int correction_17_1 = (unsigned int)(remainder_16_1 >= block_units);
                            quotient_15_1 = quotient_15_1 + correction_17_1;
                            remainder_16_1 = remainder_16_1 - correction_17_1 * block_units;
                            unsigned int quotient_18_1 = (unsigned int)((unsigned long long)quotient_15_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_19_1 = quotient_15_1 - quotient_18_1 * sequences;
                            unsigned int correction_20_1 = (unsigned int)(remainder_19_1 >= sequences);
                            quotient_18_1 = quotient_18_1 + correction_20_1;
                            remainder_19_1 = remainder_19_1 - correction_20_1 * sequences;
                            unsigned int src_21_1 = (quotient_15_1 + (quotient_18_1 * 7 + destination) * sequences) * block_units + remainder_16_1;
                            unsigned int dst_22_1 = quotient_15_1 * pitch_units + rank * block_units + remainder_16_1;
                            unsigned int _vec_load_10[4];
                            {
                                uint4 _uv4_10 = *reinterpret_cast<const uint4*>(inp_words + (src_21_1 * 4) + 0);
                                _vec_load_10[0 + 0] = _uv4_10.x;
                                _vec_load_10[0 + 1] = _uv4_10.y;
                                _vec_load_10[0 + 2] = _uv4_10.z;
                                _vec_load_10[0 + 3] = _uv4_10.w;
                            }
                            unsigned int unit_23_1 = base_1 + 1440;
                            unsigned int clamped_24_1 = ((unit_23_1 < hi) ? unit_23_1 : last_1);
                            unsigned int quotient_25_1 = (unsigned int)((unsigned long long)clamped_24_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_26_1 = clamped_24_1 - quotient_25_1 * block_units;
                            unsigned int correction_27_1 = (unsigned int)(remainder_26_1 >= block_units);
                            quotient_25_1 = quotient_25_1 + correction_27_1;
                            remainder_26_1 = remainder_26_1 - correction_27_1 * block_units;
                            unsigned int quotient_28_1 = (unsigned int)((unsigned long long)quotient_25_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_29_1 = quotient_25_1 - quotient_28_1 * sequences;
                            unsigned int correction_30_1 = (unsigned int)(remainder_29_1 >= sequences);
                            quotient_28_1 = quotient_28_1 + correction_30_1;
                            remainder_29_1 = remainder_29_1 - correction_30_1 * sequences;
                            unsigned int src_31_1 = (quotient_25_1 + (quotient_28_1 * 7 + destination) * sequences) * block_units + remainder_26_1;
                            unsigned int dst_32_1 = quotient_25_1 * pitch_units + rank * block_units + remainder_26_1;
                            unsigned int _vec_load_11[4];
                            {
                                uint4 _uv4_11 = *reinterpret_cast<const uint4*>(inp_words + (src_31_1 * 4) + 0);
                                _vec_load_11[0 + 0] = _uv4_11.x;
                                _vec_load_11[0 + 1] = _uv4_11.y;
                                _vec_load_11[0 + 2] = _uv4_11.z;
                                _vec_load_11[0 + 3] = _uv4_11.w;
                            }
                            unsigned int unit_33_1 = base_1 + 1920;
                            unsigned int clamped_34_1 = ((unit_33_1 < hi) ? unit_33_1 : last_1);
                            unsigned int quotient_35_1 = (unsigned int)((unsigned long long)clamped_34_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_36_1 = clamped_34_1 - quotient_35_1 * block_units;
                            unsigned int correction_37_1 = (unsigned int)(remainder_36_1 >= block_units);
                            quotient_35_1 = quotient_35_1 + correction_37_1;
                            remainder_36_1 = remainder_36_1 - correction_37_1 * block_units;
                            unsigned int quotient_38_1 = (unsigned int)((unsigned long long)quotient_35_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_39_1 = quotient_35_1 - quotient_38_1 * sequences;
                            unsigned int correction_40_1 = (unsigned int)(remainder_39_1 >= sequences);
                            quotient_38_1 = quotient_38_1 + correction_40_1;
                            remainder_39_1 = remainder_39_1 - correction_40_1 * sequences;
                            unsigned int src_41_1 = (quotient_35_1 + (quotient_38_1 * 7 + destination) * sequences) * block_units + remainder_36_1;
                            unsigned int dst_42_1 = quotient_35_1 * pitch_units + rank * block_units + remainder_36_1;
                            unsigned int _vec_load_12[4];
                            {
                                uint4 _uv4_12 = *reinterpret_cast<const uint4*>(inp_words + (src_41_1 * 4) + 0);
                                _vec_load_12[0 + 0] = _uv4_12.x;
                                _vec_load_12[0 + 1] = _uv4_12.y;
                                _vec_load_12[0 + 2] = _uv4_12.z;
                                _vec_load_12[0 + 3] = _uv4_12.w;
                            }
                            unsigned int unit_43_1 = base_1 + 2400;
                            unsigned int clamped_44_1 = ((unit_43_1 < hi) ? unit_43_1 : last_1);
                            unsigned int quotient_45_1 = (unsigned int)((unsigned long long)clamped_44_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_46_1 = clamped_44_1 - quotient_45_1 * block_units;
                            unsigned int correction_47_1 = (unsigned int)(remainder_46_1 >= block_units);
                            quotient_45_1 = quotient_45_1 + correction_47_1;
                            remainder_46_1 = remainder_46_1 - correction_47_1 * block_units;
                            unsigned int quotient_48_1 = (unsigned int)((unsigned long long)quotient_45_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_49_1 = quotient_45_1 - quotient_48_1 * sequences;
                            unsigned int correction_50_1 = (unsigned int)(remainder_49_1 >= sequences);
                            quotient_48_1 = quotient_48_1 + correction_50_1;
                            remainder_49_1 = remainder_49_1 - correction_50_1 * sequences;
                            unsigned int src_51_1 = (quotient_45_1 + (quotient_48_1 * 7 + destination) * sequences) * block_units + remainder_46_1;
                            unsigned int dst_52_1 = quotient_45_1 * pitch_units + rank * block_units + remainder_46_1;
                            unsigned int _vec_load_13[4];
                            {
                                uint4 _uv4_13 = *reinterpret_cast<const uint4*>(inp_words + (src_51_1 * 4) + 0);
                                _vec_load_13[0 + 0] = _uv4_13.x;
                                _vec_load_13[0 + 1] = _uv4_13.y;
                                _vec_load_13[0 + 2] = _uv4_13.z;
                                _vec_load_13[0 + 3] = _uv4_13.w;
                            }
                            unsigned int unit_53_1 = base_1 + 2880;
                            unsigned int clamped_54_1 = ((unit_53_1 < hi) ? unit_53_1 : last_1);
                            unsigned int quotient_55_1 = (unsigned int)((unsigned long long)clamped_54_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_56_1 = clamped_54_1 - quotient_55_1 * block_units;
                            unsigned int correction_57_1 = (unsigned int)(remainder_56_1 >= block_units);
                            quotient_55_1 = quotient_55_1 + correction_57_1;
                            remainder_56_1 = remainder_56_1 - correction_57_1 * block_units;
                            unsigned int quotient_58_1 = (unsigned int)((unsigned long long)quotient_55_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_59_1 = quotient_55_1 - quotient_58_1 * sequences;
                            unsigned int correction_60_1 = (unsigned int)(remainder_59_1 >= sequences);
                            quotient_58_1 = quotient_58_1 + correction_60_1;
                            remainder_59_1 = remainder_59_1 - correction_60_1 * sequences;
                            unsigned int src_61_1 = (quotient_55_1 + (quotient_58_1 * 7 + destination) * sequences) * block_units + remainder_56_1;
                            unsigned int dst_62_1 = quotient_55_1 * pitch_units + rank * block_units + remainder_56_1;
                            unsigned int _vec_load_14[4];
                            {
                                uint4 _uv4_14 = *reinterpret_cast<const uint4*>(inp_words + (src_61_1 * 4) + 0);
                                _vec_load_14[0 + 0] = _uv4_14.x;
                                _vec_load_14[0 + 1] = _uv4_14.y;
                                _vec_load_14[0 + 2] = _uv4_14.z;
                                _vec_load_14[0 + 3] = _uv4_14.w;
                            }
                            unsigned int unit_63_1 = base_1 + 3360;
                            unsigned int clamped_64_1 = ((unit_63_1 < hi) ? unit_63_1 : last_1);
                            unsigned int quotient_65_1 = (unsigned int)((unsigned long long)clamped_64_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_66_1 = clamped_64_1 - quotient_65_1 * block_units;
                            unsigned int correction_67_1 = (unsigned int)(remainder_66_1 >= block_units);
                            quotient_65_1 = quotient_65_1 + correction_67_1;
                            remainder_66_1 = remainder_66_1 - correction_67_1 * block_units;
                            unsigned int quotient_68_1 = (unsigned int)((unsigned long long)quotient_65_1 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_69_1 = quotient_65_1 - quotient_68_1 * sequences;
                            unsigned int correction_70_1 = (unsigned int)(remainder_69_1 >= sequences);
                            quotient_68_1 = quotient_68_1 + correction_70_1;
                            remainder_69_1 = remainder_69_1 - correction_70_1 * sequences;
                            unsigned int src_71_1 = (quotient_65_1 + (quotient_68_1 * 7 + destination) * sequences) * block_units + remainder_66_1;
                            unsigned int dst_72_1 = quotient_65_1 * pitch_units + rank * block_units + remainder_66_1;
                            unsigned int _vec_load_15[4];
                            {
                                uint4 _uv4_15 = *reinterpret_cast<const uint4*>(inp_words + (src_71_1 * 4) + 0);
                                _vec_load_15[0 + 0] = _uv4_15.x;
                                _vec_load_15[0 + 1] = _uv4_15.y;
                                _vec_load_15[0 + 2] = _uv4_15.z;
                                _vec_load_15[0 + 3] = _uv4_15.w;
                            }
                            if (hi > base_1) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_2 * 4) + (0)), "r"(_vec_load_8[(0) + 0]), "r"(_vec_load_8[(0) + 1]), "r"(_vec_load_8[(0) + 2]), "r"(_vec_load_8[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 480) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_12_1 * 4) + (0)), "r"(_vec_load_9[(0) + 0]), "r"(_vec_load_9[(0) + 1]), "r"(_vec_load_9[(0) + 2]), "r"(_vec_load_9[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 960) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_22_1 * 4) + (0)), "r"(_vec_load_10[(0) + 0]), "r"(_vec_load_10[(0) + 1]), "r"(_vec_load_10[(0) + 2]), "r"(_vec_load_10[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 1440) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_32_1 * 4) + (0)), "r"(_vec_load_11[(0) + 0]), "r"(_vec_load_11[(0) + 1]), "r"(_vec_load_11[(0) + 2]), "r"(_vec_load_11[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 1920) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_42_1 * 4) + (0)), "r"(_vec_load_12[(0) + 0]), "r"(_vec_load_12[(0) + 1]), "r"(_vec_load_12[(0) + 2]), "r"(_vec_load_12[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 2400) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_52_1 * 4) + (0)), "r"(_vec_load_13[(0) + 0]), "r"(_vec_load_13[(0) + 1]), "r"(_vec_load_13[(0) + 2]), "r"(_vec_load_13[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 2880) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_62_1 * 4) + (0)), "r"(_vec_load_14[(0) + 0]), "r"(_vec_load_14[(0) + 1]), "r"(_vec_load_14[(0) + 2]), "r"(_vec_load_14[(0) + 3]) : "memory");
                            }
                            if (hi > base_1 + 3360) {
                                asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(peer_words + (dst_72_1 * 4) + (0)), "r"(_vec_load_15[(0) + 0]), "r"(_vec_load_15[(0) + 1]), "r"(_vec_load_15[(0) + 2]), "r"(_vec_load_15[(0) + 3]) : "memory");
                            }
                            base_1 = base_1 + 3840;
                        }
                    } else {
                        __half* peer_elements = staging_peers.ptrs[destination];
                        #pragma unroll 1
                        for (unsigned int u_1 = lo + tid_0; u_1 < hi; u_1 += 480) {
                            unsigned int quotient_3 = (unsigned int)((unsigned long long)u_1 * (unsigned long long)units_magic >> 32);
                            unsigned int remainder_4 = u_1 - quotient_3 * block_units;
                            unsigned int correction_4 = (unsigned int)(remainder_4 >= block_units);
                            quotient_3 = quotient_3 + correction_4;
                            remainder_4 = remainder_4 - correction_4 * block_units;
                            unsigned int quotient_0_3 = (unsigned int)((unsigned long long)quotient_3 * (unsigned long long)sequences_magic >> 32);
                            unsigned int remainder_1_3 = quotient_3 - quotient_0_3 * sequences;
                            unsigned int correction_2_3 = (unsigned int)(remainder_1_3 >= sequences);
                            quotient_0_3 = quotient_0_3 + correction_2_3;
                            remainder_1_3 = remainder_1_3 - correction_2_3 * sequences;
                            unsigned int src_3 = (quotient_3 + (quotient_0_3 * 7 + destination) * sequences) * block_units + remainder_4;
                            unsigned int dst_3 = quotient_3 * pitch_units + rank * block_units + remainder_4;
                            peer_elements[dst_3] = inp[src_3];
                        }
                    }
                }
                __syncthreads();
                if (warp == 15) {
                    if (elect_sync()) {
                        asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(signals_peers.ptrs[destination] + (16 + rank * 96 + bid_1)), "r"(epoch) : "memory");
                    }
                }
            }
        }
    } else {
        unsigned int job = bid_1 - 96;
        unsigned int* staging_words = reinterpret_cast<unsigned int*>(staging_peers.ptrs[rank]);
        __half* staging_elements = staging_peers.ptrs[rank];
        #pragma unroll 1
        for (unsigned int i = job; i < 672; i += 52) {
            unsigned int step = i / 84;
            unsigned int in_step = i - step * 84;
            unsigned int other = in_step / 12;
            unsigned int lane_1 = in_step - other * 12;
            unsigned int source = other + (unsigned int)(other >= rank);
            unsigned int part = (rank + 16 - source - 1 - step) % 8 + lane_1 * 8;
            if (tid_0 == 0) {
                {
                unsigned int _acquire_observed;
                do {
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(_acquire_observed) : "l"((reinterpret_cast<unsigned int*>(signals_self) + (16 + source * 96 + part))) : "memory");
                } while (static_cast<unsigned int>(_acquire_observed - static_cast<unsigned int>(epoch)) >= static_cast<unsigned int>(1));
                }
            }
            __syncthreads();
            unsigned int lo_1 = part * part_quotient + part * part_remainder / 96;
            unsigned int hi_1 = (part + 1) * part_quotient + (part + 1) * part_remainder / 96;
            if (vector != 0) {
                unsigned int last_2 = hi_1 - 1;
                unsigned int base_2 = lo_1 + tid_0;
                while (base_2 < hi_1) {
                    unsigned int unit_2 = base_2;
                    unsigned int clamped_2 = ((unit_2 < hi_1) ? unit_2 : last_2);
                    unsigned int quotient_4 = (unsigned int)((unsigned long long)clamped_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_5 = clamped_2 - quotient_4 * block_units;
                    unsigned int correction_5 = (unsigned int)(remainder_5 >= block_units);
                    quotient_4 = quotient_4 + correction_5;
                    remainder_5 = remainder_5 - correction_5 * block_units;
                    unsigned int quotient_0_4 = (unsigned int)((unsigned long long)quotient_4 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_1_4 = quotient_4 - quotient_0_4 * sequences;
                    unsigned int correction_2_4 = (unsigned int)(remainder_1_4 >= sequences);
                    quotient_0_4 = quotient_0_4 + correction_2_4;
                    remainder_1_4 = remainder_1_4 - correction_2_4 * sequences;
                    unsigned int src_4 = (quotient_4 + (quotient_0_4 * 7 + rank) * sequences) * block_units + remainder_5;
                    unsigned int dst_4 = quotient_4 * pitch_units + source * block_units + remainder_5;
                    unsigned int _vec_load_16[4];
                    {
                        uint4 _uv4_16 = *reinterpret_cast<const uint4*>(staging_words + (dst_4 * 4) + 0);
                        _vec_load_16[0 + 0] = _uv4_16.x;
                        _vec_load_16[0 + 1] = _uv4_16.y;
                        _vec_load_16[0 + 2] = _uv4_16.z;
                        _vec_load_16[0 + 3] = _uv4_16.w;
                    }
                    unsigned int unit_3_2 = base_2 + 512;
                    unsigned int clamped_4_2 = ((unit_3_2 < hi_1) ? unit_3_2 : last_2);
                    unsigned int quotient_5_2 = (unsigned int)((unsigned long long)clamped_4_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_6_2 = clamped_4_2 - quotient_5_2 * block_units;
                    unsigned int correction_7_2 = (unsigned int)(remainder_6_2 >= block_units);
                    quotient_5_2 = quotient_5_2 + correction_7_2;
                    remainder_6_2 = remainder_6_2 - correction_7_2 * block_units;
                    unsigned int quotient_8_2 = (unsigned int)((unsigned long long)quotient_5_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_9_2 = quotient_5_2 - quotient_8_2 * sequences;
                    unsigned int correction_10_2 = (unsigned int)(remainder_9_2 >= sequences);
                    quotient_8_2 = quotient_8_2 + correction_10_2;
                    remainder_9_2 = remainder_9_2 - correction_10_2 * sequences;
                    unsigned int src_11_2 = (quotient_5_2 + (quotient_8_2 * 7 + rank) * sequences) * block_units + remainder_6_2;
                    unsigned int dst_12_2 = quotient_5_2 * pitch_units + source * block_units + remainder_6_2;
                    unsigned int _vec_load_17[4];
                    {
                        uint4 _uv4_17 = *reinterpret_cast<const uint4*>(staging_words + (dst_12_2 * 4) + 0);
                        _vec_load_17[0 + 0] = _uv4_17.x;
                        _vec_load_17[0 + 1] = _uv4_17.y;
                        _vec_load_17[0 + 2] = _uv4_17.z;
                        _vec_load_17[0 + 3] = _uv4_17.w;
                    }
                    unsigned int unit_13_2 = base_2 + 1024;
                    unsigned int clamped_14_2 = ((unit_13_2 < hi_1) ? unit_13_2 : last_2);
                    unsigned int quotient_15_2 = (unsigned int)((unsigned long long)clamped_14_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_16_2 = clamped_14_2 - quotient_15_2 * block_units;
                    unsigned int correction_17_2 = (unsigned int)(remainder_16_2 >= block_units);
                    quotient_15_2 = quotient_15_2 + correction_17_2;
                    remainder_16_2 = remainder_16_2 - correction_17_2 * block_units;
                    unsigned int quotient_18_2 = (unsigned int)((unsigned long long)quotient_15_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_19_2 = quotient_15_2 - quotient_18_2 * sequences;
                    unsigned int correction_20_2 = (unsigned int)(remainder_19_2 >= sequences);
                    quotient_18_2 = quotient_18_2 + correction_20_2;
                    remainder_19_2 = remainder_19_2 - correction_20_2 * sequences;
                    unsigned int src_21_2 = (quotient_15_2 + (quotient_18_2 * 7 + rank) * sequences) * block_units + remainder_16_2;
                    unsigned int dst_22_2 = quotient_15_2 * pitch_units + source * block_units + remainder_16_2;
                    unsigned int _vec_load_18[4];
                    {
                        uint4 _uv4_18 = *reinterpret_cast<const uint4*>(staging_words + (dst_22_2 * 4) + 0);
                        _vec_load_18[0 + 0] = _uv4_18.x;
                        _vec_load_18[0 + 1] = _uv4_18.y;
                        _vec_load_18[0 + 2] = _uv4_18.z;
                        _vec_load_18[0 + 3] = _uv4_18.w;
                    }
                    unsigned int unit_23_2 = base_2 + 1536;
                    unsigned int clamped_24_2 = ((unit_23_2 < hi_1) ? unit_23_2 : last_2);
                    unsigned int quotient_25_2 = (unsigned int)((unsigned long long)clamped_24_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_26_2 = clamped_24_2 - quotient_25_2 * block_units;
                    unsigned int correction_27_2 = (unsigned int)(remainder_26_2 >= block_units);
                    quotient_25_2 = quotient_25_2 + correction_27_2;
                    remainder_26_2 = remainder_26_2 - correction_27_2 * block_units;
                    unsigned int quotient_28_2 = (unsigned int)((unsigned long long)quotient_25_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_29_2 = quotient_25_2 - quotient_28_2 * sequences;
                    unsigned int correction_30_2 = (unsigned int)(remainder_29_2 >= sequences);
                    quotient_28_2 = quotient_28_2 + correction_30_2;
                    remainder_29_2 = remainder_29_2 - correction_30_2 * sequences;
                    unsigned int src_31_2 = (quotient_25_2 + (quotient_28_2 * 7 + rank) * sequences) * block_units + remainder_26_2;
                    unsigned int dst_32_2 = quotient_25_2 * pitch_units + source * block_units + remainder_26_2;
                    unsigned int _vec_load_19[4];
                    {
                        uint4 _uv4_19 = *reinterpret_cast<const uint4*>(staging_words + (dst_32_2 * 4) + 0);
                        _vec_load_19[0 + 0] = _uv4_19.x;
                        _vec_load_19[0 + 1] = _uv4_19.y;
                        _vec_load_19[0 + 2] = _uv4_19.z;
                        _vec_load_19[0 + 3] = _uv4_19.w;
                    }
                    unsigned int unit_33_2 = base_2 + 2048;
                    unsigned int clamped_34_2 = ((unit_33_2 < hi_1) ? unit_33_2 : last_2);
                    unsigned int quotient_35_2 = (unsigned int)((unsigned long long)clamped_34_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_36_2 = clamped_34_2 - quotient_35_2 * block_units;
                    unsigned int correction_37_2 = (unsigned int)(remainder_36_2 >= block_units);
                    quotient_35_2 = quotient_35_2 + correction_37_2;
                    remainder_36_2 = remainder_36_2 - correction_37_2 * block_units;
                    unsigned int quotient_38_2 = (unsigned int)((unsigned long long)quotient_35_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_39_2 = quotient_35_2 - quotient_38_2 * sequences;
                    unsigned int correction_40_2 = (unsigned int)(remainder_39_2 >= sequences);
                    quotient_38_2 = quotient_38_2 + correction_40_2;
                    remainder_39_2 = remainder_39_2 - correction_40_2 * sequences;
                    unsigned int src_41_2 = (quotient_35_2 + (quotient_38_2 * 7 + rank) * sequences) * block_units + remainder_36_2;
                    unsigned int dst_42_2 = quotient_35_2 * pitch_units + source * block_units + remainder_36_2;
                    unsigned int _vec_load_20[4];
                    {
                        uint4 _uv4_20 = *reinterpret_cast<const uint4*>(staging_words + (dst_42_2 * 4) + 0);
                        _vec_load_20[0 + 0] = _uv4_20.x;
                        _vec_load_20[0 + 1] = _uv4_20.y;
                        _vec_load_20[0 + 2] = _uv4_20.z;
                        _vec_load_20[0 + 3] = _uv4_20.w;
                    }
                    unsigned int unit_43_2 = base_2 + 2560;
                    unsigned int clamped_44_2 = ((unit_43_2 < hi_1) ? unit_43_2 : last_2);
                    unsigned int quotient_45_2 = (unsigned int)((unsigned long long)clamped_44_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_46_2 = clamped_44_2 - quotient_45_2 * block_units;
                    unsigned int correction_47_2 = (unsigned int)(remainder_46_2 >= block_units);
                    quotient_45_2 = quotient_45_2 + correction_47_2;
                    remainder_46_2 = remainder_46_2 - correction_47_2 * block_units;
                    unsigned int quotient_48_2 = (unsigned int)((unsigned long long)quotient_45_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_49_2 = quotient_45_2 - quotient_48_2 * sequences;
                    unsigned int correction_50_2 = (unsigned int)(remainder_49_2 >= sequences);
                    quotient_48_2 = quotient_48_2 + correction_50_2;
                    remainder_49_2 = remainder_49_2 - correction_50_2 * sequences;
                    unsigned int src_51_2 = (quotient_45_2 + (quotient_48_2 * 7 + rank) * sequences) * block_units + remainder_46_2;
                    unsigned int dst_52_2 = quotient_45_2 * pitch_units + source * block_units + remainder_46_2;
                    unsigned int _vec_load_21[4];
                    {
                        uint4 _uv4_21 = *reinterpret_cast<const uint4*>(staging_words + (dst_52_2 * 4) + 0);
                        _vec_load_21[0 + 0] = _uv4_21.x;
                        _vec_load_21[0 + 1] = _uv4_21.y;
                        _vec_load_21[0 + 2] = _uv4_21.z;
                        _vec_load_21[0 + 3] = _uv4_21.w;
                    }
                    unsigned int unit_53_2 = base_2 + 3072;
                    unsigned int clamped_54_2 = ((unit_53_2 < hi_1) ? unit_53_2 : last_2);
                    unsigned int quotient_55_2 = (unsigned int)((unsigned long long)clamped_54_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_56_2 = clamped_54_2 - quotient_55_2 * block_units;
                    unsigned int correction_57_2 = (unsigned int)(remainder_56_2 >= block_units);
                    quotient_55_2 = quotient_55_2 + correction_57_2;
                    remainder_56_2 = remainder_56_2 - correction_57_2 * block_units;
                    unsigned int quotient_58_2 = (unsigned int)((unsigned long long)quotient_55_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_59_2 = quotient_55_2 - quotient_58_2 * sequences;
                    unsigned int correction_60_2 = (unsigned int)(remainder_59_2 >= sequences);
                    quotient_58_2 = quotient_58_2 + correction_60_2;
                    remainder_59_2 = remainder_59_2 - correction_60_2 * sequences;
                    unsigned int src_61_2 = (quotient_55_2 + (quotient_58_2 * 7 + rank) * sequences) * block_units + remainder_56_2;
                    unsigned int dst_62_2 = quotient_55_2 * pitch_units + source * block_units + remainder_56_2;
                    unsigned int _vec_load_22[4];
                    {
                        uint4 _uv4_22 = *reinterpret_cast<const uint4*>(staging_words + (dst_62_2 * 4) + 0);
                        _vec_load_22[0 + 0] = _uv4_22.x;
                        _vec_load_22[0 + 1] = _uv4_22.y;
                        _vec_load_22[0 + 2] = _uv4_22.z;
                        _vec_load_22[0 + 3] = _uv4_22.w;
                    }
                    unsigned int unit_63_2 = base_2 + 3584;
                    unsigned int clamped_64_2 = ((unit_63_2 < hi_1) ? unit_63_2 : last_2);
                    unsigned int quotient_65_2 = (unsigned int)((unsigned long long)clamped_64_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_66_2 = clamped_64_2 - quotient_65_2 * block_units;
                    unsigned int correction_67_2 = (unsigned int)(remainder_66_2 >= block_units);
                    quotient_65_2 = quotient_65_2 + correction_67_2;
                    remainder_66_2 = remainder_66_2 - correction_67_2 * block_units;
                    unsigned int quotient_68_2 = (unsigned int)((unsigned long long)quotient_65_2 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_69_2 = quotient_65_2 - quotient_68_2 * sequences;
                    unsigned int correction_70_2 = (unsigned int)(remainder_69_2 >= sequences);
                    quotient_68_2 = quotient_68_2 + correction_70_2;
                    remainder_69_2 = remainder_69_2 - correction_70_2 * sequences;
                    unsigned int src_71_2 = (quotient_65_2 + (quotient_68_2 * 7 + rank) * sequences) * block_units + remainder_66_2;
                    unsigned int dst_72_2 = quotient_65_2 * pitch_units + source * block_units + remainder_66_2;
                    unsigned int _vec_load_23[4];
                    {
                        uint4 _uv4_23 = *reinterpret_cast<const uint4*>(staging_words + (dst_72_2 * 4) + 0);
                        _vec_load_23[0 + 0] = _uv4_23.x;
                        _vec_load_23[0 + 1] = _uv4_23.y;
                        _vec_load_23[0 + 2] = _uv4_23.z;
                        _vec_load_23[0 + 3] = _uv4_23.w;
                    }
                    if (hi_1 > base_2) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_4 * 4) + (0)), "r"(_vec_load_16[(0) + 0]), "r"(_vec_load_16[(0) + 1]), "r"(_vec_load_16[(0) + 2]), "r"(_vec_load_16[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 512) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_12_2 * 4) + (0)), "r"(_vec_load_17[(0) + 0]), "r"(_vec_load_17[(0) + 1]), "r"(_vec_load_17[(0) + 2]), "r"(_vec_load_17[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 1024) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_22_2 * 4) + (0)), "r"(_vec_load_18[(0) + 0]), "r"(_vec_load_18[(0) + 1]), "r"(_vec_load_18[(0) + 2]), "r"(_vec_load_18[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 1536) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_32_2 * 4) + (0)), "r"(_vec_load_19[(0) + 0]), "r"(_vec_load_19[(0) + 1]), "r"(_vec_load_19[(0) + 2]), "r"(_vec_load_19[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 2048) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_42_2 * 4) + (0)), "r"(_vec_load_20[(0) + 0]), "r"(_vec_load_20[(0) + 1]), "r"(_vec_load_20[(0) + 2]), "r"(_vec_load_20[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 2560) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_52_2 * 4) + (0)), "r"(_vec_load_21[(0) + 0]), "r"(_vec_load_21[(0) + 1]), "r"(_vec_load_21[(0) + 2]), "r"(_vec_load_21[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 3072) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_62_2 * 4) + (0)), "r"(_vec_load_22[(0) + 0]), "r"(_vec_load_22[(0) + 1]), "r"(_vec_load_22[(0) + 2]), "r"(_vec_load_22[(0) + 3]) : "memory");
                    }
                    if (hi_1 > base_2 + 3584) {
                        asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(out_words + (dst_72_2 * 4) + (0)), "r"(_vec_load_23[(0) + 0]), "r"(_vec_load_23[(0) + 1]), "r"(_vec_load_23[(0) + 2]), "r"(_vec_load_23[(0) + 3]) : "memory");
                    }
                    base_2 = base_2 + 4096;
                }
            } else {
                #pragma unroll 1
                for (unsigned int u_2 = lo_1 + tid_0; u_2 < hi_1; u_2 += 512) {
                    unsigned int quotient_6 = (unsigned int)((unsigned long long)u_2 * (unsigned long long)units_magic >> 32);
                    unsigned int remainder_7 = u_2 - quotient_6 * block_units;
                    unsigned int correction_6 = (unsigned int)(remainder_7 >= block_units);
                    quotient_6 = quotient_6 + correction_6;
                    remainder_7 = remainder_7 - correction_6 * block_units;
                    unsigned int quotient_0_5 = (unsigned int)((unsigned long long)quotient_6 * (unsigned long long)sequences_magic >> 32);
                    unsigned int remainder_1_5 = quotient_6 - quotient_0_5 * sequences;
                    unsigned int correction_2_5 = (unsigned int)(remainder_1_5 >= sequences);
                    quotient_0_5 = quotient_0_5 + correction_2_5;
                    remainder_1_5 = remainder_1_5 - correction_2_5 * sequences;
                    unsigned int src_5 = (quotient_6 + (quotient_0_5 * 7 + rank) * sequences) * block_units + remainder_7;
                    unsigned int dst_5 = quotient_6 * pitch_units + source * block_units + remainder_7;
                    out[dst_5] = staging_elements[dst_5];
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        if (elect_sync()) {
            uint32_t _atomic_inc_old_0;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_inc_old_0) : "l"(&signals_self[1]), "r"(static_cast<uint32_t>(147)) : "memory");
            if (_atomic_inc_old_0 == 147) {
                asm volatile("st.relaxed.gpu.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(signals_self) + (0))), "r"(static_cast<unsigned int>(epoch)) : "memory");
            }
        }
    }
}

} // extern "C"

