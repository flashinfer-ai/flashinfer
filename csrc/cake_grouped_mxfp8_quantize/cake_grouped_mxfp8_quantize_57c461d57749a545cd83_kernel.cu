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

// This unit serves two programs: cake_grouped_mxfp8_quantize_57c461d57749a545cd83 (CAKE_PROGRAM_VARIANT 0, the default) and
// cake_grouped_mxfp8_quantize_1dd4abd3b665713c0df0 (CAKE_PROGRAM_VARIANT 1).  CAKE_PROGRAM_KERNEL names the kernel it defines.  A build
// selects the second program with the compile definitions
// -DCAKE_PROGRAM_VARIANT=1 -DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_1dd4abd3b665713c0df0.
#ifndef CAKE_PROGRAM_VARIANT
#define CAKE_PROGRAM_VARIANT 0
#endif
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_grouped_mxfp8_quantize_57c461d57749a545cd83
#endif

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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}



__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128) void
CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ x, int* __restrict__ mask, uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales, int M, int K, int PADDED_K, int PM_TILES, int PK_TILES, CakeFastDivmod BLOCKS_PER_ROW, unsigned long long TOTAL_TASKS, unsigned int ROWS, unsigned int SUBROWS)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    unsigned int group = blockIdx.z;
    unsigned int tid_0 = (unsigned int)tid;
    unsigned int flat = (unsigned int)blockIdx.x * 128 + tid_0;
    uint32_t _fast_div_q_0 = (BLOCKS_PER_ROW.divisor != 1) ? (__umulhi((uint32_t)(flat), BLOCKS_PER_ROW.multiplier) >> BLOCKS_PER_ROW.shift_right) : (uint32_t)(flat);
    uint32_t _fast_div_r_0 = (uint32_t)(flat) - _fast_div_q_0 * (uint32_t)(BLOCKS_PER_ROW.divisor);
    if (_fast_div_q_0 < SUBROWS) {
        unsigned int first_row = (unsigned int)blockIdx.y * (ROWS * SUBROWS) + _fast_div_q_0;
        int _vec_load_0[1];
        {
            _vec_load_0[0] = *reinterpret_cast<const int*>(mask + group);
        }
        unsigned int valid_rows = (((unsigned int)_vec_load_0[0]) < ((unsigned int)M) ? ((unsigned int)_vec_load_0[0]) : ((unsigned int)M));
        if (first_row < valid_rows) {
            unsigned int k0 = _fast_div_r_0 * 32;
            unsigned int row = first_row;
            unsigned long long global_row = (unsigned long long)group * (unsigned long long)M + (unsigned long long)row;
            #pragma unroll 1
            for (unsigned int i = 0; i < ROWS; i++) {
                if (row < valid_rows) {
                    float values[32];
                    if (k0 < (unsigned int)K) {
                        unsigned long long x_offset = global_row * (unsigned long long)K + (unsigned long long)k0;
                        {
                            const void* _v8p_0 = (const void*)(x + (x_offset));
                            uint32_t _v8_0_0[8];
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_v8_0_0[0]), "=r"(_v8_0_0[1]), "=r"(_v8_0_0[2]), "=r"(_v8_0_0[3]), "=r"(_v8_0_0[4]), "=r"(_v8_0_0[5]), "=r"(_v8_0_0[6]), "=r"(_v8_0_0[7]) : "l"((const char*)_v8p_0 + 0) : "memory");
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 0]))
                                : "r"(_v8_0_0[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 2]))
                                : "r"(_v8_0_0[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 4]))
                                : "r"(_v8_0_0[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 6]))
                                : "r"(_v8_0_0[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 8]))
                                : "r"(_v8_0_0[4]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 10]))
                                : "r"(_v8_0_0[5]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 12]))
                                : "r"(_v8_0_0[6]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 14]))
                                : "r"(_v8_0_0[7]));
#else
                            (&values[0 + 0])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) << 16);
                            (&values[0 + 0])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[0]) & 0xffff0000u);
                            (&values[0 + 2])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) << 16);
                            (&values[0 + 2])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[1]) & 0xffff0000u);
                            (&values[0 + 4])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) << 16);
                            (&values[0 + 4])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[2]) & 0xffff0000u);
                            (&values[0 + 6])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) << 16);
                            (&values[0 + 6])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[3]) & 0xffff0000u);
                            (&values[0 + 8])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) << 16);
                            (&values[0 + 8])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[4]) & 0xffff0000u);
                            (&values[0 + 10])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) << 16);
                            (&values[0 + 10])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[5]) & 0xffff0000u);
                            (&values[0 + 12])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) << 16);
                            (&values[0 + 12])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[6]) & 0xffff0000u);
                            (&values[0 + 14])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) << 16);
                            (&values[0 + 14])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_0[7]) & 0xffff0000u);
#endif
                            uint32_t _v8_0_1[8];
                            asm volatile(
                                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_v8_0_1[0]), "=r"(_v8_0_1[1]), "=r"(_v8_0_1[2]), "=r"(_v8_0_1[3]), "=r"(_v8_0_1[4]), "=r"(_v8_0_1[5]), "=r"(_v8_0_1[6]), "=r"(_v8_0_1[7]) : "l"((const char*)_v8p_0 + 32) : "memory");
#if CAKE_PROGRAM_VARIANT == 1
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 16]))
                                : "r"(_v8_0_1[0]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 18]))
                                : "r"(_v8_0_1[1]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 20]))
                                : "r"(_v8_0_1[2]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 22]))
                                : "r"(_v8_0_1[3]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 24]))
                                : "r"(_v8_0_1[4]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 26]))
                                : "r"(_v8_0_1[5]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 28]))
                                : "r"(_v8_0_1[6]));
                            asm volatile(
                                "{\n\t"
                                ".reg .b16 h_lo, h_hi;\n\t"
                                ".reg .b32 f_lo, f_hi;\n\t"
                                "mov.b32 {h_lo, h_hi}, %1;\n\t"
                                "cvt.f32.f16 f_lo, h_lo;\n\t"
                                "cvt.f32.f16 f_hi, h_hi;\n\t"
                                "mov.b64 %0, {f_lo, f_hi};\n\t"
                                "}\n"
                                : "=l"(*reinterpret_cast<unsigned long long*>(&values[0 + 30]))
                                : "r"(_v8_0_1[7]));
#else
                            (&values[0 + 16])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[0]) << 16);
                            (&values[0 + 16])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[0]) & 0xffff0000u);
                            (&values[0 + 18])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[1]) << 16);
                            (&values[0 + 18])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[1]) & 0xffff0000u);
                            (&values[0 + 20])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[2]) << 16);
                            (&values[0 + 20])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[2]) & 0xffff0000u);
                            (&values[0 + 22])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[3]) << 16);
                            (&values[0 + 22])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[3]) & 0xffff0000u);
                            (&values[0 + 24])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[4]) << 16);
                            (&values[0 + 24])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[4]) & 0xffff0000u);
                            (&values[0 + 26])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[5]) << 16);
                            (&values[0 + 26])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[5]) & 0xffff0000u);
                            (&values[0 + 28])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[6]) << 16);
                            (&values[0 + 28])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[6]) & 0xffff0000u);
                            (&values[0 + 30])[0] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[7]) << 16);
                            (&values[0 + 30])[1] = __uint_as_float(static_cast<uint32_t>(_v8_0_1[7]) & 0xffff0000u);
#endif
                        }
                    } else {
                        #pragma unroll
                        for (int element = 0; element < 32; element++) {
                            values[element] = 0.0f;
                        }
                    }
                    float magnitudes[32];
                    #pragma unroll
                    for (int element_1 = 0; element_1 < 32; element_1++) {
                        magnitudes[element_1] = values[element_1];
                    }
                    magnitudes[0] = fabsf(magnitudes[0]);
                    magnitudes[1] = fabsf(magnitudes[1]);
                    magnitudes[2] = fabsf(magnitudes[2]);
                    magnitudes[3] = fabsf(magnitudes[3]);
                    magnitudes[4] = fabsf(magnitudes[4]);
                    magnitudes[5] = fabsf(magnitudes[5]);
                    magnitudes[6] = fabsf(magnitudes[6]);
                    magnitudes[7] = fabsf(magnitudes[7]);
                    magnitudes[8] = fabsf(magnitudes[8]);
                    magnitudes[9] = fabsf(magnitudes[9]);
                    magnitudes[10] = fabsf(magnitudes[10]);
                    magnitudes[11] = fabsf(magnitudes[11]);
                    magnitudes[12] = fabsf(magnitudes[12]);
                    magnitudes[13] = fabsf(magnitudes[13]);
                    magnitudes[14] = fabsf(magnitudes[14]);
                    magnitudes[15] = fabsf(magnitudes[15]);
                    magnitudes[16] = fabsf(magnitudes[16]);
                    magnitudes[17] = fabsf(magnitudes[17]);
                    magnitudes[18] = fabsf(magnitudes[18]);
                    magnitudes[19] = fabsf(magnitudes[19]);
                    magnitudes[20] = fabsf(magnitudes[20]);
                    magnitudes[21] = fabsf(magnitudes[21]);
                    magnitudes[22] = fabsf(magnitudes[22]);
                    magnitudes[23] = fabsf(magnitudes[23]);
                    magnitudes[24] = fabsf(magnitudes[24]);
                    magnitudes[25] = fabsf(magnitudes[25]);
                    magnitudes[26] = fabsf(magnitudes[26]);
                    magnitudes[27] = fabsf(magnitudes[27]);
                    magnitudes[28] = fabsf(magnitudes[28]);
                    magnitudes[29] = fabsf(magnitudes[29]);
                    magnitudes[30] = fabsf(magnitudes[30]);
                    magnitudes[31] = fabsf(magnitudes[31]);
                    float2 _reg_reduce_max2_1 = {-CAKE_INF, -CAKE_INF};
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[0], magnitudes[1]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[2], magnitudes[3]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[4], magnitudes[5]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[6], magnitudes[7]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[8], magnitudes[9]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[10], magnitudes[11]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[12], magnitudes[13]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[14], magnitudes[15]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[16], magnitudes[17]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[18], magnitudes[19]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[20], magnitudes[21]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[22], magnitudes[23]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[24], magnitudes[25]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[26], magnitudes[27]));
                    _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(magnitudes[28], magnitudes[29]));
                    _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(magnitudes[30], magnitudes[31]));
                    float absmax = row_max_reduce(_reg_reduce_max2_1);
                    float scale = absmax / 448.0f;
                    unsigned int scale_bits = __as_u32(scale);
                    unsigned int exponent = scale_bits >> 23 & 255;
                    unsigned int mantissa = scale_bits & 8388607;
                    unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
                    unsigned int normal = ((exponent != 0) ? 1 : 0);
                    unsigned int large_subnormal = ((mantissa > 4194304) ? 1 : 0);
                    unsigned int scale_byte = ((exponent + (has_mantissa & (normal | large_subnormal))) < (254) ? (exponent + (has_mantissa & (normal | large_subnormal))) : (254));
                    unsigned int inverse_nonzero_bits = 254 - scale_byte << 23;
                    unsigned int zero_bits = 0;
                    unsigned int inverse_bits = ((scale_byte == 0) ? zero_bits : inverse_nonzero_bits);
                    float inverse = 0.0f;
                    inverse = reinterpret_cast<float*>(&inverse_bits)[0];
                    const float2 _scale2_2 = {inverse, inverse};
                    #pragma unroll
                    for (int _ls = 0; _ls < 16; _ls++)
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(values)[_ls], _scale2_2);
                    unsigned long long task = global_row * (unsigned long long)((uint32_t)(BLOCKS_PER_ROW.divisor)) + (unsigned long long)_fast_div_r_0;
                    unsigned long long q_offset = task * 32;
                    unsigned int _packed_0[8];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[0]), "f"(values[1]),
                                               "f"(values[2]), "f"(values[3]));
                        _packed_0[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[4]), "f"(values[5]),
                                               "f"(values[6]), "f"(values[7]));
                        _packed_0[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[8]), "f"(values[9]),
                                               "f"(values[10]), "f"(values[11]));
                        _packed_0[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[12]), "f"(values[13]),
                                               "f"(values[14]), "f"(values[15]));
                        _packed_0[3] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[16]), "f"(values[17]),
                                               "f"(values[18]), "f"(values[19]));
                        _packed_0[4] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[20]), "f"(values[21]),
                                               "f"(values[22]), "f"(values[23]));
                        _packed_0[5] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[24]), "f"(values[25]),
                                               "f"(values[26]), "f"(values[27]));
                        _packed_0[6] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(values[28]), "f"(values[29]),
                                               "f"(values[30]), "f"(values[31]));
                        _packed_0[7] = _packed;
                    }
                    {
                        const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(_packed_0);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(quantized + q_offset)), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                    }
                    unsigned int m_tile = row / 128;
                    unsigned int row_in_tile = row - m_tile * 128;
                    unsigned int k_tile = _fast_div_r_0 / 4;
                    unsigned int scale_col = _fast_div_r_0 - k_tile * 4;
                    unsigned long long tile_linear = ((unsigned long long)group * (unsigned long long)PM_TILES + (unsigned long long)m_tile) * (unsigned long long)PK_TILES + (unsigned long long)k_tile;
                    unsigned long long scale_offset = tile_linear * 512 + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4 + (unsigned long long)scale_col;
                    *(reinterpret_cast<unsigned char*>(scales + scale_offset) + (0)) = (unsigned char)(scale_byte);
                }
                row = row + SUBROWS;
                global_row = global_row + (unsigned long long)SUBROWS;
            }
        }
    }
}

} // extern "C"
