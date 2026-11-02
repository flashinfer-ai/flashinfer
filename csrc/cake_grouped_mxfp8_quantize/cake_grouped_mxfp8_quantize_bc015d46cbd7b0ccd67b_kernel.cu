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

// This unit serves two programs: cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b (CAKE_PROGRAM_VARIANT 0, the default) and
// cake_grouped_mxfp8_quantize_977b5e967e2415146d6d (CAKE_PROGRAM_VARIANT 1).  CAKE_PROGRAM_KERNEL names the kernel it defines.  A build
// selects the second program with the compile definitions
// -DCAKE_PROGRAM_VARIANT=1 -DCAKE_PROGRAM_KERNEL=kernel_cake_grouped_mxfp8_quantize_977b5e967e2415146d6d.
#ifndef CAKE_PROGRAM_VARIANT
#define CAKE_PROGRAM_VARIANT 0
#endif
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_grouped_mxfp8_quantize_bc015d46cbd7b0ccd67b
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
#define NUM_PIPE_STAGES 1
#define SMEM_TILE_OFF 128
#define SMEM_TILE_STAGE_BYTES 8192
#define SMEM_TILE_STRIDE 8192
#define SMEM_TOTAL 8320
#define THREADS 128

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


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



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


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
CAKE_PROGRAM_KERNEL(__nv_bfloat16* __restrict__ x, int* __restrict__ mask, uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales, int M, int K, int PADDED_K, int PM_TILES, int PK_TILES, int BLOCKS_PER_ROW, unsigned long long TOTAL_TASKS)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* tile = reinterpret_cast<unsigned int*>(smem_raw + 128);
    const int tile_addr = smem + 128;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    unsigned int group = blockIdx.z;
    unsigned int row = blockIdx.y;
    unsigned int tid_0 = (unsigned int)tid;
    unsigned int first_block = blockIdx.x * 128;
    unsigned int block_col = first_block + tid_0;
    unsigned int real_blocks = (unsigned int)K >> 5;
    unsigned long long global_row = (unsigned long long)group * (unsigned long long)M + (unsigned long long)row;
    unsigned int _min_0 = ((real_blocks - first_block) < (128) ? (real_blocks - first_block) : (128));
    unsigned int tile_bytes = _min_0 * 64;
    if (warp == 0) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(full_addr, tile_bytes);
            cp_async_bulk_gmem2smem(tile_addr + (unsigned int)(0 * 8192), x + (global_row * (unsigned long long)K + (unsigned long long)first_block * 32), tile_bytes, full_addr);
        }
    }
    int _vec_load_0[1];
    {
        _vec_load_0[0] = *reinterpret_cast<const int*>(mask + group);
    }
    unsigned int valid_rows = (unsigned int)_vec_load_0[0];
    unsigned int _phase_full = 0;
    mbarrier_wait(full_addr, _phase_full);
    if (block_col < (unsigned int)BLOCKS_PER_ROW) {
        if (row < valid_rows) {
            unsigned int flip = tid_0 >> 1 & 3;
            unsigned int words[16];
            if (block_col < real_blocks) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"(tile_addr + (unsigned int)(0 * 8192) + tid_0 * 64 + ((0 ^ flip) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[4])), "=r"(*reinterpret_cast<uint32_t*>(&words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(4) + 3]))
                    : "r"(tile_addr + (unsigned int)(0 * 8192) + tid_0 * 64 + ((1 ^ flip) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[8])), "=r"(*reinterpret_cast<uint32_t*>(&words[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(8) + 3]))
                    : "r"(tile_addr + (unsigned int)(0 * 8192) + tid_0 * 64 + ((2 ^ flip) << 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[12])), "=r"(*reinterpret_cast<uint32_t*>(&words[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(12) + 3]))
                    : "r"(tile_addr + (unsigned int)(0 * 8192) + tid_0 * 64 + ((3 ^ flip) << 4)));
            } else {
                words[0] = 0;
                words[1] = 0;
                words[2] = 0;
                words[3] = 0;
                words[4] = 0;
                words[5] = 0;
                words[6] = 0;
                words[7] = 0;
                words[8] = 0;
                words[9] = 0;
                words[10] = 0;
                words[11] = 0;
                words[12] = 0;
                words[13] = 0;
                words[14] = 0;
                words[15] = 0;
            }
            float values[32];
            #pragma unroll
            for (int _pair = 0; _pair < 16; _pair++) {
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
                    : "=l"(*reinterpret_cast<unsigned long long*>(&values[_pair * 2]))
                    : "r"(words[_pair]));
#else
                (&values[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[_pair]) << 16);
                (&values[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[_pair]) & 0xffff0000u);
#endif
            }
            float magnitudes[32];
            #pragma unroll
            for (int element = 0; element < 32; element++) {
                magnitudes[element] = values[element];
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
            unsigned int loaded[8];
            uint16_t _cvt_satfinite_e4m3x2_f32_3;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_3) : "f"(values[1]), "f"(values[0]));
            unsigned int low0_1_0 = _cvt_satfinite_e4m3x2_f32_3;
            uint16_t _cvt_satfinite_e4m3x2_f32_4;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_4) : "f"(values[3]), "f"(values[2]));
            unsigned int high0_1_0 = _cvt_satfinite_e4m3x2_f32_4;
            loaded[0] = low0_1_0 | high0_1_0 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_5;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_5) : "f"(values[5]), "f"(values[4]));
            unsigned int low0_1_1 = _cvt_satfinite_e4m3x2_f32_5;
            uint16_t _cvt_satfinite_e4m3x2_f32_6;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_6) : "f"(values[7]), "f"(values[6]));
            unsigned int high0_1_1 = _cvt_satfinite_e4m3x2_f32_6;
            loaded[1] = low0_1_1 | high0_1_1 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_7;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_7) : "f"(values[9]), "f"(values[8]));
            unsigned int low0_1_2 = _cvt_satfinite_e4m3x2_f32_7;
            uint16_t _cvt_satfinite_e4m3x2_f32_8;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_8) : "f"(values[11]), "f"(values[10]));
            unsigned int high0_1_2 = _cvt_satfinite_e4m3x2_f32_8;
            loaded[2] = low0_1_2 | high0_1_2 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_9;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_9) : "f"(values[13]), "f"(values[12]));
            unsigned int low0_1_3 = _cvt_satfinite_e4m3x2_f32_9;
            uint16_t _cvt_satfinite_e4m3x2_f32_10;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_10) : "f"(values[15]), "f"(values[14]));
            unsigned int high0_1_3 = _cvt_satfinite_e4m3x2_f32_10;
            loaded[3] = low0_1_3 | high0_1_3 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_11;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_11) : "f"(values[17]), "f"(values[16]));
            unsigned int low0_1_4 = _cvt_satfinite_e4m3x2_f32_11;
            uint16_t _cvt_satfinite_e4m3x2_f32_12;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_12) : "f"(values[19]), "f"(values[18]));
            unsigned int high0_1_4 = _cvt_satfinite_e4m3x2_f32_12;
            loaded[4] = low0_1_4 | high0_1_4 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_13;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_13) : "f"(values[21]), "f"(values[20]));
            unsigned int low0_1_5 = _cvt_satfinite_e4m3x2_f32_13;
            uint16_t _cvt_satfinite_e4m3x2_f32_14;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_14) : "f"(values[23]), "f"(values[22]));
            unsigned int high0_1_5 = _cvt_satfinite_e4m3x2_f32_14;
            loaded[5] = low0_1_5 | high0_1_5 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_15;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_15) : "f"(values[25]), "f"(values[24]));
            unsigned int low0_1_6 = _cvt_satfinite_e4m3x2_f32_15;
            uint16_t _cvt_satfinite_e4m3x2_f32_16;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_16) : "f"(values[27]), "f"(values[26]));
            unsigned int high0_1_6 = _cvt_satfinite_e4m3x2_f32_16;
            loaded[6] = low0_1_6 | high0_1_6 << 16;
            uint16_t _cvt_satfinite_e4m3x2_f32_17;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_17) : "f"(values[29]), "f"(values[28]));
            unsigned int low0_1_7 = _cvt_satfinite_e4m3x2_f32_17;
            uint16_t _cvt_satfinite_e4m3x2_f32_18;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_18) : "f"(values[31]), "f"(values[30]));
            unsigned int high0_1_7 = _cvt_satfinite_e4m3x2_f32_18;
            loaded[7] = low0_1_7 | high0_1_7 << 16;
            unsigned int swap_one = flip & 1;
            unsigned int swap_two = flip & 2;
            unsigned int half[8];
            half[0] = ((swap_one != 0) ? loaded[2] : loaded[0]);
            half[1] = ((swap_one != 0) ? loaded[3] : loaded[1]);
            half[2] = ((swap_one != 0) ? loaded[0] : loaded[2]);
            half[3] = ((swap_one != 0) ? loaded[1] : loaded[3]);
            half[4] = ((swap_one != 0) ? loaded[6] : loaded[4]);
            half[5] = ((swap_one != 0) ? loaded[7] : loaded[5]);
            half[6] = ((swap_one != 0) ? loaded[4] : loaded[6]);
            half[7] = ((swap_one != 0) ? loaded[5] : loaded[7]);
            unsigned int codes[8];
            codes[0] = ((swap_two != 0) ? half[4] : half[0]);
            codes[1] = ((swap_two != 0) ? half[5] : half[1]);
            codes[2] = ((swap_two != 0) ? half[6] : half[2]);
            codes[3] = ((swap_two != 0) ? half[7] : half[3]);
            codes[4] = ((swap_two != 0) ? half[0] : half[4]);
            codes[5] = ((swap_two != 0) ? half[1] : half[5]);
            codes[6] = ((swap_two != 0) ? half[2] : half[6]);
            codes[7] = ((swap_two != 0) ? half[3] : half[7]);
            unsigned long long task = global_row * (unsigned long long)BLOCKS_PER_ROW + (unsigned long long)block_col;
            unsigned long long q_offset = task * 32;
            {
                const unsigned* _raw_stv8_19 = reinterpret_cast<const unsigned*>(codes);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(reinterpret_cast<unsigned int*>(quantized + q_offset))), "r"(_raw_stv8_19[0]), "r"(_raw_stv8_19[1]), "r"(_raw_stv8_19[2]), "r"(_raw_stv8_19[3]), "r"(_raw_stv8_19[4]), "r"(_raw_stv8_19[5]), "r"(_raw_stv8_19[6]), "r"(_raw_stv8_19[7]) : "memory");
            }
            unsigned int m_tile = row / 128;
            unsigned int row_in_tile = row - m_tile * 128;
            unsigned int k_tile = block_col / 4;
            unsigned int scale_col = block_col - k_tile * 4;
            unsigned long long tile_linear = ((unsigned long long)group * (unsigned long long)PM_TILES + (unsigned long long)m_tile) * (unsigned long long)PK_TILES + (unsigned long long)k_tile;
            unsigned long long scale_offset = tile_linear * 512 + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4 + (unsigned long long)scale_col;
            *(reinterpret_cast<unsigned char*>(scales + scale_offset) + (0)) = (unsigned char)(scale_byte);
        }
    }
    _phase_full ^= 1;

    // Cleanup
    __syncthreads();
}

} // extern "C"
