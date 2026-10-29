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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128
#define QUANT_UNITS 4
#define SF_TILE_BYTES 512

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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
kernel_cake_kimi_k3_fp8_projection_e47f4efd083cf458e167(__nv_bfloat16* __restrict__ x, uint8_t* __restrict__ a_q, uint8_t* __restrict__ a_sf, int M, int K, int units_per_row, int sf_k_sets, int sf_rows)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int half = lane >> 4;
    int half_lane = lane & 15;
    int unit = (bid * 4 + warp) * 2 + half;
    int row = unit / units_per_row;
    int row_c = ((row) < (M - 1) ? (row) : (M - 1));
    int k_block0 = (unit - row * units_per_row) * QUANT_UNITS;
    unsigned long long row_base = (unsigned long long)row_c * (unsigned long long)K;
    int m_tile = row_c / sf_rows;
    int row_in_tile = row_c - m_tile * sf_rows;
    unsigned long long sf_row_base = (unsigned long long)m_tile * (unsigned long long)sf_k_sets * (unsigned long long)SF_TILE_BYTES + (unsigned long long)(row_in_tile & 31) * 16 + (unsigned long long)(row_in_tile >> 5) * 4;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    float staged[32];
    #pragma unroll
    for (int u = 0; u < QUANT_UNITS; u++) {
        int k_ld = (k_block0 + u) * 128 + half_lane * 8;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (row_base + (unsigned long long)k_ld) + 0);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                asm volatile("ld.global.nc.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                    : "=r"(_vld_0[_blk].x), "=r"(_vld_0[_blk].y), "=r"(_vld_0[_blk].z), "=r"(_vld_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)), "l"(0x12F0000000000000ULL) : "memory");
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            staged[u * 8 + j] = _vec_load_0[j];
        }
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < QUANT_UNITS; u_1++) {
        int k_block = k_block0 + u_1;
        int k = k_block * 128 + half_lane * 8;
        float vals[8];
        float mags[8];
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            vals[j_1] = staged[u_1 * 8 + j_1];
            mags[j_1] = staged[u_1 * 8 + j_1];
        }
        mags[0] = fabsf(mags[0]);
        mags[1] = fabsf(mags[1]);
        mags[2] = fabsf(mags[2]);
        mags[3] = fabsf(mags[3]);
        mags[4] = fabsf(mags[4]);
        mags[5] = fabsf(mags[5]);
        mags[6] = fabsf(mags[6]);
        mags[7] = fabsf(mags[7]);
        float absmax = mags[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            absmax = max_noftz(absmax, mags[_lr]);
        }
        float o1 = __shfl_xor_sync(0xFFFFFFFF, absmax, 1);
        absmax = fmaxf(absmax, o1);
        float o2 = __shfl_xor_sync(0xFFFFFFFF, absmax, 2);
        absmax = fmaxf(absmax, o2);
        float o4 = __shfl_xor_sync(0xFFFFFFFF, absmax, 4);
        absmax = fmaxf(absmax, o4);
        float o8 = __shfl_xor_sync(0xFFFFFFFF, absmax, 8);
        absmax = fmaxf(absmax, o8);
        absmax = fmaxf(absmax, 1e-04f);
        float scale = absmax / 448.0f;
        unsigned int scale_bits = __as_u32(scale);
        unsigned int exponent = scale_bits >> 23 & 255;
        unsigned int mantissa = scale_bits & 8388607;
        unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
        unsigned int _max_1 = ((exponent + has_mantissa) > (1) ? (exponent + has_mantissa) : (1));
        unsigned int scale_byte = ((_max_1) < (254) ? (_max_1) : (254));
        unsigned int inverse_bits = 254 - scale_byte << 23;
        float inverse = 0.0f;
        inverse = reinterpret_cast<float*>(&inverse_bits)[0];
        const float2 _scale2_2 = {inverse, inverse};
        #pragma unroll
        for (int _ls = 0; _ls < 4; _ls++)
            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(vals)[_ls], _scale2_2);
        if (row < M) {
            unsigned int _packed_0[2];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(vals[0]), "f"(vals[1]),
                                       "f"(vals[2]), "f"(vals[3]));
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
                    : "=r"(_packed) : "f"(vals[4]), "f"(vals[5]),
                                       "f"(vals[6]), "f"(vals[7]));
                _packed_0[1] = _packed;
            }
            asm volatile("st.global.L2::cache_hint.v2.b32 [%0], {%1, %2}, %3;" :: "l"(a_q + (row_base + (unsigned long long)k)), "r"(_packed_0[(0) + 0]), "r"(_packed_0[(0) + 1]), "l"(0x14F0000000000000ULL) : "memory");
            if (half_lane == 0) {
                unsigned long long sf_addr = sf_row_base + (unsigned long long)k_block * (unsigned long long)SF_TILE_BYTES;
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    asm volatile("st.global.L2::cache_hint.b8 [%0], %1, %2;" :: "l"(a_sf + (sf_addr + (unsigned long long)c)), "r"((unsigned int)(unsigned char)(scale_byte)), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
    }
}

} // extern "C"
