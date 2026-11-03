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
#define SMEM_SMEM_RSTD_OFF 0
#define SMEM_SMEM_RSTD_STAGE_BYTES 4
#define SMEM_SMEM_RSTD_STRIDE 4
#define SMEM_SMEM_INDEX_OFF 4
#define SMEM_SMEM_INDEX_STAGE_BYTES 4
#define SMEM_SMEM_INDEX_STRIDE 4
#define SMEM_SMEM_PARTIAL_OFF 8
#define SMEM_SMEM_PARTIAL_STAGE_BYTES 84
#define SMEM_SMEM_PARTIAL_STRIDE 84
#define SMEM_TOTAL 128
#define THREADS 672
#define HIDDEN 5376
#define SCALE_COLS 168

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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

__global__ __launch_bounds__(672) void
kernel_cake_minimax_h3_mxfp8_pre_attention_0ca9783189af24b14501(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, uint8_t* __restrict__ activation_q, uint8_t* __restrict__ activation_sf, float eps, int M)
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
    float* smem_rstd = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_rstd_addr = smem + 0;
    int* smem_index = reinterpret_cast<int*>(smem_raw + 4);
    const int smem_index_addr = smem + 4;
    float* smem_partial = reinterpret_cast<float*>(smem_raw + 8);
    const int smem_partial_addr = smem + 8;

    // === Task calls (declaration order) ===
    int thread = warp * 32 + lane;
    int row = bid;
    int col = thread * 8;
    unsigned int x_words[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        x_words[j] = 0;
    }
    float sum_sq = 0.0f;
    if (row < M) {
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_words[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 4; j_1++) {
            float x_lo = __uint_as_float(x_words[j_1] << 16);
            float x_hi = __uint_as_float(x_words[j_1] & 4294901760u);
            sum_sq = __fmaf_rn(x_lo, x_lo, sum_sq);
            sum_sq = __fmaf_rn(x_hi, x_hi, sum_sq);
        }
    }
    float _shfl_xor_sync_1 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
    sum_sq += _shfl_xor_sync_1;
    float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
    sum_sq += _shfl_xor_sync_2;
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
    sum_sq += _shfl_xor_sync_3;
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
    sum_sq += _shfl_xor_sync_4;
    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
    sum_sq += _shfl_xor_sync_5;
    if (lane == 0) {
        smem_partial[warp] = sum_sq;
    }
    asm volatile("barrier.sync 1, 672;" ::: "memory");
    if (warp == 0) {
        float total_sq = 0.0f;
        if (lane < 21) {
            total_sq = smem_partial[lane];
        }
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, total_sq, 16);
        total_sq += _shfl_xor_sync_6;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, total_sq, 8);
        total_sq += _shfl_xor_sync_7;
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total_sq, 4);
        total_sq += _shfl_xor_sync_8;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total_sq, 2);
        total_sq += _shfl_xor_sync_9;
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total_sq, 1);
        total_sq += _shfl_xor_sync_10;
        if (lane == 0) {
            if (row < M) {
                float _rsqrt_11;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(_rsqrt_11) : "f"(total_sq / (float)HIDDEN + eps));
                smem_rstd[0] = _rsqrt_11;
                smem_index[0] = adaln_index[row];
            } else {
                smem_rstd[0] = 0.0f;
                smem_index[0] = 0;
            }
        }
    }
    asm volatile("barrier.sync 1, 672;" ::: "memory");
    if (row < M) {
        int table_row = smem_index[0];
        float values[8];
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            values[j_2] = 0.0f;
        }
        if (table_row >= 0 && table_row < 9) {
            unsigned long long table_base = (unsigned long long)table_row * (unsigned long long)HIDDEN;
            unsigned int words[4];
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(x_norm_weight + col);
                uint4* _vdst_12 = reinterpret_cast<uint4*>(&words[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_12[_blk] = _vptr_12[_blk];
                }
            }
            unsigned int words_0[4];
            {
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(adaln_scale + table_base + (unsigned long long)col);
                uint4* _vdst_13 = reinterpret_cast<uint4*>(&words_0[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_13[_blk] = _vptr_13[_blk];
                }
            }
            unsigned int words_1[4];
            {
                const uint4* _vptr_14 = reinterpret_cast<const uint4*>(adaln_shift + table_base + (unsigned long long)col);
                uint4* _vdst_14 = reinterpret_cast<uint4*>(&words_1[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_14[_blk] = _vptr_14[_blk];
                }
            }
            float rstd = smem_rstd[0];
            float2 _f2_0 = make_float2(rstd, rstd);
            float2 _f2_1 = make_float2(1.0f, 1.0f);
            #pragma unroll
            for (int j_3 = 0; j_3 < 4; j_3++) {
                float2 _f32x2_mul_a_15 = make_float2(__uint_as_float(x_words[j_3] << 16), __uint_as_float(x_words[j_3] & 4294901760u));
                float2 scaled;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f2_0));
                float2 _f32x2_mul_b_16 = make_float2(__uint_as_float(words[j_3] << 16), __uint_as_float(words[j_3] & 4294901760u));
                float2 weighted;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
                unsigned int normalized_word = __as_u32(__float22bfloat162_rn(make_float2(weighted.x, weighted.y)));
                float2 scale_plus_one = add_f32x2(make_float2(__uint_as_float(words_0[j_3] << 16), __uint_as_float(words_0[j_3] & 4294901760u)), _f2_1);
                unsigned int scale_word = __as_u32(__float22bfloat162_rn(make_float2(scale_plus_one.x, scale_plus_one.y)));
                float2 adaln = fma_f32x2_rn_ftz(make_float2(__uint_as_float(normalized_word << 16), __uint_as_float(normalized_word & 4294901760u)), make_float2(__uint_as_float(scale_word << 16), __uint_as_float(scale_word & 4294901760u)), make_float2(__uint_as_float(words_1[j_3] << 16), __uint_as_float(words_1[j_3] & 4294901760u)));
                values[2 * j_3] = (float)(__nv_bfloat16)adaln.x;
                values[2 * j_3 + 1] = (float)(__nv_bfloat16)adaln.y;
            }
        }
        float amax = fabsf(values[0]);
        #pragma unroll
        for (int j_4 = 1; j_4 < 8; j_4++) {
            amax = max_noftz(amax, fabsf(values[j_4]));
        }
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
        amax = max_noftz(amax, _shfl_xor_sync_17);
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, amax, 2);
        amax = max_noftz(amax, _shfl_xor_sync_18);
        float _rcp_19 = approx_rcp(448.0f);
        float block_scale = amax * _rcp_19;
        unsigned int scale_bits = __as_u32(block_scale);
        unsigned int exponent = scale_bits >> 23 & 255;
        unsigned int mantissa = scale_bits & 8388607;
        unsigned int has_mantissa = ((mantissa != 0) ? 1 : 0);
        unsigned int normal = ((exponent != 0) ? 1 : 0);
        unsigned int large_subnormal = ((mantissa > 4194304) ? 1 : 0);
        unsigned int code_raw = ((exponent + (has_mantissa & (normal | large_subnormal))) < (254) ? (exponent + (has_mantissa & (normal | large_subnormal))) : (254));
        unsigned int code_zero = 0;
        unsigned int code = ((block_scale > 0.0f) ? code_raw : code_zero);
        float scale_value = __uint_as_float(code << 23);
        float rcp_scale = approx_rcp(scale_value);
        float inverse = ((scale_value != 0.0f) ? rcp_scale : 0.0f);
        float2 _f2_2 = make_float2(inverse, inverse);
        float quant[8];
        #pragma unroll
        for (int j_5 = 0; j_5 < 4; j_5++) {
            float2 _f32x2_mul_a_20 = make_float2(values[2 * j_5], values[2 * j_5 + 1]);
            float2 scaled_q;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_q) : "l"(*(const unsigned long long*)&_f32x2_mul_a_20), "l"(*(const unsigned long long*)&_f2_2));
            quant[2 * j_5] = scaled_q.x;
            quant[2 * j_5 + 1] = scaled_q.y;
        }
        unsigned int e4m3_words[2];
        {
            uint32_t _packed;
            asm volatile("{\n\t"
                ".reg .b16 _lo;\n\t"
                ".reg .b16 _hi;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                "mov.b32 %0, {_lo, _hi};\n\t"
                "}"
                : "=r"(_packed) : "f"(quant[0]), "f"(quant[1]),
                                   "f"(quant[2]), "f"(quant[3]));
            e4m3_words[0] = _packed;
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
                : "=r"(_packed) : "f"(quant[4]), "f"(quant[5]),
                                   "f"(quant[6]), "f"(quant[7]));
            e4m3_words[1] = _packed;
        }
        reinterpret_cast<int2*>(activation_q + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)col))[0] = reinterpret_cast<int2*>(e4m3_words)[0];
        int scale_col = thread / 4;
        if (thread % 4 == 0) {
            unsigned long long scale_offset = (unsigned long long)(scale_col % 4 + scale_col / 4 * 512 + row % 32 * 16 + row % 128 / 32 * 4) + (unsigned long long)(row / 128) * (unsigned long long)(128 * SCALE_COLS);
            *(reinterpret_cast<unsigned char*>(activation_sf + scale_offset) + (0)) = (unsigned char)(code);
        }
    }
}

} // extern "C"
