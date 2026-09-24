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
#define SMEM_RING_OFF 0
#define SMEM_RING_STAGE_BYTES 28672
#define SMEM_RING_STRIDE 28672
#define SMEM_TOTAL 28672
#define THREADS 256
#define P 8
#define HEADS_PER_DESTINATION 7

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


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}


__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
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

__global__ __launch_bounds__(256, 4) void
kernel_cake_minimax_h3_nvfp4_pre_attention_341330a98b7380153c4b(__nv_bfloat16* __restrict__ qkv_bf16, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, __nv_bfloat16* __restrict__ debug_q_bf16, __nv_bfloat16* __restrict__ debug_k_bf16, int write_debug, float eps, int M, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    unsigned int* ring = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int ring_addr = smem + 0;

    // === Task calls (dependency order) ===
    int row_slot = lane / 8;
    int lane8 = lane % 8;
    int rows_per_token = HEADS_PER_DESTINATION * 3;
    int token_groups = (M + 16 - 1) / 16;
    int warps_per_destination = token_groups * rows_per_token;
    int destination = blockIdx.y;
    int warp_in_destination = bid * 8 + warp;
    if (warp_in_destination < warps_per_destination) {
        int token_group = warp_in_destination / rows_per_token;
        int head_kind = warp_in_destination % rows_per_token;
        int local_head = head_kind / 3;
        int kind = head_kind % 3;
        int head = destination * HEADS_PER_DESTINATION + local_head;
        int group_token = token_group * 16;
        int dim = lane8 * 16;
        int rope_col = ((dim < 48) ? dim : dim - 48);
        rope_col = ((dim < 96) ? rope_col : 0);
        int stages_smem = ring_addr + (unsigned int)(warp * 3584);
        int source_lane_off = row_slot * 256 + lane8 * 32;
        int table_row_off = 1024 + row_slot * 192;
        int row_base = destination * ROWS_PER_DESTINATION + head_kind;
        int lane_q_off = lane8 * 8;
        unsigned long long scale_lane_base = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)(lane8 / 4 * 512) + (unsigned long long)(lane8 % 4);
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int token_base = group_token + s * 4;
            int token = token_base + row_slot;
            int load_token = ((token < M) ? token : M - 1);
            unsigned long long source_base = (((unsigned long long)load_token * 56 + (unsigned long long)head) * 3 + (unsigned long long)kind) * 128 + (unsigned long long)dim;
            unsigned long long table_src = (unsigned long long)load_token * 96 + (unsigned long long)(lane8 * 16);
            int stage_smem = stages_smem + s * 1792;
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((token_base < M) ? 1 : 0), "r"(stage_smem + source_lane_off + h * 16), "l"(qkv_bf16 + (source_base + (unsigned long long)(h * 8))));
            }
            #pragma unroll
            for (int h_1 = 0; h_1 < 2; h_1++) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((token_base < M && (lane8 < 6 && kind < 2)) ? 1 : 0), "r"(stage_smem + table_row_off + lane8 * 32 + h_1 * 16), "l"(rope_cos_sin + (table_src + (unsigned long long)(h_1 * 8))));
            }
            asm volatile("cp.async.commit_group;");
        }
        unsigned int weight_words[8];
        if (kind < 2) {
            #pragma unroll
            for (int h_2 = 0; h_2 < 2; h_2++) {
                {
                    const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + dim + h_2 * 8);
                    uint4* _vdst_0 = reinterpret_cast<uint4*>(&weight_words[4 * h_2]);
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vdst_0[_blk] = _vptr_0[_blk];
                    }
                }
            }
        }
        unsigned int sin_sign = ((lane8 < 3) ? (unsigned int)2147483648 : (unsigned int)0);
        int partner_lane = ((lane8 < 3) ? lane + 3 : ((lane8 < 6) ? lane - 3 : lane));
        float global_scale = out_global_scale[0];
        float _rcp_0 = approx_rcp(global_scale);
        float global_scale_rcp = _rcp_0;
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            asm volatile("cp.async.wait_group 1;");
            __syncwarp();
            int token_base_1 = group_token + q * 4;
            int token_1 = token_base_1 + row_slot;
            if (token_base_1 < M) {
                int stage_smem_1 = stages_smem + q % 2 * 1792;
                int row_in_destination = token_1 * rows_per_token + head_kind;
                int output_row = row_base + token_1 * rows_per_token;
                unsigned int source_words[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&source_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&source_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&source_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&source_words[(0) + 3]))
                    : "r"(stage_smem_1 + source_lane_off));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&source_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&source_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&source_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&source_words[(4) + 3]))
                    : "r"(stage_smem_1 + source_lane_off + 16));
                float values[16];
                float absolute[16];
                float quant_values[16];
                unsigned int packed[2];
                if (kind < 2) {
                    int table_smem = stage_smem_1 + table_row_off + rope_col * 2;
                    unsigned int cos_words[8];
                    unsigned int sin_words[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&cos_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&cos_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cos_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cos_words[(0) + 3]))
                        : "r"(table_smem));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&cos_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&cos_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&cos_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&cos_words[(4) + 3]))
                        : "r"(table_smem + 16));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sin_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&sin_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sin_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sin_words[(0) + 3]))
                        : "r"(table_smem + 96));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&sin_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&sin_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sin_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sin_words[(4) + 3]))
                        : "r"(table_smem + 96 + 16));
                    float source_values[16];
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        source_values[2 * j] = __uint_as_float(source_words[j] << 16);
                        source_values[2 * j + 1] = __uint_as_float(source_words[j] & 4294901760);
                    }
                    float sum_lo = 0.0f;
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        float _fma_0 = __fmaf_rn(source_values[j_1], source_values[j_1], sum_lo);
                        sum_lo = _fma_0;
                    }
                    float sum_hi = 0.0f;
                    #pragma unroll
                    for (int j_2 = 8; j_2 < 16; j_2++) {
                        float _fma_1 = __fmaf_rn(source_values[j_2], source_values[j_2], sum_hi);
                        sum_hi = _fma_1;
                    }
                    float sum_sq = sum_lo + sum_hi;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
                    sum_sq += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
                    sum_sq += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
                    sum_sq += _shfl_xor_2;
                    float _fdiv_rn_0 = __fdiv_rn(sum_sq, 128.0f);
                    float mean_sq = _fdiv_rn_0;
                    float _rsqrt_0 = rsqrtf(mean_sq + eps);
                    float rstd = _rsqrt_0;
                    float2 _f2_0 = make_float2(rstd, rstd);
                    unsigned int normalized_words[8];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 8; j_3++) {
                        float2 _f2_1 = make_float2(source_values[2 * j_3], source_values[2 * j_3 + 1]);
                        float2 _mul_f32x2_0;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f2_2 = make_float2(__uint_as_float(weight_words[j_3] << 16), __uint_as_float(weight_words[j_3] & 4294901760));
                        float2 _mul_f32x2_1;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_mul_f32x2_0), "l"(*(const unsigned long long*)&_f2_2));
                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_mul_f32x2_1.x, _mul_f32x2_1.y));
                        normalized_words[j_3] = __as_u32(_bf16x2_0);
                    }
                    unsigned int output_words[8];
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 8; j_4++) {
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, normalized_words[j_4], partner_lane);
                        unsigned int partner_word = _shfl_0;
                        float2 _f2_3 = make_float2(__uint_as_float(sin_words[j_4] << 16 ^ sin_sign), __uint_as_float(sin_words[j_4] & 4294901760 ^ sin_sign));
                        float2 _f2_4 = make_float2(__uint_as_float(partner_word << 16), __uint_as_float(partner_word & 4294901760));
                        float2 _mul_f32x2_2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_4), "l"(*(const unsigned long long*)&_f2_3));
                        float2 _f2_5 = make_float2(__uint_as_float(cos_words[j_4] << 16), __uint_as_float(cos_words[j_4] & 4294901760));
                        float2 _f2_6 = make_float2(__uint_as_float(normalized_words[j_4] << 16), __uint_as_float(normalized_words[j_4] & 4294901760));
                        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                        #error "Packed FP32x2 arithmetic requires SM100 or newer"
                        #endif
                        float2 _packed_fma_f32x2_0;
                        {
                            float2 _packed_f32x2_1_0 = _f2_5;
                            float2 _packed_f32x2_1_1 = _f2_6;
                            float2 _packed_f32x2_1_2 = _mul_f32x2_2;
                            asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_0)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_1_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_1_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_1_2)));
                        }
                        __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_0.x, _packed_fma_f32x2_0.y));
                        unsigned int rotated_word = __as_u32(_bf16x2_1);
                        output_words[j_4] = ((dim < 96) ? rotated_word : normalized_words[j_4]);
                    }
                    if (write_debug != 0) {
                        if (token_1 < M) {
                            float debug_values[16];
                            #pragma unroll
                            for (int j_5 = 0; j_5 < 8; j_5++) {
                                debug_values[2 * j_5] = __uint_as_float(output_words[j_5] << 16);
                                debug_values[2 * j_5 + 1] = __uint_as_float(output_words[j_5] & 4294901760);
                            }
                            unsigned long long debug_base = ((unsigned long long)token_1 * 56 + (unsigned long long)head) * 128 + (unsigned long long)dim;
                            if (kind == 0) {
                                {
                                    __nv_bfloat162 _pk[8];
                                    _pk[0] = __floats2bfloat162_rn(debug_values[0 + 0], debug_values[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(debug_values[0 + 2], debug_values[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(debug_values[0 + 4], debug_values[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(debug_values[0 + 6], debug_values[0 + 7]);
                                    _pk[4] = __floats2bfloat162_rn(debug_values[0 + 8], debug_values[0 + 9]);
                                    _pk[5] = __floats2bfloat162_rn(debug_values[0 + 10], debug_values[0 + 11]);
                                    _pk[6] = __floats2bfloat162_rn(debug_values[0 + 12], debug_values[0 + 13]);
                                    _pk[7] = __floats2bfloat162_rn(debug_values[0 + 14], debug_values[0 + 15]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_q_bf16 + debug_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_q_bf16 + debug_base))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                }
                            } else {
                                {
                                    __nv_bfloat162 _pk[8];
                                    _pk[0] = __floats2bfloat162_rn(debug_values[0 + 0], debug_values[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(debug_values[0 + 2], debug_values[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(debug_values[0 + 4], debug_values[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(debug_values[0 + 6], debug_values[0 + 7]);
                                    _pk[4] = __floats2bfloat162_rn(debug_values[0 + 8], debug_values[0 + 9]);
                                    _pk[5] = __floats2bfloat162_rn(debug_values[0 + 10], debug_values[0 + 11]);
                                    _pk[6] = __floats2bfloat162_rn(debug_values[0 + 12], debug_values[0 + 13]);
                                    _pk[7] = __floats2bfloat162_rn(debug_values[0 + 14], debug_values[0 + 15]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_k_bf16 + debug_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(debug_k_bf16 + debug_base))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        values[2 * j_6] = __uint_as_float(output_words[j_6] << 16);
                        values[2 * j_6 + 1] = __uint_as_float(output_words[j_6] & 4294901760);
                    }
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 16; j_7++) {
                        absolute[j_7] = values[j_7];
                    }
                    float _fabs_0 = fabsf(absolute[0]);
                    absolute[0] = _fabs_0;
                    float _fabs_1 = fabsf(absolute[1]);
                    absolute[1] = _fabs_1;
                    float _fabs_2 = fabsf(absolute[2]);
                    absolute[2] = _fabs_2;
                    float _fabs_3 = fabsf(absolute[3]);
                    absolute[3] = _fabs_3;
                    float _fabs_4 = fabsf(absolute[4]);
                    absolute[4] = _fabs_4;
                    float _fabs_5 = fabsf(absolute[5]);
                    absolute[5] = _fabs_5;
                    float _fabs_6 = fabsf(absolute[6]);
                    absolute[6] = _fabs_6;
                    float _fabs_7 = fabsf(absolute[7]);
                    absolute[7] = _fabs_7;
                    float _fabs_8 = fabsf(absolute[8]);
                    absolute[8] = _fabs_8;
                    float _fabs_9 = fabsf(absolute[9]);
                    absolute[9] = _fabs_9;
                    float _fabs_10 = fabsf(absolute[10]);
                    absolute[10] = _fabs_10;
                    float _fabs_11 = fabsf(absolute[11]);
                    absolute[11] = _fabs_11;
                    float _fabs_12 = fabsf(absolute[12]);
                    absolute[12] = _fabs_12;
                    float _fabs_13 = fabsf(absolute[13]);
                    absolute[13] = _fabs_13;
                    float _fabs_14 = fabsf(absolute[14]);
                    absolute[14] = _fabs_14;
                    float _fabs_15 = fabsf(absolute[15]);
                    absolute[15] = _fabs_15;
                    float absolute_max = absolute[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        absolute_max = max_noftz(absolute_max, absolute[_lr]);
                    }
                    float amax = absolute_max;
                    float _rcp_1 = approx_rcp(6.0f);
                    float sf_value = global_scale * (amax * _rcp_1);
                    float _fp8_rt_0;
                    uint16_t _e4m3x2_2;
                    uint32_t _f16x2_2;
                    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_2) : "f"(0.0f), "f"(sf_value));
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_2) : "h"(_e4m3x2_2));
                    uint16_t _fp8_h0_2 = (uint16_t)(_f16x2_2 & 0xFFFFu);
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_2));
                    float sf_rounded = _fp8_rt_0;
                    float _rcp_2 = approx_rcp(sf_rounded * global_scale_rcp);
                    float _min_0 = fminf(_rcp_2, 3.4028234663852886e+38f);
                    float output_scale = _min_0;
                    float2 _f2_7 = make_float2(output_scale, output_scale);
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 8; j_8++) {
                        float2 _f2_8 = make_float2(values[2 * j_8], values[2 * j_8 + 1]);
                        float2 _mul_f32x2_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_7));
                        quant_values[2 * j_8] = _mul_f32x2_3.x;
                        quant_values[2 * j_8 + 1] = _mul_f32x2_3.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
                    if (token_1 < M) {
                        *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)lane_q_off)) + (0)) = packed[0];
                        *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)lane_q_off + 4)) + (0)) = packed[1];
                        unsigned int sr = (unsigned int)row_in_destination;
                        unsigned int scale_swizzle = sr >> 7 << 10 | (sr & 31) << 4 | (sr >> 5 & 3) << 2;
                        unsigned long long scale_offset = scale_lane_base + (unsigned long long)scale_swizzle;
                        {
                            unsigned short _sf_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value));
                            *(reinterpret_cast<unsigned char*>(out_sf + scale_offset) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 8; j_9++) {
                        values[2 * j_9] = __uint_as_float(source_words[j_9] << 16);
                        values[2 * j_9 + 1] = __uint_as_float(source_words[j_9] & 4294901760);
                    }
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 16; j_10++) {
                        absolute[j_10] = values[j_10];
                    }
                    float _fabs_16 = fabsf(absolute[0]);
                    absolute[0] = _fabs_16;
                    float _fabs_17 = fabsf(absolute[1]);
                    absolute[1] = _fabs_17;
                    float _fabs_18 = fabsf(absolute[2]);
                    absolute[2] = _fabs_18;
                    float _fabs_19 = fabsf(absolute[3]);
                    absolute[3] = _fabs_19;
                    float _fabs_20 = fabsf(absolute[4]);
                    absolute[4] = _fabs_20;
                    float _fabs_21 = fabsf(absolute[5]);
                    absolute[5] = _fabs_21;
                    float _fabs_22 = fabsf(absolute[6]);
                    absolute[6] = _fabs_22;
                    float _fabs_23 = fabsf(absolute[7]);
                    absolute[7] = _fabs_23;
                    float _fabs_24 = fabsf(absolute[8]);
                    absolute[8] = _fabs_24;
                    float _fabs_25 = fabsf(absolute[9]);
                    absolute[9] = _fabs_25;
                    float _fabs_26 = fabsf(absolute[10]);
                    absolute[10] = _fabs_26;
                    float _fabs_27 = fabsf(absolute[11]);
                    absolute[11] = _fabs_27;
                    float _fabs_28 = fabsf(absolute[12]);
                    absolute[12] = _fabs_28;
                    float _fabs_29 = fabsf(absolute[13]);
                    absolute[13] = _fabs_29;
                    float _fabs_30 = fabsf(absolute[14]);
                    absolute[14] = _fabs_30;
                    float _fabs_31 = fabsf(absolute[15]);
                    absolute[15] = _fabs_31;
                    float absolute_max_1 = absolute[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        absolute_max_1 = max_noftz(absolute_max_1, absolute[_lr]);
                    }
                    float amax_1 = absolute_max_1;
                    float _rcp_3 = approx_rcp(6.0f);
                    float sf_value_1 = global_scale * (amax_1 * _rcp_3);
                    float _fp8_rt_1;
                    uint16_t _e4m3x2_3;
                    uint32_t _f16x2_3;
                    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_3) : "f"(0.0f), "f"(sf_value_1));
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_3) : "h"(_e4m3x2_3));
                    uint16_t _fp8_h0_3 = (uint16_t)(_f16x2_3 & 0xFFFFu);
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_3));
                    float sf_rounded_1 = _fp8_rt_1;
                    float _rcp_4 = approx_rcp(sf_rounded_1 * global_scale_rcp);
                    float _min_1 = fminf(_rcp_4, 3.4028234663852886e+38f);
                    float output_scale_1 = _min_1;
                    float2 _f2_9 = make_float2(output_scale_1, output_scale_1);
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 8; j_11++) {
                        float2 _f2_10 = make_float2(values[2 * j_11], values[2 * j_11 + 1]);
                        float2 _mul_f32x2_4;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_10), "l"(*(const unsigned long long*)&_f2_9));
                        quant_values[2 * j_11] = _mul_f32x2_4.x;
                        quant_values[2 * j_11 + 1] = _mul_f32x2_4.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
                    if (token_1 < M) {
                        *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)lane_q_off)) + (0)) = packed[0];
                        *(reinterpret_cast<int*>(out_q + ((unsigned long long)output_row * 64 + (unsigned long long)lane_q_off + 4)) + (0)) = packed[1];
                        unsigned int sr_1 = (unsigned int)row_in_destination;
                        unsigned int scale_swizzle_1 = sr_1 >> 7 << 10 | (sr_1 & 31) << 4 | (sr_1 >> 5 & 3) << 2;
                        unsigned long long scale_offset_1 = scale_lane_base + (unsigned long long)scale_swizzle_1;
                        {
                            unsigned short _sf_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(sf_value_1));
                            *(reinterpret_cast<unsigned char*>(out_sf + scale_offset_1) + (0)) = (unsigned char)(_sf_pair & 0x7F);
                        }
                    }
                }
            }
            __syncwarp();
            if (q + 2 < 4) {
                int token_base_0 = group_token + (q + 2) * 4;
                int token_1_1 = token_base_0 + row_slot;
                int load_token_1 = ((token_1_1 < M) ? token_1_1 : M - 1);
                unsigned long long source_base_1 = (((unsigned long long)load_token_1 * 56 + (unsigned long long)head) * 3 + (unsigned long long)kind) * 128 + (unsigned long long)dim;
                unsigned long long table_src_1 = (unsigned long long)load_token_1 * 96 + (unsigned long long)(lane8 * 16);
                int stage_smem_2 = stages_smem + q % 2 * 1792;
                #pragma unroll
                for (int h_3 = 0; h_3 < 2; h_3++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                        "}"
                        :: "r"((token_base_0 < M) ? 1 : 0), "r"(stage_smem_2 + source_lane_off + h_3 * 16), "l"(qkv_bf16 + (source_base_1 + (unsigned long long)(h_3 * 8))));
                }
                #pragma unroll
                for (int h_4 = 0; h_4 < 2; h_4++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 16;\n\t"
                        "}"
                        :: "r"((token_base_0 < M && (lane8 < 6 && kind < 2)) ? 1 : 0), "r"(stage_smem_2 + table_row_off + lane8 * 32 + h_4 * 16), "l"(rope_cos_sin + (table_src_1 + (unsigned long long)(h_4 * 8))));
                }
            }
            asm volatile("cp.async.commit_group;");
        }
    }
}

} // extern "C"
