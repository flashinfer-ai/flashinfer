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

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


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



// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_minimax_h3_varlen_attention_5e9badc10a1a02a423c0(__half* __restrict__ partial_O, float* __restrict__ partial_ML, int* __restrict__ combine_table, int* __restrict__ seg_begin, int* __restrict__ seg_len, __nv_bfloat16* __restrict__ O, int num_heads)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int unit = blockIdx.x / 128;
    int cluster_row = blockIdx.x % 128 * 4 + warp;
    int rec = unit * 4;
    int seg = combine_table[rec];
    int packed = combine_table[rec + 1];
    int first_slot = combine_table[rec + 2];
    int splits = combine_table[rec + 3];
    int head = packed >> 16;
    int c = packed & 65535;
    int doc_begin = seg_begin[seg];
    int doc_len = seg_len[seg];
    int local_row = c * 512 + cluster_row;
    int d_base = lane * 4;
    float lane_ml[2];
    lane_ml[0] = -CUDART_INF_F;
    lane_ml[1] = 0.0f;
    if (splits > lane) {
        int lane_row = (first_slot + lane) * 512 + cluster_row;
        {
            float2 _v2_0 = *reinterpret_cast<const float2*>(partial_ML + lane_row * 2);
            lane_ml[0] = _v2_0.x;
            lane_ml[0 + 1] = _v2_0.y;
        }
    }
    unsigned int split_o[16];
    long long split_word = (long long)(first_slot * 512 + cluster_row) * 64 + (long long)(lane * 2);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (splits > i) {
            {
                const uint2* _ivptr_1 = reinterpret_cast<const uint2*>(reinterpret_cast<unsigned int*>(partial_O) + (split_word + (long long)(i * 32768)) + 0);
                uint2 _ivld_1;
                _ivld_1 = *_ivptr_1;
                split_o[i * 2 + 0] = _ivld_1.x;
                split_o[i * 2 + 1] = _ivld_1.y;
            }
        }
    }
    if (local_row < doc_len) {
        float row_max = lane_ml[0];
        float _shfl_xor_sync_2 = __shfl_xor_sync(0xFFFFFFFF, row_max, 16);
        row_max = max_noftz(row_max, _shfl_xor_sync_2);
        float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, row_max, 8);
        row_max = max_noftz(row_max, _shfl_xor_sync_3);
        float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, row_max, 4);
        row_max = max_noftz(row_max, _shfl_xor_sync_4);
        float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, row_max, 2);
        row_max = max_noftz(row_max, _shfl_xor_sync_5);
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, row_max, 1);
        row_max = max_noftz(row_max, _shfl_xor_sync_6);
        float _exp2_7 = approx_exp2(lane_ml[0] - row_max);
        float lane_weight = ((lane_ml[0] == -CUDART_INF_F) ? 0.0f : _exp2_7 * lane_ml[1]);
        float acc[4];
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            acc[e] = 0.0f;
        }
        float denom = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
            if (splits > i_1) {
                float weight = __shfl_sync(0xFFFFFFFF, lane_weight, i_1);
                denom = denom + weight;
                float _unpack_f16_0[4];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_0[_pair * 2]))
                        : "r"((split_o + i_1 * 2)[_pair]));
                }
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b16 h_lo, h_hi;\n\t"
                        ".reg .b32 f_lo, f_hi;\n\t"
                        "mov.b32 {h_lo, h_hi}, %1;\n\t"
                        "cvt.f32.f16 f_lo, h_lo;\n\t"
                        "cvt.f32.f16 f_hi, h_hi;\n\t"
                        "mov.b64 %0, {f_lo, f_hi};\n\t"
                        "}\n"
                        : "=l"(*reinterpret_cast<unsigned long long*>(&_unpack_f16_0[2 + _pair * 2]))
                        : "r"((split_o + i_1 * 2)[1 + _pair]));
                }
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    acc[e_1] = __fmaf_rn(_unpack_f16_0[e_1], weight, acc[e_1]);
                }
            }
        }
        float _rcp_8 = approx_rcp(denom);
        float inv = ((denom > 0.0f) ? _rcp_8 : 0.0f);
        int out_row = (doc_begin + local_row) * num_heads + head;
        float _scaled_0[4];
        const float2 _scale2_9 = {inv, inv};
        #pragma unroll
        for (int _ls = 0; _ls < 2; _ls++)
            reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(acc)[_ls], _scale2_9);
        unsigned int _packed_0[2];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int2*>(O + ((long long)out_row * 128 + (long long)d_base))[0] = reinterpret_cast<int2*>(_packed_0)[0];
    }
}

} // extern "C"
