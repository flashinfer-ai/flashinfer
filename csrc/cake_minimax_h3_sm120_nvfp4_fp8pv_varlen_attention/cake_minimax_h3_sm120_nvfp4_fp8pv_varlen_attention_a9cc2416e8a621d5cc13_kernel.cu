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
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_a9cc2416e8a621d5cc13(__nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K, __nv_bfloat16* __restrict__ V, int* __restrict__ tok_seg, float* __restrict__ mean_k, float* __restrict__ v_scale, unsigned int* __restrict__ q4, unsigned int* __restrict__ k4, uint8_t* __restrict__ q_sf, uint8_t* __restrict__ k_sf, unsigned int* __restrict__ vt8, float* __restrict__ q_scale, float* __restrict__ k_scale, int total_tokens, int total_padded, int num_heads, int num_kblocks)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int sub = tid & 7;
    int trow = tid >> 3;
    int col16 = sub * 16;
    float sgn_hi = (((105 >> sub & 1) == 1) ? -1.0f : 1.0f);
    int num_tblk = total_padded / 32;
    #pragma unroll 1
    for (int work = bid; work < num_tblk * num_heads; work += num_bids) {
        int tblk = work / num_heads;
        int head = work % num_heads;
        int tok_base = tblk * 32;
        int tok = tok_base + trow;
        int valid = ((tok < total_tokens) ? 1 : 0);
        int tok_c = ((tok < total_tokens) ? tok : total_tokens - 1);
        int base = (tok_c * num_heads + head) * 128 + col16;
        float _vec_load_0[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(Q + base + 0);
            uint4 _vld_0[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_1[16];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(K + base + 0);
            uint4 _vld_1[2];
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        int seg = tok_seg[tok_c];
        int ch = trow * 4;
        int chunk = sub >> 2;
        int t8 = sub & 3;
        int vlive[4];
        int vseg[4];
        vlive[0] = ((tok_base + (chunk * 16 + 2 * t8) < total_tokens) ? 1 : 0);
        vseg[0] = tok_seg[((tok_base + (chunk * 16 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8) : total_tokens - 1)];
        float _vec_load_2[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
        vlive[1] = ((tok_base + (chunk * 16 + 2 * t8 + 1) < total_tokens) ? 1 : 0);
        vseg[1] = tok_seg[((tok_base + (chunk * 16 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8 + 1) : total_tokens - 1)];
        float _vec_load_3[4];
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 2 * t8 + 1) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_3[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                (&_vec_load_3[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
            }
        }
        vlive[2] = ((tok_base + (chunk * 16 + 8 + 2 * t8) < total_tokens) ? 1 : 0);
        vseg[2] = tok_seg[((tok_base + (chunk * 16 + 8 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8) : total_tokens - 1)];
        float _vec_load_4[4];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 8 + 2 * t8) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_4[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&_vec_load_4[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
        vlive[3] = ((tok_base + (chunk * 16 + 8 + 2 * t8 + 1) < total_tokens) ? 1 : 0);
        vseg[3] = tok_seg[((tok_base + (chunk * 16 + 8 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8 + 1) : total_tokens - 1)];
        float _vec_load_5[4];
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(V + ((((tok_base + (chunk * 16 + 8 + 2 * t8 + 1) < total_tokens) ? tok_base + (chunk * 16 + 8 + 2 * t8 + 1) : total_tokens - 1) * num_heads + head) * 128 + ch) + 0);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_5[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&_vec_load_5[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
        __syncthreads();
        asm volatile("griddepcontrol.wait;" ::: "memory");
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        float tmp[16];
        float qf[16];
        for (int e = 0; e < 16; e++) {
            qf[e] = _vec_load_0[e];
        }
        tmp[0] = qf[0] + qf[1];
        tmp[1] = qf[0] - qf[1];
        tmp[2] = qf[2] + qf[3];
        tmp[3] = qf[2] - qf[3];
        tmp[4] = qf[4] + qf[5];
        tmp[5] = qf[4] - qf[5];
        tmp[6] = qf[6] + qf[7];
        tmp[7] = qf[6] - qf[7];
        tmp[8] = qf[8] + qf[9];
        tmp[9] = qf[8] - qf[9];
        tmp[10] = qf[10] + qf[11];
        tmp[11] = qf[10] - qf[11];
        tmp[12] = qf[12] + qf[13];
        tmp[13] = qf[12] - qf[13];
        tmp[14] = qf[14] + qf[15];
        tmp[15] = qf[14] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        tmp[0] = qf[0] + qf[2];
        tmp[1] = qf[1] + qf[3];
        tmp[2] = qf[0] - qf[2];
        tmp[3] = qf[1] - qf[3];
        tmp[4] = qf[4] + qf[6];
        tmp[5] = qf[5] + qf[7];
        tmp[6] = qf[4] - qf[6];
        tmp[7] = qf[5] - qf[7];
        tmp[8] = qf[8] + qf[10];
        tmp[9] = qf[9] + qf[11];
        tmp[10] = qf[8] - qf[10];
        tmp[11] = qf[9] - qf[11];
        tmp[12] = qf[12] + qf[14];
        tmp[13] = qf[13] + qf[15];
        tmp[14] = qf[12] - qf[14];
        tmp[15] = qf[13] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        tmp[0] = qf[0] + qf[4];
        tmp[1] = qf[1] + qf[5];
        tmp[2] = qf[2] + qf[6];
        tmp[3] = qf[3] + qf[7];
        tmp[4] = qf[0] - qf[4];
        tmp[5] = qf[1] - qf[5];
        tmp[6] = qf[2] - qf[6];
        tmp[7] = qf[3] - qf[7];
        tmp[8] = qf[8] + qf[12];
        tmp[9] = qf[9] + qf[13];
        tmp[10] = qf[10] + qf[14];
        tmp[11] = qf[11] + qf[15];
        tmp[12] = qf[8] - qf[12];
        tmp[13] = qf[9] - qf[13];
        tmp[14] = qf[10] - qf[14];
        tmp[15] = qf[11] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        tmp[0] = qf[0] + qf[8];
        tmp[1] = qf[1] + qf[9];
        tmp[2] = qf[2] + qf[10];
        tmp[3] = qf[3] + qf[11];
        tmp[4] = qf[4] + qf[12];
        tmp[5] = qf[5] + qf[13];
        tmp[6] = qf[6] + qf[14];
        tmp[7] = qf[7] + qf[15];
        tmp[8] = qf[0] - qf[8];
        tmp[9] = qf[1] - qf[9];
        tmp[10] = qf[2] - qf[10];
        tmp[11] = qf[3] - qf[11];
        tmp[12] = qf[4] - qf[12];
        tmp[13] = qf[5] - qf[13];
        tmp[14] = qf[6] - qf[14];
        tmp[15] = qf[7] - qf[15];
        qf[0] = tmp[0];
        qf[1] = tmp[1];
        qf[2] = tmp[2];
        qf[3] = tmp[3];
        qf[4] = tmp[4];
        qf[5] = tmp[5];
        qf[6] = tmp[6];
        qf[7] = tmp[7];
        qf[8] = tmp[8];
        qf[9] = tmp[9];
        qf[10] = tmp[10];
        qf[11] = tmp[11];
        qf[12] = tmp[12];
        qf[13] = tmp[13];
        qf[14] = tmp[14];
        qf[15] = tmp[15];
        float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, qf[0], 1);
        tmp[0] = _shfl_xor_sync_6;
        float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, qf[1], 1);
        tmp[1] = _shfl_xor_sync_7;
        float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, qf[2], 1);
        tmp[2] = _shfl_xor_sync_8;
        float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, qf[3], 1);
        tmp[3] = _shfl_xor_sync_9;
        float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, qf[4], 1);
        tmp[4] = _shfl_xor_sync_10;
        float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, qf[5], 1);
        tmp[5] = _shfl_xor_sync_11;
        float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, qf[6], 1);
        tmp[6] = _shfl_xor_sync_12;
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, qf[7], 1);
        tmp[7] = _shfl_xor_sync_13;
        float _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, qf[8], 1);
        tmp[8] = _shfl_xor_sync_14;
        float _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, qf[9], 1);
        tmp[9] = _shfl_xor_sync_15;
        float _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, qf[10], 1);
        tmp[10] = _shfl_xor_sync_16;
        float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, qf[11], 1);
        tmp[11] = _shfl_xor_sync_17;
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, qf[12], 1);
        tmp[12] = _shfl_xor_sync_18;
        float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, qf[13], 1);
        tmp[13] = _shfl_xor_sync_19;
        float _shfl_xor_sync_20 = __shfl_xor_sync(0xFFFFFFFF, qf[14], 1);
        tmp[14] = _shfl_xor_sync_20;
        float _shfl_xor_sync_21 = __shfl_xor_sync(0xFFFFFFFF, qf[15], 1);
        tmp[15] = _shfl_xor_sync_21;
        qf[0] = (((sub & 1) == 1) ? tmp[0] - qf[0] : qf[0] + tmp[0]);
        qf[1] = (((sub & 1) == 1) ? tmp[1] - qf[1] : qf[1] + tmp[1]);
        qf[2] = (((sub & 1) == 1) ? tmp[2] - qf[2] : qf[2] + tmp[2]);
        qf[3] = (((sub & 1) == 1) ? tmp[3] - qf[3] : qf[3] + tmp[3]);
        qf[4] = (((sub & 1) == 1) ? tmp[4] - qf[4] : qf[4] + tmp[4]);
        qf[5] = (((sub & 1) == 1) ? tmp[5] - qf[5] : qf[5] + tmp[5]);
        qf[6] = (((sub & 1) == 1) ? tmp[6] - qf[6] : qf[6] + tmp[6]);
        qf[7] = (((sub & 1) == 1) ? tmp[7] - qf[7] : qf[7] + tmp[7]);
        qf[8] = (((sub & 1) == 1) ? tmp[8] - qf[8] : qf[8] + tmp[8]);
        qf[9] = (((sub & 1) == 1) ? tmp[9] - qf[9] : qf[9] + tmp[9]);
        qf[10] = (((sub & 1) == 1) ? tmp[10] - qf[10] : qf[10] + tmp[10]);
        qf[11] = (((sub & 1) == 1) ? tmp[11] - qf[11] : qf[11] + tmp[11]);
        qf[12] = (((sub & 1) == 1) ? tmp[12] - qf[12] : qf[12] + tmp[12]);
        qf[13] = (((sub & 1) == 1) ? tmp[13] - qf[13] : qf[13] + tmp[13]);
        qf[14] = (((sub & 1) == 1) ? tmp[14] - qf[14] : qf[14] + tmp[14]);
        qf[15] = (((sub & 1) == 1) ? tmp[15] - qf[15] : qf[15] + tmp[15]);
        float _shfl_xor_sync_22 = __shfl_xor_sync(0xFFFFFFFF, qf[0], 2);
        tmp[0] = _shfl_xor_sync_22;
        float _shfl_xor_sync_23 = __shfl_xor_sync(0xFFFFFFFF, qf[1], 2);
        tmp[1] = _shfl_xor_sync_23;
        float _shfl_xor_sync_24 = __shfl_xor_sync(0xFFFFFFFF, qf[2], 2);
        tmp[2] = _shfl_xor_sync_24;
        float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, qf[3], 2);
        tmp[3] = _shfl_xor_sync_25;
        float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, qf[4], 2);
        tmp[4] = _shfl_xor_sync_26;
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, qf[5], 2);
        tmp[5] = _shfl_xor_sync_27;
        float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, qf[6], 2);
        tmp[6] = _shfl_xor_sync_28;
        float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, qf[7], 2);
        tmp[7] = _shfl_xor_sync_29;
        float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, qf[8], 2);
        tmp[8] = _shfl_xor_sync_30;
        float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, qf[9], 2);
        tmp[9] = _shfl_xor_sync_31;
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, qf[10], 2);
        tmp[10] = _shfl_xor_sync_32;
        float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, qf[11], 2);
        tmp[11] = _shfl_xor_sync_33;
        float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, qf[12], 2);
        tmp[12] = _shfl_xor_sync_34;
        float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, qf[13], 2);
        tmp[13] = _shfl_xor_sync_35;
        float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, qf[14], 2);
        tmp[14] = _shfl_xor_sync_36;
        float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, qf[15], 2);
        tmp[15] = _shfl_xor_sync_37;
        qf[0] = (((sub >> 1 & 1) == 1) ? tmp[0] - qf[0] : qf[0] + tmp[0]);
        qf[1] = (((sub >> 1 & 1) == 1) ? tmp[1] - qf[1] : qf[1] + tmp[1]);
        qf[2] = (((sub >> 1 & 1) == 1) ? tmp[2] - qf[2] : qf[2] + tmp[2]);
        qf[3] = (((sub >> 1 & 1) == 1) ? tmp[3] - qf[3] : qf[3] + tmp[3]);
        qf[4] = (((sub >> 1 & 1) == 1) ? tmp[4] - qf[4] : qf[4] + tmp[4]);
        qf[5] = (((sub >> 1 & 1) == 1) ? tmp[5] - qf[5] : qf[5] + tmp[5]);
        qf[6] = (((sub >> 1 & 1) == 1) ? tmp[6] - qf[6] : qf[6] + tmp[6]);
        qf[7] = (((sub >> 1 & 1) == 1) ? tmp[7] - qf[7] : qf[7] + tmp[7]);
        qf[8] = (((sub >> 1 & 1) == 1) ? tmp[8] - qf[8] : qf[8] + tmp[8]);
        qf[9] = (((sub >> 1 & 1) == 1) ? tmp[9] - qf[9] : qf[9] + tmp[9]);
        qf[10] = (((sub >> 1 & 1) == 1) ? tmp[10] - qf[10] : qf[10] + tmp[10]);
        qf[11] = (((sub >> 1 & 1) == 1) ? tmp[11] - qf[11] : qf[11] + tmp[11]);
        qf[12] = (((sub >> 1 & 1) == 1) ? tmp[12] - qf[12] : qf[12] + tmp[12]);
        qf[13] = (((sub >> 1 & 1) == 1) ? tmp[13] - qf[13] : qf[13] + tmp[13]);
        qf[14] = (((sub >> 1 & 1) == 1) ? tmp[14] - qf[14] : qf[14] + tmp[14]);
        qf[15] = (((sub >> 1 & 1) == 1) ? tmp[15] - qf[15] : qf[15] + tmp[15]);
        float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, qf[0], 4);
        tmp[0] = _shfl_xor_sync_38;
        float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, qf[1], 4);
        tmp[1] = _shfl_xor_sync_39;
        float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, qf[2], 4);
        tmp[2] = _shfl_xor_sync_40;
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, qf[3], 4);
        tmp[3] = _shfl_xor_sync_41;
        float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, qf[4], 4);
        tmp[4] = _shfl_xor_sync_42;
        float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, qf[5], 4);
        tmp[5] = _shfl_xor_sync_43;
        float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, qf[6], 4);
        tmp[6] = _shfl_xor_sync_44;
        float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, qf[7], 4);
        tmp[7] = _shfl_xor_sync_45;
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, qf[8], 4);
        tmp[8] = _shfl_xor_sync_46;
        float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, qf[9], 4);
        tmp[9] = _shfl_xor_sync_47;
        float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, qf[10], 4);
        tmp[10] = _shfl_xor_sync_48;
        float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, qf[11], 4);
        tmp[11] = _shfl_xor_sync_49;
        float _shfl_xor_sync_50 = __shfl_xor_sync(0xFFFFFFFF, qf[12], 4);
        tmp[12] = _shfl_xor_sync_50;
        float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, qf[13], 4);
        tmp[13] = _shfl_xor_sync_51;
        float _shfl_xor_sync_52 = __shfl_xor_sync(0xFFFFFFFF, qf[14], 4);
        tmp[14] = _shfl_xor_sync_52;
        float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, qf[15], 4);
        tmp[15] = _shfl_xor_sync_53;
        qf[0] = (((sub >> 2 & 1) == 1) ? tmp[0] - qf[0] : qf[0] + tmp[0]);
        qf[1] = (((sub >> 2 & 1) == 1) ? tmp[1] - qf[1] : qf[1] + tmp[1]);
        qf[2] = (((sub >> 2 & 1) == 1) ? tmp[2] - qf[2] : qf[2] + tmp[2]);
        qf[3] = (((sub >> 2 & 1) == 1) ? tmp[3] - qf[3] : qf[3] + tmp[3]);
        qf[4] = (((sub >> 2 & 1) == 1) ? tmp[4] - qf[4] : qf[4] + tmp[4]);
        qf[5] = (((sub >> 2 & 1) == 1) ? tmp[5] - qf[5] : qf[5] + tmp[5]);
        qf[6] = (((sub >> 2 & 1) == 1) ? tmp[6] - qf[6] : qf[6] + tmp[6]);
        qf[7] = (((sub >> 2 & 1) == 1) ? tmp[7] - qf[7] : qf[7] + tmp[7]);
        qf[8] = (((sub >> 2 & 1) == 1) ? tmp[8] - qf[8] : qf[8] + tmp[8]);
        qf[9] = (((sub >> 2 & 1) == 1) ? tmp[9] - qf[9] : qf[9] + tmp[9]);
        qf[10] = (((sub >> 2 & 1) == 1) ? tmp[10] - qf[10] : qf[10] + tmp[10]);
        qf[11] = (((sub >> 2 & 1) == 1) ? tmp[11] - qf[11] : qf[11] + tmp[11]);
        qf[12] = (((sub >> 2 & 1) == 1) ? tmp[12] - qf[12] : qf[12] + tmp[12]);
        qf[13] = (((sub >> 2 & 1) == 1) ? tmp[13] - qf[13] : qf[13] + tmp[13]);
        qf[14] = (((sub >> 2 & 1) == 1) ? tmp[14] - qf[14] : qf[14] + tmp[14]);
        qf[15] = (((sub >> 2 & 1) == 1) ? tmp[15] - qf[15] : qf[15] + tmp[15]);
        qf[0] = qf[0] * (sgn_hi * 0.08838834764831843f);
        qf[1] = qf[1] * (sgn_hi * -0.08838834764831843f);
        qf[2] = qf[2] * (sgn_hi * 0.08838834764831843f);
        qf[3] = qf[3] * (sgn_hi * 0.08838834764831843f);
        qf[4] = qf[4] * (sgn_hi * -0.08838834764831843f);
        qf[5] = qf[5] * (sgn_hi * 0.08838834764831843f);
        qf[6] = qf[6] * (sgn_hi * -0.08838834764831843f);
        qf[7] = qf[7] * (sgn_hi * -0.08838834764831843f);
        qf[8] = qf[8] * (sgn_hi * 0.08838834764831843f);
        qf[9] = qf[9] * (sgn_hi * 0.08838834764831843f);
        qf[10] = qf[10] * (sgn_hi * -0.08838834764831843f);
        qf[11] = qf[11] * (sgn_hi * 0.08838834764831843f);
        qf[12] = qf[12] * (sgn_hi * -0.08838834764831843f);
        qf[13] = qf[13] * (sgn_hi * -0.08838834764831843f);
        qf[14] = qf[14] * (sgn_hi * 0.08838834764831843f);
        qf[15] = qf[15] * (sgn_hi * -0.08838834764831843f);
        float qamax = 0.0f;
        float bmax = 0.0f;
        for (int e_1 = 0; e_1 < 16; e_1++) {
            bmax = fmaxf(bmax, fabsf(qf[e_1]));
        }
        float _shfl_xor_sync_54 = __shfl_xor_sync(0xFFFFFFFF, bmax, 4);
        qamax = fmaxf(bmax, _shfl_xor_sync_54);
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, qamax, 2);
        qamax = fmaxf(qamax, _shfl_xor_sync_55);
        float _shfl_xor_sync_56 = __shfl_xor_sync(0xFFFFFFFF, qamax, 1);
        qamax = fmaxf(qamax, _shfl_xor_sync_56);
        qamax = fmaxf(qamax, 1e-12f);
        float qs = __fdiv_rn(qamax, 2688.0f);
        float qg = __fdiv_rn(2688.0f, qamax);
        float qsf_val = bmax * qg * 0.16666666666666666f;
        uint16_t _cvt_satfinite_e4m3x2_f32_57;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_57) : "f"(0.0f), "f"(qsf_val));
        uint16_t _e4m3x2_decode_58 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_57 & 0xFFu);
        uint32_t _f16x2_decode_58;
        float qsf_dec;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_58) : "h"(_e4m3x2_decode_58));
        uint16_t _f16_decode_58 = (uint16_t)_f16x2_decode_58;
        asm("cvt.f32.f16 %0, %1;" : "=f"(qsf_dec) : "h"(_f16_decode_58));
        float _fdiv_59 = __fdiv_rn(qg, qsf_dec);
        float qout = ((qsf_dec != 0.0f) ? _fdiv_59 : 0.0f);
        float qn[16];
        for (int e_2 = 0; e_2 < 16; e_2++) {
            qn[e_2] = qf[e_2] * qout;
        }
        unsigned int qw[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[0]) : "f"(qn[0]), "f"(qn[1]), "f"(qn[2]), "f"(qn[3]), "f"(qn[4]), "f"(qn[5]), "f"(qn[6]), "f"(qn[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(qw[1]) : "f"(qn[8]), "f"(qn[9]), "f"(qn[10]), "f"(qn[11]), "f"(qn[12]), "f"(qn[13]), "f"(qn[14]), "f"(qn[15]));
        unsigned int qstore[4];
        qstore[0] = qw[0];
        qstore[1] = qw[1];
        unsigned int _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, qw[0], 1);
        qstore[2] = _shfl_xor_sync_60;
        unsigned int _shfl_xor_sync_61 = __shfl_xor_sync(0xFFFFFFFF, qw[1], 1);
        qstore[3] = _shfl_xor_sync_61;
        if (valid == 1) {
            {
                unsigned short _sf_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(qsf_val));
                *(reinterpret_cast<unsigned char*>(q_sf + ((tok * num_heads + head) * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
            }
            if ((sub & 1) == 0) {
                reinterpret_cast<int4*>(q4 + (((tok * num_heads + head) * 64 + sub * 8) / 4))[0] = reinterpret_cast<int4*>(qstore)[0];
            }
            if (sub == 0) {
                *(reinterpret_cast<float*>(q_scale + (tok * num_heads + head)) + (0)) = qs;
            }
        }
        int mbase = (seg * num_heads + head) * 128 + col16;
        float kf16[16];
        for (int quad = 0; quad < 4; quad++) {
            float _vec_load_6[4];
            {
                float4 _v4 = *reinterpret_cast<const float4*>(mean_k + (mbase + quad * 4) + 0);
                _vec_load_6[0 + 0] = _v4.x;
                _vec_load_6[0 + 1] = _v4.y;
                _vec_load_6[0 + 2] = _v4.z;
                _vec_load_6[0 + 3] = _v4.w;
            }
            for (int e_3 = 0; e_3 < 4; e_3++) {
                float kf = _vec_load_1[quad * 4 + e_3];
                float kc = kf - _vec_load_6[e_3];
                kf16[quad * 4 + e_3] = ((valid == 1) ? kc : 0.0f);
            }
        }
        tmp[0] = kf16[0] + kf16[1];
        tmp[1] = kf16[0] - kf16[1];
        tmp[2] = kf16[2] + kf16[3];
        tmp[3] = kf16[2] - kf16[3];
        tmp[4] = kf16[4] + kf16[5];
        tmp[5] = kf16[4] - kf16[5];
        tmp[6] = kf16[6] + kf16[7];
        tmp[7] = kf16[6] - kf16[7];
        tmp[8] = kf16[8] + kf16[9];
        tmp[9] = kf16[8] - kf16[9];
        tmp[10] = kf16[10] + kf16[11];
        tmp[11] = kf16[10] - kf16[11];
        tmp[12] = kf16[12] + kf16[13];
        tmp[13] = kf16[12] - kf16[13];
        tmp[14] = kf16[14] + kf16[15];
        tmp[15] = kf16[14] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        tmp[0] = kf16[0] + kf16[2];
        tmp[1] = kf16[1] + kf16[3];
        tmp[2] = kf16[0] - kf16[2];
        tmp[3] = kf16[1] - kf16[3];
        tmp[4] = kf16[4] + kf16[6];
        tmp[5] = kf16[5] + kf16[7];
        tmp[6] = kf16[4] - kf16[6];
        tmp[7] = kf16[5] - kf16[7];
        tmp[8] = kf16[8] + kf16[10];
        tmp[9] = kf16[9] + kf16[11];
        tmp[10] = kf16[8] - kf16[10];
        tmp[11] = kf16[9] - kf16[11];
        tmp[12] = kf16[12] + kf16[14];
        tmp[13] = kf16[13] + kf16[15];
        tmp[14] = kf16[12] - kf16[14];
        tmp[15] = kf16[13] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        tmp[0] = kf16[0] + kf16[4];
        tmp[1] = kf16[1] + kf16[5];
        tmp[2] = kf16[2] + kf16[6];
        tmp[3] = kf16[3] + kf16[7];
        tmp[4] = kf16[0] - kf16[4];
        tmp[5] = kf16[1] - kf16[5];
        tmp[6] = kf16[2] - kf16[6];
        tmp[7] = kf16[3] - kf16[7];
        tmp[8] = kf16[8] + kf16[12];
        tmp[9] = kf16[9] + kf16[13];
        tmp[10] = kf16[10] + kf16[14];
        tmp[11] = kf16[11] + kf16[15];
        tmp[12] = kf16[8] - kf16[12];
        tmp[13] = kf16[9] - kf16[13];
        tmp[14] = kf16[10] - kf16[14];
        tmp[15] = kf16[11] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        tmp[0] = kf16[0] + kf16[8];
        tmp[1] = kf16[1] + kf16[9];
        tmp[2] = kf16[2] + kf16[10];
        tmp[3] = kf16[3] + kf16[11];
        tmp[4] = kf16[4] + kf16[12];
        tmp[5] = kf16[5] + kf16[13];
        tmp[6] = kf16[6] + kf16[14];
        tmp[7] = kf16[7] + kf16[15];
        tmp[8] = kf16[0] - kf16[8];
        tmp[9] = kf16[1] - kf16[9];
        tmp[10] = kf16[2] - kf16[10];
        tmp[11] = kf16[3] - kf16[11];
        tmp[12] = kf16[4] - kf16[12];
        tmp[13] = kf16[5] - kf16[13];
        tmp[14] = kf16[6] - kf16[14];
        tmp[15] = kf16[7] - kf16[15];
        kf16[0] = tmp[0];
        kf16[1] = tmp[1];
        kf16[2] = tmp[2];
        kf16[3] = tmp[3];
        kf16[4] = tmp[4];
        kf16[5] = tmp[5];
        kf16[6] = tmp[6];
        kf16[7] = tmp[7];
        kf16[8] = tmp[8];
        kf16[9] = tmp[9];
        kf16[10] = tmp[10];
        kf16[11] = tmp[11];
        kf16[12] = tmp[12];
        kf16[13] = tmp[13];
        kf16[14] = tmp[14];
        kf16[15] = tmp[15];
        float _shfl_xor_sync_63 = __shfl_xor_sync(0xFFFFFFFF, kf16[0], 1);
        tmp[0] = _shfl_xor_sync_63;
        float _shfl_xor_sync_64 = __shfl_xor_sync(0xFFFFFFFF, kf16[1], 1);
        tmp[1] = _shfl_xor_sync_64;
        float _shfl_xor_sync_65 = __shfl_xor_sync(0xFFFFFFFF, kf16[2], 1);
        tmp[2] = _shfl_xor_sync_65;
        float _shfl_xor_sync_66 = __shfl_xor_sync(0xFFFFFFFF, kf16[3], 1);
        tmp[3] = _shfl_xor_sync_66;
        float _shfl_xor_sync_67 = __shfl_xor_sync(0xFFFFFFFF, kf16[4], 1);
        tmp[4] = _shfl_xor_sync_67;
        float _shfl_xor_sync_68 = __shfl_xor_sync(0xFFFFFFFF, kf16[5], 1);
        tmp[5] = _shfl_xor_sync_68;
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, kf16[6], 1);
        tmp[6] = _shfl_xor_sync_69;
        float _shfl_xor_sync_70 = __shfl_xor_sync(0xFFFFFFFF, kf16[7], 1);
        tmp[7] = _shfl_xor_sync_70;
        float _shfl_xor_sync_71 = __shfl_xor_sync(0xFFFFFFFF, kf16[8], 1);
        tmp[8] = _shfl_xor_sync_71;
        float _shfl_xor_sync_72 = __shfl_xor_sync(0xFFFFFFFF, kf16[9], 1);
        tmp[9] = _shfl_xor_sync_72;
        float _shfl_xor_sync_73 = __shfl_xor_sync(0xFFFFFFFF, kf16[10], 1);
        tmp[10] = _shfl_xor_sync_73;
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, kf16[11], 1);
        tmp[11] = _shfl_xor_sync_74;
        float _shfl_xor_sync_75 = __shfl_xor_sync(0xFFFFFFFF, kf16[12], 1);
        tmp[12] = _shfl_xor_sync_75;
        float _shfl_xor_sync_76 = __shfl_xor_sync(0xFFFFFFFF, kf16[13], 1);
        tmp[13] = _shfl_xor_sync_76;
        float _shfl_xor_sync_77 = __shfl_xor_sync(0xFFFFFFFF, kf16[14], 1);
        tmp[14] = _shfl_xor_sync_77;
        float _shfl_xor_sync_78 = __shfl_xor_sync(0xFFFFFFFF, kf16[15], 1);
        tmp[15] = _shfl_xor_sync_78;
        kf16[0] = (((sub & 1) == 1) ? tmp[0] - kf16[0] : kf16[0] + tmp[0]);
        kf16[1] = (((sub & 1) == 1) ? tmp[1] - kf16[1] : kf16[1] + tmp[1]);
        kf16[2] = (((sub & 1) == 1) ? tmp[2] - kf16[2] : kf16[2] + tmp[2]);
        kf16[3] = (((sub & 1) == 1) ? tmp[3] - kf16[3] : kf16[3] + tmp[3]);
        kf16[4] = (((sub & 1) == 1) ? tmp[4] - kf16[4] : kf16[4] + tmp[4]);
        kf16[5] = (((sub & 1) == 1) ? tmp[5] - kf16[5] : kf16[5] + tmp[5]);
        kf16[6] = (((sub & 1) == 1) ? tmp[6] - kf16[6] : kf16[6] + tmp[6]);
        kf16[7] = (((sub & 1) == 1) ? tmp[7] - kf16[7] : kf16[7] + tmp[7]);
        kf16[8] = (((sub & 1) == 1) ? tmp[8] - kf16[8] : kf16[8] + tmp[8]);
        kf16[9] = (((sub & 1) == 1) ? tmp[9] - kf16[9] : kf16[9] + tmp[9]);
        kf16[10] = (((sub & 1) == 1) ? tmp[10] - kf16[10] : kf16[10] + tmp[10]);
        kf16[11] = (((sub & 1) == 1) ? tmp[11] - kf16[11] : kf16[11] + tmp[11]);
        kf16[12] = (((sub & 1) == 1) ? tmp[12] - kf16[12] : kf16[12] + tmp[12]);
        kf16[13] = (((sub & 1) == 1) ? tmp[13] - kf16[13] : kf16[13] + tmp[13]);
        kf16[14] = (((sub & 1) == 1) ? tmp[14] - kf16[14] : kf16[14] + tmp[14]);
        kf16[15] = (((sub & 1) == 1) ? tmp[15] - kf16[15] : kf16[15] + tmp[15]);
        float _shfl_xor_sync_79 = __shfl_xor_sync(0xFFFFFFFF, kf16[0], 2);
        tmp[0] = _shfl_xor_sync_79;
        float _shfl_xor_sync_80 = __shfl_xor_sync(0xFFFFFFFF, kf16[1], 2);
        tmp[1] = _shfl_xor_sync_80;
        float _shfl_xor_sync_81 = __shfl_xor_sync(0xFFFFFFFF, kf16[2], 2);
        tmp[2] = _shfl_xor_sync_81;
        float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, kf16[3], 2);
        tmp[3] = _shfl_xor_sync_82;
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, kf16[4], 2);
        tmp[4] = _shfl_xor_sync_83;
        float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, kf16[5], 2);
        tmp[5] = _shfl_xor_sync_84;
        float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, kf16[6], 2);
        tmp[6] = _shfl_xor_sync_85;
        float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, kf16[7], 2);
        tmp[7] = _shfl_xor_sync_86;
        float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, kf16[8], 2);
        tmp[8] = _shfl_xor_sync_87;
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, kf16[9], 2);
        tmp[9] = _shfl_xor_sync_88;
        float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, kf16[10], 2);
        tmp[10] = _shfl_xor_sync_89;
        float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, kf16[11], 2);
        tmp[11] = _shfl_xor_sync_90;
        float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, kf16[12], 2);
        tmp[12] = _shfl_xor_sync_91;
        float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, kf16[13], 2);
        tmp[13] = _shfl_xor_sync_92;
        float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, kf16[14], 2);
        tmp[14] = _shfl_xor_sync_93;
        float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, kf16[15], 2);
        tmp[15] = _shfl_xor_sync_94;
        kf16[0] = (((sub >> 1 & 1) == 1) ? tmp[0] - kf16[0] : kf16[0] + tmp[0]);
        kf16[1] = (((sub >> 1 & 1) == 1) ? tmp[1] - kf16[1] : kf16[1] + tmp[1]);
        kf16[2] = (((sub >> 1 & 1) == 1) ? tmp[2] - kf16[2] : kf16[2] + tmp[2]);
        kf16[3] = (((sub >> 1 & 1) == 1) ? tmp[3] - kf16[3] : kf16[3] + tmp[3]);
        kf16[4] = (((sub >> 1 & 1) == 1) ? tmp[4] - kf16[4] : kf16[4] + tmp[4]);
        kf16[5] = (((sub >> 1 & 1) == 1) ? tmp[5] - kf16[5] : kf16[5] + tmp[5]);
        kf16[6] = (((sub >> 1 & 1) == 1) ? tmp[6] - kf16[6] : kf16[6] + tmp[6]);
        kf16[7] = (((sub >> 1 & 1) == 1) ? tmp[7] - kf16[7] : kf16[7] + tmp[7]);
        kf16[8] = (((sub >> 1 & 1) == 1) ? tmp[8] - kf16[8] : kf16[8] + tmp[8]);
        kf16[9] = (((sub >> 1 & 1) == 1) ? tmp[9] - kf16[9] : kf16[9] + tmp[9]);
        kf16[10] = (((sub >> 1 & 1) == 1) ? tmp[10] - kf16[10] : kf16[10] + tmp[10]);
        kf16[11] = (((sub >> 1 & 1) == 1) ? tmp[11] - kf16[11] : kf16[11] + tmp[11]);
        kf16[12] = (((sub >> 1 & 1) == 1) ? tmp[12] - kf16[12] : kf16[12] + tmp[12]);
        kf16[13] = (((sub >> 1 & 1) == 1) ? tmp[13] - kf16[13] : kf16[13] + tmp[13]);
        kf16[14] = (((sub >> 1 & 1) == 1) ? tmp[14] - kf16[14] : kf16[14] + tmp[14]);
        kf16[15] = (((sub >> 1 & 1) == 1) ? tmp[15] - kf16[15] : kf16[15] + tmp[15]);
        float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, kf16[0], 4);
        tmp[0] = _shfl_xor_sync_95;
        float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, kf16[1], 4);
        tmp[1] = _shfl_xor_sync_96;
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, kf16[2], 4);
        tmp[2] = _shfl_xor_sync_97;
        float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, kf16[3], 4);
        tmp[3] = _shfl_xor_sync_98;
        float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, kf16[4], 4);
        tmp[4] = _shfl_xor_sync_99;
        float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, kf16[5], 4);
        tmp[5] = _shfl_xor_sync_100;
        float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, kf16[6], 4);
        tmp[6] = _shfl_xor_sync_101;
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, kf16[7], 4);
        tmp[7] = _shfl_xor_sync_102;
        float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, kf16[8], 4);
        tmp[8] = _shfl_xor_sync_103;
        float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, kf16[9], 4);
        tmp[9] = _shfl_xor_sync_104;
        float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, kf16[10], 4);
        tmp[10] = _shfl_xor_sync_105;
        float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, kf16[11], 4);
        tmp[11] = _shfl_xor_sync_106;
        float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, kf16[12], 4);
        tmp[12] = _shfl_xor_sync_107;
        float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, kf16[13], 4);
        tmp[13] = _shfl_xor_sync_108;
        float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, kf16[14], 4);
        tmp[14] = _shfl_xor_sync_109;
        float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, kf16[15], 4);
        tmp[15] = _shfl_xor_sync_110;
        kf16[0] = (((sub >> 2 & 1) == 1) ? tmp[0] - kf16[0] : kf16[0] + tmp[0]);
        kf16[1] = (((sub >> 2 & 1) == 1) ? tmp[1] - kf16[1] : kf16[1] + tmp[1]);
        kf16[2] = (((sub >> 2 & 1) == 1) ? tmp[2] - kf16[2] : kf16[2] + tmp[2]);
        kf16[3] = (((sub >> 2 & 1) == 1) ? tmp[3] - kf16[3] : kf16[3] + tmp[3]);
        kf16[4] = (((sub >> 2 & 1) == 1) ? tmp[4] - kf16[4] : kf16[4] + tmp[4]);
        kf16[5] = (((sub >> 2 & 1) == 1) ? tmp[5] - kf16[5] : kf16[5] + tmp[5]);
        kf16[6] = (((sub >> 2 & 1) == 1) ? tmp[6] - kf16[6] : kf16[6] + tmp[6]);
        kf16[7] = (((sub >> 2 & 1) == 1) ? tmp[7] - kf16[7] : kf16[7] + tmp[7]);
        kf16[8] = (((sub >> 2 & 1) == 1) ? tmp[8] - kf16[8] : kf16[8] + tmp[8]);
        kf16[9] = (((sub >> 2 & 1) == 1) ? tmp[9] - kf16[9] : kf16[9] + tmp[9]);
        kf16[10] = (((sub >> 2 & 1) == 1) ? tmp[10] - kf16[10] : kf16[10] + tmp[10]);
        kf16[11] = (((sub >> 2 & 1) == 1) ? tmp[11] - kf16[11] : kf16[11] + tmp[11]);
        kf16[12] = (((sub >> 2 & 1) == 1) ? tmp[12] - kf16[12] : kf16[12] + tmp[12]);
        kf16[13] = (((sub >> 2 & 1) == 1) ? tmp[13] - kf16[13] : kf16[13] + tmp[13]);
        kf16[14] = (((sub >> 2 & 1) == 1) ? tmp[14] - kf16[14] : kf16[14] + tmp[14]);
        kf16[15] = (((sub >> 2 & 1) == 1) ? tmp[15] - kf16[15] : kf16[15] + tmp[15]);
        kf16[0] = kf16[0] * (sgn_hi * 0.08838834764831843f);
        kf16[1] = kf16[1] * (sgn_hi * -0.08838834764831843f);
        kf16[2] = kf16[2] * (sgn_hi * 0.08838834764831843f);
        kf16[3] = kf16[3] * (sgn_hi * 0.08838834764831843f);
        kf16[4] = kf16[4] * (sgn_hi * -0.08838834764831843f);
        kf16[5] = kf16[5] * (sgn_hi * 0.08838834764831843f);
        kf16[6] = kf16[6] * (sgn_hi * -0.08838834764831843f);
        kf16[7] = kf16[7] * (sgn_hi * -0.08838834764831843f);
        kf16[8] = kf16[8] * (sgn_hi * 0.08838834764831843f);
        kf16[9] = kf16[9] * (sgn_hi * 0.08838834764831843f);
        kf16[10] = kf16[10] * (sgn_hi * -0.08838834764831843f);
        kf16[11] = kf16[11] * (sgn_hi * 0.08838834764831843f);
        kf16[12] = kf16[12] * (sgn_hi * -0.08838834764831843f);
        kf16[13] = kf16[13] * (sgn_hi * -0.08838834764831843f);
        kf16[14] = kf16[14] * (sgn_hi * 0.08838834764831843f);
        kf16[15] = kf16[15] * (sgn_hi * -0.08838834764831843f);
        float kbmax = 0.0f;
        for (int e_4 = 0; e_4 < 16; e_4++) {
            kbmax = fmaxf(kbmax, fabsf(kf16[e_4]));
        }
        float ks = k_scale[seg * num_heads + head];
        float kg = __fdiv_rn(1.0f, ks);
        float ksf_val = kbmax * kg * 0.16666666666666666f;
        uint16_t _cvt_satfinite_e4m3x2_f32_111;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_111) : "f"(0.0f), "f"(ksf_val));
        uint16_t _e4m3x2_decode_112 = (uint16_t)((unsigned int)_cvt_satfinite_e4m3x2_f32_111 & 0xFFu);
        uint32_t _f16x2_decode_112;
        float ksf_dec;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_112) : "h"(_e4m3x2_decode_112));
        uint16_t _f16_decode_112 = (uint16_t)_f16x2_decode_112;
        asm("cvt.f32.f16 %0, %1;" : "=f"(ksf_dec) : "h"(_f16_decode_112));
        float _fdiv_113 = __fdiv_rn(kg, ksf_dec);
        float kout = ((ksf_dec != 0.0f) ? _fdiv_113 : 0.0f);
        float kn[16];
        for (int e_5 = 0; e_5 < 16; e_5++) {
            kn[e_5] = kf16[e_5] * kout;
        }
        unsigned int kw[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[0]) : "f"(kn[0]), "f"(kn[1]), "f"(kn[2]), "f"(kn[3]), "f"(kn[4]), "f"(kn[5]), "f"(kn[6]), "f"(kn[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(kw[1]) : "f"(kn[8]), "f"(kn[9]), "f"(kn[10]), "f"(kn[11]), "f"(kn[12]), "f"(kn[13]), "f"(kn[14]), "f"(kn[15]));
        unsigned int kstore[4];
        unsigned int _shfl_xor_sync_114 = __shfl_xor_sync(0xFFFFFFFF, kw[0], 1);
        kstore[0] = _shfl_xor_sync_114;
        unsigned int _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, kw[1], 1);
        kstore[1] = _shfl_xor_sync_115;
        kstore[2] = kw[0];
        kstore[3] = kw[1];
        {
            unsigned short _sf_pair;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_sf_pair) : "f"(ksf_val));
            *(reinterpret_cast<unsigned char*>(k_sf + (((head * num_kblocks + (tok >> 7)) * 128 + (tok & 127)) * 8 + sub)) + (0)) = (unsigned char)(_sf_pair & 0x7F);
        }
        if (valid == 1) {
            if ((sub & 1) == 1) {
                reinterpret_cast<int4*>(k4 + (((tok * num_heads + head) * 64 + (sub - 1) * 8) / 4))[0] = reinterpret_cast<int4*>(kstore)[0];
            }
        }
        float vq[16];
        float vf4[4];
        float _vec_load_7[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[0] * num_heads + head) * 128 + ch) + 0);
            _vec_load_7[0 + 0] = _v4.x;
            _vec_load_7[0 + 1] = _v4.y;
            _vec_load_7[0 + 2] = _v4.z;
            _vec_load_7[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_2[0];
        vf4[1] = _vec_load_2[1];
        vf4[2] = _vec_load_2[2];
        vf4[3] = _vec_load_2[3];
        float _fdiv_117 = __fdiv_rn(vf4[0], _vec_load_7[0]);
        vq[0] = ((vlive[0] == 1) ? _fdiv_117 : 0.0f);
        float _fdiv_118 = __fdiv_rn(vf4[1], _vec_load_7[1]);
        vq[1] = ((vlive[0] == 1) ? _fdiv_118 : 0.0f);
        float _fdiv_119 = __fdiv_rn(vf4[2], _vec_load_7[2]);
        vq[2] = ((vlive[0] == 1) ? _fdiv_119 : 0.0f);
        float _fdiv_120 = __fdiv_rn(vf4[3], _vec_load_7[3]);
        vq[3] = ((vlive[0] == 1) ? _fdiv_120 : 0.0f);
        float _vec_load_8[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[1] * num_heads + head) * 128 + ch) + 0);
            _vec_load_8[0 + 0] = _v4.x;
            _vec_load_8[0 + 1] = _v4.y;
            _vec_load_8[0 + 2] = _v4.z;
            _vec_load_8[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_3[0];
        vf4[1] = _vec_load_3[1];
        vf4[2] = _vec_load_3[2];
        vf4[3] = _vec_load_3[3];
        float _fdiv_122 = __fdiv_rn(vf4[0], _vec_load_8[0]);
        vq[4] = ((vlive[1] == 1) ? _fdiv_122 : 0.0f);
        float _fdiv_123 = __fdiv_rn(vf4[1], _vec_load_8[1]);
        vq[5] = ((vlive[1] == 1) ? _fdiv_123 : 0.0f);
        float _fdiv_124 = __fdiv_rn(vf4[2], _vec_load_8[2]);
        vq[6] = ((vlive[1] == 1) ? _fdiv_124 : 0.0f);
        float _fdiv_125 = __fdiv_rn(vf4[3], _vec_load_8[3]);
        vq[7] = ((vlive[1] == 1) ? _fdiv_125 : 0.0f);
        float _vec_load_9[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[2] * num_heads + head) * 128 + ch) + 0);
            _vec_load_9[0 + 0] = _v4.x;
            _vec_load_9[0 + 1] = _v4.y;
            _vec_load_9[0 + 2] = _v4.z;
            _vec_load_9[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_4[0];
        vf4[1] = _vec_load_4[1];
        vf4[2] = _vec_load_4[2];
        vf4[3] = _vec_load_4[3];
        float _fdiv_127 = __fdiv_rn(vf4[0], _vec_load_9[0]);
        vq[8] = ((vlive[2] == 1) ? _fdiv_127 : 0.0f);
        float _fdiv_128 = __fdiv_rn(vf4[1], _vec_load_9[1]);
        vq[9] = ((vlive[2] == 1) ? _fdiv_128 : 0.0f);
        float _fdiv_129 = __fdiv_rn(vf4[2], _vec_load_9[2]);
        vq[10] = ((vlive[2] == 1) ? _fdiv_129 : 0.0f);
        float _fdiv_130 = __fdiv_rn(vf4[3], _vec_load_9[3]);
        vq[11] = ((vlive[2] == 1) ? _fdiv_130 : 0.0f);
        float _vec_load_10[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(v_scale + ((vseg[3] * num_heads + head) * 128 + ch) + 0);
            _vec_load_10[0 + 0] = _v4.x;
            _vec_load_10[0 + 1] = _v4.y;
            _vec_load_10[0 + 2] = _v4.z;
            _vec_load_10[0 + 3] = _v4.w;
        }
        vf4[0] = _vec_load_5[0];
        vf4[1] = _vec_load_5[1];
        vf4[2] = _vec_load_5[2];
        vf4[3] = _vec_load_5[3];
        float _fdiv_132 = __fdiv_rn(vf4[0], _vec_load_10[0]);
        vq[12] = ((vlive[3] == 1) ? _fdiv_132 : 0.0f);
        float _fdiv_133 = __fdiv_rn(vf4[1], _vec_load_10[1]);
        vq[13] = ((vlive[3] == 1) ? _fdiv_133 : 0.0f);
        float _fdiv_134 = __fdiv_rn(vf4[2], _vec_load_10[2]);
        vq[14] = ((vlive[3] == 1) ? _fdiv_134 : 0.0f);
        float _fdiv_135 = __fdiv_rn(vf4[3], _vec_load_10[3]);
        vq[15] = ((vlive[3] == 1) ? _fdiv_135 : 0.0f);
        int vt_word = (head * 128 + ch) * total_padded + tok_base + chunk * 16 + 4 * t8 >> 2;
        for (int e_6 = 0; e_6 < 4; e_6++) {
            float four[4];
            for (int i = 0; i < 4; i++) {
                four[i] = vq[i * 4 + e_6];
            }
            unsigned int word[1];
            {
                uint32_t _packed;
                asm volatile("{\n\t"
                    ".reg .b16 _lo;\n\t"
                    ".reg .b16 _hi;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                    "mov.b32 %0, {_lo, _hi};\n\t"
                    "}"
                    : "=r"(_packed) : "f"(four[0]), "f"(four[1]),
                                       "f"(four[2]), "f"(four[3]));
                word[0] = _packed;
            }
            *(reinterpret_cast<unsigned int*>(vt8 + (vt_word + e_6 * (total_padded >> 2))) + (0)) = word[0];
        }
    }
}

} // extern "C"
