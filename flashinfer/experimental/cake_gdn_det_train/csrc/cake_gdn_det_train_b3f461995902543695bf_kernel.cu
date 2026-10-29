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
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_b3f461995902543695bf(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k, __nv_bfloat16* __restrict__ dq_hv, __nv_bfloat16* __restrict__ dk_hv, __nv_bfloat16* __restrict__ dk2, float* __restrict__ dg1, float* __restrict__ dg2, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, float* __restrict__ dg_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int normalize_qk)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (declaration order) ===
    int chunk = blockIdx.x;
    int hq = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int group = num_v_heads / num_heads;
    int hv0 = hq * group;
    int lane_0 = lane;
    float dq_a[4];
    float dk_a[4];
    float dq_b[4];
    float dk_b[4];
    float yq_a[4];
    float yk_a[4];
    float yq_b[4];
    float yk_b[4];
    float pq_a0[4];
    float pk_a0[4];
    float p2_a0[4];
    float pq_b0[4];
    float pk_b0[4];
    float p2_b0[4];
    float pq_a1[4];
    float pk_a1[4];
    float p2_a1[4];
    float pq_b1[4];
    float pk_b1[4];
    float p2_b1[4];
    float pq_a2[4];
    float pk_a2[4];
    float p2_a2[4];
    float pq_b2[4];
    float pk_b2[4];
    float p2_b2[4];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        int row_a = warp * 16 + r;
        int row_b = row_a + 8;
        if (row_a < n_valid) {
            int row_b_src = row_b;
            if (row_b >= n_valid) {
                row_b_src = n_valid - 1;
            }
            long long tok_a = (long long)(tok0 + row_a);
            long long tok_b = (long long)(tok0 + row_b_src);
            long long qbase_a = (tok_a * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
            long long qbase_b = (tok_b * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
            long long vbase_a = (tok_a * (long long)num_v_heads + (long long)hv0) * 128 + (long long)(lane_0 * 4);
            long long vbase_b = (tok_b * (long long)num_v_heads + (long long)hv0) * 128 + (long long)(lane_0 * 4);
            {
                uint2 _vld_0;
                _vld_0 = *reinterpret_cast<const uint2*>(qn + qbase_a);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_a[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&yq_a[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_1;
                _vld_1 = *reinterpret_cast<const uint2*>(kn + qbase_a);
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_a[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&yk_a[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_2;
                _vld_2 = *reinterpret_cast<const uint2*>(qn + qbase_b);
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yq_b[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&yq_b[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_3;
                _vld_3 = *reinterpret_cast<const uint2*>(kn + qbase_b);
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&yk_b[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&yk_b[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
            float rq_a = rstd_q[tok_a * (long long)num_heads + (long long)hq];
            float rk_a = rstd_k[tok_a * (long long)num_heads + (long long)hq];
            float rq_b = rstd_q[tok_b * (long long)num_heads + (long long)hq];
            float rk_b = rstd_k[tok_b * (long long)num_heads + (long long)hq];
            {
                uint2 _vld_4;
                _vld_4 = *reinterpret_cast<const uint2*>(dq_hv + vbase_a);
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pq_a0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&pq_a0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_5;
                _vld_5 = *reinterpret_cast<const uint2*>(dk_hv + vbase_a);
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pk_a0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&pk_a0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_6;
                _vld_6 = *reinterpret_cast<const uint2*>(dk2 + vbase_a);
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&p2_a0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                    (&p2_a0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_7;
                _vld_7 = *reinterpret_cast<const uint2*>(dq_hv + vbase_b);
                uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pq_b0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                    (&pq_b0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_8;
                _vld_8 = *reinterpret_cast<const uint2*>(dk_hv + vbase_b);
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pk_b0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&pk_b0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_9;
                _vld_9 = *reinterpret_cast<const uint2*>(dk2 + vbase_b);
                uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&p2_b0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                    (&p2_b0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_10;
                _vld_10 = *reinterpret_cast<const uint2*>(dq_hv + vbase_a + 128);
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pq_a1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&pq_a1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_11;
                _vld_11 = *reinterpret_cast<const uint2*>(dk_hv + vbase_a + 128);
                uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pk_a1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                    (&pk_a1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_12;
                _vld_12 = *reinterpret_cast<const uint2*>(dk2 + vbase_a + 128);
                uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&p2_a1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) << 16);
                    (&p2_a1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_12[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_13;
                _vld_13 = *reinterpret_cast<const uint2*>(dq_hv + vbase_b + 128);
                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pq_b1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                    (&pq_b1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_14;
                _vld_14 = *reinterpret_cast<const uint2*>(dk_hv + vbase_b + 128);
                uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pk_b1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) << 16);
                    (&pk_b1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_14[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_15;
                _vld_15 = *reinterpret_cast<const uint2*>(dk2 + vbase_b + 128);
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&p2_b1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                    (&p2_b1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_16;
                _vld_16 = *reinterpret_cast<const uint2*>(dq_hv + vbase_a + 256);
                uint32_t* _vpairs_16 = reinterpret_cast<uint32_t*>(&_vld_16);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pq_a2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) << 16);
                    (&pq_a2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_16[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_17;
                _vld_17 = *reinterpret_cast<const uint2*>(dk_hv + vbase_a + 256);
                uint32_t* _vpairs_17 = reinterpret_cast<uint32_t*>(&_vld_17);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pk_a2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) << 16);
                    (&pk_a2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_17[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_18;
                _vld_18 = *reinterpret_cast<const uint2*>(dk2 + vbase_a + 256);
                uint32_t* _vpairs_18 = reinterpret_cast<uint32_t*>(&_vld_18);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&p2_a2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) << 16);
                    (&p2_a2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_18[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_19;
                _vld_19 = *reinterpret_cast<const uint2*>(dq_hv + vbase_b + 256);
                uint32_t* _vpairs_19 = reinterpret_cast<uint32_t*>(&_vld_19);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pq_b2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) << 16);
                    (&pq_b2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_19[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_20;
                _vld_20 = *reinterpret_cast<const uint2*>(dk_hv + vbase_b + 256);
                uint32_t* _vpairs_20 = reinterpret_cast<uint32_t*>(&_vld_20);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&pk_b2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) << 16);
                    (&pk_b2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_20[_pair]) & 0xffff0000u);
                }
            }
            {
                uint2 _vld_21;
                _vld_21 = *reinterpret_cast<const uint2*>(dk2 + vbase_b + 256);
                uint32_t* _vpairs_21 = reinterpret_cast<uint32_t*>(&_vld_21);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&p2_b2[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) << 16);
                    (&p2_b2[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_21[_pair]) & 0xffff0000u);
                }
            }
            dq_a[0] = 0.0f;
            dq_a[1] = 0.0f;
            dq_a[2] = 0.0f;
            dq_a[3] = 0.0f;
            dk_a[0] = 0.0f;
            dk_a[1] = 0.0f;
            dk_a[2] = 0.0f;
            dk_a[3] = 0.0f;
            dq_b[0] = 0.0f;
            dq_b[1] = 0.0f;
            dq_b[2] = 0.0f;
            dq_b[3] = 0.0f;
            dk_b[0] = 0.0f;
            dk_b[1] = 0.0f;
            dk_b[2] = 0.0f;
            dk_b[3] = 0.0f;
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_a[_la] = dq_a[_la] + pq_a0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_a[_la] = dk_a[_la] + pk_a0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_a[_la] = dk_a[_la] + p2_a0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_b[_la] = dq_b[_la] + pq_b0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_b[_la] = dk_b[_la] + pk_b0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_b[_la] = dk_b[_la] + p2_b0[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_a[_la] = dq_a[_la] + pq_a1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_a[_la] = dk_a[_la] + pk_a1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_a[_la] = dk_a[_la] + p2_a1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_b[_la] = dq_b[_la] + pq_b1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_b[_la] = dk_b[_la] + pk_b1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_b[_la] = dk_b[_la] + p2_b1[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_a[_la] = dq_a[_la] + pq_a2[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_a[_la] = dk_a[_la] + pk_a2[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_a[_la] = dk_a[_la] + p2_a2[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dq_b[_la] = dq_b[_la] + pq_b2[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_b[_la] = dk_b[_la] + pk_b2[_la];
            #pragma unroll
            for (int _la = 0; _la < 4; _la++)
                dk_b[_la] = dk_b[_la] + p2_b2[_la];
            float inner_q_a = 0.0f;
            float inner_k_a = 0.0f;
            float inner_q_b = 0.0f;
            float inner_k_b = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                inner_q_a = __fmaf_rn(dq_a[i], yq_a[i], inner_q_a);
                inner_k_a = __fmaf_rn(dk_a[i], yk_a[i], inner_k_a);
                inner_q_b = __fmaf_rn(dq_b[i], yq_b[i], inner_q_b);
                inner_k_b = __fmaf_rn(dk_b[i], yk_b[i], inner_k_b);
            }
            float _warp_reduce_22 = inner_q_a;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_22 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_22, offset);
            inner_q_a = _warp_reduce_22;
            float _warp_reduce_23 = inner_k_a;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_23 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_23, offset);
            inner_k_a = _warp_reduce_23;
            float _warp_reduce_24 = inner_q_b;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_24 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_24, offset);
            inner_q_b = _warp_reduce_24;
            float _warp_reduce_25 = inner_k_b;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_25 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_25, offset);
            inner_k_b = _warp_reduce_25;
            if (normalize_qk != 0) {
                #pragma unroll
                for (int i2 = 0; i2 < 4; i2++) {
                    dq_a[i2] = __fmaf_rn(dq_a[i2], rq_a, -(inner_q_a * yq_a[i2] * rq_a));
                    dk_a[i2] = __fmaf_rn(dk_a[i2], rk_a, -(inner_k_a * yk_a[i2] * rk_a));
                    dq_b[i2] = __fmaf_rn(dq_b[i2], rq_b, -(inner_q_b * yq_b[i2] * rq_b));
                    dk_b[i2] = __fmaf_rn(dk_b[i2], rk_b, -(inner_k_b * yk_b[i2] * rk_b));
                }
            }
            unsigned int _packed_0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_a[_lp*2 + 0], dq_a[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_a[_lp*2 + 2], dq_a[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dq_out + (qbase_a))[0] = reinterpret_cast<int2*>(_packed_0)[0];
            unsigned int _packed_1[2];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_a[_lp*2 + 0], dk_a[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_a[_lp*2 + 2], dk_a[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int2*>(dk_out + (qbase_a))[0] = reinterpret_cast<int2*>(_packed_1)[0];
            if (row_b < n_valid) {
                unsigned int _packed_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_b[_lp*2 + 0], dq_b[_lp*2+1 + 0]));
                    _packed_2[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_b[_lp*2 + 2], dq_b[_lp*2+1 + 2]));
                    _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dq_out + (qbase_b))[0] = reinterpret_cast<int2*>(_packed_2)[0];
                unsigned int _packed_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_b[_lp*2 + 0], dk_b[_lp*2+1 + 0]));
                    _packed_3[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dk_b[_lp*2 + 2], dk_b[_lp*2+1 + 2]));
                    _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int2*>(dk_out + (qbase_b))[0] = reinterpret_cast<int2*>(_packed_3)[0];
            }
        }
    }
    #pragma unroll 1
    for (int gi = warp; gi < group; gi += 4) {
        int hv = hv0 + gi;
        float d_lo = 0.0f;
        float d_hi = 0.0f;
        long long gate_index_lo = (long long)(tok0 + lane_0) * (long long)num_v_heads + (long long)hv;
        long long gate_index_hi = (long long)(tok0 + lane_0 + 32) * (long long)num_v_heads + (long long)hv;
        if (lane_0 < n_valid) {
            d_lo = dg1[gate_index_lo] + dg2[gate_index_lo];
        }
        if (n_valid > lane_0 + 32) {
            d_hi = dg1[gate_index_hi] + dg2[gate_index_hi];
        }
        float self_lo = d_lo;
        float self_hi = d_hi;
        #pragma unroll
        for (int delta = 0; delta < 5; delta++) {
            const int shift = 1 << delta;
            float up_lo = __shfl_up_sync(0xFFFFFFFF, d_lo, shift, 32);
            float up_hi = __shfl_up_sync(0xFFFFFFFF, d_hi, shift, 32);
            if (lane_0 >= shift) {
                d_lo = d_lo + up_lo;
                d_hi = d_hi + up_hi;
            }
        }
        float total_lo = __shfl_sync(0xFFFFFFFF, d_lo, 31);
        float total_hi = __shfl_sync(0xFFFFFFFF, d_hi, 31);
        float total = total_lo + total_hi;
        if (lane_0 < n_valid) {
            dg_out[gate_index_lo] = total - d_lo + self_lo;
        }
        if (n_valid > lane_0 + 32) {
            dg_out[gate_index_hi] = total - total_lo - d_hi + self_hi;
        }
    }
}

} // extern "C"
