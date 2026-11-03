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

// Common preamble (typedefs, tensor-map ABI, compiler helpers) shared by this export's kernels.
#include "cake_mla_nvfp4_paged_decode_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_mla_nvfp4_paged_decode_a9bf6a63f937e7cacaba(__nv_bfloat16* __restrict__ q_bf16, unsigned int* __restrict__ q_nope, uint8_t* __restrict__ q_sf, unsigned int* __restrict__ q_rope, float* __restrict__ q_scale, int rows, float c_nope, float c_rope, float kpe_scale, float ckv_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int row = blockIdx.x * 4 + warp;
    if (row < rows) {
        int lane_0 = lane;
        int base = row * 576;
        float v[16];
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q_bf16 + base + lane_0 * 16);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_1[8];
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(q_bf16 + base + lane_0 * 16 + 8);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            v[j] = _vec_load_0[j];
            v[8 + j] = _vec_load_1[j];
        }
        float _vec_load_2[2];
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(q_bf16 + base + 512 + lane_0 * 2);
            (&_vec_load_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) << 16);
            (&_vec_load_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_bits_2) & 0xffff0000u);
        }
        float amax_b = 0.0f;
        #pragma unroll
        for (int j_1 = 0; j_1 < 16; j_1++) {
            float _fabs_0 = fabsf(v[j_1]);
            float _max_0 = max_noftz(amax_b, _fabs_0);
            amax_b = _max_0;
        }
        float _fabs_1 = fabsf(_vec_load_2[0]);
        float _fabs_2 = fabsf(_vec_load_2[1]);
        float _max_1 = max_noftz(_fabs_1, _fabs_2);
        float amax_r_l = _max_1;
        float _warp_reduce_0 = amax_b;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
        float amax_n = _warp_reduce_0;
        float _warp_reduce_1 = amax_r_l;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 = max_noftz(_warp_reduce_1, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset));
        float amax_r = _warp_reduce_1;
        float _max_2 = max_noftz(amax_n * c_nope, amax_r * c_rope);
        float qs_raw = _max_2;
        float qs = ((qs_raw > 0.0f) ? qs_raw : 1.0f);
        float _rcp_0 = __frcp_rn(6.0f * qs);
        float inv6 = _rcp_0;
        float sf_val = ((amax_b > 0.0f) ? amax_b * inv6 : 0.0f);
        uint16_t _e4m3x2_f32_0;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(0.0f), "f"(sf_val));
        uint16_t sf_pair = _e4m3x2_f32_0;
        unsigned int sf_byte = (unsigned int)sf_pair & 255;
        float2 _fp8x2_decode_0;
        asm("{ .reg .b32 pair; .reg .b16 lo, hi;\n"
            "cvt.rn.f16x2.e4m3x2 pair, %2;\n"
            "mov.b32 {lo, hi}, pair;\n"
            "cvt.f32.f16 %0, lo; cvt.f32.f16 %1, hi; }"
            : "=f"(_fp8x2_decode_0.x), "=f"(_fp8x2_decode_0.y) : "h"(sf_pair));
        float sf_f = _fp8x2_decode_0.x;
        float _rcp_1 = __frcp_rn(sf_f * qs);
        float out_scale_raw = _rcp_1;
        float out_scale = ((sf_f > 0.0f) ? out_scale_raw : 0.0f);
        #pragma unroll
        for (int j_2 = 0; j_2 < 16; j_2++) {
            v[j_2] = v[j_2] * out_scale;
        }
        unsigned int w0 = 0;
        unsigned int w1 = 0;
        #pragma unroll
        for (int j_3 = 0; j_3 < 4; j_3++) {
            uint32_t _fp4_pair_0;
            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_0) : "f"(v[2 * j_3]), "f"(v[2 * j_3 + 1]));
            w0 = w0 | _fp4_pair_0 << (unsigned int)(8 * j_3);
            uint32_t _fp4_pair_1;
            asm("{\n"             ".reg .b8 byte0;\n"             "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"             "mov.b32 %0, {byte0, 0, 0, 0};\n"             "}\n"             : "=r"(_fp4_pair_1) : "f"(v[8 + 2 * j_3]), "f"(v[8 + 2 * j_3 + 1]));
            w1 = w1 | _fp4_pair_1 << (unsigned int)(8 * j_3);
        }
        *(reinterpret_cast<unsigned int*>(q_nope + (row * 64 + lane_0 * 2)) + (0)) = w0;
        *(reinterpret_cast<unsigned int*>(q_nope + (row * 64 + lane_0 * 2 + 1)) + (0)) = w1;
        *(reinterpret_cast<unsigned char*>(q_sf + (row * 32 + lane_0)) + (0)) = (unsigned char)(sf_byte);
        float _rcp_2 = __frcp_rn(qs * ckv_scale);
        float rope_mul = kpe_scale * _rcp_2;
        uint16_t _e4m3x2_f32_1;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_vec_load_2[1] * rope_mul), "f"(_vec_load_2[0] * rope_mul));
        uint16_t rpair = _e4m3x2_f32_1;
        unsigned int rw = (unsigned int)rpair;
        unsigned int _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, rw, 1, 32);
        unsigned int rw_hi = _shfl_down_0;
        if ((lane_0 & 1) == 0) {
            *(reinterpret_cast<unsigned int*>(q_rope + (row * 16 + (lane_0 >> 1))) + (0)) = rw | rw_hi << 16;
        }
        if (lane_0 == 0) {
            *(reinterpret_cast<float*>(q_scale + row) + (0)) = qs;
        }
    }
}

} // extern "C"
