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

#include "cake_dsa_h64_train_device.cuh"

#define LAUNCH_MIN_BLOCKS 1

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_h64_nope_train_ae9db52dff326ae8409b(__nv_bfloat16* __restrict__ dout, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ delta, int num_rows, int dout_token_stride, int dout_head_stride, int dout_offset)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    // === Task calls (declaration order) ===
    int row = blockIdx.x * 8 + warp;
    if (row < num_rows) {
        long long base = (long long)row * 512 + (long long)(lane * 16);
        int token = row / 64;
        int head = row % 64;
        long long gbase = (long long)dout_offset + (long long)token * (long long)dout_token_stride + (long long)head * (long long)dout_head_stride + (long long)(lane * 16);
        float acc = 0.0f;
        float _vec_load_0[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(dout + gbase + 0);
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
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(out + base + 0);
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
        float _vec_load_2[8];
        {
            const uint4* _vptr_2 = reinterpret_cast<const uint4*>(o_lo + base + 0);
            uint4 _vld_2[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_2[_blk] = _vptr_2[_blk];
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                }
            }
        }
        acc = __fmaf_rn(_vec_load_0[0], _vec_load_1[0] + _vec_load_2[0], acc);
        acc = __fmaf_rn(_vec_load_0[1], _vec_load_1[1] + _vec_load_2[1], acc);
        acc = __fmaf_rn(_vec_load_0[2], _vec_load_1[2] + _vec_load_2[2], acc);
        acc = __fmaf_rn(_vec_load_0[3], _vec_load_1[3] + _vec_load_2[3], acc);
        acc = __fmaf_rn(_vec_load_0[4], _vec_load_1[4] + _vec_load_2[4], acc);
        acc = __fmaf_rn(_vec_load_0[5], _vec_load_1[5] + _vec_load_2[5], acc);
        acc = __fmaf_rn(_vec_load_0[6], _vec_load_1[6] + _vec_load_2[6], acc);
        acc = __fmaf_rn(_vec_load_0[7], _vec_load_1[7] + _vec_load_2[7], acc);
        float _vec_load_3[8];
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(dout + (gbase + 8) + 0);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_4[8];
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(out + (base + 8) + 0);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        float _vec_load_5[8];
        {
            const uint4* _vptr_5 = reinterpret_cast<const uint4*>(o_lo + (base + 8) + 0);
            uint4 _vld_5[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_5[_blk] = _vptr_5[_blk];
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                }
            }
        }
        acc = __fmaf_rn(_vec_load_3[0], _vec_load_4[0] + _vec_load_5[0], acc);
        acc = __fmaf_rn(_vec_load_3[1], _vec_load_4[1] + _vec_load_5[1], acc);
        acc = __fmaf_rn(_vec_load_3[2], _vec_load_4[2] + _vec_load_5[2], acc);
        acc = __fmaf_rn(_vec_load_3[3], _vec_load_4[3] + _vec_load_5[3], acc);
        acc = __fmaf_rn(_vec_load_3[4], _vec_load_4[4] + _vec_load_5[4], acc);
        acc = __fmaf_rn(_vec_load_3[5], _vec_load_4[5] + _vec_load_5[5], acc);
        acc = __fmaf_rn(_vec_load_3[6], _vec_load_4[6] + _vec_load_5[6], acc);
        acc = __fmaf_rn(_vec_load_3[7], _vec_load_4[7] + _vec_load_5[7], acc);
        float total = acc;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            total += __shfl_xor_sync(0xFFFFFFFF, total, offset);
        if (lane == 0) {
            delta[row] = total;
        }
    }
}

} // extern "C"
