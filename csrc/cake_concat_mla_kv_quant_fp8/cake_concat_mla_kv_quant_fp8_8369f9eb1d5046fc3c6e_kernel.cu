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
#include "cake_concat_mla_kv_quant_fp8_device_common.cuh"

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_concat_mla_kv_quant_fp8_8369f9eb1d5046fc3c6e(const __nv_bfloat16* __restrict__ kv_nope, const __nv_bfloat16* __restrict__ k_pe, uint8_t* __restrict__ key, uint8_t* __restrict__ value, int num_tokens, int num_heads, int head_pairs, int warps_per_token, int k_pe_row_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // === Task calls (dependency order) ===
    int lane_0 = lane;
    int head_select = lane_0 >> 4;
    int sub = lane_0 & 15;
    int is_value = sub >> 3;
    int piece = sub & 7;
    int rope_slot = lane_0 >> 3;
    int rope_hsel = lane_0 >> 2 & 1;
    int rope_piece = lane_0 & 3;
    int warp_global = bid * 8 + warp;
    int token = warp_global / warps_per_token;
    int pair0 = (warp_global - token * warps_per_token) * 3;
    if (token < num_tokens) {
        long long kv_elems_per_token = (long long)num_heads * 256;
        long long key_elems_per_token = (long long)num_heads * 192;
        long long lane_head_stride = (long long)(((is_value == 1) ? 128 : 192));
        long long lane_elems_per_token = ((is_value == 1) ? (long long)num_heads * 128 : key_elems_per_token);
        long long kv_off = (long long)token * kv_elems_per_token + (long long)(pair0 * 512 + head_select * 256 + sub * 16);
        long long dst_off = (long long)token * lane_elems_per_token + (long long)(pair0 * 2 + head_select) * lane_head_stride + (long long)(piece * 16);
        long long dst_pair_step = 2 * lane_head_stride;
        long long rope_src = (long long)token * (long long)k_pe_row_stride + (long long)(rope_piece * 16);
        long long rope_dst = (long long)token * key_elems_per_token + (long long)(pair0 * 2 + rope_hsel) * 192 + (long long)(128 + rope_piece * 16);
        int head0 = pair0 * 2 + head_select;
        unsigned int words[24];
        unsigned int rope_words[8];
        float vals[16];
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            if (head0 + 2 * i < num_heads) {
                {
                    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(words[i * 8 + 0]), "=r"(words[i * 8 + 1]), "=r"(words[i * 8 + 2]), "=r"(words[i * 8 + 3]), "=r"(words[i * 8 + 4]), "=r"(words[i * 8 + 5]), "=r"(words[i * 8 + 6]), "=r"(words[i * 8 + 7]) : "l"((const void*)((const char*)(kv_nope + kv_off + (long long)(i * 512)) + 0)) : "memory");
                }
            }
        }
        {
            asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(rope_words[0 + 0]), "=r"(rope_words[0 + 1]), "=r"(rope_words[0 + 2]), "=r"(rope_words[0 + 3]), "=r"(rope_words[0 + 4]), "=r"(rope_words[0 + 5]), "=r"(rope_words[0 + 6]), "=r"(rope_words[0 + 7]) : "l"((const void*)((const char*)(k_pe + rope_src) + 0)) : "memory");
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 3; i_1++) {
            if (head0 + 2 * i_1 < num_heads) {
                vals[0] = __uint_as_float(words[i_1 * 8] << 16);
                vals[1] = __uint_as_float(words[i_1 * 8] & 4294901760u);
                vals[2] = __uint_as_float(words[i_1 * 8 + 1] << 16);
                vals[3] = __uint_as_float(words[i_1 * 8 + 1] & 4294901760u);
                vals[4] = __uint_as_float(words[i_1 * 8 + 2] << 16);
                vals[5] = __uint_as_float(words[i_1 * 8 + 2] & 4294901760u);
                vals[6] = __uint_as_float(words[i_1 * 8 + 3] << 16);
                vals[7] = __uint_as_float(words[i_1 * 8 + 3] & 4294901760u);
                vals[8] = __uint_as_float(words[i_1 * 8 + 4] << 16);
                vals[9] = __uint_as_float(words[i_1 * 8 + 4] & 4294901760u);
                vals[10] = __uint_as_float(words[i_1 * 8 + 5] << 16);
                vals[11] = __uint_as_float(words[i_1 * 8 + 5] & 4294901760u);
                vals[12] = __uint_as_float(words[i_1 * 8 + 6] << 16);
                vals[13] = __uint_as_float(words[i_1 * 8 + 6] & 4294901760u);
                vals[14] = __uint_as_float(words[i_1 * 8 + 7] << 16);
                vals[15] = __uint_as_float(words[i_1 * 8 + 7] & 4294901760u);
                {
                    unsigned int _fp8_pk[4];
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[0]) : "f"(vals[0 + 0]), "f"(vals[0 + 1]), "f"(vals[0 + 2]), "f"(vals[0 + 3]));
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[1]) : "f"(vals[0 + 4]), "f"(vals[0 + 5]), "f"(vals[0 + 6]), "f"(vals[0 + 7]));
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[2]) : "f"(vals[0 + 8]), "f"(vals[0 + 9]), "f"(vals[0 + 10]), "f"(vals[0 + 11]));
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[3]) : "f"(vals[0 + 12]), "f"(vals[0 + 13]), "f"(vals[0 + 14]), "f"(vals[0 + 15]));
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(((is_value == 1) ? value : key) + (dst_off + (long long)i_1 * dst_pair_step)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                }
            }
        }
        vals[0] = __uint_as_float(rope_words[0] << 16);
        vals[1] = __uint_as_float(rope_words[0] & 4294901760u);
        vals[2] = __uint_as_float(rope_words[1] << 16);
        vals[3] = __uint_as_float(rope_words[1] & 4294901760u);
        vals[4] = __uint_as_float(rope_words[2] << 16);
        vals[5] = __uint_as_float(rope_words[2] & 4294901760u);
        vals[6] = __uint_as_float(rope_words[3] << 16);
        vals[7] = __uint_as_float(rope_words[3] & 4294901760u);
        vals[8] = __uint_as_float(rope_words[4] << 16);
        vals[9] = __uint_as_float(rope_words[4] & 4294901760u);
        vals[10] = __uint_as_float(rope_words[5] << 16);
        vals[11] = __uint_as_float(rope_words[5] & 4294901760u);
        vals[12] = __uint_as_float(rope_words[6] << 16);
        vals[13] = __uint_as_float(rope_words[6] & 4294901760u);
        vals[14] = __uint_as_float(rope_words[7] << 16);
        vals[15] = __uint_as_float(rope_words[7] & 4294901760u);
        #pragma unroll
        for (int k = 0; k < 1; k++) {
            int slot = k * 4 + rope_slot;
            if (slot < 3 && pair0 * 2 + rope_hsel + 2 * slot < num_heads) {
                {
                    unsigned int _fp8_pk[4];
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[0]) : "f"(vals[0 + 0]), "f"(vals[0 + 1]), "f"(vals[0 + 2]), "f"(vals[0 + 3]));
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[1]) : "f"(vals[0 + 4]), "f"(vals[0 + 5]), "f"(vals[0 + 6]), "f"(vals[0 + 7]));
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[2]) : "f"(vals[0 + 8]), "f"(vals[0 + 9]), "f"(vals[0 + 10]), "f"(vals[0 + 11]));
                    asm("{\n\t"
                        ".reg .b16 _lo, _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}\n"
                        : "=r"(_fp8_pk[3]) : "f"(vals[0 + 12]), "f"(vals[0 + 13]), "f"(vals[0 + 14]), "f"(vals[0 + 15]));
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(key + (rope_dst + (long long)slot * 384)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                }
            }
        }
    }
}

} // extern "C"
