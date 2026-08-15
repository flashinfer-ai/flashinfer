/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256
#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 4
#endif
#ifndef PAGE_SIZE
#define PAGE_SIZE 16
#endif
#ifndef VPT
#define VPT 4
#endif

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_fmha_context_nvfp4_dequant(uint8_t* __restrict__ K_packed, uint8_t* __restrict__ V_packed, uint8_t* __restrict__ K_scales, uint8_t* __restrict__ V_scales, uint8_t* __restrict__ K_output, uint8_t* __restrict__ V_output, int total_groups, int output_page_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int groups_per_head = PAGE_SIZE * 8;
    int groups_per_page = NUM_KV_HEADS * groups_per_head;
    #pragma unroll 1
    for (int item = 0; item < VPT; item++) {
        int group_idx = (bid * VPT + item) * 256 + tid;
        if (group_idx < total_groups) {
            int page = group_idx / groups_per_page;
            int within_page = group_idx % groups_per_page;
            int head = within_page / groups_per_head;
            int within_head = within_page % groups_per_head;
            int token = within_head / 8;
            int scale_group = within_head % 8;
            int output_idx = page * output_page_stride + (head * PAGE_SIZE + token) * 128 + scale_group * 16;
            float _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const unsigned int*>(K_packed + group_idx * 8);
            }
            float _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const unsigned int*>(K_packed + group_idx * 8 + 4);
            }
            float _vec_load_2[1];
            {
                uint8_t _fp8_byte_0 = *reinterpret_cast<const uint8_t*>(K_scales + group_idx);
                uint16_t _e4m3x2_0 = (uint16_t)_fp8_byte_0;
                uint32_t _f16x2_0;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                uint16_t _h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_2[0]) : "h"(_h0_0));
            }
            unsigned int p0 = _vec_load_0[0];
            unsigned int p1 = _vec_load_1[0];
            float sf = _vec_load_2[0];
            float values[16];
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned int word = ((j < 8) ? p0 : p1);
                unsigned int code = word >> (unsigned int)(j % 8 * 4) & 15;
                unsigned int magnitude = code & 7;
                unsigned int bits = (code & 8) << 28;
                if (magnitude != 0) {
                    unsigned int exponent = 125 + (magnitude + 1) / 2;
                    bits = bits | exponent << 23;
                    if (magnitude > 1) {
                        bits = bits | (magnitude & 1) << 22;
                    }
                }
                float value;
                value = reinterpret_cast<float*>(&bits)[0];
                values[j] = value * sf;
            }
            unsigned int encoded[4];
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values[1]), "f"(values[0]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values[3]), "f"(values[2]));
                encoded[0] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values[5]), "f"(values[4]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values[7]), "f"(values[6]));
                encoded[1] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values[9]), "f"(values[8]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values[11]), "f"(values[10]));
                encoded[2] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values[13]), "f"(values[12]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values[15]), "f"(values[14]));
                encoded[3] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                int4 _iv4 = make_int4(encoded[0 + 0], encoded[0 + 1], encoded[0 + 2], encoded[0 + 3]);
                *reinterpret_cast<int4*>(K_output + output_idx + 0) = _iv4;
            }
            int swizzled_token = token & -4 | scale_group >> 1;
            int swizzled_group = (scale_group & 1) << 2 | token & 3;
            int v_scale_idx = page * groups_per_page + head * groups_per_head + swizzled_token * 8 + swizzled_group;
            float _vec_load_3[1];
            {
                _vec_load_3[0] = *reinterpret_cast<const unsigned int*>(V_packed + group_idx * 8);
            }
            float _vec_load_4[1];
            {
                _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(V_packed + group_idx * 8 + 4);
            }
            float _vec_load_5[1];
            {
                uint8_t _fp8_byte_1 = *reinterpret_cast<const uint8_t*>(V_scales + v_scale_idx);
                uint16_t _e4m3x2_1 = (uint16_t)_fp8_byte_1;
                uint32_t _f16x2_1;
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                uint16_t _h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_vec_load_5[0]) : "h"(_h0_1));
            }
            unsigned int p0_0 = _vec_load_3[0];
            unsigned int p1_1 = _vec_load_4[0];
            float sf_2 = _vec_load_5[0];
            float values_3[16];
            #pragma unroll
            for (int j_1 = 0; j_1 < 16; j_1++) {
                unsigned int word_1 = ((j_1 < 8) ? p0_0 : p1_1);
                unsigned int code_1 = word_1 >> (unsigned int)(j_1 % 8 * 4) & 15;
                unsigned int magnitude_1 = code_1 & 7;
                unsigned int bits_1 = (code_1 & 8) << 28;
                if (magnitude_1 != 0) {
                    unsigned int exponent_1 = 125 + (magnitude_1 + 1) / 2;
                    bits_1 = bits_1 | exponent_1 << 23;
                    if (magnitude_1 > 1) {
                        bits_1 = bits_1 | (magnitude_1 & 1) << 22;
                    }
                }
                float value_1;
                value_1 = reinterpret_cast<float*>(&bits_1)[0];
                values_3[j_1] = value_1 * sf_2;
            }
            unsigned int encoded_4[4];
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values_3[1]), "f"(values_3[0]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values_3[3]), "f"(values_3[2]));
                encoded_4[0] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values_3[5]), "f"(values_3[4]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values_3[7]), "f"(values_3[6]));
                encoded_4[1] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values_3[9]), "f"(values_3[8]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values_3[11]), "f"(values_3[10]));
                encoded_4[2] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                unsigned short _lo, _hi;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_lo) : "f"(values_3[13]), "f"(values_3[12]));
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_hi) : "f"(values_3[15]), "f"(values_3[14]));
                encoded_4[3] = (unsigned)_lo | ((unsigned)_hi << 16);
            }
            {
                int4 _iv4 = make_int4(encoded_4[0 + 0], encoded_4[0 + 1], encoded_4[0 + 2], encoded_4[0 + 3]);
                *reinterpret_cast<int4*>(V_output + output_idx + 0) = _iv4;
            }
        }
    }
}

} // extern "C"

