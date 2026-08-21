/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>
#include <cuda_fp16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define NUM_MAIN_STAGES 1
#define THREADS 32

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(32) void
kernel_flashinfer_mnnvl_moe_alltoall_quantize_combine(float* __restrict__ accumulated, uint8_t* __restrict__ quantized_fp8, uint8_t* __restrict__ quantized_packed, uint8_t* __restrict__ scales_u8, uint8_t* __restrict__ scales_fp8, int elements_per_token, int payload_dtype_code, int quant_mode, int scale_layout, float output_scalar_scale, int blocks_per_row, int padded_scale_cols, bool enable_pdl)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    if (enable_pdl) {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int logical_block = bid;
    int token = logical_block / blocks_per_row;
    int block_column = logical_block - token * blocks_per_row;
    int block_size = 32;
    if (quant_mode == 3) {
        block_size = 16;
    }
    int column = block_column * block_size + lane;
    int active_lane = ((block_size > lane && elements_per_token > column) ? 1 : 0);
    float value = 0.0f;
    if (active_lane != 0) {
        value = accumulated[token * elements_per_token + column];
        if (payload_dtype_code == 0) {
            value = __bfloat162float(__float2bfloat16(value));
        }
        if (payload_dtype_code == 1) {
            value = __half2float(__float2half_rn(value));
        }
    }
    float scaled_value = value;
    if (quant_mode == 3) {
        scaled_value = value * output_scalar_scale;
    }
    float _fabs_0 = fabsf(scaled_value);
    float block_max = _fabs_0;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, block_max, 16);
    float _max_0 = max_noftz(block_max, _shfl_xor_0);
    block_max = _max_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_max, 8);
    float _max_1 = max_noftz(block_max, _shfl_xor_1);
    block_max = _max_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, block_max, 4);
    float _max_2 = max_noftz(block_max, _shfl_xor_2);
    block_max = _max_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_max, 2);
    float _max_3 = max_noftz(block_max, _shfl_xor_3);
    block_max = _max_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, block_max, 1);
    float _max_4 = max_noftz(block_max, _shfl_xor_4);
    block_max = _max_4;
    float _shfl_0 = __shfl_sync(0xFFFFFFFF, block_max, 0);
    block_max = _shfl_0;
    int scale_byte = 0;
    float actual_scale = 0.0f;
    float fp8_scale = 0.0f;
    if (quant_mode == 3) {
        float _fp8_rt_0;
        uint16_t _e4m3x2_0;
        uint32_t _f16x2_0;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_0) : "f"(0.0f), "f"(block_max * 0.16666666666666666f));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
        uint16_t _fp8_h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_0));
        fp8_scale = _fp8_rt_0;
        float _max_5 = max_noftz(fp8_scale, 1e-12f);
        actual_scale = _max_5;
    } else {
        float denominator = 6.0f;
        if (quant_mode == 1) {
            denominator = 448.0f;
        }
        float raw_scale = block_max / denominator;
        int raw_bits = 0;
        raw_bits = reinterpret_cast<int*>(&raw_scale)[0];
        int exponent = raw_bits >> 23 & 255;
        int mantissa = raw_bits & 8388607;
        int round_up = 0;
        if (mantissa != 0) {
            if (exponent != 0) {
                round_up = 1;
            } else if (mantissa > 4194304) {
                round_up = 1;
            }
        }
        if (raw_scale > 0.0f) {
            scale_byte = exponent + round_up;
            if (scale_byte > 254) {
                scale_byte = 254;
            }
        }
        int scale_bits = scale_byte << 23;
        actual_scale = reinterpret_cast<float*>(&scale_bits)[0];
        if (scale_byte == 0) {
            actual_scale = 1e-12f;
        }
    }
    int scale_index = token * blocks_per_row + block_column;
    if (scale_layout == 0) {
        scale_index = block_column % 4 + block_column / 4 * 512 + token % 32 * 16 + token % 128 / 32 * 4 + token / 128 * (128 * padded_scale_cols);
    }
    if (scale_layout == 1) {
        int tiles = padded_scale_cols / 4;
        scale_index = token / 8 * (tiles * 32) + block_column / 4 * 32 + token % 8 * 4 + block_column % 4;
    }
    if (lane == 0) {
        if (quant_mode == 3) {
            {
                unsigned short _fp8_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(fp8_scale));
                *(reinterpret_cast<unsigned char*>(scales_fp8 + scale_index) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
            }
        } else {
            scales_u8[scale_index] = scale_byte;
        }
    }
    float normalized = scaled_value / actual_scale;
    if (quant_mode == 1) {
        if (active_lane != 0) {
            {
                unsigned short _fp8_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(normalized));
                *(reinterpret_cast<unsigned char*>(quantized_fp8 + (token * elements_per_token + column)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
            }
        }
    } else {
        int bits = 0;
        bits = reinterpret_cast<int*>(&normalized)[0];
        int absolute_bits = bits & 2147483647;
        int sign = bits >> 28 & 8;
        if (absolute_bits == 0) {
            sign = 0;
        }
        int magnitude = 0;
        if (sign == 0) {
            magnitude = ((absolute_bits > 1048576000) ? 1 : 0) + ((absolute_bits > 1061158912) ? 1 : 0) + ((absolute_bits > 1067450368) ? 1 : 0) + ((absolute_bits > 1071644672) ? 1 : 0) + ((absolute_bits > 1075838976) ? 1 : 0) + ((absolute_bits > 1080033280) ? 1 : 0) + ((absolute_bits > 1084227584) ? 1 : 0);
        } else {
            magnitude = ((absolute_bits >= 1048576000) ? 1 : 0) + ((absolute_bits >= 1061158912) ? 1 : 0) + ((absolute_bits >= 1067450368) ? 1 : 0) + ((absolute_bits >= 1071644672) ? 1 : 0) + ((absolute_bits >= 1075838976) ? 1 : 0) + ((absolute_bits >= 1080033280) ? 1 : 0) + ((absolute_bits >= 1084227584) ? 1 : 0);
        }
        if (magnitude == 0) {
            sign = 0;
        }
        int code = magnitude + sign;
        int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, code, 1);
        int partner = _shfl_xor_5;
        if (active_lane != 0) {
            if ((lane & 1) == 0 && column + 1 < elements_per_token) {
                int packed = code | partner << 4;
                int packed_column = block_column * (block_size / 2) + lane / 2;
                quantized_packed[token * (elements_per_token / 2) + packed_column] = packed;
            }
        }
    }
    if (enable_pdl) {
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
}

} // extern "C"
