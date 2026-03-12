/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <string>
#include <stdexcept>

////////////////////////////////////////////////////////////////////////////////////////////////////
// Quantization Configuration and Helper Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Supported quantization data types
enum class QuantDType {
    BF16,     // No quantization (native bfloat16)
    FP8_E4M3, // FP8 E4M3 format (4-bit exponent, 3-bit mantissa)
    FP8_E5M2, // FP8 E5M2 format (5-bit exponent, 2-bit mantissa)
    MXFP4     // Microscaling FP4 (block-wise quantization)
};

// Configuration for quantization
struct QuantConfig {
    QuantDType cache_input_dtype;    // feat_cache INPUT quantization type (how it's stored)
    QuantDType cache_output_dtype;   // feat_cache OUTPUT quantization type (how to write it to output)
    QuantDType output_dtype;          // final output quantization type
    float cache_input_scale;          // Scale factor for dequantizing feat_cache input (for FP8/MXFP4)
    float cache_output_scale;         // Scale factor for quantizing cache to output (for FP8/MXFP4)
    float output_scale;               // Scale factor for final output (for FP8/MXFP4)
    
    // Constructor with defaults (no quantization)
    QuantConfig()
        : cache_input_dtype(QuantDType::BF16)
        , cache_output_dtype(QuantDType::BF16)
        , output_dtype(QuantDType::BF16)
        , cache_input_scale(1.0f)
        , cache_output_scale(1.0f)
        , output_scale(1.0f)
    {}
    
    // Check if quantization is enabled
    bool has_cache_input_quant() const { return cache_input_dtype != QuantDType::BF16; }
    bool has_cache_output_quant() const { return cache_output_dtype != QuantDType::BF16; }
    bool has_output_quant() const { return output_dtype != QuantDType::BF16; }
};

// String to QuantDType conversion
inline QuantDType string_to_quant_dtype(const std::string& dtype_str) {
    if (dtype_str == "bf16" || dtype_str == "bfloat16") {
        return QuantDType::BF16;
    } else if (dtype_str == "fp8" || dtype_str == "fp8_e4m3" || dtype_str == "e4m3") {
        return QuantDType::FP8_E4M3;
    } else if (dtype_str == "fp8_e5m2" || dtype_str == "e5m2") {
        return QuantDType::FP8_E5M2;
    } else if (dtype_str == "mxfp4" || dtype_str == "fp4") {
        return QuantDType::MXFP4;
    } else {
        throw std::runtime_error("Unsupported quantization dtype: " + dtype_str);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Device-side quantization/dequantization functions
// Only available when compiling with nvcc (__CUDACC__ defined)
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

// FP8 E4M3 quantization (float -> __nv_fp8_e4m3) with optional scaling
__device__ __forceinline__ __nv_fp8_e4m3 quantize_fp8_e4m3(float val, float scale = 1.0f) {
    return __nv_fp8_e4m3(val * scale);
}

// FP8 E4M3 dequantization (__nv_fp8_e4m3 -> float) with optional scaling
__device__ __forceinline__ float dequantize_fp8_e4m3(__nv_fp8_e4m3 val, float scale = 1.0f) {
    return float(val) / scale;  // Implicit conversion then descale
}

// FP8 E5M2 quantization (float -> __nv_fp8_e5m2) with optional scaling
__device__ __forceinline__ __nv_fp8_e5m2 quantize_fp8_e5m2(float val, float scale = 1.0f) {
    return __nv_fp8_e5m2(val * scale);
}

// FP8 E5M2 dequantization (__nv_fp8_e5m2 -> float) with optional scaling
__device__ __forceinline__ float dequantize_fp8_e5m2(__nv_fp8_e5m2 val, float scale = 1.0f) {
    return float(val) / scale;  // Implicit conversion then descale
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MXFP4 Block-wise Quantization
// Microscaling: Store 4-bit mantissas with shared 8-bit block exponent
// Block size: 32 elements (1 warp)
////////////////////////////////////////////////////////////////////////////////////////////////////

// MXFP4: Pack two 4-bit values into one uint8_t
__device__ __forceinline__ uint8_t pack_mxfp4_pair(uint8_t val0, uint8_t val1) {
    return (val0 & 0xF) | ((val1 & 0xF) << 4);
}


// MXFP4: Quantize float to 4-bit mantissa (0-15 range)
// Uses shared exponent across block
__device__ __forceinline__ uint8_t quantize_mxfp4_mantissa(float val, float block_scale) {
    // Normalize by block scale
    float normalized = val * block_scale;
    // Quantize to 4-bit (0-15 range)
    int quantized = __float2int_rn(normalized * 15.0f);
    // Clamp to [0, 15]
    quantized = (quantized < 0) ? 0 : ((quantized > 15) ? 15 : quantized);
    return static_cast<uint8_t>(quantized);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Vectorized FP8 conversion helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert float2 to FP8 E4M3 pair (packed into uint16_t) with optional scaling
__device__ __forceinline__ uint16_t float2_to_fp8_e4m3_packed(float2 val, float scale = 1.0f) {
    __nv_fp8_e4m3 fp8_0 = quantize_fp8_e4m3(val.x, scale);
    __nv_fp8_e4m3 fp8_1 = quantize_fp8_e4m3(val.y, scale);
    return (*reinterpret_cast<uint8_t*>(&fp8_0)) | 
           ((*reinterpret_cast<uint8_t*>(&fp8_1)) << 8);
}

// Convert 4 FP32 values to 4 FP8 E4M3 values (packed into uint32_t) with optional scaling
// Uses PTX inline assembly to force vectorized PACK_AB_MERGE_C pattern
// Output order: [fp8_f0, fp8_f1, fp8_f2, fp8_f3] (little-endian byte order)
// Uses cvt.rn.satfinite.e4m3x2.f32 which converts 2 FP32 -> 2 FP8 (16 bits)
__device__ __forceinline__ uint32_t float4_to_fp8_e4m3_packed(
    float f0, float f1, float f2, float f3, float scale = 1.0f) {
    // Apply scaling
    float scaled0 = f0 * scale;
    float scaled1 = f1 * scale;
    float scaled2 = f2 * scale;
    float scaled3 = f3 * scale;
    
    uint32_t result;
    
    // Use PTX inline assembly to force optimal PACK_AB_MERGE_C pattern
    // Expected SASS (matching CUTLASS pattern):
    //   F2FP.E4M3.F32.PACK_AB_MERGE_C Dst, Fp2, Fp3, RZ
    //   F2FP.E4M3.F32.PACK_AB_MERGE_C Dst, Fp0, Fp1, Dst.H0
    // Pattern: First conversion writes lower 16 bits, second merges with it
    // Based on CUTLASS float8.h implementation approach
    asm volatile(
        "{\n\t"
        ".reg .f32 f0, f1, f2, f3;\n\t"
        ".reg .b16 r_lo, r_hi;\n\t"
        ".reg .b32 r_result;\n\t"
        "mov.f32 f0, %1;\n\t"
        "mov.f32 f1, %2;\n\t"
        "mov.f32 f2, %3;\n\t"
        "mov.f32 f3, %4;\n\t"
        // First PACK_AB_MERGE_C: convert f2, f3 -> lower 16 bits (RZ = 0)
        // Target: F2FP.E4M3.F32.PACK_AB_MERGE_C Dst, Fp2, Fp3, RZ
        "cvt.rn.satfinite.e4m3x2.f32 r_lo, f3, f2;\n\t"
        // Second PACK_AB_MERGE_C: convert f0, f1 -> upper 16 bits
        // Target: F2FP.E4M3.F32.PACK_AB_MERGE_C Dst, Fp0, Fp1, Dst.H0
        // Convert f0, f1 -> upper 16 bits
        "cvt.rn.satfinite.e4m3x2.f32 r_hi, f1, f0;\n\t"
        // Combine: [r_hi (f0,f1), r_lo (f2,f3)] = [fp8_f0, fp8_f1, fp8_f2, fp8_f3]
        // Structure: first conversion produces r_lo, second produces r_hi
        // The compiler should recognize both target same register and optimize to merge
        "mov.b32 r_result, {r_hi, r_lo};\n\t"
        "mov.b32 %0, r_result;\n\t"
        "}"
        : "=r"(result)
        : "f"(scaled0), "f"(scaled1), "f"(scaled2), "f"(scaled3)
    );
    
    return result;
}

// Convert 4 float2 (8 floats) to FP8 E4M3 (packed into uint2 = 8 bytes) with optional scaling
// Optimized to use float4_to_fp8_e4m3_packed for better SASS code generation
__device__ __forceinline__ uint2 float8_to_fp8_e4m3_packed(
    float2 val0, float2 val1, float2 val2, float2 val3, float scale = 1.0f) {
    // Convert 4 FP32 values at a time to encourage PACK_AB_MERGE_C pattern
    uint32_t packed_lo = float4_to_fp8_e4m3_packed(
        val0.x, val0.y, val1.x, val1.y, scale);
    uint32_t packed_hi = float4_to_fp8_e4m3_packed(
        val2.x, val2.y, val3.x, val3.y, scale);
    
    return make_uint2(packed_lo, packed_hi);
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// Template-based quantized store function
// Supports BF16, FP8_E4M3, and MXFP4 output formats
////////////////////////////////////////////////////////////////////////////////////////////////////

// Template specialization for BF16 output (no quantization)
template<QuantDType OUTPUT_DTYPE, bool USE_UINT4>
struct QuantizedStore {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale = 1.0f);  // Unused for BF16
};

// Specialization: BF16 output with UINT4 vectorization
template<>
struct QuantizedStore<QuantDType::BF16, true> {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale) {
        // Convert float2 to bfloat16
        __nv_bfloat162 out0 = __float22bfloat162_rn(v0);
        __nv_bfloat162 out1 = __float22bfloat162_rn(v1);
        __nv_bfloat162 out2 = __float22bfloat162_rn(v2);
        __nv_bfloat162 out3 = __float22bfloat162_rn(v3);
        
        uint4 packed_output = make_uint4(
            *reinterpret_cast<uint const*>(&out0),
            *reinterpret_cast<uint const*>(&out1),
            *reinterpret_cast<uint const*>(&out2),
            *reinterpret_cast<uint const*>(&out3)
        );
        *reinterpret_cast<uint4*>(ptr) = packed_output;
    }
};

// Specialization: BF16 output with UINT2 vectorization
template<>
struct QuantizedStore<QuantDType::BF16, false> {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale) {
        // Convert float2 to bfloat16 (only first 2 float2s for UINT2)
        __nv_bfloat162 out0 = __float22bfloat162_rn(v0);
        __nv_bfloat162 out1 = __float22bfloat162_rn(v1);
        
        uint2 packed_output = make_uint2(
            *reinterpret_cast<uint const*>(&out0),
            *reinterpret_cast<uint const*>(&out1)
        );
        *reinterpret_cast<uint2*>(ptr) = packed_output;
    }
};

// Specialization: FP8_E4M3 output with UINT4 vectorization
template<>
struct QuantizedStore<QuantDType::FP8_E4M3, true> {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale) {
        // Convert 4 float2 (8 floats) to 8 FP8 values (8 bytes = uint2) with scaling
        uint16_t packed0 = float2_to_fp8_e4m3_packed(v0, block_scale);
        uint16_t packed1 = float2_to_fp8_e4m3_packed(v1, block_scale);
        uint16_t packed2 = float2_to_fp8_e4m3_packed(v2, block_scale);
        uint16_t packed3 = float2_to_fp8_e4m3_packed(v3, block_scale);
        
        uint2 output = make_uint2(
            packed0 | (packed1 << 16),
            packed2 | (packed3 << 16)
        );
        *reinterpret_cast<uint2*>(ptr) = output;
    }
};

// Specialization: FP8_E4M3 output with UINT2 vectorization
template<>
struct QuantizedStore<QuantDType::FP8_E4M3, false> {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale) {
        // Convert 2 float2 (4 floats) to 4 FP8 values (4 bytes = uint) with scaling
        uint16_t packed0 = float2_to_fp8_e4m3_packed(v0, block_scale);
        uint16_t packed1 = float2_to_fp8_e4m3_packed(v1, block_scale);
        
        uint output = packed0 | (packed1 << 16);
        *reinterpret_cast<uint*>(ptr) = output;
    }
};

// Specialization: MXFP4 output with UINT4 vectorization
template<>
struct QuantizedStore<QuantDType::MXFP4, true> {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale) {
        // Quantize 8 floats to 4-bit mantissas (4 bytes total)
        uint8_t m0 = quantize_mxfp4_mantissa(v0.x, block_scale);
        uint8_t m1 = quantize_mxfp4_mantissa(v0.y, block_scale);
        uint8_t m2 = quantize_mxfp4_mantissa(v1.x, block_scale);
        uint8_t m3 = quantize_mxfp4_mantissa(v1.y, block_scale);
        uint8_t m4 = quantize_mxfp4_mantissa(v2.x, block_scale);
        uint8_t m5 = quantize_mxfp4_mantissa(v2.y, block_scale);
        uint8_t m6 = quantize_mxfp4_mantissa(v3.x, block_scale);
        uint8_t m7 = quantize_mxfp4_mantissa(v3.y, block_scale);
        
        // Pack pairs of 4-bit values into bytes
        uint output = pack_mxfp4_pair(m0, m1) | 
                     (pack_mxfp4_pair(m2, m3) << 8) |
                     (pack_mxfp4_pair(m4, m5) << 16) |
                     (pack_mxfp4_pair(m6, m7) << 24);
        
        *reinterpret_cast<uint*>(ptr) = output;
    }
};

// Specialization: MXFP4 output with UINT2 vectorization
template<>
struct QuantizedStore<QuantDType::MXFP4, false> {
    __device__ __forceinline__ static void store(
        float2 const& v0,
        float2 const& v1,
        float2 const& v2,
        float2 const& v3,
        void* __restrict__ ptr,
        float block_scale) {
        // Quantize 4 floats to 4-bit mantissas (2 bytes total)
        uint8_t m0 = quantize_mxfp4_mantissa(v0.x, block_scale);
        uint8_t m1 = quantize_mxfp4_mantissa(v0.y, block_scale);
        uint8_t m2 = quantize_mxfp4_mantissa(v1.x, block_scale);
        uint8_t m3 = quantize_mxfp4_mantissa(v1.y, block_scale);
        
        // Pack pairs of 4-bit values into bytes
        uint16_t output = pack_mxfp4_pair(m0, m1) | 
                         (pack_mxfp4_pair(m2, m3) << 8);
        
        *reinterpret_cast<uint16_t*>(ptr) = output;
    }
};

#endif // __CUDACC__

