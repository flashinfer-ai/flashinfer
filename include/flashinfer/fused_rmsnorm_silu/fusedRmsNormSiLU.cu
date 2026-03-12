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

 #include "fusedRMSNormSiLU.h"
 #include "quantization.h"
 #include "tensorrt_llm_utils.h"
 #include <cmath>
 #include <cstdint>
 #include <cuda_bf16.h>
 #include <cuda_fp16.h>
 #include <cuda_fp8.h>
 #include <cuda_runtime.h>
 
 
 constexpr int THREADS_PER_WARP = 32;
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Magic Division Infrastructure for fast integer division
 // Replaces expensive integer division with multiply + shift operations
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
 // Magic division parameters for unsigned 32-bit division
 struct MagicDiv32 {
     uint32_t magic;      // Magic multiplier
     uint32_t shift;      // Shift amount (includes the +32 for high bits)
     uint32_t divisor;    // Original divisor (for modulo computation)
 };
 
 // Host function to compute magic division parameters
 // Uses the classic algorithm: x/d = (x * magic) >> (32 + shift)
 __host__ inline MagicDiv32 computeMagicDiv32(uint32_t d) {
     MagicDiv32 result;
     result.divisor = d;
     
     if (d == 1) {
         result.magic = 1;
         result.shift = 0;
         return result;
     }
     
     // Find shift such that 2^(32+shift) / d fits in 32 bits when rounded up
     uint32_t shift = 0;
     uint64_t nc = (1ULL << 32) - 1 - ((1ULL << 32) % d);  // Number of "ceiling" cases
     
     // Binary search for the smallest shift that works
     while ((1ULL << (32 + shift)) < (nc * (d - 1))) {
         shift++;
     }
     
     // Compute magic number: ceil(2^(32+shift) / d)
     uint64_t m = ((1ULL << (32 + shift)) + d - 1) / d;
     
     result.magic = (uint32_t)m;
     result.shift = shift;
     return result;
 }
 
 // Device function to perform fast unsigned division using precomputed magic numbers
 __device__ __forceinline__ uint32_t magicDiv(uint32_t n, MagicDiv32 const& md) {
     // Compute (n * magic) >> (32 + shift)
     // Use 64-bit multiply to get the high 32 bits
     uint64_t tmp = (uint64_t)n * (uint64_t)md.magic;
     return (uint32_t)(tmp >> (32 + md.shift));
 }
 
 // Device function to compute both quotient and remainder efficiently
 __device__ __forceinline__ void magicDivMod(uint32_t n, MagicDiv32 const& md, 
                                             uint32_t& quotient, uint32_t& remainder) {
     quotient = magicDiv(n, md);
     remainder = n - quotient * md.divisor;
 }
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // FFMA2 intrinsics for Blackwell (SM100+) architecture
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
 // Blackwell (SM100+) has native FFMA2 instructions
 __device__ __forceinline__ float2 fmul2(const float2& a, const float2& b) {
     uint64_t c;
     asm volatile("mul.f32x2 %0, %1, %2;\n"
                   : "=l"(c)
                   : "l"(reinterpret_cast<const uint64_t&>(a))
                   , "l"(reinterpret_cast<const uint64_t&>(b)));
     return reinterpret_cast<float2&>(c);
 }
 
 __device__ __forceinline__ float2 ffma2(const float2& a, const float2& b, const float2& c) {
     uint64_t d;
     asm volatile("fma.rn.f32x2 %0, %1, %2, %3;\n"
                   : "=l"(d)
                   : "l"(reinterpret_cast<const uint64_t&>(a))
                   , "l"(reinterpret_cast<const uint64_t&>(b))
                   , "l"(reinterpret_cast<const uint64_t&>(c)));
     return reinterpret_cast<float2&>(d);
 }
 
 __device__ __forceinline__ float2 fadd2(const float2& a, const float2& b) {
     uint64_t c;
     asm volatile("add.f32x2 %0, %1, %2;\n"
                   : "=l"(c)
                   : "l"(reinterpret_cast<const uint64_t&>(a))
                   , "l"(reinterpret_cast<const uint64_t&>(b)));
     return reinterpret_cast<float2&>(c);
 }
 #else
 // Fallback for older architectures
 __device__ __forceinline__ float2 fmul2(const float2& a, const float2& b) {
     return make_float2(a.x * b.x, a.y * b.y);
 }
 
 __device__ __forceinline__ float2 ffma2(const float2& a, const float2& b, const float2& c) {
     return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
 }
 
 __device__ __forceinline__ float2 fadd2(const float2& a, const float2& b) {
     return make_float2(a.x + b.x, a.y + b.y);
 }
 #endif
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Inline device helper functions for vectorized operations (unified template versions)
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
 // Load function: template specialization for uint4 (true) and uint2 (false)
 template<bool USE_UINT4>
 __device__ __forceinline__ void load(
     __nv_bfloat16 const* __restrict__ ptr,
     float2& out0,
     float2& out1,
     float2& out2,
     float2& out3
 );
 
 template<>
 __device__ __forceinline__ void load<true>(  // uint4 version
     __nv_bfloat16 const* __restrict__ ptr,
     float2& out0,
     float2& out1,
     float2& out2,
     float2& out3
 ) {
     uint4 packed = *reinterpret_cast<uint4 const*>(ptr);
     out0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed.x));
     out1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed.y));
     out2 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed.z));
     out3 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed.w));
 }
 
 template<>
 __device__ __forceinline__ void load<false>(  // uint2 version
     __nv_bfloat16 const* __restrict__ ptr,
     float2& out0,
     float2& out1,
     float2& out2,
     float2& out3
 ) {
     // For uint2, only use first 2 float2s
     uint2 packed = *reinterpret_cast<uint2 const*>(ptr);
     out0 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed.x));
     out1 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed.y));
     // out2 and out3 are unused for uint2
 }
 
 // Accumulate sum of squares function: template specialization
 template<bool USE_UINT4>
 __device__ __forceinline__ float accumulate_sum_squares(
     float sum,
     float2 const& v0,
     float2 const& v1,
     float2 const& v2,
     float2 const& v3
 );
 
 template<>
 __device__ __forceinline__ float accumulate_sum_squares<true>(  // uint4 version
     float sum,
     float2 const& v0,
     float2 const& v1,
     float2 const& v2,
     float2 const& v3
 ) {
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
     // Blackwell (SM100+): Use FFMA2 for vectorized operations
     float2 lo0 = fmul2(v0, v0);
     float2 lo1 = fmul2(v1, v1);
     float2 hi0 = fmul2(v2, v2);
     float2 hi1 = fmul2(v3, v3);
     
     // Add squared values together (not multiply again!)
     lo0 = fadd2(lo0, hi0);
     lo1 = fadd2(lo1, hi1);
     lo0 = fadd2(lo0, lo1);
     
     // The compiler may move the sum + into an FFMA
     return sum + lo0.x + lo0.y;
 #else
     // Older architectures: Use scalar FMA
     return __fmaf_rn(v3.y, v3.y,
            __fmaf_rn(v3.x, v3.x,
            __fmaf_rn(v2.y, v2.y,
            __fmaf_rn(v2.x, v2.x,
            __fmaf_rn(v1.y, v1.y,
            __fmaf_rn(v1.x, v1.x,
            __fmaf_rn(v0.y, v0.y,
            __fmaf_rn(v0.x, v0.x, sum))))))));
 #endif
 }
 
 template<>
 __device__ __forceinline__ float accumulate_sum_squares<false>(  // uint2 version
     float sum,
     float2 const& v0,
     float2 const& v1,
     float2 const& v2,
     float2 const& v3
 ) {
     // For uint2, only use first 2 float2s (v0, v1)
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
     // Blackwell (SM100+): Use FFMA2 for vectorized operations
     float2 lo0 = fmul2(v0, v0);
     float2 lo1 = fmul2(v1, v1);
     lo0 = fadd2(lo0, lo1);  // Add squared values
     
     // The compiler may move the sum + into an FFMA
     return sum + lo0.x + lo0.y;
 #else
     // Older architectures: Use scalar FMA
     return __fmaf_rn(v1.y, v1.y,
            __fmaf_rn(v1.x, v1.x,
            __fmaf_rn(v0.y, v0.y,
            __fmaf_rn(v0.x, v0.x, sum))));
 #endif
 }
 
 // Apply normalization, weight multiplication, and SiLU activation: template specialization
 template<bool USE_UINT4>
 __device__ __forceinline__ void apply_norm_weight_silu(
     float2& v0,
     float2& v1,
     float2& v2,
     float2& v3,
     float2 const& w0,
     float2 const& w1,
     float2 const& w2,
     float2 const& w3,
     float norm_scale
 );
 
 template<>
 __device__ __forceinline__ void apply_norm_weight_silu<true>(  // uint4 version
     float2& v0,
     float2& v1,
     float2& v2,
     float2& v3,
     float2 const& w0,
     float2 const& w1,
     float2 const& w2,
     float2 const& w3,
     float norm_scale
 ) {
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
     // Blackwell (SM100+): Use FFMA2 for vectorized operations
     float2 norm_scale_vec = make_float2(norm_scale, norm_scale);
     float2 zero = make_float2(0.0f, 0.0f);
     
     v0 = fmul2(v0, norm_scale_vec);
     v1 = fmul2(v1, norm_scale_vec);
     v2 = fmul2(v2, norm_scale_vec);
     v3 = fmul2(v3, norm_scale_vec);
     
     v0 = ffma2(v0, w0, zero);
     v1 = ffma2(v1, w1, zero);
     v2 = ffma2(v2, w2, zero);
     v3 = ffma2(v3, w3, zero);
 #else
     // Older architectures: Use scalar FMA
     // Apply: (x * norm_scale) * weight
     v0.x = __fmaf_rn(v0.x * norm_scale, w0.x, 0.0f);
     v0.y = __fmaf_rn(v0.y * norm_scale, w0.y, 0.0f);
     v1.x = __fmaf_rn(v1.x * norm_scale, w1.x, 0.0f);
     v1.y = __fmaf_rn(v1.y * norm_scale, w1.y, 0.0f);
     v2.x = __fmaf_rn(v2.x * norm_scale, w2.x, 0.0f);
     v2.y = __fmaf_rn(v2.y * norm_scale, w2.y, 0.0f);
     v3.x = __fmaf_rn(v3.x * norm_scale, w3.x, 0.0f);
     v3.y = __fmaf_rn(v3.y * norm_scale, w3.y, 0.0f);
 #endif
     
     // Apply SiLU: x * sigmoid(x) (scalar, no vectorized version available)
     v0.x *= __frcp_rn(1.0f + __expf(-v0.x));
     v0.y *= __frcp_rn(1.0f + __expf(-v0.y));
     v1.x *= __frcp_rn(1.0f + __expf(-v1.x));
     v1.y *= __frcp_rn(1.0f + __expf(-v1.y));
     v2.x *= __frcp_rn(1.0f + __expf(-v2.x));
     v2.y *= __frcp_rn(1.0f + __expf(-v2.y));
     v3.x *= __frcp_rn(1.0f + __expf(-v3.x));
     v3.y *= __frcp_rn(1.0f + __expf(-v3.y));
 }
 
 template<>
 __device__ __forceinline__ void apply_norm_weight_silu<false>(  // uint2 version
     float2& v0,
     float2& v1,
     float2& v2,
     float2& v3,
     float2 const& w0,
     float2 const& w1,
     float2 const& w2,
     float2 const& w3,
     float norm_scale
 ) {
     // For uint2, only use first 2 float2s (v0, v1)
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
     // Blackwell (SM100+): Use FFMA2 for vectorized operations
     float2 norm_scale_vec = make_float2(norm_scale, norm_scale);
     float2 zero = make_float2(0.0f, 0.0f);
     
     v0 = fmul2(v0, norm_scale_vec);
     v1 = fmul2(v1, norm_scale_vec);
     v0 = ffma2(v0, w0, zero);
     v1 = ffma2(v1, w1, zero);
 #else
     // Older architectures: Use scalar FMA
     v0.x = __fmaf_rn(v0.x * norm_scale, w0.x, 0.0f);
     v0.y = __fmaf_rn(v0.y * norm_scale, w0.y, 0.0f);
     v1.x = __fmaf_rn(v1.x * norm_scale, w1.x, 0.0f);
     v1.y = __fmaf_rn(v1.y * norm_scale, w1.y, 0.0f);
 #endif
     
     // Apply SiLU: x * sigmoid(x) (scalar, no vectorized version available)
     v0.x *= __frcp_rn(1.0f + __expf(-v0.x));
     v0.y *= __frcp_rn(1.0f + __expf(-v0.y));
     v1.x *= __frcp_rn(1.0f + __expf(-v1.x));
     v1.y *= __frcp_rn(1.0f + __expf(-v1.y));
 }
 
 // Scalar store function: single element with quantization support
 // OUTPUT_DTYPE: Quantization type for output (BF16, FP8_E4M3, MXFP4)
 template<QuantDType OUTPUT_DTYPE>
 __device__ __forceinline__ void store_scalar(
     float val,
     void* __restrict__ output,
     int outputOffset,
     float output_quant_scale = 1.0f
 ) {
     if constexpr (OUTPUT_DTYPE == QuantDType::BF16) {
         reinterpret_cast<__nv_bfloat16*>(output)[outputOffset] = __float2bfloat16_rn(val);
     } else if constexpr (OUTPUT_DTYPE == QuantDType::FP8_E4M3) {
         reinterpret_cast<__nv_fp8_e4m3*>(output)[outputOffset] = quantize_fp8_e4m3(val, output_quant_scale);
     } else if constexpr (OUTPUT_DTYPE == QuantDType::MXFP4) {
         // MXFP4 stores 2 elements per byte
         int byte_offset = outputOffset / 2;
         uint8_t quant = quantize_mxfp4_mantissa(val, output_quant_scale);
         if (outputOffset % 2 == 0) {
             reinterpret_cast<uint8_t*>(output)[byte_offset] = 
                 (reinterpret_cast<uint8_t*>(output)[byte_offset] & 0xF0) | (quant & 0x0F);
         } else {
             reinterpret_cast<uint8_t*>(output)[byte_offset] = 
                 (reinterpret_cast<uint8_t*>(output)[byte_offset] & 0x0F) | ((quant & 0x0F) << 4);
         }
     }
 }
 
 // Vectorized pair store function: two consecutive elements with quantization support
 // OUTPUT_DTYPE: Quantization type for output (BF16, FP8_E4M3, MXFP4)
 template<QuantDType OUTPUT_DTYPE>
 __device__ __forceinline__ void store_pair(
     float val0,
     float val1,
     void* __restrict__ output,
     int outputOffset,
     float output_quant_scale = 1.0f
 ) {
     if constexpr (OUTPUT_DTYPE == QuantDType::BF16) {
         __nv_bfloat162 out = __float22bfloat162_rn(make_float2(val0, val1));
         *reinterpret_cast<uint*>(reinterpret_cast<__nv_bfloat16*>(output) + outputOffset) = 
             *reinterpret_cast<uint const*>(&out);
     } else if constexpr (OUTPUT_DTYPE == QuantDType::FP8_E4M3) {
         uint16_t packed = float2_to_fp8_e4m3_packed(make_float2(val0, val1), output_quant_scale);
         *reinterpret_cast<uint16_t*>(reinterpret_cast<__nv_fp8_e4m3*>(output) + outputOffset) = packed;
     } else if constexpr (OUTPUT_DTYPE == QuantDType::MXFP4) {
         uint8_t m0 = quantize_mxfp4_mantissa(val0, output_quant_scale);
         uint8_t m1 = quantize_mxfp4_mantissa(val1, output_quant_scale);
         reinterpret_cast<uint8_t*>(output)[outputOffset / 2] = pack_mxfp4_pair(m0, m1);
     }
 }
 
 // Vectorized store function: template specialization with quantization support
 // OUTPUT_DTYPE: Quantization type for output (BF16, FP8_E4M3, MXFP4)
 // USE_UINT4: Vectorization width (true = uint4, false = uint2)
 template<QuantDType OUTPUT_DTYPE = QuantDType::BF16, bool USE_UINT4 = true>
 __device__ __forceinline__ void store(
     float2 const& v0,
     float2 const& v1,
     float2 const& v2,
     float2 const& v3,
     void* __restrict__ ptr,
     float block_scale = 1.0f  // Used for MXFP4 quantization
 ) {
     // Delegate to QuantizedStore template from quantization.h
     QuantizedStore<OUTPUT_DTYPE, USE_UINT4>::store(v0, v1, v2, v3, ptr, block_scale);
 }
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Helper function to compute output base offset
 // Template parameter USE_NDHWC selects between standard and NDHWC output layouts
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 template <bool USE_NDHWC>
 __device__ __forceinline__ int64_t computeOutputBaseOffset(
     int const tokenIdx,
     // Standard layout parameters (used when USE_NDHWC == false)
     int const output_stride_token,
     // NDHWC layout parameters (used when USE_NDHWC == true)
     int const DHW, int const HW, int const W,
     int const output_stride_N, int const output_stride_D,
     int const output_stride_H, int const output_stride_W,
     int const output_D_offset)
 {
     if constexpr (USE_NDHWC) {
         // Decompose tokenIdx into NDHW coordinates
         // Input is permuted from [N, C, D, H, W] to [N, D, H, W, C] then flattened
         // So tokenIdx = n * (D*H*W) + d * (H*W) + h * W + w
         int const n = tokenIdx / DHW;
         int const d = (tokenIdx % DHW) / HW;
         int const h = (tokenIdx % HW) / W;
         int const w = tokenIdx % W;
         
         // Compute output base address using NDHWC strides (without channel offset)
         // output_addr = n*stride_N + (d_offset + d)*stride_D + h*stride_H + w*stride_W
         int const output_D_idx = output_D_offset + d;
         return (int64_t)n * output_stride_N + 
                (int64_t)output_D_idx * output_stride_D + 
                (int64_t)h * output_stride_H + 
                (int64_t)w * output_stride_W;
     } else {
         // Standard layout: outputBaseOffset = tokenIdx * output_stride_token
         return (int64_t)tokenIdx * output_stride_token;
     }
 }
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 //
 // Unified Fused RMSNorm + SiLU kernel with automatic vectorization selection
 // Non-persistent, global memory weights
 // - Automatically selects uint4 (16-byte) for dims divisible by 256, uint2 (8-byte) otherwise
 // - Uses uint4 (8 bf16) vectorized loads for maximum throughput when possible
 // - Falls back to uint2 (4 bf16) vectorized loads for flexibility
 // - Supports: 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048, etc.
 //   (must be divisible by 128; divisible by 256 uses faster uint4 path)
 //
 // Formula: SiLU(L2Norm(x) * scale * gamma + bias)
 // where L2Norm(x) = x / sqrt(sum(x^2) + eps)
 // This matches the open source WAN implementation: F.normalize(x, dim=1) * scale * gamma + bias
 //
 // OUTPUT_DTYPE: Quantization type for output (BF16, FP8_E4M3, MXFP4)
 template <int hidden_dim, QuantDType OUTPUT_DTYPE = QuantDType::BF16, bool USE_NDHWC = false, int WARPS_PER_CTA = 8>
 __global__ void fusedRMSNormSiLUKernel(
     __nv_bfloat16 const* __restrict__ input,   // Input tensor [num_tokens, hidden_dim]
     void* __restrict__ output,                 // Output tensor [num_tokens, hidden_dim] or NDHWC layout (quantized)
     int const num_tokens,                      // Number of tokens
     int const input_stride_token,              // Stride between tokens in input (in elements)
     float const eps,                           // Epsilon for L2 normalization
     __nv_bfloat16 const* __restrict__ weight,  // Gamma weights [hidden_dim]
     float const scale,                         // Scale factor (typically sqrt(hidden_dim))
     __nv_bfloat16 const* __restrict__ bias,   // Optional bias [hidden_dim] (can be nullptr)
     // Standard layout parameters (used when USE_NDHWC == false)
     int const output_stride_token,             // Stride between tokens in output (in elements)
     // NDHWC layout parameters (used when USE_NDHWC == true)
     int const DHW,                             // Depth * Height * Width dimension (for input tokens)
     int const HW,                              // Height * Width dimension
     int const W,                               // Width dimension
     int const output_stride_N,                 // Stride for N dimension in output (in elements)
     int const output_stride_D,                 // Stride for D dimension in output (in elements)
     int const output_stride_H,                 // Stride for H dimension in output (in elements)
     int const output_stride_W,                 // Stride for W dimension in output (in elements)
     int const output_D_offset,                 // Offset in D dimension where to start writing
     float const output_quant_scale             // Quantization scale for MXFP4 output
 )
 {
     // hidden_dim must be divisible by 128 (32 threads * 4 bf16 per uint2 load)
     static_assert(hidden_dim % 128 == 0, "hidden_dim must be divisible by 128");
     
     // Determine vectorization type: use uint4 if divisible by 256, otherwise uint2
     constexpr bool USE_UINT4 = (hidden_dim % 256 == 0);
     
     // Coalesced memory access: when hidden_dim >= 256, fetch elements by groups
     // This ensures consecutive memory access across threads for better coalescing
     constexpr int NUM_BF16s_PER_UINT4 = 8;  // uint4 = 16 bytes = 8 bf16
     constexpr int NUM_BF16s_PER_UINT2 = 4;  // uint2 = 8 bytes = 4 bf16
     constexpr int NUM_BF16s_PER_LOAD = USE_UINT4 ? NUM_BF16s_PER_UINT4 : NUM_BF16s_PER_UINT2;
     
     constexpr int NUM_LOADS = hidden_dim / (THREADS_PER_WARP * NUM_BF16s_PER_LOAD);
     
     int const laneId = threadIdx.x % THREADS_PER_WARP;
     
     // Each warp processes one token; 8 warps per CTA for latency hiding
     int const tokenIdx = blockIdx.x * WARPS_PER_CTA + threadIdx.x / THREADS_PER_WARP;
     
     // Early exit if this warp has no work
     if (tokenIdx >= num_tokens)
         return;
     
     // Use stride_token for input offset calculation (supports non-contiguous token layouts)
     // stride_hidden must be 1 for vectorized loads to work (checked in C++ wrapper)
     int const tokenBaseOffset = tokenIdx * input_stride_token;
     
     // Compute output base offset using helper function (without channel offset)
     int64_t const outputBaseOffset = computeOutputBaseOffset<USE_NDHWC>(
         tokenIdx,
         output_stride_token,
         DHW, HW, W,
         output_stride_N, output_stride_D, output_stride_H, output_stride_W,
         output_D_offset);
     
     float sumOfSquares = 0.0f;
     
     // ========== Unified path: works for both uint4 and uint2 ==========
     // Use float2 arrays - each load produces 4 float2s for uint4, 2 float2s for uint2
     constexpr int NUM_FLOAT2_PER_LOAD = USE_UINT4 ? 4 : 2;  // uint4: 4 float2s, uint2: 2 float2s
     constexpr int NUM_FLOAT2 = NUM_LOADS * NUM_FLOAT2_PER_LOAD;
     float2 vals[NUM_FLOAT2];
     
     // ========== Pass 1: Load input, compute sum of squares (coalesced access) ==========
     #pragma unroll
     for (int ii = 0; ii < NUM_LOADS; ++ii) {
         // Coalesced access pattern: consecutive addresses across threads
         int offset = tokenBaseOffset + (ii * THREADS_PER_WARP + laneId) * NUM_BF16s_PER_LOAD;
         int valIdx = ii * NUM_FLOAT2_PER_LOAD;
         
         // Load input using unified template function
         load<USE_UINT4>(&input[offset], vals[valIdx], vals[valIdx + 1], vals[valIdx + 2], vals[valIdx + 3]);
         
         // Accumulate sum of squares using unified template function
         sumOfSquares = accumulate_sum_squares<USE_UINT4>(sumOfSquares, vals[valIdx], vals[valIdx + 1], vals[valIdx + 2], vals[valIdx + 3]);
     }
     
     // Warp reduction for sum of squares using __shfl_xor_sync (naturally broadcasts to all lanes)
     #pragma unroll
     for (int step = THREADS_PER_WARP / 2; step > 0; step /= 2) {
         sumOfSquares += __shfl_xor_sync(0xffffffff, sumOfSquares, step);
     }
     
     // Compute L2 norm reciprocal * scale (sumOfSquares already in all lanes from xor reduction)
     // Formula matches open source WAN: F.normalize(x, dim=1) = x / sqrt(sum(x^2) + eps)
     float norm_scale = rsqrtf(sumOfSquares + eps) * scale;
     
     // ========== Pass 2: Apply normalization, weight, SiLU, store ==========
     // OPTIMIZATION: For FP8 with USE_UINT4, process 2 iterations together to use uint4 stores (16 bytes)
     // This matches BF16's store width and improves memory throughput from ~27% to ~45%
     if constexpr (OUTPUT_DTYPE == QuantDType::FP8_E4M3 && USE_UINT4) {
         // Process pairs of iterations to store 16 FP8 values as uint4 (16 bytes)
         constexpr int NUM_PAIRS = NUM_LOADS / 2;
         constexpr bool HAS_ODD_LOAD = (NUM_LOADS % 2) != 0;
         
         // Process pairs: each pair stores 16 FP8 values as uint4 (16 bytes)
         #pragma unroll
         for (int pair = 0; pair < NUM_PAIRS; ++pair) {
             int ii = pair * 2;
             // First iteration: process 8 floats
             int elemOffset0 = (ii * THREADS_PER_WARP + laneId) * NUM_BF16s_PER_LOAD;
             int valIdx0 = ii * NUM_FLOAT2_PER_LOAD;
             
             // Second iteration: process next 8 floats
             int elemOffset1 = ((ii + 1) * THREADS_PER_WARP + laneId) * NUM_BF16s_PER_LOAD;
             int valIdx1 = (ii + 1) * NUM_FLOAT2_PER_LOAD;
             
             // Load weights for both iterations
             float2 w0_0, w1_0, w2_0, w3_0;
             float2 w0_1, w1_1, w2_1, w3_1;
             load<USE_UINT4>(&weight[elemOffset0], w0_0, w1_0, w2_0, w3_0);
             load<USE_UINT4>(&weight[elemOffset1], w0_1, w1_1, w2_1, w3_1);
             
             // Apply normalization, weight, and SiLU for first iteration
             apply_norm_weight_silu<USE_UINT4>(vals[valIdx0], vals[valIdx0 + 1], vals[valIdx0 + 2], vals[valIdx0 + 3],
                                               w0_0, w1_0, w2_0, w3_0, norm_scale);
             
             // Apply normalization, weight, and SiLU for second iteration
             apply_norm_weight_silu<USE_UINT4>(vals[valIdx1], vals[valIdx1 + 1], vals[valIdx1 + 2], vals[valIdx1 + 3],
                                               w0_1, w1_1, w2_1, w3_1, norm_scale);
             
             // Store 16 FP8 values at their correct offsets
             // Each iteration writes 8 FP8 values (8 bytes) at its own offset
             // This ensures correct data placement while still writing 16 bytes total per pair
             uint2 packed_lo = float8_to_fp8_e4m3_packed(vals[valIdx0], vals[valIdx0 + 1], vals[valIdx0 + 2], vals[valIdx0 + 3], output_quant_scale);
             uint2 packed_hi = float8_to_fp8_e4m3_packed(vals[valIdx1], vals[valIdx1 + 1], vals[valIdx1 + 2], vals[valIdx1 + 3], output_quant_scale);
             
             // Store first iteration at its correct offset
             int64_t outputOffset0_bytes = outputBaseOffset + elemOffset0;
             *reinterpret_cast<uint2*>(reinterpret_cast<char*>(output) + outputOffset0_bytes) = packed_lo;
             
             // Store second iteration at its correct offset (not consecutive, but still coalesced)
             int64_t outputOffset1_bytes = outputBaseOffset + elemOffset1;
             *reinterpret_cast<uint2*>(reinterpret_cast<char*>(output) + outputOffset1_bytes) = packed_hi;
         }
         
         // Handle odd last iteration (if NUM_LOADS is odd, e.g., dim=1280)
         if constexpr (HAS_ODD_LOAD) {
             int ii = NUM_PAIRS * 2;
             int elemOffset = (ii * THREADS_PER_WARP + laneId) * NUM_BF16s_PER_LOAD;
             int valIdx = ii * NUM_FLOAT2_PER_LOAD;
             int64_t outputOffset = outputBaseOffset + elemOffset;
             
             float2 w0, w1, w2, w3;
             load<USE_UINT4>(&weight[elemOffset], w0, w1, w2, w3);
             apply_norm_weight_silu<USE_UINT4>(vals[valIdx], vals[valIdx + 1], vals[valIdx + 2], vals[valIdx + 3],
                                               w0, w1, w2, w3, norm_scale);
             
             // Store last 8 FP8 values as uint2 (8 bytes)
             uint2 packed = float8_to_fp8_e4m3_packed(vals[valIdx], vals[valIdx + 1], vals[valIdx + 2], vals[valIdx + 3], output_quant_scale);
             *reinterpret_cast<uint2*>(reinterpret_cast<char*>(output) + outputOffset) = packed;
         }
     } else {
         // Standard path: BF16 or FP8 with USE_UINT2 (process one iteration at a time)
         #pragma unroll
         for (int ii = 0; ii < NUM_LOADS; ++ii) {
             // Compute channel offset within the token
             int elemOffset = (ii * THREADS_PER_WARP + laneId) * NUM_BF16s_PER_LOAD;
             int valIdx = ii * NUM_FLOAT2_PER_LOAD;
             
             // Compute output offset: outputBaseOffset already computed by helper function
             // For NDHWC layout, output_stride_C is always 1 (enforced in header)
             // For quantized output, stride is adjusted based on dtype size
             int64_t outputBaseOffset_bytes;
             int elemOffset_bytes;
             
             if constexpr (OUTPUT_DTYPE == QuantDType::BF16) {
                 outputBaseOffset_bytes = outputBaseOffset * 2;  // BF16: 2 bytes per element
                 elemOffset_bytes = elemOffset * 2;
             } else if constexpr (OUTPUT_DTYPE == QuantDType::FP8_E4M3) {
                 outputBaseOffset_bytes = outputBaseOffset;      // FP8: 1 byte per element
                 elemOffset_bytes = elemOffset;
             } else {  // MXFP4
                 outputBaseOffset_bytes = outputBaseOffset / 2;  // MXFP4: 0.5 bytes per element
                 elemOffset_bytes = elemOffset / 2;
             }
             
             int64_t outputOffset = outputBaseOffset_bytes + elemOffset_bytes;
             
             // Load weights using unified template function (weights are contiguous)
             float2 w0, w1, w2, w3;
             load<USE_UINT4>(&weight[elemOffset], w0, w1, w2, w3);
             
             // Apply normalization, weight, and SiLU using unified template function
             apply_norm_weight_silu<USE_UINT4>(vals[valIdx], vals[valIdx + 1], vals[valIdx + 2], vals[valIdx + 3],
                                               w0, w1, w2, w3, norm_scale);
             
             // Store results using unified template function with quantization
             store<OUTPUT_DTYPE, USE_UINT4>(vals[valIdx], vals[valIdx + 1], vals[valIdx + 2], vals[valIdx + 3], 
                                            reinterpret_cast<char*>(output) + outputOffset, output_quant_scale);
         }
     }
 }
 
 // Unified multi-token kernel for dim=64 (4 tok/warp) and dim=128 (2 tok/warp)
 // Uses uint4 loads, sub-warp reduction, magic division for NDHWC
 template <int HIDDEN_DIM, QuantDType OUTPUT_DTYPE = QuantDType::BF16, bool USE_NDHWC = false, int WARPS_PER_CTA = 32>
 __global__ void __launch_bounds__(1024) fusedRMSNormSiLUKernelMultiTok(
     __nv_bfloat16 const* __restrict__ input,
     void* __restrict__ output,
     int const num_tokens,
     int const input_stride_token,
     float const eps,
     __nv_bfloat16 const* __restrict__ weight,
     float const scale,
     __nv_bfloat16 const* __restrict__ bias,
     int const output_stride_token,
     int const DHW, int const HW, int const W,
     int const output_stride_N, int const output_stride_D,
     int const output_stride_H, int const output_stride_W,
     int const output_D_offset,
     float const output_quant_scale,
     MagicDiv32 const magic_DHW, MagicDiv32 const magic_HW, MagicDiv32 const magic_W)
 {
     static_assert(HIDDEN_DIM == 64 || HIDDEN_DIM == 128, "HIDDEN_DIM must be 64 or 128");
     constexpr int TOKENS_PER_WARP = (HIDDEN_DIM == 64) ? 4 : 2;
     constexpr int THREADS_PER_TOKEN = THREADS_PER_WARP / TOKENS_PER_WARP;
     constexpr int ELEMS_PER_THREAD = HIDDEN_DIM / THREADS_PER_TOKEN;
     
     int const warpId = threadIdx.x / THREADS_PER_WARP;
     int const laneId = threadIdx.x % THREADS_PER_WARP;
     int const tokenInWarp = laneId / THREADS_PER_TOKEN;
     int const threadInToken = laneId % THREADS_PER_TOKEN;
     int const channelOffset = threadInToken * ELEMS_PER_THREAD;
     
     uint4 packed_weight = *reinterpret_cast<uint4 const*>(&weight[channelOffset]);
     __nv_bfloat162 const* wgt_bf16_2 = reinterpret_cast<__nv_bfloat162 const*>(&packed_weight);
     float wgt[8];
     #pragma unroll
     for (int i = 0; i < 4; i++) {
         float2 w2 = __bfloat1622float2(wgt_bf16_2[i]);
         wgt[i*2] = w2.x;
         wgt[i*2+1] = w2.y;
     }
     
     int const totalWarps = gridDim.x * WARPS_PER_CTA;
     int const globalWarpId = blockIdx.x * WARPS_PER_CTA + warpId;
     int const tokenStride = totalWarps * TOKENS_PER_WARP;
     
     if constexpr (USE_NDHWC) {
         for (int baseTokenIdx = globalWarpId * TOKENS_PER_WARP; baseTokenIdx < num_tokens; baseTokenIdx += tokenStride) {
             int const tokenIdx = baseTokenIdx + tokenInWarp;
             bool const validToken = tokenIdx < num_tokens;
             int const inputOffset = tokenIdx * input_stride_token + channelOffset;
             
             uint4 packed_input;
             if (validToken) {
                 packed_input = *reinterpret_cast<uint4 const*>(&input[inputOffset]);
             } else {
                 packed_input = make_uint4(0, 0, 0, 0);
             }
             
             __nv_bfloat162 const* input_bf16_2 = reinterpret_cast<__nv_bfloat162 const*>(&packed_input);
             float vals[8];
             float sumOfSquares = 0.0f;
             #pragma unroll
             for (int i = 0; i < 4; i++) {
                 float2 v2 = __bfloat1622float2(input_bf16_2[i]);
                 vals[i*2] = v2.x;
                 vals[i*2+1] = v2.y;
                 sumOfSquares = __fmaf_rn(v2.x, v2.x, sumOfSquares);
                 sumOfSquares = __fmaf_rn(v2.y, v2.y, sumOfSquares);
             }
             
             #pragma unroll
             for (int step = THREADS_PER_TOKEN / 2; step > 0; step /= 2) {
                 sumOfSquares += __shfl_xor_sync(0xffffffff, sumOfSquares, step);
             }
             
             // Formula matches open source WAN: F.normalize(x, dim=1) = x / sqrt(sum(x^2) + eps)
             float norm_scale = rsqrtf(sumOfSquares + eps) * scale;
             
             float out[8];
             #pragma unroll
             for (int i = 0; i < 8; i++) {
                 float v = vals[i] * norm_scale * wgt[i];
                 out[i] = v * __frcp_rn(1.0f + __expf(-v));
             }
             
             if (validToken) {
                 uint32_t n, rem_dhw;
                 magicDivMod((uint32_t)tokenIdx, magic_DHW, n, rem_dhw);
                 uint32_t d, rem_hw;
                 magicDivMod(rem_dhw, magic_HW, d, rem_hw);
                 uint32_t h, w;
                 magicDivMod(rem_hw, magic_W, h, w);
                 
                 int64_t outputBaseOffset = (int64_t)n * output_stride_N + 
                                            (int64_t)(output_D_offset + d) * output_stride_D + 
                                            (int64_t)h * output_stride_H + 
                                            (int64_t)w * output_stride_W;
                 #pragma unroll
                 for (int i = 0; i < 4; i++) {
                     int outOff = outputBaseOffset + channelOffset + i * 2;
                     store_pair<OUTPUT_DTYPE>(out[i*2], out[i*2+1], output, outOff, output_quant_scale);
                 }
             }
         }
     } else {
         for (int baseTokenIdx = globalWarpId * TOKENS_PER_WARP; baseTokenIdx < num_tokens; baseTokenIdx += tokenStride) {
             int const tokenIdx = baseTokenIdx + tokenInWarp;
             bool const validToken = tokenIdx < num_tokens;
             int const inputOffset = tokenIdx * input_stride_token + channelOffset;
             
             uint4 packed_input;
             if (validToken) {
                 packed_input = *reinterpret_cast<uint4 const*>(&input[inputOffset]);
             } else {
                 packed_input = make_uint4(0, 0, 0, 0);
             }
             
             __nv_bfloat162 const* input_bf16_2 = reinterpret_cast<__nv_bfloat162 const*>(&packed_input);
             float vals[8];
             float sumOfSquares = 0.0f;
             #pragma unroll
             for (int i = 0; i < 4; i++) {
                 float2 v2 = __bfloat1622float2(input_bf16_2[i]);
                 vals[i*2] = v2.x;
                 vals[i*2+1] = v2.y;
                 sumOfSquares = __fmaf_rn(v2.x, v2.x, sumOfSquares);
                 sumOfSquares = __fmaf_rn(v2.y, v2.y, sumOfSquares);
             }
             
             #pragma unroll
             for (int step = THREADS_PER_TOKEN / 2; step > 0; step /= 2) {
                 sumOfSquares += __shfl_xor_sync(0xffffffff, sumOfSquares, step);
             }
             
             // Formula matches open source WAN: F.normalize(x, dim=1) = x / sqrt(sum(x^2) + eps)
             float norm_scale = rsqrtf(sumOfSquares + eps) * scale;
             
             float out[8];
             #pragma unroll
             for (int i = 0; i < 8; i++) {
                 float v = vals[i] * norm_scale * wgt[i];
                 out[i] = v * __frcp_rn(1.0f + __expf(-v));
             }
             
             if (validToken) {
                 int64_t outputBaseOffset = (int64_t)tokenIdx * output_stride_token;
                 #pragma unroll
                 for (int i = 0; i < 4; i++) {
                     int outOff = outputBaseOffset + channelOffset + i * 2;
                     store_pair<OUTPUT_DTYPE>(out[i*2], out[i*2+1], output, outOff, output_quant_scale);
                 }
             }
         }
     }
 }
 
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 //
 // Fused RMSNorm + SiLU kernel for small hidden_dim (160, 256, 320)
 // Persistent CTA version with flexible vectorization
 // - Persistent CTAs: gridDim.x = 2 * num_SMs, each CTA loops through all assigned tokens
 // - Weights loaded into shared memory ONCE, reused for ALL token iterations
 // - 1 warp processes 1 token (entire hidden_dim)
 // - Uses uint (2 bf16) vectorized loads for flexibility (handles odd ELEMS_PER_THREAD)
 // - Supports: 160, 256, 320 (must be divisible by 32)
 // - Note: dim=64 and dim=128 use fusedRMSNormSiLUKernelMultiTok for better performance
 //
 // Formula: SiLU(L2Norm(x) * scale * gamma + bias)
 // where L2Norm(x) = x / sqrt(sum(x^2) + eps)
 //
 // Unified Small Fused RMSNorm + SiLU kernel
 // Supports both standard and NDHWC output layouts via template parameter
 // - Uses scalar loads/stores (handles odd element counts like 160)
 // - Uses persistent CTA with 32 warps per CTA (1024 threads)
 // - Supports: 160, 256, 320 (must be divisible by 32)
 // OUTPUT_DTYPE: Quantization type for output (BF16, FP8_E4M3, MXFP4)
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 template <int hidden_dim, QuantDType OUTPUT_DTYPE = QuantDType::BF16, bool USE_NDHWC = false, int WARPS_PER_CTA = 32>
 __global__ void __launch_bounds__(1024) fusedRMSNormSiLUKernelSmall(
     __nv_bfloat16 const* __restrict__ input,   // Input tensor [num_tokens, hidden_dim]
     void* __restrict__ output,                 // Output tensor [num_tokens, hidden_dim] or NDHWC layout (quantized)
     int const num_tokens,                      // Number of tokens
     int const input_stride_token,              // Stride between tokens in input (in elements)
     float const eps,                           // Epsilon for L2 normalization
     __nv_bfloat16 const* __restrict__ weight,  // Gamma weights [hidden_dim]
     float const scale,                         // Scale factor (typically sqrt(hidden_dim))
     __nv_bfloat16 const* __restrict__ bias,   // Optional bias [hidden_dim] (can be nullptr)
     // Standard layout parameters (used when USE_NDHWC == false)
     int const output_stride_token,             // Stride between tokens in output (in elements)
     // NDHWC layout parameters (used when USE_NDHWC == true)
     int const DHW,                             // Depth * Height * Width dimension (for input tokens)
     int const HW,                              // Height * Width dimension
     int const W,                               // Width dimension
     int const output_stride_N,                 // Stride for N dimension in output (in elements)
     int const output_stride_D,                 // Stride for D dimension in output (in elements)
     int const output_stride_H,                 // Stride for H dimension in output (in elements)
     int const output_stride_W,                 // Stride for W dimension in output (in elements)
     int const output_D_offset,                 // Offset in D dimension where to start writing
     float const output_quant_scale             // Quantization scale for MXFP4 output
 )
 {
     static_assert(hidden_dim % 32 == 0, "hidden_dim must be divisible by 32");
     
     constexpr int ELEMS_PER_THREAD = hidden_dim / THREADS_PER_WARP;  // 4, 5, 8, 10 for 128, 160, 256, 320
     constexpr bool HAS_ODD = (ELEMS_PER_THREAD % 2) != 0;  // True for 160 (5 elements)
     // For odd ELEMS_PER_THREAD, some threads have misaligned addresses for uint loads/stores
     // Use scalar loads/stores for all elements when HAS_ODD is true
     constexpr int NUM_PAIRS = HAS_ODD ? 0 : ELEMS_PER_THREAD;
     
     // Shared memory for weights (loaded ONCE, used for ALL token iterations)
     // Store as float for flexibility (handles odd element counts)
     __shared__ float s_weight[hidden_dim];
     
     int const warpId = threadIdx.x / THREADS_PER_WARP;
     int const laneId = threadIdx.x % THREADS_PER_WARP;
     int const channelOffset = laneId * ELEMS_PER_THREAD;  // Starting channel index for this thread
     
     // ========== ONE-TIME: Cooperative weight load into shared memory ==========
     // All threads cooperatively load weights (2 bf16 per thread using uint)
     constexpr int NUM_WEIGHT_PAIRS = hidden_dim / 2;
     int const index = threadIdx.x << 1;
     if (threadIdx.x < NUM_WEIGHT_PAIRS) {
         uint packed_w = *reinterpret_cast<uint const*>(&weight[index]);
         float2 w = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed_w));
         s_weight[index] = w.x;
         s_weight[index + 1] = w.y;
     }
     __syncthreads();
     
     // ========== ONCE: Load weights from shared memory into registers ==========
     float r_weight[ELEMS_PER_THREAD];
     #pragma unroll
     for (int i = 0; i < ELEMS_PER_THREAD; i++) {
         r_weight[i] = s_weight[channelOffset + i];
     }
     
     // ========== Persistent CTA: Loop through all tokens ==========
     int const totalWarps = gridDim.x * WARPS_PER_CTA;
     int const globalWarpId = blockIdx.x * WARPS_PER_CTA + warpId;
     
     for (int tokenIdx = globalWarpId; tokenIdx < num_tokens; tokenIdx += totalWarps) {
         
         // Use stride_token for input offset calculation (supports non-contiguous token layouts)
         // stride_hidden == 1 is required for vectorized loads to work (checked in C++ wrapper)
         int const inputBaseOffset = tokenIdx * input_stride_token + channelOffset;
         
         // Compute output base offset using helper function (without channel offset)
         int64_t const outputBaseOffset = computeOutputBaseOffset<USE_NDHWC>(
             tokenIdx,
             output_stride_token,
             DHW, HW, W,
             output_stride_N, output_stride_D, output_stride_H, output_stride_W,
             output_D_offset);
         
         // Per-token register storage for input values
         float vals[ELEMS_PER_THREAD];
         
         // ========== Pass 1: Load input, compute sum of squares ==========
         float sumOfSquares = 0.0f;
         
         if constexpr (HAS_ODD) {
             // For odd ELEMS_PER_THREAD (e.g., 160), use scalar loads to avoid alignment issues
             #pragma unroll
             for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                 float v = __bfloat162float(input[inputBaseOffset + i]);
                 vals[i] = v;
                 sumOfSquares = __fmaf_rn(v, v, sumOfSquares);
             }
         } else {
             // For even ELEMS_PER_THREAD, use vectorized uint (2 bf16) loads
             // stride_hidden == 1 is required for vectorized loads (checked in C++ wrapper)
             #pragma unroll
             for (int i = 0; i < NUM_PAIRS; i += 2) {
                 uint packed_input = *reinterpret_cast<uint const*>(&input[inputBaseOffset + i]);
                 float2 v = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&packed_input));
                 vals[i] = v.x;
                 vals[i + 1] = v.y;
                 sumOfSquares = __fmaf_rn(v.y, v.y, __fmaf_rn(v.x, v.x, sumOfSquares));
             }
         }
         
         // Warp reduction for sum of squares using __shfl_xor_sync (naturally broadcasts to all lanes)
         #pragma unroll
         for (int step = THREADS_PER_WARP / 2; step > 0; step /= 2) {
             sumOfSquares += __shfl_xor_sync(0xffffffff, sumOfSquares, step);
         }
         
         // Compute L2 norm reciprocal * scale (sumOfSquares already in all lanes from xor reduction)
         // Formula matches open source WAN: F.normalize(x, dim=1) = x / sqrt(sum(x^2) + eps)
         float norm_scale = rsqrtf(sumOfSquares + eps) * scale;
         
         // ========== Pass 2: Apply normalization, weight, SiLU, store ==========
         #pragma unroll
         for (int i = 0; i < ELEMS_PER_THREAD; i++) {
             float r = __fmaf_rn(vals[i] * norm_scale, r_weight[i], 0.0f);
             r *= __frcp_rn(1.0f + __expf(-r));
             vals[i] = r;
         }
         
         // Store to output (outputBaseOffset already includes channelOffset for standard layout)
         if constexpr (HAS_ODD) {
             // For odd ELEMS_PER_THREAD, use scalar stores to avoid alignment issues
             #pragma unroll
             for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                 int outputOffset = outputBaseOffset + channelOffset + i;
                 store_scalar<OUTPUT_DTYPE>(vals[i], output, outputOffset, output_quant_scale);
             }
         } else {
             // For even ELEMS_PER_THREAD, use vectorized uint (2 bf16) stores
             // stride_hidden == 1 is required for vectorized stores (checked in C++ wrapper)
             #pragma unroll
             for (int i = 0; i < NUM_PAIRS; i += 2) {
                 int outputOffset = outputBaseOffset + channelOffset + i;
                 store_pair<OUTPUT_DTYPE>(vals[i], vals[i + 1], output, outputOffset, output_quant_scale);
             }
         }
     }  // End of persistent token loop
 }
 
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Launch function for fused RMSNorm + SiLU
 // Computes RMS normalization across hidden dimension, applies weight/scale, then SiLU activation
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Unified launch function template
 // Handles both standard and NDHWC output layouts via template parameter USE_NDHWC
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 template <bool USE_NDHWC>
 void launchFusedRMSNormSiLU(
     void const* input,
     void* output,
     int const num_tokens,
     int const hidden_dim,
     int const input_stride_token,
     int const num_sms,
     float const eps,
     void const* weight,
     float const scale,
     void const* bias,
     // Standard layout parameters (used when USE_NDHWC == false)
     int const output_stride_token,
     // NDHWC layout parameters (used when USE_NDHWC == true)
     int const DHW,
     int const HW,
     int const W,
     int const output_stride_N,
     int const output_stride_D,
     int const output_stride_H,
     int const output_stride_W,
     int const output_D_offset,
     cudaStream_t stream)
 {
     constexpr int WARPS_PER_CTA_SMALL = 32;
     constexpr int WARPS_PER_CTA_LARGE = 8;
     // For MultiTok kernels (C=64,128) with few tokens: use fewer warps/CTA to improve
     // SM utilization. With 32 warps/CTA and 4 tokens/warp, 1560 tokens only fills 13 CTAs
     // (390 active warps out of 1568 = 25% utilization). Using 8 warps/CTA gives 98 CTAs
     // with nearly 100% warp utilization.
     constexpr int WARPS_PER_CTA_MULTITOK_SMALL = 8;
     constexpr int blockSizeSmall = WARPS_PER_CTA_SMALL * THREADS_PER_WARP;
     constexpr int blockSizeLarge = WARPS_PER_CTA_LARGE * THREADS_PER_WARP;
     constexpr int blockSizeMultiTokSmall = WARPS_PER_CTA_MULTITOK_SMALL * THREADS_PER_WARP;
     
     int const gridSizeStandardSmall = tensorrt_llm::common::divUp(num_tokens, WARPS_PER_CTA_SMALL);
     int const gridSizeSmall = min(2 * num_sms, gridSizeStandardSmall);
     int const gridSizeLarge = tensorrt_llm::common::divUp(num_tokens, WARPS_PER_CTA_LARGE);
     int const gridSizeMultiTokSmall = min(2 * num_sms, (int)tensorrt_llm::common::divUp(num_tokens, WARPS_PER_CTA_MULTITOK_SMALL));
     
     // Use fewer warps/CTA for MultiTok when token count is small enough that
     // the 32-warp CTA would have low warp utilization (< ~50% warps active).
     bool const useSmallMultiTok64 = (num_tokens < 4 * WARPS_PER_CTA_SMALL * num_sms / 2);
     bool const useSmallMultiTok128 = (num_tokens < 2 * WARPS_PER_CTA_SMALL * num_sms / 2);
     
     // Dummy values for unused parameters (compiler will optimize away)
     constexpr int DUMMY_OUTPUT_STRIDE_TOKEN = 0;
     constexpr int DUMMY_DHW = 0, DUMMY_HW = 0, DUMMY_W = 0;
     constexpr int DUMMY_STRIDE_N = 0, DUMMY_STRIDE_D = 0, DUMMY_STRIDE_H = 0, DUMMY_STRIDE_W = 0;
     constexpr int DUMMY_D_OFFSET = 0;
     constexpr float DUMMY_QUANT_SCALE = 1.0f;  // No quantization for BF16 output
     
     // Precompute magic division parameters for NDHWC coordinate computation (dim64 kernel only)
     // These replace expensive integer division with multiply+shift operations
     MagicDiv32 magic_DHW = USE_NDHWC ? computeMagicDiv32(DHW > 0 ? DHW : 1) : MagicDiv32{1, 0, 1};
     MagicDiv32 magic_HW = USE_NDHWC ? computeMagicDiv32(HW > 0 ? HW : 1) : MagicDiv32{1, 0, 1};
     MagicDiv32 magic_W = USE_NDHWC ? computeMagicDiv32(W > 0 ? W : 1) : MagicDiv32{1, 0, 1};
     
     // Dispatch based on hidden dimension (matches open source WAN implementation)
     switch (hidden_dim) {
         case 64:
             // Use optimized kernel for dim=64: 4 tokens per warp with uint4 loads
             // Uses magic division for fast NDHWC coordinate computation
             if (useSmallMultiTok64) {
                 // Few tokens: use 8 warps/CTA for better SM utilization
                 fusedRMSNormSiLUKernelMultiTok<64, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_MULTITOK_SMALL><<<gridSizeMultiTokSmall, blockSizeMultiTokSmall, 0, stream>>>(
                     reinterpret_cast<__nv_bfloat16 const*>(input),
                     reinterpret_cast<__nv_bfloat16*>(output),
                     num_tokens, input_stride_token, eps,
                     reinterpret_cast<__nv_bfloat16 const*>(weight),
                     scale,
                     reinterpret_cast<__nv_bfloat16 const*>(bias),
                     USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                     USE_NDHWC ? DHW : DUMMY_DHW,
                     USE_NDHWC ? HW : DUMMY_HW,
                     USE_NDHWC ? W : DUMMY_W,
                     USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                     USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                     USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                     USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                     USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                     DUMMY_QUANT_SCALE,
                     magic_DHW, magic_HW, magic_W);
             } else {
                 // Many tokens: use 32 warps/CTA for maximum throughput
                 fusedRMSNormSiLUKernelMultiTok<64, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                     reinterpret_cast<__nv_bfloat16 const*>(input),
                     reinterpret_cast<__nv_bfloat16*>(output),
                     num_tokens, input_stride_token, eps,
                     reinterpret_cast<__nv_bfloat16 const*>(weight),
                     scale,
                     reinterpret_cast<__nv_bfloat16 const*>(bias),
                     USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                     USE_NDHWC ? DHW : DUMMY_DHW,
                     USE_NDHWC ? HW : DUMMY_HW,
                     USE_NDHWC ? W : DUMMY_W,
                     USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                     USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                     USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                     USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                     USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                     DUMMY_QUANT_SCALE,
                     magic_DHW, magic_HW, magic_W);
             }
             break;
         case 128:
             // Use optimized kernel for dim=128: 2 tokens per warp with uint4 loads
             // Uses magic division for fast NDHWC coordinate computation
             if (useSmallMultiTok128) {
                 // Few tokens: use 8 warps/CTA for better SM utilization
                 fusedRMSNormSiLUKernelMultiTok<128, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_MULTITOK_SMALL><<<gridSizeMultiTokSmall, blockSizeMultiTokSmall, 0, stream>>>(
                     reinterpret_cast<__nv_bfloat16 const*>(input),
                     reinterpret_cast<__nv_bfloat16*>(output),
                     num_tokens, input_stride_token, eps,
                     reinterpret_cast<__nv_bfloat16 const*>(weight),
                     scale,
                     reinterpret_cast<__nv_bfloat16 const*>(bias),
                     USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                     USE_NDHWC ? DHW : DUMMY_DHW,
                     USE_NDHWC ? HW : DUMMY_HW,
                     USE_NDHWC ? W : DUMMY_W,
                     USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                     USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                     USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                     USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                     USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                     DUMMY_QUANT_SCALE,
                     magic_DHW, magic_HW, magic_W);
             } else {
                 // Many tokens: use 32 warps/CTA for maximum throughput
                 fusedRMSNormSiLUKernelMultiTok<128, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                     reinterpret_cast<__nv_bfloat16 const*>(input),
                     reinterpret_cast<__nv_bfloat16*>(output),
                     num_tokens, input_stride_token, eps,
                     reinterpret_cast<__nv_bfloat16 const*>(weight),
                     scale,
                     reinterpret_cast<__nv_bfloat16 const*>(bias),
                     USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                     USE_NDHWC ? DHW : DUMMY_DHW,
                     USE_NDHWC ? HW : DUMMY_HW,
                     USE_NDHWC ? W : DUMMY_W,
                     USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                     USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                     USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                     USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                     USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                     DUMMY_QUANT_SCALE,
                     magic_DHW, magic_HW, magic_W);
             }
             break;
         case 160:
             fusedRMSNormSiLUKernelSmall<160, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 256:
             fusedRMSNormSiLUKernelSmall<256, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 320:
             fusedRMSNormSiLUKernelSmall<320, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 384:
             fusedRMSNormSiLUKernel<384, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 512:
             fusedRMSNormSiLUKernel<512, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 640:
             fusedRMSNormSiLUKernel<640, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 768:
             fusedRMSNormSiLUKernel<768, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 896:
             fusedRMSNormSiLUKernel<896, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 1024:
             fusedRMSNormSiLUKernel<1024, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 1280:
             fusedRMSNormSiLUKernel<1280, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 1536:
             fusedRMSNormSiLUKernel<1536, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         case 2048:
             fusedRMSNormSiLUKernel<2048, QuantDType::BF16, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input),
                 reinterpret_cast<__nv_bfloat16*>(output),
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight),
                 scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW,
                 USE_NDHWC ? HW : DUMMY_HW,
                 USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 DUMMY_QUANT_SCALE);
             break;
         default:
             TLLM_THROW("Unsupported hidden dimension for fusedRMSNormSiLU: %d. "
                     "Supported: 64, 128, 160, 256, 320, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048", hidden_dim);
     }
 }
 
 // Note: The template function is called directly from C++ code, so explicit template
 // instantiations are provided at the end of this file for linking.
 
 
 __global__ void feature_cat_kernel(__nv_bfloat16* __restrict__ output, __nv_bfloat16 const* __restrict__ input, __nv_bfloat16 const* __restrict__ feat_cache, 
                            const int C, const int H, const int W, const int D, const int D_cache) {
     int ni = blockIdx.x;
     int hi = blockIdx.y;
     int wi = blockIdx.z;
 
     const int channels_per_thread = sizeof(float4)/sizeof(__nv_bfloat16);
 
     int ci = threadIdx.x*channels_per_thread;
 
     float4* output_vec = (float4*)output;
     float4* input_vec = (float4*)input;
     float4* feat_cache_vec = (float4*)feat_cache;
 
     // TODO: get strides from the tensors
     int64_t batch_stride_cache = D_cache*H*W*C;
     int64_t batch_stride_input = D*H*W*C;
     int64_t batch_stride_output = (D+D_cache)*H*W*C;
 
     int64_t offset_cache = ni*batch_stride_cache + hi*W*C + wi*C + ci;
     int64_t offset_input = ni*batch_stride_input + hi*W*C + wi*C + ci;
     int64_t offset_output = ni*batch_stride_output + hi*W*C + wi*C + ci; 
 
     int di = 0;
 #pragma unroll
     for (int i=0; i<D_cache; i++) {
             output_vec[(offset_output + di*H*W*C)/channels_per_thread] = feat_cache_vec[(offset_cache + i*H*W*C)/channels_per_thread];
         di += 1;
     }
 #pragma unroll
     for (int i=0; i<D; i++) {
             output_vec[(offset_output + di*H*W*C)/channels_per_thread] = input_vec[(offset_input + i*H*W*C)/channels_per_thread];
         di += 1;
     }
 }
 
 extern "C" void launchFeatureCat(
     void* output, void const* input, void const* feat_cache, 
     int const N, int const C, int const D, int const H, int const W, int const D_cache, 
     cudaStream_t stream) {
 
     dim3 grid(N,H,W);
 
     const int channels_per_thread = sizeof(float4)/sizeof(__nv_bfloat16);
     assert((C%channels_per_thread)==0);
 
     dim3 block(C/channels_per_thread,1,1);
 
     feature_cat_kernel<<<grid,block,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(output), reinterpret_cast<const __nv_bfloat16*>(input), reinterpret_cast<const __nv_bfloat16*>(feat_cache), C, H, W, D, D_cache);
 
 }
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Helper launch function with quantization support (template)
 // Launches kernels with specific OUTPUT_DTYPE parameter
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 template <bool USE_NDHWC, QuantDType OUTPUT_DTYPE>
 void launchFusedRMSNormSiLUWithQuant(
     void const* input,
     void* output,
     int const num_tokens,
     int const hidden_dim,
     int const input_stride_token,
     int const num_sms,
     float const eps,
     void const* weight,
     float const scale,
     void const* bias,
     int const output_stride_token,
     int const DHW,
     int const HW,
     int const W,
     int const output_stride_N,
     int const output_stride_D,
     int const output_stride_H,
     int const output_stride_W,
     int const output_D_offset,
     float const output_quant_scale,
     cudaStream_t stream)
 {
     constexpr int WARPS_PER_CTA_SMALL = 32;
     constexpr int WARPS_PER_CTA_LARGE = 8;
     // For MultiTok kernels (C=64,128) with few tokens: use fewer warps/CTA to improve
     // SM utilization. With 32 warps/CTA and 4 tokens/warp, 1560 tokens only fills 13 CTAs
     // (390 active warps out of 1568 = 25% utilization). Using 8 warps/CTA gives 98 CTAs
     // with nearly 100% warp utilization.
     constexpr int WARPS_PER_CTA_MULTITOK_SMALL = 8;
     constexpr int blockSizeSmall = WARPS_PER_CTA_SMALL * THREADS_PER_WARP;
     constexpr int blockSizeLarge = WARPS_PER_CTA_LARGE * THREADS_PER_WARP;
     constexpr int blockSizeMultiTokSmall = WARPS_PER_CTA_MULTITOK_SMALL * THREADS_PER_WARP;
     
     int const gridSizeStandardSmall = tensorrt_llm::common::divUp(num_tokens, WARPS_PER_CTA_SMALL);
     int const gridSizeSmall = min(2 * num_sms, gridSizeStandardSmall);
     int const gridSizeLarge = tensorrt_llm::common::divUp(num_tokens, WARPS_PER_CTA_LARGE);
     int const gridSizeMultiTokSmall = min(2 * num_sms, (int)tensorrt_llm::common::divUp(num_tokens, WARPS_PER_CTA_MULTITOK_SMALL));
     
     // Use fewer warps/CTA for MultiTok when token count is small enough that
     // the 32-warp CTA would have low warp utilization (< ~50% warps active).
     bool const useSmallMultiTok64 = (num_tokens < 4 * WARPS_PER_CTA_SMALL * num_sms / 2);
     bool const useSmallMultiTok128 = (num_tokens < 2 * WARPS_PER_CTA_SMALL * num_sms / 2);
     
     // Dummy values for unused parameters
     constexpr int DUMMY_OUTPUT_STRIDE_TOKEN = 0;
     constexpr int DUMMY_DHW = 0, DUMMY_HW = 0, DUMMY_W = 0;
     constexpr int DUMMY_STRIDE_N = 0, DUMMY_STRIDE_D = 0, DUMMY_STRIDE_H = 0, DUMMY_STRIDE_W = 0;
     constexpr int DUMMY_D_OFFSET = 0;
     
     // Precompute magic division parameters for NDHWC coordinate computation (dim64 kernel only)
     // These replace expensive integer division with multiply+shift operations
     MagicDiv32 magic_DHW = USE_NDHWC ? computeMagicDiv32(DHW > 0 ? DHW : 1) : MagicDiv32{1, 0, 1};
     MagicDiv32 magic_HW = USE_NDHWC ? computeMagicDiv32(HW > 0 ? HW : 1) : MagicDiv32{1, 0, 1};
     MagicDiv32 magic_W = USE_NDHWC ? computeMagicDiv32(W > 0 ? W : 1) : MagicDiv32{1, 0, 1};
     
     // Dispatch based on hidden dimension (matches open source WAN implementation)
     switch (hidden_dim) {
         case 64:
             // Use optimized kernel for dim=64: 4 tokens per warp with uint4 loads
             // Uses magic division for fast NDHWC coordinate computation
             fusedRMSNormSiLUKernelMultiTok<64, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale,
                 magic_DHW, magic_HW, magic_W);
             break;
         case 128:
             // Use optimized kernel for dim=128: 2 tokens per warp with uint4 loads
             // Uses magic division for fast NDHWC coordinate computation
             fusedRMSNormSiLUKernelMultiTok<128, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale,
                 magic_DHW, magic_HW, magic_W);
             break;
         case 160:
             fusedRMSNormSiLUKernelSmall<160, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 256:
             fusedRMSNormSiLUKernelSmall<256, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 320:
             fusedRMSNormSiLUKernelSmall<320, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_SMALL><<<gridSizeSmall, blockSizeSmall, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 384:
             fusedRMSNormSiLUKernel<384, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 512:
             fusedRMSNormSiLUKernel<512, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 640:
             fusedRMSNormSiLUKernel<640, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 768:
             fusedRMSNormSiLUKernel<768, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 896:
             fusedRMSNormSiLUKernel<896, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 1024:
             fusedRMSNormSiLUKernel<1024, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 1280:
             fusedRMSNormSiLUKernel<1280, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 1536:
             fusedRMSNormSiLUKernel<1536, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         case 2048:
             fusedRMSNormSiLUKernel<2048, OUTPUT_DTYPE, USE_NDHWC, WARPS_PER_CTA_LARGE><<<gridSizeLarge, blockSizeLarge, 0, stream>>>(
                 reinterpret_cast<__nv_bfloat16 const*>(input), output,
                 num_tokens, input_stride_token, eps,
                 reinterpret_cast<__nv_bfloat16 const*>(weight), scale,
                 reinterpret_cast<__nv_bfloat16 const*>(bias),
                 USE_NDHWC ? DUMMY_OUTPUT_STRIDE_TOKEN : output_stride_token,
                 USE_NDHWC ? DHW : DUMMY_DHW, USE_NDHWC ? HW : DUMMY_HW, USE_NDHWC ? W : DUMMY_W,
                 USE_NDHWC ? output_stride_N : DUMMY_STRIDE_N,
                 USE_NDHWC ? output_stride_D : DUMMY_STRIDE_D,
                 USE_NDHWC ? output_stride_H : DUMMY_STRIDE_H,
                 USE_NDHWC ? output_stride_W : DUMMY_STRIDE_W,
                 USE_NDHWC ? output_D_offset : DUMMY_D_OFFSET,
                 output_quant_scale);
             break;
         default:
             TLLM_THROW("Unsupported hidden dimension for fusedRMSNormSiLU: %d. "
                     "Supported: 64, 128, 160, 256, 320, 384, 512, 640, 768, 896, 1024, 1280, 1536, 2048", hidden_dim);
     }
 }
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // Overloaded launch function with quantization support
 // Dispatches to appropriate quantized kernel based on QuantConfig
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 template <bool USE_NDHWC>
 void launchFusedRMSNormSiLU(
     void const* input,
     void* output,
     int const num_tokens,
     int const hidden_dim,
     int const input_stride_token,
     int const num_sms,
     float const eps,
     void const* weight,
     float const scale,
     void const* bias,
     int const output_stride_token,
     int const DHW,
     int const HW,
     int const W,
     int const output_stride_N,
     int const output_stride_D,
     int const output_stride_H,
     int const output_stride_W,
     int const output_D_offset,
     QuantConfig const& quant_config,
     cudaStream_t stream)
 {
     // Dispatch based on output quantization type
     switch (quant_config.output_dtype) {
         case QuantDType::BF16:
             launchFusedRMSNormSiLUWithQuant<USE_NDHWC, QuantDType::BF16>(
                 input, output, num_tokens, hidden_dim, input_stride_token, num_sms, eps,
                 weight, scale, bias, output_stride_token, DHW, HW, W,
                 output_stride_N, output_stride_D, output_stride_H, output_stride_W, output_D_offset,
                 quant_config.output_scale, stream);
             break;
         case QuantDType::FP8_E4M3:
             launchFusedRMSNormSiLUWithQuant<USE_NDHWC, QuantDType::FP8_E4M3>(
                 input, output, num_tokens, hidden_dim, input_stride_token, num_sms, eps,
                 weight, scale, bias, output_stride_token, DHW, HW, W,
                 output_stride_N, output_stride_D, output_stride_H, output_stride_W, output_D_offset,
                 quant_config.output_scale, stream);
             break;
         case QuantDType::FP8_E5M2:
             // E5M2 not commonly used for inference; falls back to E4M3
             // This reduces compilation time by avoiding E5M2 template instantiations (25% faster compile)
             launchFusedRMSNormSiLUWithQuant<USE_NDHWC, QuantDType::FP8_E4M3>(
                 input, output, num_tokens, hidden_dim, input_stride_token, num_sms, eps,
                 weight, scale, bias, output_stride_token, DHW, HW, W,
                 output_stride_N, output_stride_D, output_stride_H, output_stride_W, output_D_offset,
                 quant_config.output_scale, stream);
             break;
         case QuantDType::MXFP4:
             launchFusedRMSNormSiLUWithQuant<USE_NDHWC, QuantDType::MXFP4>(
                 input, output, num_tokens, hidden_dim, input_stride_token, num_sms, eps,
                 weight, scale, bias, output_stride_token, DHW, HW, W,
                 output_stride_N, output_stride_D, output_stride_H, output_stride_W, output_D_offset,
                 quant_config.output_scale, stream);
             break;
         default:
             TLLM_THROW("Unsupported output quantization type");
     }
 }
 
 // Explicit template instantiations to ensure symbols are exported for linking
 template void launchFusedRMSNormSiLU<false>(
     void const*, void*, int const, int const, int const, int const, float const,
     void const*, float const, void const*, int const, int const, int const, int const,
     int const, int const, int const, int const, int const, cudaStream_t);
 
 template void launchFusedRMSNormSiLU<true>(
     void const*, void*, int const, int const, int const, int const, float const,
     void const*, float const, void const*, int const, int const, int const, int const,
     int const, int const, int const, int const, int const, cudaStream_t);
 
 // Explicit template instantiations for quantized versions
 template void launchFusedRMSNormSiLU<false>(
     void const*, void*, int const, int const, int const, int const, float const,
     void const*, float const, void const*, int const, int const, int const, int const,
     int const, int const, int const, int const, int const, QuantConfig const&, cudaStream_t);
 
 template void launchFusedRMSNormSiLU<true>(
     void const*, void*, int const, int const, int const, int const, float const,
     void const*, float const, void const*, int const, int const, int const, int const,
     int const, int const, int const, int const, int const, QuantConfig const&, cudaStream_t);
 
 