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
#ifndef FLASHINFER_MLA_KV_PACK_FP8_CUH_
#define FLASHINFER_MLA_KV_PACK_FP8_CUH_

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flashinfer {

// Fused MLA context K/V pack: bf16 [k_nope | v] per head + shared bf16 k_pe
// -> fp8 e4m3 key = [k_nope | k_pe broadcast] and fp8 e4m3 value = v, in one
// pass over HBM. The conversion is the saturating e4m3 cast
// (cvt.rn.satfinite: RNE, finite overflow and +-inf -> +-448, NaN -> 0x7F),
// i.e. torch >= 2.13 Tensor.to(float8_e4m3fn) on the GPU.
//
// Head geometry is fixed at (nope 128 | v 128) x rope 64 (DeepSeek/Kimi MLA);
// the head COUNT is a runtime argument, with a fully unrolled instantiation
// for 12 local heads (96 heads / TP8).
namespace mla_kv_pack {

constexpr int kNopeDim = 128;
constexpr int kRopeDim = 64;
constexpr int kVDim = 128;
constexpr int kKVDim = kNopeDim + kVDim;     // 256 bf16 per head in
constexpr int kQKDim = kNopeDim + kRopeDim;  // 192 fp8 per head out
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = 32 * kWarpsPerBlock;

// 16 B granules per head/row.
constexpr int kKVUint4PerHead = kKVDim * 2 / 16;     // 32
constexpr int kKeyUint2PerHead = kQKDim / 8;         // 24
constexpr int kValUint2PerHead = kVDim / 8;          // 16
constexpr int kRopeUint4PerRow = kRopeDim * 2 / 16;  // 8

// Eight bf16 (one 16 B load) -> eight e4m3 (saturating RNE), packed into 8 B.
// bf16 -> f32 is an exact bit shift, so the f32-source cvt (PTX ISA 7.8+,
// every CUDA >= 11.8 toolkit) gives the same bytes as a bf16x2-source cvt
// while building on the deployment toolkit; the first source operand lands in
// the high byte (element 1). The four 16-bit results are packed with explicit
// mov.b32 register pairs instead of shift/or arithmetic.
__device__ __forceinline__ uint2 cvt_octet_satfinite(const uint4 bits) {
  uint2 out;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  // No fp8 cvt before sm_89; the Python guard never dispatches here. The asm
  // is excluded (not just skipped) so ptxas never sees it for older targets.
  __trap();
  out = make_uint2(0u, 0u);
#else
  asm("{\n\t"
      ".reg .b16 p0, p1, p2, p3;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 p0, %2, %3;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 p1, %4, %5;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 p2, %6, %7;\n\t"
      "cvt.rn.satfinite.e4m3x2.f32 p3, %8, %9;\n\t"
      "mov.b32 %0, {p0, p1};\n\t"
      "mov.b32 %1, {p2, p3};\n\t"
      "}"
      : "=r"(out.x), "=r"(out.y)
      : "f"(__uint_as_float(bits.x & 0xFFFF0000u)), "f"(__uint_as_float(bits.x << 16)),
        "f"(__uint_as_float(bits.y & 0xFFFF0000u)), "f"(__uint_as_float(bits.y << 16)),
        "f"(__uint_as_float(bits.z & 0xFFFF0000u)), "f"(__uint_as_float(bits.z << 16)),
        "f"(__uint_as_float(bits.w & 0xFFFF0000u)), "f"(__uint_as_float(bits.w << 16)));
#endif
  return out;
}

// One warp per token. Lanes 0-15 handle even heads, 16-31 odd heads; within a
// half-warp, lanes 0-7 write the key nope part and 8-15 the value part, each
// lane converting 16 bf16 (two 16 B loads) into one fully covered 16 B store.
// The rope row is converted once per token by lanes 0-7 and shuffle-broadcast
// to every head. kHeadsStatic > 0 fixes the head count at compile time.
template <int kHeadsStatic>
__global__ __launch_bounds__(kThreads, 2) void MLAKVPackFP8Kernel(
    const uint4* __restrict__ kv, const uint4* __restrict__ pe, uint2* __restrict__ key,
    uint2* __restrict__ value, int64_t tokens, int num_heads_rt) {
  const int num_heads = kHeadsStatic > 0 ? kHeadsStatic : num_heads_rt;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  int64_t token = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  const int64_t token_stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  const int rope_piece = lane & 7;
  const int rope_head_group = lane >> 3;
  const int head_select = lane >> 4;
  const int sub = lane & 15;
  const int is_value = sub >> 3;
  const int piece = sub & 7;

  for (; token < tokens; token += token_stride) {
    const uint4* kv_row = kv + token * (num_heads * kKVUint4PerHead);
    uint2* key_row = key + token * (num_heads * kKeyUint2PerHead);
    uint2* value_row = value + token * (num_heads * kValUint2PerHead);

    uint2 rope = make_uint2(0, 0);
    if (lane < 8) rope = cvt_octet_satfinite(pe[token * kRopeUint4PerRow + lane]);

#pragma unroll
    for (int head = head_select; head < num_heads; head += 2) {
      const uint4* src = kv_row + head * kKVUint4PerHead + sub * 2;
      const uint2 lo = cvt_octet_satfinite(src[0]);
      const uint2 hi = cvt_octet_satfinite(src[1]);
      const uint4 packed = make_uint4(lo.x, lo.y, hi.x, hi.y);
      if (is_value) {
        *reinterpret_cast<uint4*>(value_row + head * kValUint2PerHead + piece * 2) = packed;
      } else {
        *reinterpret_cast<uint4*>(key_row + head * kKeyUint2PerHead + piece * 2) = packed;
      }
    }

    uint2 rope_piece_value;
    rope_piece_value.x = __shfl_sync(0xffffffffu, rope.x, rope_piece);
    rope_piece_value.y = __shfl_sync(0xffffffffu, rope.y, rope_piece);
#pragma unroll
    for (int head = rope_head_group; head < num_heads; head += 4) {
      key_row[head * kKeyUint2PerHead + (kNopeDim / 8) + rope_piece] = rope_piece_value;
    }
  }
}

}  // namespace mla_kv_pack

/*!
 * \brief Launch the fused bf16 -> saturating-e4m3 MLA context K/V pack.
 * \param kv_nope [tokens, num_heads, 256] bf16, contiguous ([k_nope | v] per head)
 * \param k_pe    [tokens, 64] bf16, contiguous (shared across heads)
 * \param key     [tokens, num_heads, 192] fp8 e4m3, contiguous (out)
 * \param value   [tokens, num_heads, 128] fp8 e4m3, contiguous (out)
 */
inline cudaError_t MLAKVPackFP8(const void* kv_nope, const void* k_pe, void* key, void* value,
                                int64_t tokens, int num_heads, cudaStream_t stream) {
  using namespace mla_kv_pack;
  if (tokens <= 0) return cudaSuccess;
  if (num_heads <= 0) return cudaErrorInvalidValue;
  const int64_t blocks64 = (tokens + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const unsigned blocks = static_cast<unsigned>(blocks64 > 0x7fffffffLL ? 0x7fffffffLL : blocks64);
  const uint4* kv = static_cast<const uint4*>(kv_nope);
  const uint4* pe = static_cast<const uint4*>(k_pe);
  uint2* k = static_cast<uint2*>(key);
  uint2* v = static_cast<uint2*>(value);
  if (num_heads == 12) {
    MLAKVPackFP8Kernel<12><<<blocks, kThreads, 0, stream>>>(kv, pe, k, v, tokens, num_heads);
  } else {
    MLAKVPackFP8Kernel<0><<<blocks, kThreads, 0, stream>>>(kv, pe, k, v, tokens, num_heads);
  }
  return cudaGetLastError();
}

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_KV_PACK_FP8_CUH_
