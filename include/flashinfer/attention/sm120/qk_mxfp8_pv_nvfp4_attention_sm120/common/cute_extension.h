/*
 * Copyright (c) 2025 by SageAttention team.
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
#pragma once

#include "cute/arch/mma_sm120.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/atom/mma_traits_sm120.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"

namespace cute::SM120::BLOCKSCALED {

using cutlass::float_e2m1_t;
using cutlass::float_ue4m3_t;

// MMA.SF 16x32x64 TN E2M1 x E2M1 with SF E4M3
struct SM120_16x32x64_TN_VS_NVFP4 {
  using DRegisters = float[16];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[8];
  using CRegisters = float[16];

  static constexpr int SFBits = 32;
  using RegTypeSF = cute::uint_bit_t<SFBits>;

  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];

  CUTE_HOST_DEVICE static void fma(
      float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7,
      float& d8, float& d9, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& b4, uint32_t const& b5, uint32_t const& b6, uint32_t const& b7,
      float const& c0, float const& c1, float const& c2, float const& c3, float const& c4,
      float const& c5, float const& c6, float const& c7, float const& c8, float const& c9,
      float const& c10, float const& c11, float const& c12, float const& c13, float const& c14,
      float const& c15, RegTypeSF const& sfa0, RegTypeSF const& sfb0) {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t bidB = 0;
    static constexpr uint16_t tidB0 = 0;
    static constexpr uint16_t tidB1 = 1;
    static constexpr uint16_t tidB2 = 2;
    static constexpr uint16_t tidB3 = 3;

    // ---- each pair of independent mma.sync instructions.
    // ---- The 4 MMAs have NO register dependency on each other (different d/c regs).
    // ---- If total time doesn't increase, MUFU executes for free during MMA pipeline
    // ---- latency, confirming instruction-level overlap is possible on SM120a.

#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    // MMA 1/4: d0,d1,d8,d9
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d8), "=f"(d9)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c8), "f"(c9),
          "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB0));

    // MMA 2/4: d2,d3,d10,d11
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d2), "=f"(d3), "=f"(d10), "=f"(d11)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "f"(c2), "f"(c3), "f"(c10),
          "f"(c11), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(tidB1));

    // MMA 3/4: d4,d5,d12,d13
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d4), "=f"(d5), "=f"(d12), "=f"(d13)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5), "f"(c4), "f"(c5), "f"(c12),
          "f"(c13), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(tidB2));

    // MMA 4/4: d6,d7,d14,d15
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d6), "=f"(d7), "=f"(d14), "=f"(d15)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7), "f"(c6), "f"(c7), "f"(c14),
          "f"(c15), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(tidB3));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM120::BLOCKSCALED::SM120_16x8x64_TN_VS without "
        "CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED");
#endif
  }
};

// MMA.SF 16x64x64 TN E2M1 x E2M1 with SF E4M3
// Wider N: 8 PTX mma.sync instructions (tidB0-7) instead of 4.
// Doubles TC work per atom call, fills pipeline better during softmax overlap.
struct SM120_16x64x64_TN_VS_NVFP4 {
  using DRegisters = float[32];     // 2× (16×64 output vs 16×32)
  using ARegisters = uint32_t[4];   // same (M×K unchanged)
  using BRegisters = uint32_t[16];  // 2× (64N needs 2× B data)
  using CRegisters = float[32];     // 2× accumulator

  static constexpr int SFBits = 32;
  using RegTypeSF = cute::uint_bit_t<SFBits>;
  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];  // Same as 16x32: SFB is K-dependent, not N-dependent

  CUTE_HOST_DEVICE static void fma(
      float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7,
      float& d8, float& d9, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
      float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22,
      float& d23, float& d24, float& d25, float& d26, float& d27, float& d28, float& d29,
      float& d30, float& d31, uint32_t const& a0, uint32_t const& a1, uint32_t const& a2,
      uint32_t const& a3, uint32_t const& b0, uint32_t const& b1, uint32_t const& b2,
      uint32_t const& b3, uint32_t const& b4, uint32_t const& b5, uint32_t const& b6,
      uint32_t const& b7, uint32_t const& b8, uint32_t const& b9, uint32_t const& b10,
      uint32_t const& b11, uint32_t const& b12, uint32_t const& b13, uint32_t const& b14,
      uint32_t const& b15, float const& c0, float const& c1, float const& c2, float const& c3,
      float const& c4, float const& c5, float const& c6, float const& c7, float const& c8,
      float const& c9, float const& c10, float const& c11, float const& c12, float const& c13,
      float const& c14, float const& c15, float const& c16, float const& c17, float const& c18,
      float const& c19, float const& c20, float const& c21, float const& c22, float const& c23,
      float const& c24, float const& c25, float const& c26, float const& c27, float const& c28,
      float const& c29, float const& c30, float const& c31, RegTypeSF const& sfa0,
      RegTypeSF const& sfb0) {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t bidB = 0;

#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    // 8 PTX mma.sync instructions covering N=0..63 (8 × n8 = 64N)

    // MMA 1/8: tidB=0, N=[0,8)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d16), "=f"(d17)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c16),
          "f"(c17), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(0)));

    // MMA 2/8: tidB=1, N=[8,16)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d2), "=f"(d3), "=f"(d18), "=f"(d19)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "f"(c2), "f"(c3), "f"(c18),
          "f"(c19), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(1)));

    // MMA 3/8: tidB=2, N=[16,24)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d4), "=f"(d5), "=f"(d20), "=f"(d21)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5), "f"(c4), "f"(c5), "f"(c20),
          "f"(c21), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(2)));

    // MMA 4/8: tidB=3, N=[24,32)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d6), "=f"(d7), "=f"(d22), "=f"(d23)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7), "f"(c6), "f"(c7), "f"(c22),
          "f"(c23), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(3)));

    // MMA 5/8: tidB=4, N=[32,40) — second SF block
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d8), "=f"(d9), "=f"(d24), "=f"(d25)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b8), "r"(b9), "f"(c8), "f"(c9), "f"(c24),
          "f"(c25), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(0)));

    // MMA 6/8: tidB=5, N=[40,48)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d10), "=f"(d11), "=f"(d26), "=f"(d27)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b10), "r"(b11), "f"(c10), "f"(c11), "f"(c26),
          "f"(c27), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(1)));

    // MMA 7/8: tidB=6, N=[48,56)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d12), "=f"(d13), "=f"(d28), "=f"(d29)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b12), "r"(b13), "f"(c12), "f"(c13), "f"(c28),
          "f"(c29), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(2)));

    // MMA 8/8: tidB=7, N=[56,64)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1."
        "f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d14), "=f"(d15), "=f"(d30), "=f"(d31)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b14), "r"(b15), "f"(c14), "f"(c15), "f"(c30),
          "f"(c31), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(uint16_t(3)));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM120::BLOCKSCALED::SM120_16x64x64_TN_VS without "
        "CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED");
#endif
  }
};

// MMA.SF 16x32x32 TN E4M3 x E4M3 with SF UE8M0
// Composite atom: 4 PTX mma.sync instructions (tidB0-3) covering N=32.
struct SM120_16x32x32_TN_VS_FP8 {
  using DRegisters = float[16];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[8];
  using CRegisters = float[16];

  static constexpr int SFBits = 8;
  using RegTypeSF = uint8_t;

  using SFARegisters = RegTypeSF[1];
  using SFBRegisters = RegTypeSF[1];

  CUTE_HOST_DEVICE static void fma(
      float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7,
      float& d8, float& d9, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
      uint32_t const& b4, uint32_t const& b5, uint32_t const& b6, uint32_t const& b7,
      float const& c0, float const& c1, float const& c2, float const& c3, float const& c4,
      float const& c5, float const& c6, float const& c7, float const& c8, float const& c9,
      float const& c10, float const& c11, float const& c12, float const& c13, float const& c14,
      float const& c15, RegTypeSF const& sfa0, RegTypeSF const& sfb0) {
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t bidB = 0;
    static constexpr uint16_t tidB0 = 0;
    static constexpr uint16_t tidB1 = 1;
    static constexpr uint16_t tidB2 = 2;
    static constexpr uint16_t tidB3 = 3;

    // ---- each pair of independent mma.sync instructions.

#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    // MMA 1/4: d0,d1,d8,d9
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
        "f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d8), "=f"(d9)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c8), "f"(c9),
          "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB0));

    // MMA 2/4: d2,d3,d10,d11
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
        "f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d2), "=f"(d3), "=f"(d10), "=f"(d11)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "f"(c2), "f"(c3), "f"(c10),
          "f"(c11), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(tidB1));

    // MMA 3/4: d4,d5,d12,d13
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
        "f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d4), "=f"(d5), "=f"(d12), "=f"(d13)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5), "f"(c4), "f"(c5), "f"(c12),
          "f"(c13), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(tidB2));

    // MMA 4/4: d6,d7,d14,d15
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
        "f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d6), "=f"(d7), "=f"(d14), "=f"(d15)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7), "f"(c6), "f"(c7), "f"(c14),
          "f"(c15), "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB),
          "h"(tidB3));
#else
    CUTE_INVALID_CONTROL_PATH(
        "Attempting to use SM120::BLOCKSCALED::SM120_16x32x32_TN_VS_FP8 without "
        "CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
  }
};

// Variant of the composite FP8 atom that exposes the three issue gaps between
// its four independent m16n8k32 instructions.  The normal CuTe atom emits the
// same instructions back-to-back, which leaves no source-level place to
// schedule scalar softmax work on the otherwise independent score slot.
template <typename GapFn>
CUTE_HOST_DEVICE static void fma_fp8_with_interleave(
    float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7,
    float& d8, float& d9, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
    uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
    uint32_t const& b0, uint32_t const& b1, uint32_t const& b2, uint32_t const& b3,
    uint32_t const& b4, uint32_t const& b5, uint32_t const& b6, uint32_t const& b7, float const& c0,
    float const& c1, float const& c2, float const& c3, float const& c4, float const& c5,
    float const& c6, float const& c7, float const& c8, float const& c9, float const& c10,
    float const& c11, float const& c12, float const& c13, float const& c14, float const& c15,
    uint8_t const& sfa0, uint8_t const& sfb0, GapFn&& gap_fn) {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
  static constexpr uint16_t tidA = 0;
  static constexpr uint16_t bidA = 0;
  static constexpr uint16_t bidB = 0;
  static constexpr uint16_t tidB0 = 0;
  static constexpr uint16_t tidB1 = 1;
  static constexpr uint16_t tidB2 = 2;
  static constexpr uint16_t tidB3 = 3;

  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
      "f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d8), "=f"(d9)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c8), "f"(c9),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB0));

  gap_fn(cute::Int<0>{});

  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
      "f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(d2), "=f"(d3), "=f"(d10), "=f"(d11)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b2), "r"(b3), "f"(c2), "f"(c3), "f"(c10), "f"(c11),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB1));

  gap_fn(cute::Int<1>{});

  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
      "f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(d4), "=f"(d5), "=f"(d12), "=f"(d13)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b4), "r"(b5), "f"(c4), "f"(c5), "f"(c12), "f"(c13),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB2));

  gap_fn(cute::Int<2>{});

  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3."
      "f32.ue8m0 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
      : "=f"(d6), "=f"(d7), "=f"(d14), "=f"(d15)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b6), "r"(b7), "f"(c6), "f"(c7), "f"(c14), "f"(c15),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB3));
#else
  CUTE_INVALID_CONTROL_PATH("fma_fp8_with_interleave requires CUTE_ARCH_MXF8F6F4_MMA_ENABLED");
#endif
}

}  // namespace cute::SM120::BLOCKSCALED

namespace cute {

// MMA FP8 Block-Scaled 16x32x32 TN
//
// Composite of 4 PTX mxf8f6f4 m16n8k32 instructions covering N=32.
// A/B/C layouts derived from SM80_16x8x32_S32S8S8S32_TN (the base for all
// SM120 16x8x32 variants), expanded to N=32.
// SFA/SFB layouts from CUTLASS mma_traits_sm120.hpp:194-198.
//
// Base SM80_16x8x32 block-scaled layouts (from mma_traits_sm80.hpp):
//   ALayout: (T32, V16) -> (M16, K32)
//     Shape<Shape<_4,_8>, Shape<_4,_2,_2>>, Stride<Stride<_64,_1>, Stride<_16,_8,_256>>
//   BLayout: (T32, V8) -> (N8, K32)
//     Shape<Shape<_4,_8>, Shape<_4,_2>>, Stride<Stride<_32,_1>, Stride<_8,_128>>
//   CLayout: SM80_16x8_Row (T32, V4) -> (M16, N8)
//
// Composite N=32 expansion (4 tidB groups, same as NVFP4 pattern):
//   BLayout: add _4 in value dim, strides scaled for 4 tidB groups
//   CLayout: same as NVFP4 composite (M16×N32, same output tile)
template <>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x32x32_TN_VS_FP8> {
  using ValTypeA = uint8_t;
  using ValTypeB = uint8_t;
  using ValTypeD = float;
  using ValTypeC = float;

  using ValTypeSF = cutlass::float_ue8m0_t;
  constexpr static int SFVecSize = 32;

  using Shape_MNK = Shape<_16, _32, _32>;
  using ThrID = Layout<_32>;

  // ALayout: (T32, V16) -> (M16, K32)
  // Same as base SM80_16x8x32 (A shared across all 4 tidB instructions)
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>,
                         Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;

  // BLayout: (T32, V32) -> (N32, K32)
  // Base: (T32, V8) Shape<Shape<_4,_8>, Shape<_4,_2>> Stride<Stride<_32,_1>, Stride<_8,_128>>
  // Composite 4-wide N: all base strides ×4 (N: 8→32), new tidB mode stride _8.
  // Col-major B tensor (K=32, N=32): flat = k*32 + n.  tidB shifts n by 8 → stride _8.
  // Matches NVFP4 pattern: v0 stride = base_v0×4, v1 stride = base_v1×4, v2(tidB) = _8.
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _4>>,
                         Stride<Stride<_128, _1>, Stride<_32, _512, _8>>>;

  // SFALayout: (T32, V32) -> (M16, K32) — from CUTLASS base block-scaled
  using SFALayout = Layout<Shape<Shape<_2, _2, _8>, _32>, Stride<Stride<_8, _0, _1>, _16>>;

  // SFBLayout: N-expanded for smem copy compatibility (same pattern as NVFP4 composite).
  // All 4 tidB groups share one sfb0 register in fma(), but the layout must be
  // non-zero strided for smem copy to work.  mma_unpack extracts element 0.
  using SFBLayout = Layout<Shape<Shape<_4, _8>, _32>, Stride<Stride<_8, _1>, _32>>;

  // CLayout: (T32, V16) -> (M16, N32)
  // Same as NVFP4 composite (16 float outputs, 4 per tidB × 4 tidB groups)
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<Shape<_2, _4>, _2>>,
                         Stride<Stride<_32, _1>, Stride<Stride<_16, _128>, _8>>>;
};

// CuTe's generic SM120 block-scaled unpacker assumes a single N8 MMA atom.
// The FP8 atom above is a source-level composite of four N8 instructions, so
// its expanded B/C layouts intentionally contain more values than the base
// atom assertions allow. Keep the specialization local to this operation and
// extract the one UE8M0 scale register consumed by all four instructions.
namespace SM120::BLOCKSCALED {

template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC,
          class CLayout>
CUTE_HOST_DEVICE constexpr void mma_unpack(MMA_Traits<SM120_16x32x32_TN_VS_FP8> const&,
                                           Tensor<TD, DLayout>& D,
                                           Tensor<TA, ALayout> const& A_zipped,
                                           Tensor<TB, BLayout> const& B_zipped,
                                           Tensor<TC, CLayout> const& C) {
  using MMAOp = SM120_16x32x32_TN_VS_FP8;
  using RegTypeD = typename remove_extent<typename MMAOp::DRegisters>::type;
  using RegTypeA = typename remove_extent<typename MMAOp::ARegisters>::type;
  using RegTypeB = typename remove_extent<typename MMAOp::BRegisters>::type;
  using RegTypeC = typename remove_extent<typename MMAOp::CRegisters>::type;
  using RegTypeSFA = typename remove_extent<typename MMAOp::SFARegisters>::type;
  using RegTypeSFB = typename remove_extent<typename MMAOp::SFBRegisters>::type;

  constexpr int RegNumD = extent<typename MMAOp::DRegisters>::value;
  constexpr int RegNumA = extent<typename MMAOp::ARegisters>::value;
  constexpr int RegNumB = extent<typename MMAOp::BRegisters>::value;
  constexpr int RegNumC = extent<typename MMAOp::CRegisters>::value;
  constexpr int RegNumSFA = extent<typename MMAOp::SFARegisters>::value;
  constexpr int RegNumSFB = extent<typename MMAOp::SFBRegisters>::value;

  auto [A, SFA] = unzip_tensor(A_zipped);
  auto [B, SFB] = unzip_tensor(B_zipped);
  Tensor rA = recast<RegTypeA>(A);
  Tensor rB = recast<RegTypeB>(B);
  Tensor rD = recast<RegTypeD>(D);
  Tensor rC = recast<RegTypeC>(C);

  auto filtered_sfa = recast<RegTypeSFA>(filter_zeros(SFA));
  auto filtered_sfb = recast<RegTypeSFB>(filter_zeros(SFB));
  RegTypeSFA sfa = filtered_sfa(0);
  RegTypeSFB sfb = filtered_sfb(0);
  Tensor rSFA = make_tensor(make_rmem_ptr<RegTypeSFA>(&sfa), Int<RegNumSFA>{});
  Tensor rSFB = make_tensor(make_rmem_ptr<RegTypeSFB>(&sfb), Int<RegNumSFB>{});

  detail::explode(MMAOp::fma, rD, make_int_sequence<RegNumD>{}, rA, make_int_sequence<RegNumA>{},
                  rB, make_int_sequence<RegNumB>{}, rC, make_int_sequence<RegNumC>{}, rSFA,
                  make_int_sequence<RegNumSFA>{}, rSFB, make_int_sequence<RegNumSFB>{});
}

}  // namespace SM120::BLOCKSCALED

// MMA NVFP4 16x32x64 TN
template <>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4> {
  // The MMA accepts 4-bit inputs regardless of the types for A and B
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;

  using ValTypeD = float;
  using ValTypeC = float;

  using ValTypeSF = cutlass::float_ue4m3_t;
  constexpr static int SFVecSize = 16;

  using Shape_MNK = Shape<_16, _32, _64>;
  using ThrID = Layout<_32>;

  // (T32,V32) -> (M16,K64)
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _2, _2>>,
                         Stride<Stride<_128, _1>, Stride<_16, _8, _512>>>;
  // (T32,V64) -> (N32,K64)
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _2, _4>>,
                         Stride<Stride<_256, _1>, Stride<_32, _1024, _8>>>;
  // (T32,V64) -> (M16,K64)
  using SFALayout = Layout<Shape<Shape<_2, _2, _8>, _64>, Stride<Stride<_8, _0, _1>, _16>>;
  // (T32,V64) -> (N32,K64)
  using SFBLayout = Layout<Shape<Shape<_4, _8>, _64>, Stride<Stride<_8, _1>, _32>>;
  // (T32,V16) -> (M16,N32)
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<Shape<_2, _4>, _2>>,
                         Stride<Stride<_32, _1>, Stride<Stride<_16, _128>, _8>>>;
};

// MMA NVFP4 16x64x64 TN (wider N variant, 8 PTX instructions)
// Derived from 16x32x64 by doubling N: _4→_8 in value shapes, strides unchanged.
template <>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x64x64_TN_VS_NVFP4> {
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;
  using ValTypeD = float;
  using ValTypeC = float;
  using ValTypeSF = cutlass::float_ue4m3_t;
  constexpr static int SFVecSize = 16;

  using Shape_MNK = Shape<_16, _64, _64>;
  using ThrID = Layout<_32>;

  // A layout: same as 16x32x64 (M×K unchanged)
  // (T32,V32) -> (M16,K64)
  using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _2, _2>>,
                         Stride<Stride<_128, _1>, Stride<_16, _8, _512>>>;
  // B layout: 8 tidB groups (was 4) → (T32,V128) -> (N64,K64)
  // Pattern: base BLayout × 8 N-copies. Stride _8 per tidB in N dimension.
  using BLayout = Layout<Shape<Shape<_4, _8>, Shape<_8, _2, _8>>,
                         Stride<Stride<_512, _1>, Stride<_64, _2048, _8>>>;
  // SFA layout: same as 16x32x64 (M×K only, no N dependency)
  // (T32,V64) -> (M16,K64)
  using SFALayout = Layout<Shape<Shape<_2, _2, _8>, _64>, Stride<Stride<_8, _0, _1>, _16>>;
  // SFB layout: V64 (= K dimension, NOT N). N covered by thread stride.
  // (T32,V64) -> (N64,K64). Thread stride _0,_1 means 4:0 mode (8 active threads).
  // Same value dimension as 16x32 (SFB logical size = K = 64).
  using SFBLayout = Layout<Shape<Shape<_4, _8>, _64>, Stride<Stride<_0, _1>, _8>>;
  // C layout: 8 tidB groups (was 4) → (T32,V32) -> (M16,N64)
  // _4→_8 in value inner shape. Thread stride unchanged (_32).
  // N-major linearization: index = n*16 + m. Stride _128 = 8N_per_tidB × 16M.
  using CLayout = Layout<Shape<Shape<_4, _8>, Shape<Shape<_2, _8>, _2>>,
                         Stride<Stride<_32, _1>, Stride<Stride<_16, _128>, _8>>>;
};

template <class SFATensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_SFA(SFATensor&& sfatensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfatensor) >= Int<2>{});

  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFA_TV = typename Atom::Traits::SFALayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // Reorder the tensor for the TiledAtom
  auto t_tile = make_tile(get<0>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfatensor, t_tile);  // (PermM,PermK)

  // Tile the tensor for the Atom
  auto a_tile =
      make_tile(make_layout(size<0>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomM,AtomK),(RestM,RestK))

  // Transform the Atom mode from (M,K) to (Thr,Val)
  auto tv_tensor = a_tensor.compose(AtomLayoutSFA_TV{}, _);  // ((ThrV,FrgV),(RestM,RestK))

  // Tile the tensor for the Thread
  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<1>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor =
      zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))

  return thr_tensor;
}

template <class SFBTensor, class Atom, class TiledThr, class TiledPerm>
CUTE_HOST_DEVICE constexpr auto thrfrg_SFB(SFBTensor&& sfbtensor,
                                           TiledMMA<Atom, TiledThr, TiledPerm>& mma) {
  CUTE_STATIC_ASSERT_V(rank(sfbtensor) >= Int<2>{});

  using AtomShape_MNK = typename Atom::Shape_MNK;
  using AtomLayoutSFB_TV = typename Atom::Traits::SFBLayout;

  auto permutation_mnk = TiledPerm{};
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // Reorder the tensor for the TiledAtom
  auto t_tile = make_tile(get<1>(permutation_mnk), get<2>(permutation_mnk));
  auto t_tensor = logical_divide(sfbtensor, t_tile);  // (PermN,PermK)

  // Tile the tensor for the Atom
  auto a_tile =
      make_tile(make_layout(size<1>(AtomShape_MNK{})), make_layout(size<2>(AtomShape_MNK{})));
  auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomN,AtomK),(RestN,RestK))

  // Transform the Atom mode from (M,K) to (Thr,Val)
  auto tv_tensor = a_tensor.compose(AtomLayoutSFB_TV{}, _);  // ((ThrV,FrgV),(RestN,RestK))

  // Tile the tensor for the Thread
  auto thr_tile = make_tile(
      _, make_tile(make_layout(size<2>(thr_layout_vmnk)), make_layout(size<3>(thr_layout_vmnk))));
  auto thr_tensor =
      zipped_divide(tv_tensor, thr_tile);  // ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK)))
  return thr_tensor;
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFATensor&&>(sfatensor).data(),
                                thrfrg_SFA(sfatensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vmk = make_coord(get<0>(thr_vmnk), make_coord(get<1>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vmk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

template <class SFATensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_SFA(SFATensor&& sfatensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFA(sfatensor, thread_mma));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  auto thr_tensor = make_tensor(static_cast<SFBTensor&&>(sfbtensor).data(),
                                thrfrg_SFB(sfbtensor.layout(), thread_mma));
  auto thr_vmnk = thread_mma.thr_vmnk_;
  auto thr_vnk = make_coord(get<0>(thr_vmnk), make_coord(get<2>(thr_vmnk), get<3>(thr_vmnk)));
  return thr_tensor(thr_vnk, make_coord(_, repeat<rank<1, 1>(thr_tensor)>(_)));
}

template <class SFBTensor, class ThrMma>
CUTE_HOST_DEVICE constexpr auto partition_fragment_SFB(SFBTensor&& sfbtensor, ThrMma& thread_mma) {
  using ValTypeSF = typename ThrMma::Atom::Traits::ValTypeSF;
  return make_fragment_like<ValTypeSF>(partition_SFB(sfbtensor, thread_mma));
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto get_layoutSFA_TV(TiledMma& mma) {
  // (M,K) -> (M,K)
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_A = make_layout(make_shape(size<0>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
  auto atile = make_tile(
      _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                               make_stride(Int<1>{}, Int<0>{})),
                   _));

  // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  // (thr_idx,val) -> (M,K)
  return thrfrg_SFA(ref_A, mma).compose(atile, _).compose(thridx_2_thrid, _);
}

template <class TiledMma>
CUTE_HOST_DEVICE constexpr auto get_layoutSFB_TV(TiledMma& mma) {
  // (N,K) -> (N,K)
  auto tile_shape_mnk = tile_shape(mma);
  auto ref_B = make_layout(make_shape(size<1>(tile_shape_mnk), size<2>(tile_shape_mnk)));
  auto thr_layout_vmnk = mma.get_thr_layout_vmnk();

  // (ThrV,(ThrM,ThrK)) -> (ThrV,(ThrM,ThrN,ThrK))
  auto btile = make_tile(
      _, make_tile(make_layout(make_shape(size<1>(thr_layout_vmnk), size<2>(thr_layout_vmnk)),
                               make_stride(Int<0>{}, Int<1>{})),
                   _));

  // thr_idx -> (ThrV,ThrM,ThrN,ThrK)
  auto thridx_2_thrid = right_inverse(thr_layout_vmnk);
  // (thr_idx,val) -> (M,K)
  return thrfrg_SFB(ref_B, mma).compose(btile, _).compose(thridx_2_thrid, _);
}

}  // namespace cute
