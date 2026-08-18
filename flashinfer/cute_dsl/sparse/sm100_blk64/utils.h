/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 ******************************************************************************/
// Self-contained SM100 PTX utility helpers for fused attention kernel.
// All functions inlined — no external dependencies beyond CUTLASS/CuTe headers.
#pragma once

#include <cutlass/arch/barrier.h>

#include <cute/arch/copy_sm100.hpp>
#include <cute/tensor.hpp>

// ============================================================================
// MMA trait structs (must be in cute namespace for MMA_Traits specialization)
// ============================================================================

namespace cute {

// TS (TMEM A, SMEM desc B) NOELECT MMA
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_TS_NOELECT {
  static_assert(M == 32 || M == 64 || M == 128, "WS TS MMA requires M in {32,64,128}");
  static_assert(N == 64 || N == 128 || N == 256, "WS TS MMA requires N in {64,128,256}");
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];
  CUTE_HOST_DEVICE static void fma(uint32_t const& tmem_a, uint64_t const& desc_b,
                                   uint32_t const& tmem_c, uint32_t const& scale_c,
                                   uint64_t const& idesc_e) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(uint32_t(idesc_e >> 32)), "r"(scale_c));
  }
};

// SS (SMEM desc A, SMEM desc B) NOELECT MMA
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg = UMMA::ScaleIn::One,
          UMMA::ScaleIn b_neg = UMMA::ScaleIn::One>
struct SM100_MMA_F16BF16_WS_SS_NOELECT {
  static_assert(M == 32 || M == 64 || M == 128, "WS SS MMA requires M in {32,64,128}");
  static_assert(N == 64 || N == 128 || N == 256, "WS SS MMA requires N in {64,128,256}");
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = uint32_t[1];
  CUTE_HOST_DEVICE static void fma(uint64_t const& desc_a, uint64_t const& desc_b,
                                   uint32_t const& tmem_c, uint32_t const& scale_c,
                                   uint64_t const& idesc_e) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(uint32_t(idesc_e >> 32)), "r"(scale_c));
  }
};

// MMA_Traits for TS NOELECT
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<
    SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == 16 && cute::sizeof_bits_v<b_type> == 16);
  using FrgTypeA = UMMA::tmem_frg_1sm<a_type, a_type, UMMA::TmemAllocMode::NonInterleaved>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;
  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();
  template <class TD, class DLayout, class TA, class ALayoutX, class TB, class BLayoutX, class TC,
            class CLayoutX>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits,
                                                    Tensor<TD, DLayout>& d,
                                                    Tensor<TA, ALayoutX> const& a,
                                                    Tensor<TB, BLayoutX> const& b,
                                                    Tensor<TC, CLayoutX> const& c) {
    static_assert(is_tmem<TD>::value && is_tmem<TA>::value && is_rmem<TB>::value &&
                  is_tmem<TC>::value);
    SM100_MMA_F16BF16_WS_TS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
                                    b_neg>::fma(raw_pointer_cast(a.data()), b[0],
                                                raw_pointer_cast(d.data()),
                                                uint32_t(traits.accumulate_),
                                                UMMA::make_runtime_instr_desc<>(traits.idesc_));
  }
};

// MMA_Traits for SS NOELECT
template <class a_type, class b_type, class c_type, int M, int N, UMMA::Major a_major,
          UMMA::Major b_major, UMMA::ScaleIn a_neg, UMMA::ScaleIn b_neg>
struct MMA_Traits<
    SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>> {
  using ValTypeD = c_type;
  using ValTypeA = a_type;
  using ValTypeB = b_type;
  using ValTypeC = c_type;
  static_assert(cute::sizeof_bits_v<a_type> == 16 && cute::sizeof_bits_v<b_type> == 16);
  using FrgTypeA = UMMA::smem_desc<a_major>;
  using FrgTypeB = UMMA::smem_desc<b_major>;
  using FrgTypeC = UMMA::tmem_frg_ws_1sm<c_type>;
  static constexpr int K = 256 / cute::sizeof_bits<ValTypeA>::value;
  using Shape_MNK = Shape<Int<M>, Int<N>, Int<K>>;
  using ThrID = Layout<_1>;
  using ALayout = Layout<Shape<_1, Shape<Int<M>, Int<K>>>, Stride<_0, Stride<_1, Int<M>>>>;
  using BLayout = Layout<Shape<_1, Shape<Int<N>, Int<K>>>, Stride<_0, Stride<_1, Int<N>>>>;
  using CLayout = Layout<Shape<_1, Shape<Int<M>, Int<N>>>, Stride<_0, Stride<_1, Int<M>>>>;
  UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::One;
  UMMA::InstrDescriptor idesc_ =
      UMMA::make_instr_desc<a_type, b_type, c_type, M, N, a_major, b_major, a_neg, b_neg>();
  template <class TD, class DLayout, class TA, class ALayoutX, class TB, class BLayoutX, class TC,
            class CLayoutX>
  CUTE_HOST_DEVICE constexpr friend void mma_unpack(MMA_Traits const& traits,
                                                    Tensor<TD, DLayout>& d,
                                                    Tensor<TA, ALayoutX> const& a,
                                                    Tensor<TB, BLayoutX> const& b,
                                                    Tensor<TC, CLayoutX> const& c) {
    static_assert(is_tmem<TD>::value && is_rmem<TA>::value && is_rmem<TB>::value &&
                  is_tmem<TC>::value);
    SM100_MMA_F16BF16_WS_SS_NOELECT<a_type, b_type, c_type, M, N, a_major, b_major, a_neg,
                                    b_neg>::fma(a[0], b[0], raw_pointer_cast(d.data()),
                                                uint32_t(traits.accumulate_),
                                                UMMA::make_runtime_instr_desc<>(traits.idesc_));
  }
};

}  // namespace cute

// ============================================================================
// flash namespace: PTX helpers and MMA wrappers
// ============================================================================

namespace flash {

// ---- TMEM load ----

template <int kNumElements>
__device__ __forceinline__ void tmem_load(uint32_t tmem_start, float* data_) {
  uint32_t* data = static_cast<uint32_t*>(static_cast<void*>(data_));
  static_assert(kNumElements == 1 || kNumElements == 2 || kNumElements == 4 || kNumElements == 8 ||
                kNumElements == 16 || kNumElements == 32 || kNumElements == 64 ||
                kNumElements == 128);
#ifndef __VSCODE_IDE__
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    if constexpr (kNumElements == 1) {
      cute::SM100_TMEM_LOAD_32dp32b1x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 2) {
      cute::SM100_TMEM_LOAD_32dp32b2x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 4) {
      cute::SM100_TMEM_LOAD_32dp32b4x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 8) {
      cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 16) {
      cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 32) {
      cute::SM100_TMEM_LOAD_32dp32b32x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 64) {
      cute::SM100_TMEM_LOAD_32dp32b64x::copy(tmem_start, data[Is]...);
    } else if constexpr (kNumElements == 128) {
      cute::SM100_TMEM_LOAD_32dp32b128x::copy(tmem_start, data[Is]...);
    }
  }(cute::make_index_sequence<kNumElements>{});
#endif
}

// ---- Fence/commit ----

__device__ __forceinline__ void tcgen05_commit() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tcgen05_fence_before_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

// ---- SMEM load/store ----

__device__ __forceinline__ void smem_store_float4(float* addr, float4 val) {
  asm volatile("st.shared.b128 [%0], %1;"
               :
               : "l"(__cvta_generic_to_shared(addr)), "q"(*reinterpret_cast<__int128_t*>(&val)));
}

__device__ __forceinline__ float4 smem_load_float4(const float* addr) {
  __int128_t temp;
  asm volatile("ld.shared.b128 %0, [%1];" : "=q"(temp) : "l"(__cvta_generic_to_shared(addr)));
  return *reinterpret_cast<float4*>(&temp);
}

// ---- Arithmetic ----

__device__ __forceinline__ float2 float2_add(float2 const& a, float2 const& b) {
  float2 c;
  asm volatile("add.f32x2 %0, %1, %2;\n"
               : "=l"(reinterpret_cast<uint64_t&>(c))
               : "l"(reinterpret_cast<uint64_t const&>(a)),
                 "l"(reinterpret_cast<uint64_t const&>(b)));
  return c;
}

// ---- UMMA arrive (no elect) ----

__device__ __forceinline__ void umma_arrive(cute::uint64_t& bar) {
  uint32_t bar_addr = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
               :
               : "r"(bar_addr));
}

// ---- MMA helpers ----

template <typename TiledMma, typename TensorA, typename TensorB, typename TensorFragC>
__device__ __forceinline__ void utcmma_ss(TiledMma& tiled_mma, TensorA sA, TensorB sB,
                                          TensorFragC tC_frag, bool clear_accum) {
  using namespace cute;
  tiled_mma.accumulate_ = clear_accum ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
  auto thr_mma = tiled_mma.get_slice(_0{});
  auto sA_frag = thr_mma.partition_fragment_A(sA);
  auto sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(sA_frag) == size<2>(sB_frag));
  constexpr int kARank = decltype(rank(sA_frag))::value;
  constexpr int kBRank = decltype(rank(sB_frag))::value;
  constexpr int kAExt3 = []() constexpr {
    if constexpr (kARank > 3) {
      return decltype(size<3>(sA_frag))::value;
    } else {
      return 1;
    }
  }();
  constexpr int kAExt4 = []() constexpr {
    if constexpr (kARank > 4) {
      return decltype(size<4>(sA_frag))::value;
    } else {
      return 1;
    }
  }();
  constexpr int kBExt3 = []() constexpr {
    if constexpr (kBRank > 3) {
      return decltype(size<3>(sB_frag))::value;
    } else {
      return 1;
    }
  }();

  CUTE_UNROLL
  for (int k = 0; k < size<2>(sA_frag); ++k) {
    if constexpr (kARank == 5 && kBRank == 4) {
      if constexpr (kAExt4 == 1 && kAExt3 == kBExt3) {
        CUTE_UNROLL
        for (int dual_k = 0; dual_k < kAExt3; ++dual_k) {
          cute::gemm(tiled_mma, sA_frag(_, _, k, dual_k, Int<0>{}), sB_frag(_, _, k, dual_k),
                     tC_frag);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
      } else if constexpr (kAExt3 == 1 && kAExt4 == kBExt3) {
        CUTE_UNROLL
        for (int dual_k = 0; dual_k < kAExt4; ++dual_k) {
          cute::gemm(tiled_mma, sA_frag(_, _, k, Int<0>{}, dual_k), sB_frag(_, _, k, dual_k),
                     tC_frag);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
      }
    } else if constexpr (kARank == 5) {
      if constexpr (kAExt4 > 1 && kAExt3 == 1) {
        CUTE_UNROLL
        for (int dual_k = 0; dual_k < kAExt4; ++dual_k) {
          cute::gemm(tiled_mma, sA_frag(_, _, k, Int<0>{}, dual_k), sB_frag(_, _, k), tC_frag);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
      } else if constexpr (kAExt3 > 1 && kAExt4 == 1) {
        CUTE_UNROLL
        for (int dual_k = 0; dual_k < kAExt3; ++dual_k) {
          cute::gemm(tiled_mma, sA_frag(_, _, k, dual_k, Int<0>{}), sB_frag(_, _, k), tC_frag);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
      } else {
        cute::gemm(tiled_mma, sA_frag(_, _, k, Int<0>{}, Int<0>{}), sB_frag(_, _, k), tC_frag);
      }
    } else if constexpr (kARank == 4 && kBRank == 4) {
      cute::gemm(tiled_mma, sA_frag(_, _, k, Int<0>{}), sB_frag(_, _, k, Int<0>{}), tC_frag);
    } else if constexpr (kARank == 4) {
      cute::gemm(tiled_mma, sA_frag(_, _, k, Int<0>{}), sB_frag(_, _, k), tC_frag);
    } else if constexpr (kBRank == 4) {
      cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k, Int<0>{}), tC_frag);
    } else {
      cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), tC_frag);
    }
    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
  }
}

// ---- SMEM layout helpers ----

template <int M, int N, int K, typename Element>
auto make_umma_k_major_layout() {
  using namespace cute;
  using base_atom_type = std::conditional_t<
      K == 0 || K == 16, UMMA::Layout_K_INTER_Atom<Element>,
      std::conditional_t<
          K == 32, UMMA::Layout_K_SW32_Atom<Element>,
          std::conditional_t<
              K == 64, UMMA::Layout_K_SW64_Atom<Element>,
              std::conditional_t<K == 128, UMMA::Layout_K_SW128_Atom<Element>, void>>>>;
  static_assert(!std::is_same_v<base_atom_type, void>, "Invalid swizzle value");
  return coalesce(tile_to_shape(base_atom_type{}, Shape<Int<M>, Int<N>>{}, Step<_1, _2>{}),
                  Shape<_1, _1>{});
}

}  // namespace flash
