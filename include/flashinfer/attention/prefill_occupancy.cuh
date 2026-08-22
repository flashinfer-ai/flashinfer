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
#ifndef FLASHINFER_ATTENTION_PREFILL_OCCUPANCY_CUH_
#define FLASHINFER_ATTENTION_PREFILL_OCCUPANCY_CUH_

#include <cuda_runtime.h>
#if CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#include <cstdint>
#include <type_traits>

namespace flashinfer {

// Number of NVFP4 elements sharing one scale factor (UE4M3 byte).
constexpr uint32_t NVFP4_SF_VEC_SIZE = 16;

// Type trait to detect packed NVFP4 KV cache types (__nv_fp4x2_e2m1 stores 2 FP4 per byte).
template <typename T>
struct is_fp4_type : std::false_type {};
#if CUDA_VERSION >= 12080
template <>
struct is_fp4_type<__nv_fp4x2_e2m1> : std::true_type {};
#endif
template <typename T>
inline constexpr bool is_fp4_type_v = is_fp4_type<T>::value;

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q == 32) {
    return 1;  // HEAD_DIM_VO >= 512
  }
  if (cta_tile_q > 16) {
    return 4;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) {
  return 4 / get_num_warps_q(cta_tile_kv);
}

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  if (cta_tile_q == 32) {
    return 2;  // HEAD_DIM_VO >= 512
  }
  if (cta_tile_q > 64) {
    return 2;
  } else {
    return 1;
  }
}

/*!
 * \brief The dtype facts FA2PrefillCtaSmemLowerBound needs, derived from the types.
 *
 * These three values must describe the same pair of types, and two of them cannot
 * be recovered from each other: FP8 and NVFP4 KV are both 1 byte, so `kv_is_fp4`
 * is independent information, and there is no value of it that is safe to guess
 * (guessing `false` for an FP4 cache turns on `kUseRepack`, which the ragged and
 * paged launchers do not charge, and the bound then *exceeds* the launcher's
 * requirement -- the one direction that is unsound).
 *
 * Callers therefore never spell the fields out; they call
 * MakeFA2PrefillDTypeInfo<DTypeQ, DTypeKV>(), which computes all three from the
 * types themselves. There is no defaulted overload on purpose, and the members
 * are private behind a friend-only constructor so that "spell the fields out"
 * is a compile error rather than a convention.
 */
class FA2PrefillDTypeInfo;

template <typename DTypeQ, typename DTypeKV>
constexpr FA2PrefillDTypeInfo MakeFA2PrefillDTypeInfo();

class FA2PrefillDTypeInfo {
 public:
  constexpr uint32_t q_dtype_bytes() const { return q_dtype_bytes_; }
  constexpr uint32_t kv_dtype_bytes() const { return kv_dtype_bytes_; }
  constexpr bool kv_is_fp4() const { return kv_is_fp4_; }

 private:
  // Private tag: the constructor cannot be named outside the friend factory, so
  // this is not an aggregate, has no default constructor, and exposes no mutable
  // member. Aggregate init, designated init, value init and copy-then-mutate are
  // all compile errors -- the only way to obtain a value is from the types.
  struct FromTypes {};
  constexpr FA2PrefillDTypeInfo(FromTypes, uint32_t q_bytes, uint32_t kv_bytes, bool is_fp4)
      : q_dtype_bytes_(q_bytes), kv_dtype_bytes_(kv_bytes), kv_is_fp4_(is_fp4) {}

  template <typename DTypeQ, typename DTypeKV>
  friend constexpr FA2PrefillDTypeInfo MakeFA2PrefillDTypeInfo();

  uint32_t q_dtype_bytes_;
  uint32_t kv_dtype_bytes_;
  bool kv_is_fp4_;
};

template <typename DTypeQ, typename DTypeKV>
constexpr FA2PrefillDTypeInfo MakeFA2PrefillDTypeInfo() {
  return FA2PrefillDTypeInfo(FA2PrefillDTypeInfo::FromTypes{},
                             static_cast<uint32_t>(sizeof(DTypeQ)),
                             static_cast<uint32_t>(sizeof(DTypeKV)), is_fp4_type_v<DTypeKV>);
}

/*!
 * \brief Lower bound, in bytes, on the shared memory one FA2 prefill CTA needs.
 *
 * The kernel launchers already decide how many CTAs fit on an SM:
 *
 *   num_ctas_per_sm =
 *       max_smem_per_sm >= 2 * (kFixedSmem + kMinValidMmaKV * kKVSmemPerMmaKV) ? 2 : 1;
 *
 * (BatchPrefillWithRaggedKVCacheDispatched / BatchPrefillWithPagedKVCacheDispatched
 * in prefill.cuh).  The planner cannot evaluate that expression directly:
 * PrefillPlanImpl is templated only on <MATERIALIZE, IdType> and the plan entry
 * point is shared by the ragged and paged run paths.  This function reproduces
 * the same arithmetic from quantities the plan-time translation unit does have,
 * and for every quantity that is still not observable it substitutes the choice
 * that makes the requirement *smallest*.  The result is therefore always <= the
 * launcher's value; prefill.cuh pins that per instantiation with a static_assert.
 *
 * The inequality direction is the load-bearing property.  The planner only uses
 * this to lower num_blocks_per_sm from 2 to 1, so an under-estimate merely keeps
 * the historical behaviour, whereas an over-estimate would shrink the split-KV
 * grid on a configuration that really does hold two CTAs per SM.
 *
 * Quantities that remain unobservable at plan time, and how each is handled:
 *   - AttentionVariant::use_softmax (bound to the run-time mask mode) and the
 *     ragged/paged launcher identity (one plan entry point serves both):
 *     kVOSplitDispatch is treated as false, dropping both the per-tile and the
 *     fixed VO-split allocations, and the wider (ragged) kLargeHeadWarpSplit
 *     predicate is taken, halving NUM_WARPS_KV.  Every remaining term is
 *     non-decreasing in NUM_WARPS_KV, so both substitutions only remove bytes.
 *   - NVFP4 scale-factor staging: dropped, because the ragged and paged
 *     launchers do not count it either (only the single-prefill launcher does).
 *     See the note at the term itself.
 *   - POS_ENCODING_MODE: kSharedRopeFreqSmem is dropped (it is >= 0).  It is
 *     visible in the plan TU but is at most 4 * (HEAD_DIM_QK / 32) * 16 bytes
 *     and never changes the predicate's outcome, so it is left out to keep the
 *     shared contract narrow.
 *   - MASK_MODE and USE_FP16_QK_REDUCTION do not enter the launcher's
 *     expression at all.
 *
 * \param cta_tile_q   CTA_TILE_Q the run path will dispatch on, i.e. plan_info.cta_tile_q
 * \param head_dim_qk  HEAD_DIM_QK
 * \param head_dim_vo  HEAD_DIM_VO
 * \param dtype_info   from MakeFA2PrefillDTypeInfo<DTypeQ, DTypeKV>()
 */
constexpr uint32_t FA2PrefillCtaSmemLowerBound(uint32_t cta_tile_q, uint32_t head_dim_qk,
                                               uint32_t head_dim_vo,
                                               FA2PrefillDTypeInfo dtype_info) {
  const uint32_t q_dtype_bytes = dtype_info.q_dtype_bytes();
  const uint32_t kv_dtype_bytes = dtype_info.kv_dtype_bytes();
  const bool kv_is_fp4 = dtype_info.kv_is_fp4();
  const uint32_t num_mma_d_vo = head_dim_vo / 16;
  // kLargeHeadWarpSplit / kBf16VOSplit, taken unconditionally: it gives
  // NUM_WARPS_KV = 2 where the paged FP8 path would use 4.
  const bool large_head_warp_split = (head_dim_vo >= 512) && (cta_tile_q == 32);
  const uint32_t num_warps_q = large_head_warp_split ? 2u : get_num_warps_q(cta_tile_q);
  const uint32_t num_warps_kv = large_head_warp_split ? 2u : get_num_warps_kv(cta_tile_q);

  // kUseRepack
  const bool use_repack = (kv_dtype_bytes == 1) && !kv_is_fp4 && (head_dim_vo != 64) &&
                          (head_dim_vo <= 256) && (cta_tile_q > 16);
  // kKVShared
  const bool kv_shared = !kv_is_fp4 && (num_mma_d_vo > 16) && (num_mma_d_vo % num_warps_kv == 0) &&
                         (head_dim_qk == head_dim_vo) && (kv_dtype_bytes == 2 || cta_tile_q > 16);

  // kKVSmemPerMmaKV, without the kVOSplitDispatch term.
  uint32_t kv_smem_per_mma_kv =
      kv_shared ? (head_dim_qk * 16 * num_warps_kv * kv_dtype_bytes)
                : ((head_dim_qk + head_dim_vo) * 16 * num_warps_kv * kv_dtype_bytes);
  if (use_repack) {
    kv_smem_per_mma_kv +=
        (head_dim_qk > head_dim_vo ? head_dim_qk : head_dim_vo) * 16 * num_warps_kv * q_dtype_bytes;
  }
  // NOTE: the NVFP4 scale-factor staging term is deliberately NOT counted. Only
  // SinglePrefillWithKVCacheDispatched includes it in its occupancy budget; the
  // ragged and paged launchers -- the only two this planner feeds -- omit it.
  // Adding it here would make the estimate exceed their requirement, i.e. break
  // the direction this whole function depends on. Leaving it out also stays
  // correct if the launchers later start counting it, since the term is
  // non-negative.
  // kFixedSmem, without kVOSplitFixedSmem and kSharedRopeFreqSmem.
  const uint32_t fixed_smem = cta_tile_q * head_dim_qk * q_dtype_bytes;
  // kMinValidMmaKV
  const uint32_t min_valid_mma_kv =
      (kv_dtype_bytes == 1 && num_warps_q > 2) ? (num_warps_q / 2) : 1u;

  return fixed_smem + min_valid_mma_kv * kv_smem_per_mma_kv;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_PREFILL_OCCUPANCY_CUH_
