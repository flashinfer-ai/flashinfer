/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_DECODE_TRAITS_CUH_
#define FLASHINFER_DECODE_TRAITS_CUH_

#include "../page.cuh"
#include "../pos_enc.cuh"
#include "logits_post_hook.cuh"

namespace flashinfer {

template <LogitsPostHook LOGITS_POST_HOOK_, PosEncodingMode POS_ENCODING_MODE_,
          uint32_t NUM_STAGES_SMEM_, uint32_t TILE_SIZE_PER_BDX_, uint32_t VEC_SIZE_, uint32_t BDX_,
          uint32_t BDY_, uint32_t BDZ_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct DecodeTraitsBase {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;

  static constexpr LogitsPostHook LOGITS_POST_HOOK = LOGITS_POST_HOOK_;
  static constexpr PosEncodingMode POS_ENCODING_MODE = POS_ENCODING_MODE_;
  static constexpr uint32_t NUM_STAGES_SMEM = NUM_STAGES_SMEM_;
  static constexpr uint32_t TILE_SIZE_PER_BDX = TILE_SIZE_PER_BDX_;
  static constexpr uint32_t VEC_SIZE = VEC_SIZE_;
  static constexpr uint32_t BDX = BDX_;
  static constexpr uint32_t BDY = BDY_;
  static constexpr uint32_t BDZ = BDZ_;
};

template <LogitsPostHook LOGITS_POST_HOOK_, PosEncodingMode POS_ENCODING_MODE_,
          uint32_t NUM_STAGES_SMEM_, uint32_t TILE_SIZE_PER_BDX_, uint32_t VEC_SIZE_, uint32_t BDX_,
          uint32_t BDY_, uint32_t BDZ_, PageStorage PAGE_STORAGE_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename IdType_>
struct BatchDecodeTraits
    : DecodeTraitsBase<LOGITS_POST_HOOK_, POS_ENCODING_MODE_, NUM_STAGES_SMEM_, TILE_SIZE_PER_BDX_,
                       VEC_SIZE_, BDX_, BDY_, BDZ_, DTypeQ_, DTypeKV_, DTypeO_> {
  using IdType = IdType_;
  static constexpr PageStorage PAGE_STORAGE = PAGE_STORAGE_;
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_TRAITS_CUH_