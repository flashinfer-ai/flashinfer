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

#ifndef FLASHINFER_FUSED_MOE_DA_CONFIG_CUH_
#define FLASHINFER_FUSED_MOE_DA_CONFIG_CUH_

namespace flashinfer::da_moe {

/** Maximum local-expert domain supported by the distribution-aware selector. */
inline constexpr int kDAMaxExperts = 512;

/** Immutable maximum number of uploaded distribution exemplar rows. */
inline constexpr int kDAMaxExemplars = 8;

/** Immutable maximum number of unique conditional child bodies. */
inline constexpr int kDAMaxBodies = 8;

}  // namespace flashinfer::da_moe

#endif  // FLASHINFER_FUSED_MOE_DA_CONFIG_CUH_
