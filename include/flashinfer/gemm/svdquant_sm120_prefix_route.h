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

// What used to live here: a nine-entry ladder naming, per exact (M, K), which
// prefix had measured fastest, plus an environment override grammar for A/B
// runs and a reconciliation that threw if the ladder and a second copy of the
// gate in csrc ever disagreed. All of it existed to keep hand-written "measured
// faster" lists in step with each other.
//
// The prefix is now a producer family the tactic names, enumerated by
// sm120_producer_variants() on the Python side, so there is no route to look up
// and nothing to reconcile. What remains is the packed-tactic grammar the two
// languages still have to agree on.
#ifndef FLASHINFER_GEMM_SVDQUANT_SM120_PREFIX_ROUTE_H_
#define FLASHINFER_GEMM_SVDQUANT_SM120_PREFIX_ROUTE_H_

#include <cstdint>

namespace flashinfer {
namespace gemm {
namespace svdquant_sm120_prefix_route {

// Mirror of SM120_PRODUCER_SHIFT / SM120_K3_ROW_MASK in
// flashinfer/gemm/svdquant_sm120_routes.py. Python packs, this unpacks.
inline constexpr int kProducerShift = 8;
inline constexpr std::int64_t kK3RowMask = (std::int64_t{1} << kProducerShift) - 1;

inline int k3_row_of(std::int64_t tactic) {
  // A negative tactic is the autotuner's "no selection" sentinel and must reach
  // the consumer unchanged rather than being masked into a valid row.
  return tactic < 0 ? static_cast<int>(tactic) : static_cast<int>(tactic & kK3RowMask);
}

}  // namespace svdquant_sm120_prefix_route
}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_SVDQUANT_SM120_PREFIX_ROUTE_H_
