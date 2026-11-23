/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define LL_POLL_HD __host__ __device__
#else
#define LL_POLL_HD
#endif

namespace flashinfer::glm5_tp8::ll_wire {

// Check the latest loaded tags, including the final allowed reload.  The
// predicate is host/device so it can be checked on the CPU; device code calls
// it only inside the cold deadline branch of the poll loops.
LL_POLL_HD constexpr bool deadline_missing_tags(uint32_t polls, uint32_t limit, uint32_t tag0,
                                                uint32_t tag1, uint32_t expected) {
  return polls >= limit && (tag0 != expected || tag1 != expected);
}

#if defined(__CUDACC__)
// A missing publication is an execution error.  Called only from the timeout
// branches, so the hot poll/consume path carries no extra code.
__device__ __forceinline__ void timeout_trap() { __trap(); }
#endif

}  // namespace flashinfer::glm5_tp8::ll_wire

#undef LL_POLL_HD
