// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cute/tensor.hpp>

namespace cute {
// SM120 requires CTA scope here. SM90_BULK_COPY_G2S uses cluster scope, which
// ptxas lowers to an external call on SM120 and disables setmaxnreg for the CTA.
// Keep the CuTe bulk-copy interface while selecting the native instruction.
struct SM120_BULK_COPY_G2S {
  CUTE_DEVICE static void copy(void const* source, uint64_t* ready, void* destination,
                               int32_t bytes) {
    uint32_t dst = cast_smem_ptr_to_uint(destination);
    uint32_t bar = cast_smem_ptr_to_uint(ready);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
            dst),
        "l"(source), "r"(bytes), "r"(bar)
        : "memory");
  }
};
}  // namespace cute
