// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Shared zero row for masked (-1) sparse candidates. Gathering a real cache
// slot for a masked lane would make the value MMA read mutable cache contents:
// a NaN in that slot contaminates valid outputs through 0 * NaN. A zeroed row
// keeps every masked value and scale finite (0.0), so masked lanes contribute
// exactly nothing. Sized to the largest single gather in the family
// (DOTS3_SWA decode: 1024-byte nope + 128-byte rope); use sites
// static_assert against it. `static` gives each translation unit its own
// zero-initialized .rodata copy; it is never written.
constexpr int SPARSE_MLA_ZERO_ROW_BYTES = 1152;
static __device__ __align__(16) const uint8_t sparse_mla_zero_row[SPARSE_MLA_ZERO_ROW_BYTES] = {};
