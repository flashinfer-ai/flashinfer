// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// Module-owned padding cannot be overwritten by cache allocation or graph
// dummy writes. GLM NoPE gathers 512 FP8 bytes and four inline FP32 scales.
// Both payload and scales are finite zero, so masked value MMA lanes cannot
// produce 0 * NaN from an unrelated cache row.
static __device__ __align__(16) const uint8_t glm53_padding_kv[528] = {};
