// Copyright (c) 2025 FlashInfer Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "flashinfer/ep/types.h"

namespace flashinfer {
namespace ep {

/// Pre-computed routing metadata. Shared between dispatch and combine.
struct LayoutInfo {
  EpTensorView  num_tokens_per_rank;       // [world_size], int32
  EpTensorView  num_tokens_per_expert;     // [num_experts], int32

  // DeepEP-only (zeroed for NCCL-EP)
  EpTensorView  num_tokens_per_rdma_rank;  // [num_rdma_ranks], int32
  EpTensorView  is_token_in_rank;          // [num_tokens, world_size], bool

  StreamDep*    dep;                       // non-null when async=true
};

/// Analyze routing decisions and pre-compute communication pattern.
///
/// WARNING (Issue #4): This function performs a device-to-host memcpy
/// (token counts must reach the CPU). This BREAKS CUDA graph capture.
/// Call this OUTSIDE the graph capture region.
///
/// @param group           Communication group
/// @param topk_idx        [num_tokens, top_k] expert assignments
/// @param previous_dep    Dependency on prior async operation (default: nullptr)
/// @param async           If true, returns immediately with dep (default: true)
/// @param status          Optional output for error status
/// @return                Pre-computed LayoutInfo containing token distribution
///                        and communication metadata
LayoutInfo get_dispatch_layout(EpGroup* group,
                               const EpTensorView& topk_idx,
                               StreamDep* previous_dep = nullptr,
                               bool async = true,
                               EpStatus* status = nullptr);

}  // namespace ep
}  // namespace flashinfer
