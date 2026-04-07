/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "flashinfer/ep/types.h"
#include "flashinfer/ep/ops.h"
#include <memory>

namespace flashinfer {
namespace ep {

/// Abstract backend interface for Expert Parallelism operations.
///
/// Defines the contract for pluggable backends (DeepEP, NCCL-EP) to implement
/// both High-Throughput (HT) and Low-Latency (LL) dispatch/combine operations.
/// A single EpGroup instance serves both modes via runtime dispatch selection
/// (Issue #2).
///
/// All virtual methods are pure and must be implemented by concrete backends.
/// This is a PyTorch-free C++ interface; PyTorch coupling lives in pybind11 layer.
class EpBackendImpl {
 public:
  virtual ~EpBackendImpl() = default;

  /// Initialize backend with communicator and CUDA stream.
  ///
  /// Called once per EpGroup creation. Backend allocates internal buffers
  /// (e.g., deep_ep::Buffer or ncclEpGroup_t objects).
  ///
  /// \param config      Group configuration (expert count, hidden dim, etc.)
  /// \param comm        Communicator (rank IDs, send/recv lists, etc.)
  /// \param stream      CUDA stream (void* to avoid C++ CUDA coupling)
  /// \return            EpStatus with error_code and optional error_cstr
  virtual EpStatus init(const EpGroupConfig& config,
                        const EpCommunicator& comm,
                        void* stream) = 0;

  /// Destroy backend and release all resources.
  ///
  /// Called once per EpGroup destruction. Cleans up allocations made in init().
  ///
  /// \return            EpStatus with error code
  virtual EpStatus destroy() = 0;

  /// Compute the scatter layout for the given routing decisions.
  ///
  /// High-Throughput only. Must be called OUTSIDE CUDA graph capture (Issue #4).
  /// Performs device-to-host memcpy to read topk_idx shape.
  ///
  /// \param topk_idx    Router output: shape [batch_size, top_k]
  /// \param prev        Stream dependency from prior op (or null)
  /// \param async       (Legacy; often false for layout computation)
  /// \param status      Output: pointer to EpStatus for error reporting
  /// \return            LayoutInfo with scatter plan (num_tokens_per_rank, etc.)
  virtual LayoutInfo get_dispatch_layout(
      const EpTensorView& topk_idx,
      StreamDep* prev, bool async, EpStatus* status) = 0;

  /// High-Throughput dispatch: scatter tokens to expert-owning GPUs.
  ///
  /// Issue #24: Output written via DispatchResult* pointer (zero-copy across
  /// vtable boundary). Caller provides heap-allocated DispatchResult; backend
  /// fills in output_tokens, output_indices, etc.
  ///
  /// Dynamic output size: total tokens sent = sum(num_tokens_per_rank from layout).
  /// Not CUDA graph capturable due to dynamic shape (Issue #4).
  ///
  /// \param out              Output struct: caller allocates, backend fills
  /// \param hidden           Input tokens: shape [batch_size, hidden_dim]
  /// \param scales           Optional scaling factors (or null)
  /// \param topk_idx         Router output: shape [batch_size, top_k]
  /// \param topk_weights     Router weights: shape [batch_size, top_k]
  /// \param layout           Scatter layout from get_dispatch_layout()
  /// \param out_layout       Output layout: FLAT_2D or EXPERT_MAJOR_3D
  /// \param prev             Stream dependency from prior op
  /// \param prev_handle      Output: EpHandle for depend-on by combine()
  /// \param alloc_comm       Whether to allocate new communication buffers
  /// \param async            Async execution flag
  virtual void dispatch(
      DispatchResult* out,
      const EpTensorView& hidden, const EpTensorView* scales,
      const EpTensorView& topk_idx, const EpTensorView& topk_weights,
      const LayoutInfo& layout, OutputLayout out_layout,
      StreamDep* prev, EpHandle* prev_handle,
      bool alloc_comm, bool async) = 0;

  /// High-Throughput combine: gather expert outputs and apply weighting.
  ///
  /// Issue #24: Output written via CombineResult* pointer (zero-copy across
  /// vtable boundary). Caller provides heap-allocated CombineResult; backend
  /// fills in output, output_indices, etc.
  ///
  /// Inverse of dispatch(): reconstructs original token order and applies
  /// topk_weights for final expert mixing.
  ///
  /// \param out              Output struct: caller allocates, backend fills
  /// \param expert_output    Expert computation results: shape [num_experts, max_tok, hidden]
  /// \param handle           EpHandle from dispatch() (carries deferred state)
  /// \param topk_weights     Router weights for weighted reduction
  /// \param prev             Stream dependency from expert computation
  /// \param alloc_comm       Whether to allocate new communication buffers
  /// \param async            Async execution flag
  virtual void combine(
      CombineResult* out,
      const EpTensorView& expert_output, EpHandle* handle,
      const EpTensorView* topk_weights,
      StreamDep* prev, bool alloc_comm, bool async) = 0;

  /// Low-Latency dispatch: scatter tokens with pre-allocated fixed-size output.
  ///
  /// Issue #24: Output written via DispatchResult* pointer.
  /// Issue #39: Optional RoutingCache input (NCCL-EP only; ignored by DeepEP).
  ///
  /// Static output size: [num_local_experts, max_tokens_per_rank, hidden_dim]
  /// (in EXPERT_MAJOR_3D mode).
  /// CUDA graph capturable when cuda_graph_max_tokens > 0 (Issue #4).
  ///
  /// Issue #43A: If cached_routing is provided, backend checks its validity
  /// via is_routing_cache_valid(cache, topk_idx) to skip expensive
  /// ncclEpCreateHandle() calls. The validity check is SHAPE-ONLY and performs
  /// no device-to-host copies.
  ///
  /// \param out                  Output struct: caller allocates, backend fills
  /// \param hidden               Input tokens: shape [num_tokens, hidden_dim]
  /// \param topk_idx             Router output: shape [num_tokens, top_k]
  /// \param max_tokens_per_rank  Pre-allocated max tokens per rank
  /// \param out_layout           Output layout: FLAT_2D or EXPERT_MAJOR_3D
  /// \param prev                 Stream dependency from prior op
  /// \param prev_handle          Output: EpHandle for depend-on by low_latency_combine()
  /// \param alloc_comm           Whether to allocate new communication buffers
  /// \param async                Async execution flag
  /// \param return_hook          Whether to return a hook for deferred synchronization
  /// \param cached_routing       Optional RoutingCache from prior create_routing_cache()
  virtual void low_latency_dispatch(
      DispatchResult* out,
      const EpTensorView& hidden, const EpTensorView& topk_idx,
      int max_tokens_per_rank, OutputLayout out_layout,
      StreamDep* prev, EpHandle* prev_handle,
      bool alloc_comm, bool async, bool return_hook,
      RoutingCache* cached_routing = nullptr) = 0;

  /// Create a RoutingCache for repeated dispatch calls with identical topk_idx.
  ///
  /// Issue #39: NCCL-EP optimizes repeated dispatch() calls by caching the
  /// ncclEpCreateHandle output. DeepEP ignores this (no-op cache).
  ///
  /// Issue #43A: Cache includes cached_shape for shape-only validity checks
  /// in the hot path. Saves ~1-2 μs per decode step by skipping handle creation
  /// on cache hit.
  ///
  /// Default implementation: returns a valid no-op cache with cached_shape
  /// populated so is_routing_cache_valid() shape checks work for backends
  /// that don't override this method.
  ///
  /// \param topk_idx    Router output to cache: shape [batch_size, top_k]
  /// \param status      Output: error code (0 on success)
  /// \return            Newly allocated RoutingCache (caller owns; delete via
  ///                    destroy_routing_cache())
  virtual RoutingCache* create_routing_cache(
      const EpTensorView& topk_idx, EpStatus* status) {
    // Default: return a valid but no-op cache (DeepEP behavior).
    // Issue #43A: Still populate cached_shape so is_routing_cache_valid()
    // shape check works correctly even for no-op caches.
    auto* cache = new RoutingCache{
        nullptr, nullptr,
        {topk_idx.shape[0], topk_idx.shape[1]},  // cached_shape
        0,     // topology_hash (unused for DeepEP)
        true   // valid
    };
    if (status) {
      *status = EpStatus::Ok();
    }
    return cache;
  }

  /// Destroy a RoutingCache created via create_routing_cache().
  ///
  /// Default implementation: simple delete. Backends may override to release
  /// backend-specific resources (e.g., NCCL handles stored in cache->backend_data).
  ///
  /// \param cache    RoutingCache to destroy (may be null)
  /// \return         EpStatus with error code
  virtual EpStatus destroy_routing_cache(RoutingCache* cache) {
    delete cache;
    return EpStatus::Ok();
  }

  /// Low-Latency combine: gather expert outputs with static-size input.
  ///
  /// Issue #24: Output written via CombineResult* pointer.
  ///
  /// Inverse of low_latency_dispatch(). Input shape is static:
  /// [num_local_experts, max_tokens_per_rank, hidden_dim].
  /// CUDA graph capturable when cuda_graph_max_tokens > 0 (Issue #4).
  ///
  /// \param out              Output struct: caller allocates, backend fills
  /// \param expert_output    Expert computation results: static shape
  /// \param topk_idx         Router output: [num_tokens, top_k]
  /// \param topk_weights     Router weights for weighted reduction
  /// \param handle           EpHandle from low_latency_dispatch()
  /// \param prev             Stream dependency from expert computation
  /// \param alloc_comm       Whether to allocate new communication buffers
  /// \param async            Async execution flag
  /// \param return_hook      Whether to return a hook for deferred synchronization
  virtual void low_latency_combine(
      CombineResult* out,
      const EpTensorView& expert_output, const EpTensorView& topk_idx,
      const EpTensorView& topk_weights, EpHandle* handle,
      StreamDep* prev, bool alloc_comm, bool async, bool return_hook) = 0;
};

/// DeepEP backend implementation.
///
/// Wraps deep_ep::Buffer. A single Buffer instance serves both High-Throughput
/// and Low-Latency modes via internal low_latency_mode toggle. Which mode is
/// active is determined by which dispatch/combine method is called.
///
/// Does not benefit from RoutingCache (no-op override of create_routing_cache()).
class DeepEPBackend : public EpBackendImpl {
  // Implementation in separate translation unit (deep_ep_backend.cu / .cc)
};

/// NCCL-EP backend implementation.
///
/// Wraps two ncclEpGroup_t objects: one for Low-Latency mode, one for
/// High-Throughput mode. Both groups share the same EpCommunicator but have
/// separate algorithm configs (NCCL_EP_ALGO_* enums).
///
/// Overrides create_routing_cache() to store ncclEpHandle_t for hot-path
/// reuse. Issue #43A: Hot-path validity check is shape-only; content hash
/// is computed at cache creation time but not re-checked.
class NcclEPBackend : public EpBackendImpl {
  // Implementation in separate translation unit (nccl_ep_backend.cu / .cc)
};

/// Register a backend implementation.
///
/// Stores the implementation in a global registry (map by Backend enum).
/// Called once during module initialization to register concrete backends.
/// Ownership of impl is transferred to the registry (unique_ptr moved).
///
/// \param id      Backend identifier (Backend::DEEP_EP or Backend::NCCL_EP)
/// \param impl    Unique pointer to backend implementation
void register_backend(Backend id, std::unique_ptr<EpBackendImpl> impl);

/// Retrieve a registered backend implementation.
///
/// Returns a raw pointer to the backend; registry retains ownership.
/// Returns null if backend not found in registry.
///
/// \param id      Backend identifier to retrieve
/// \return        Raw pointer to EpBackendImpl (registry owned; do not delete)
EpBackendImpl* get_backend(Backend id);

}  // namespace ep
}  // namespace flashinfer
