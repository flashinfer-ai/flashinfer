// Copyright (c) 2025 FlashInfer Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "flashinfer/ep/types.h"

namespace flashinfer {
namespace ep {

// --- Results -----------------------------------------------------------
//
// Issue #24: DispatchResult (~600 bytes in v3) was returned by value
// across virtual dispatch (EpBackendImpl::low_latency_dispatch). NRVO is
// NOT guaranteed across virtual boundaries — the compiler cannot see the
// concrete return site at the caller. This added ~0.5-1 μs per dispatch
// from memcpy of 5 EpTensorViews + handle + status.
//
// Fix: All backend virtual methods now take a DispatchResult*/CombineResult*
// output pointer. The caller stack-allocates the result struct and passes
// its address. Zero copies regardless of optimization level.
//
// Issue #23: DeferredRecv is no longer a std::function field here.
// The deferred callback is stored directly in EpHandle (see types.h).
// Call handle->invoke_deferred() instead of result.deferred_recv().

struct DispatchResult {
  EpTensorView  recv_hidden;               // shape depends on output_layout
  EpTensorView  recv_topk_idx;
  EpTensorView  recv_topk_weights;
  EpTensorView  recv_expert_counts;        // [num_local_experts], int32
  EpTensorView  recv_scales;               // FP8 quant scales (empty if not FP8)
  EpHandle*     handle;                    // pass to combine(); owns deferred_recv
  StreamDep*    dep;                       // for async overlap
  EpStatus      status;
};

struct CombineResult {
  EpTensorView  combined_hidden;           // [num_tokens, hidden_dim]
  StreamDep*    dep;
  EpStatus      status;
};


// --- High-Throughput Operations (prefill / training) -------------------
//
// NOT CUDA-graph-capturable (Issue #4): output shape is dynamic.

/// Dispatch tokens to remote experts (HT mode).
///
/// Issue #11 — Runtime dtype: The wire dtype is inferred from hidden.scalar_type.
///   - BF16 hidden (scalar_type::kBFloat16): standard path, scales ignored.
///   - FP8  hidden (scalar_type::kFloat8E4M3): scales MUST be non-null.
///   - The backend validates that hidden.scalar_type <= max_dispatch_elem_size
///     configured on the group. Oversized elements return EpStatus::Error.
///
/// @param group               Communication group
/// @param out                 Caller-allocated output struct (Issue #24)
/// @param hidden              [num_tokens, hidden_dim]. Dtype determines wire format.
/// @param scales              FP8 quantization scales (nullptr if hidden is not FP8)
/// @param topk_idx            [num_tokens, top_k] expert assignments
/// @param topk_weights        [num_tokens, top_k] router weights
/// @param layout              Pre-computed from get_dispatch_layout()
/// @param output_layout       FLAT_2D or EXPERT_MAJOR_3D (Issue #1, default: kFlat2D)
/// @param prev_dep            Dependency on prior async operation (default: nullptr)
/// @param prev_handle         Prior micro-batch handle for double buffering (Issue #9)
/// @param alloc_on_comm_stream  Allocate output on comm stream (default: true, Issue #5)
/// @param async               If true, returns immediately with dep (default: true)
void dispatch(EpGroup* group,
              DispatchResult* out,                          // caller-allocated
              const EpTensorView& hidden,
              const EpTensorView* scales,
              const EpTensorView& topk_idx,
              const EpTensorView& topk_weights,
              const LayoutInfo& layout,
              OutputLayout output_layout = OutputLayout::kFlat2D,
              StreamDep* prev_dep = nullptr,
              EpHandle* prev_handle = nullptr,
              bool alloc_on_comm_stream = true,
              bool async = true);

/// Combine expert outputs back to original token ordering (HT mode).
/// Issue #24: Output via pointer — guaranteed zero-copy across virtual dispatch.
///
/// @param group               Communication group
/// @param out                 Caller-allocated output struct
/// @param expert_output       Expert computation results
/// @param handle              Handle from dispatch(), owns communication state
/// @param topk_weights        [num_tokens, top_k] router weights for weighted reduction
/// @param prev_dep            Dependency on prior async operation (default: nullptr)
/// @param alloc_on_comm_stream  Allocate output on comm stream (default: true, Issue #5)
/// @param async               If true, returns immediately with dep (default: true)
void combine(EpGroup* group,
             CombineResult* out,                            // caller-allocated
             const EpTensorView& expert_output,
             EpHandle* handle,
             const EpTensorView* topk_weights,
             StreamDep* prev_dep = nullptr,
             bool alloc_on_comm_stream = true,
             bool async = true);


// --- Low-Latency Operations (decode) -----------------------------------
//
// CUDA-graph-capturable when cuda_graph_max_tokens > 0 (Issue #4).

/// Dispatch tokens to remote experts (LL mode).
///
/// @param group               Communication group
/// @param out                 Caller-allocated output struct (Issue #24)
/// @param hidden              [num_tokens, hidden_dim]. Dtype determines wire format (Issue #11).
/// @param topk_idx            [num_tokens, top_k] expert assignments
/// @param max_tokens_per_rank  Upper bound. Validated at runtime (Issue #7).
/// @param output_layout       FLAT_2D or EXPERT_MAJOR_3D (Issue #1, default: kFlat2D).
///                            Prefer EXPERT_MAJOR_3D for static shapes on both backends.
/// @param prev_dep            Dependency on prior async operation (default: nullptr)
/// @param prev_handle         Double buffering (default: nullptr, Issue #9)
/// @param alloc_on_comm_stream  Allocate output on comm stream (default: true, Issue #5)
/// @param async               If true, returns immediately with dep (default: false for LL)
/// @param return_recv_hook    If true, returns DeferredRecv callable (default: false, Issue #3)
/// @param cached_routing      Issue #39: Pre-built routing cache. When non-null,
///                            the NCCL-EP backend skips ncclEpCreateHandle()
///                            (~1-2 μs savings per dispatch). DeepEP ignores
///                            this parameter (routing is implicit). The caller
///                            is responsible for invalidating the cache when
///                            topk_idx topology changes. In static CUDA graph
///                            replay, the routing is identical every step —
///                            this parameter eliminates redundant handle setup.
///
/// CUDA GRAPH (Issue #4): capturable when cuda_graph_max_tokens > 0.
///   max_tokens_per_rank must match at capture and replay.
///   Prefer EXPERT_MAJOR_3D for static shapes on both backends.
///   Issue #19: The backend records hidden.scalar_type on first graph-mode
///   dispatch and validates it on subsequent calls. Dtype mismatch returns
///   EpStatus::Error("graph dtype mismatch") — not silent corruption.
void low_latency_dispatch(EpGroup* group,
                          DispatchResult* out,               // caller-allocated
                          const EpTensorView& hidden,
                          const EpTensorView& topk_idx,
                          int max_tokens_per_rank,
                          OutputLayout output_layout = OutputLayout::kFlat2D,
                          StreamDep* prev_dep = nullptr,
                          EpHandle* prev_handle = nullptr,
                          bool alloc_on_comm_stream = true,
                          bool async = false,
                          bool return_recv_hook = false,
                          RoutingCache* cached_routing = nullptr);  // Issue #39

/// Combine expert outputs (LL mode).
/// Issue #24: Output via pointer — guaranteed zero-copy across virtual dispatch.
///
/// @param group               Communication group
/// @param out                 Caller-allocated output struct
/// @param expert_output       Expert computation results
/// @param topk_idx            [num_tokens, top_k] expert assignments
/// @param topk_weights        [num_tokens, top_k] router weights for weighted reduction
/// @param handle              Handle from low_latency_dispatch(), owns communication state
/// @param prev_dep            Dependency on prior async operation (default: nullptr)
/// @param alloc_on_comm_stream  Allocate output on comm stream (default: true, Issue #5)
/// @param async               If true, returns immediately with dep (default: false for LL)
/// @param return_recv_hook    If true, returns DeferredRecv callable (default: false, Issue #3)
void low_latency_combine(EpGroup* group,
                         CombineResult* out,                  // caller-allocated
                         const EpTensorView& expert_output,
                         const EpTensorView& topk_idx,
                         const EpTensorView& topk_weights,
                         EpHandle* handle,
                         StreamDep* prev_dep = nullptr,
                         bool alloc_on_comm_stream = true,
                         bool async = false,
                         bool return_recv_hook = false);


// --- Handle Management -------------------------------------------------

/// Destroy a dispatch handle and free backend resources.
///
/// @param handle              Handle from dispatch() or low_latency_dispatch()
/// @return                    Status of the destroy operation
EpStatus destroy_handle(EpHandle* handle);

/// Block until incoming transfers finish.
/// Issue #7: max_tokens_per_rank and complete() timeout are runtime validated.
///
/// @param handle              Handle from dispatch() or low_latency_dispatch()
/// @param stream              CUDA stream to synchronize with (void* = cudaStream_t)
/// @param timeout_ms          Override group timeout. 0 = use group default (default: 0)
/// @return                    Status of the complete operation
EpStatus complete(EpHandle* handle, void* /*cudaStream_t*/ stream,
                  int timeout_ms = 0);

/// Get the number of tokens received at this GPU for this dispatch.
///
/// @param handle              Handle from dispatch() or low_latency_dispatch()
/// @return                    Number of tokens in recv_hidden
int get_num_recv_tokens(EpHandle* handle);


// --- Routing Cache Management (Issue #39) --------------------------------
//
// For NCCL-EP, each low_latency_dispatch() call builds a routing table
// via ncclEpCreateHandle() (~1-2 μs). In steady-state decode or CUDA graph
// replay, the routing topology (topk_idx shape and expert mapping) is
// identical across steps. The routing cache lets frameworks pre-build this
// table once and reuse it, eliminating ~1-2 μs per dispatch.
//
// DeepEP: create_routing_cache returns a valid but no-op cache (routing
// is implicit in the NVSHMEM put addresses). The cache is accepted by
// low_latency_dispatch but has no effect.

/// Create a routing cache from a representative topk_idx tensor.
/// The cache is valid as long as the routing topology is unchanged
/// (same expert assignments, same batch structure).
///
/// @param group     Communication group
/// @param topk_idx  [num_tokens, top_k] representative routing
/// @param status    Optional output for error status (default: nullptr)
/// @return          Opaque cache (caller owns), nullptr on failure
RoutingCache* create_routing_cache(EpGroup* group,
                                   const EpTensorView& topk_idx,
                                   EpStatus* status = nullptr);

/// Destroy a routing cache and free backend resources.
///
/// @param cache     Routing cache from create_routing_cache()
/// @return          Status of the destroy operation
EpStatus destroy_routing_cache(RoutingCache* cache);

/// Check if a routing cache is still valid for a given topk_idx.
/// Returns true if the cache can be reused (topology match).
///
/// Issue #43A: This is the HOT-PATH check — called every decode step.
/// It performs a SHAPE-ONLY comparison (two int64 compares, ~0 μs).
/// NO device-to-host copies. The content-sampled hash computed at
/// create_routing_cache() time is stored in the cache but is NOT
/// recomputed here — doing so would add ~0.5 μs of blocking D2H per
/// step, undoing the ~1-2 μs savings that RoutingCache exists to provide.
///
/// The contract is: if the shape changes (different batch size or top_k),
/// the cache is invalidated and the caller must recreate it. If the shape
/// is the same but expert assignments differ (same batch size, different
/// routing), the caller is responsible for explicit invalidation — e.g.,
/// by calling destroy + recreate when the router produces new assignments
/// that differ from the cached topology. In practice, CUDA graph replay
/// uses identical routing every step, and eager-mode decode changes
/// batch size when sequences complete, which this shape check catches.
///
/// @param cache     Routing cache from create_routing_cache()
/// @param topk_idx  [num_tokens, top_k] current routing to check
/// @return          true if cache is still valid, false if topology changed
bool is_routing_cache_valid(const RoutingCache* cache,
                            const EpTensorView& topk_idx);


// --- Stream Dependency Management (Issue #3) ---------------------------
//
// Issue #3: DeferredRecv allows overlapping the wait for incoming data
// with other compute. create_stream_dep / record_stream_dep / wait_stream_dep
// form the core of async dependency tracking. is_stream_dep_complete checks
// if a dependency is ready without blocking.

/// Create a stream dependency object.
///
/// @param stream             CUDA stream to track (void* = cudaStream_t)
/// @return                   Opaque dependency object (caller owns)
StreamDep* create_stream_dep(void* /*cudaStream_t*/ stream);

/// Record an event on a stream into a dependency object.
///
/// @param dep                Dependency object from create_stream_dep()
/// @param stream             CUDA stream to record (void* = cudaStream_t)
/// @return                   Status of the record operation
EpStatus record_stream_dep(StreamDep* dep, void* /*cudaStream_t*/ stream);

/// Wait for a dependency on a target stream.
///
/// @param dep                Dependency object from create_stream_dep()
/// @param target_stream      CUDA stream to wait on (void* = cudaStream_t)
/// @return                   Status of the wait operation
EpStatus wait_stream_dep(StreamDep* dep, void* /*cudaStream_t*/ target_stream);

/// Check if a dependency is complete without blocking.
///
/// @param dep                Dependency object from create_stream_dep()
/// @return                   true if the dependency's event has completed
bool is_stream_dep_complete(const StreamDep* dep);

/// Destroy a stream dependency object and free resources.
///
/// @param dep                Dependency object from create_stream_dep()
void destroy_stream_dep(StreamDep* dep);

}  // namespace ep
}  // namespace flashinfer
