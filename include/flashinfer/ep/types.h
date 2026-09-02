#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace flashinfer {
namespace ep {

// --- Backend and Layout Enums ------------------------------------------

enum class Backend : int {
  kDeepEP = 0,
  kNcclEP = 1,
};

enum class OutputLayout : int {
  kFlat2D         = 0,  // [total_recv_tokens, hidden_dim] — universal default
  kExpertMajor3D  = 1,  // [num_local_experts, max_tok_per_expert, hidden] — grouped GEMM
};

// --- Scalar type (Issue #8: no custom Dtype enum) ----------------------
// Uses int constants matching PyTorch's ScalarType for ABI stability,
// without requiring the PyTorch header.

namespace scalar_type {
  constexpr int kFloat32    = 6;
  constexpr int kFloat16    = 5;
  constexpr int kBFloat16   = 15;
  constexpr int kFloat8E4M3 = 23;
  constexpr int kFloat8E5M2 = 24;
}  // namespace scalar_type


// --- PyTorch-free tensor view ------------------------------------------
//
// Issue #32: EP tensors are always 2D or 3D. shape/strides are sized to 3
// instead of 4 to reduce per-tensor copy overhead (~16 bytes saved per view,
// significant when 8 views are constructed per dispatch call).

struct EpTensorView {
  void*    data_ptr;
  int      scalar_type;            // one of scalar_type:: constants
  int64_t  shape[3];               // up to 3D, unused dims = 0
  int      ndim;
  int64_t  strides[3];             // optional, 0 = contiguous assumed
  int      device_id;              // CUDA device ordinal
};


// --- Backend-agnostic communicator (Issue #6) --------------------------

struct EpCommunicator {
  // Required by all backends
  int     rank;
  int     world_size;
  int     device_id;               // local CUDA device

  // NCCL-EP backend: set this to the raw ncclComm_t
  void*   nccl_comm;               // nullptr if not NCCL-EP

  // DeepEP backend: set this to the NVSHMEM team handle
  void*   nvshmem_team;            // nullptr if not DeepEP

  // Intra-node topology (used by both backends for NVLink detection)
  int     local_rank;              // rank within the node
  int     local_world_size;        // GPUs per node
};


// --- Error handling (Issue #7, Issue #22) -------------------------------
//
// Issue #22: Hot-path overhead. The original EpStatus constructed a
// std::string + std::vector<int> on EVERY call — even successes. On the
// LL decode path (~191 μs budget), this added ~1-3 μs of pure host
// overhead across 4 calls per step.
//
// Fix: Two-tier error reporting.
//   - EpStatus (lightweight): returned from every call. int error_code
//     + const char* for the fast path. Zero heap allocation on success.
//   - EpStatusDetail (heavyweight): constructed lazily only on error,
//     carries std::string + std::vector for rich diagnostics.

struct EpStatusDetail {
  std::string          error_msg;
  std::vector<int>     failed_ranks;
};

struct EpStatus {
  int                  error_code;     // 0 = ok, nonzero = error
  const char*          error_cstr;     // static or thread_local error string (nullptr on ok)
  EpStatusDetail*      detail;         // non-null only on error (caller does NOT own)

  bool ok() const { return error_code == 0; }

  // Hot path: zero allocation
  static EpStatus Ok() { return {0, nullptr, nullptr}; }

  // Error path: allocates detail lazily. Uses thread_local storage
  // so the EpStatusDetail* is valid until the next error on the same thread.
  // Caller does NOT own or free the detail pointer.
  //
  // Issue #44: Uses ep_error::kGeneric (not -1) so error_code is always
  // a positive constant from the ep_error namespace. This ensures
  // error_code checks like `status.error_code == ep_error::kBufferOverflow`
  // work consistently — no silent mismatch between -1 and the positive codes.
  static EpStatus Error(const char* msg) { return {ep_error::kGeneric, msg, nullptr}; }

  // Issue #38: Implementation uses thread_local to avoid heap leak.
  // The returned EpStatusDetail* points into a thread_local static —
  // it is valid until the next call to Error() on the SAME thread.
  // This is safe because EP dispatch is single-threaded per GPU.
  static EpStatus Error(const std::string& msg, std::vector<int> failed = {}) {
    thread_local EpStatusDetail tl_detail;
    tl_detail.error_msg = msg;
    tl_detail.failed_ranks = std::move(failed);
    return {ep_error::kBackendError, tl_detail.error_msg.c_str(), &tl_detail};
  }
};

// Error code constants for common fast-path checks
namespace ep_error {
  constexpr int kOk              = 0;
  constexpr int kBufferOverflow  = 1;
  constexpr int kTimeout         = 2;
  constexpr int kDtypeMismatch   = 3;
  constexpr int kBackendError    = 4;
  constexpr int kRankFailure     = 5;
  constexpr int kGeneric         = 6;  // Issue #44: catch-all for Error(const char*)
}  // namespace ep_error


// --- Configuration -----------------------------------------------------

struct EpGroupConfig {
  Backend   backend;
  int       num_experts;           // total experts across all ranks
  int       num_local_experts;     // experts on this rank
  int       top_k;                 // router top-K
  int       hidden_dim;            // model hidden dimension

  // Issue #11: Dtype is now a RUNTIME parameter — inferred from the hidden
  // tensor passed to dispatch/combine at call time. This field sizes the
  // buffer for the largest element type that will ever be dispatched.
  // Use 2 for BF16-only workloads, 2 for mixed FP8+BF16 (BF16 combine
  // dominates), 1 if you are certain only FP8 will be used on both paths.
  int       max_dispatch_elem_size;  // bytes per element for buffer sizing (default: 2)

  // Buffer sizing — covers BOTH HT and LL modes in a single allocation
  // Call get_buffer_size_hint() to compute these.
  size_t    nvl_buffer_bytes;
  size_t    rdma_buffer_bytes;

  // DeepEP-specific (ignored by NCCL-EP)
  // Issue #13: num_qps_per_rank MUST equal num_local_experts for LL mode.
  // DeepEP's LL kernel assigns exactly one QP per local expert.
  // Default of 0 means auto-set to num_local_experts at create_group() time.
  // See: https://github.com/deepseek-ai/DeepEP/issues/46
  int       num_qps_per_rank;      // RDMA queue pairs (0 = auto = num_local_experts)
  int       num_sms;               // SM budget for HT kernels (0 = auto)

  // NCCL-EP-specific (ignored by DeepEP)
  bool      enable_pcie_fallback;  // graceful PCIe degradation

  // CUDA graph support (Issue #4)
  int       cuda_graph_max_tokens; // 0 = disabled. If > 0, enables graph capture
                                   // with static output tensor allocation at this
                                   // upper bound. Required for LL graph capture.
                                   // HT mode is NOT graph-capturable (see 4.6).

  // Timeout for blocking operations (Issue #7)
  int       timeout_ms;            // default: 30000 (30s). 0 = no timeout.

  // Issue #50: device_id propagated from EpCommunicator at create_group time.
  // Used by make_view_3d/make_view_2d to tag output EpTensorViews correctly.
  int       device_id;             // CUDA device ordinal (set by create_group)
};


// --- Opaque handles ----------------------------------------------------

struct EpGroup;

/// Issue #23: EpHandle now owns the deferred recv callback directly,
/// avoiding std::function heap allocation on every dispatch.
/// The deferred_fn pointer + opaque context replace std::function<EpStatus()>.
///
/// Issue #33: The deferred context is stored INLINE in a small union
/// (16 bytes) rather than via a heap-allocated void*. Both backends'
/// context types fit: DeepEP hook closure is a single std::function*
/// (8 bytes), NCCL-EP NcclDeferCtx is {ncclEpHandle_t, cudaStream_t}
/// (16 bytes). This eliminates the `new`/`delete` per dispatch that
/// contradicted the zero-heap-alloc goal of Issue #23.
struct EpHandle {
  void*   impl_handle;             // backend-specific handle (deep_ep or ncclEp)
  void*   group;                   // back-pointer to owning EpGroup

  // Issue #23 + #33: Deferred recv callback stored fully inline.
  // The context union holds backend-specific state without heap alloc.
  EpStatus (*deferred_fn)(void* ctx);  // function pointer (no std::function)

  // Issue #33: Inline context union — 16 bytes covers both backends.
  // DeepEP: stores the hook_fn pointer (8 bytes).
  // NCCL-EP: stores {ncclEpHandle_t, cudaStream_t} (16 bytes).
  // No `new`/`delete` on the dispatch hot path.
  static constexpr size_t kDeferCtxSize = 16;
  alignas(8) char deferred_ctx_storage[kDeferCtxSize];

  template <typename T>
  T& deferred_ctx_as() {
    static_assert(sizeof(T) <= kDeferCtxSize, "deferred context too large");
    return *reinterpret_cast<T*>(deferred_ctx_storage);
  }

  // Scratch slot index for ping-pong buffer tracking (Issue #34).
  // Saved at dispatch time, read back at combine time to ensure
  // combine uses the same scratch buffer as its paired dispatch.
  int scratch_slot;

  // Invoke deferred receive. Returns Ok if no hook was registered.
  EpStatus invoke_deferred() {
    if (deferred_fn) return deferred_fn(deferred_ctx_storage);
    return EpStatus::Ok();
  }
};


// --- Routing Cache (Issue #39) -------------------------------------------
//
// NCCL-EP's ncclEpCreateHandle() builds a per-dispatch routing table
// (~1-2 μs). In static CUDA graph replay or steady-state decode where
// topk_idx is structurally identical across steps, this work is redundant.
//
// RoutingCache is an opaque handle that backends can populate once and
// reuse across multiple dispatches with the same routing topology.
//
// Lifecycle:
//   1. Framework calls create_routing_cache(group, topk_idx) ONCE.
//   2. Passes the cache to low_latency_dispatch() on each step.
//   3. Backend skips ncclEpCreateHandle() when cache is valid.
//   4. Framework calls destroy_routing_cache() when topology changes
//      or at shutdown.
//
// DeepEP: no-op (routing is implicit in the NVSHMEM put addresses).
// NCCL-EP: caches the ncclEpHandle_t for reuse.

struct RoutingCache {
  void*    impl_cache;             // backend-specific cached state
  void*    group;                  // back-pointer to owning EpGroup
  int64_t  cached_shape[2];        // Issue #43A: topk_idx shape at creation time
  uint64_t topology_hash;          // Issue #43: content-sampled hash (computed ONCE
                                   // at create_routing_cache time via D2H; NEVER
                                   // recomputed on the hot path)
  bool     valid;                  // false after invalidation
};


// --- Async primitives (Issue #3: split EventSync) ----------------------

/// StreamDep: thin wrapper around CUDA stream ordering.
/// Used to sequence dispatch -> expert compute -> combine across streams.
/// Both backends implement this via cudaEvent_t under the hood.
struct StreamDep {
  void* cuda_event;                // opaque cudaEvent_t
  void* stream;                    // stream the event was recorded on
};


// --- Tensor tags for NCCL-EP interop -----------------------------------

enum class TensorTag : int {
  kTokens             = 0,
  kTopkIdx            = 1,
  kTopkWeights        = 2,
  kScales             = 3,
  kExpertCountDevice  = 4,
  kExpertCountHost    = 5,
};

}  // namespace ep
}  // namespace flashinfer
