#pragma once
#include "flashinfer/ep/types.h"

namespace flashinfer {
namespace ep {

/// Create a communication group. Collective across all ranks.
/// A single group supports BOTH HT and LL modes (Issue #2).
/// The active mode is determined by which API is called.
///
/// @param config      Fully populated configuration struct
/// @param comm        Backend-agnostic communicator (Issue #6)
/// @param stream      CUDA stream for communication operations
/// @param[out] status Error status
/// @return            Opaque group handle (caller owns), nullptr on failure
EpGroup* create_group(const EpGroupConfig& config,
                      const EpCommunicator& comm,
                      void* /*cudaStream_t*/ stream,
                      EpStatus* status);

/// Destroy group and release all resources. Collective.
EpStatus destroy_group(EpGroup* group);

/// Query buffer size hints. Call BEFORE create_group to populate config.
/// Returns sizes large enough for BOTH HT and LL modes (Issue #2).
///
/// Issue #11: Uses max_elem_size (bytes per element) to compute worst-case
/// buffer requirements. Pass 2 for BF16, 1 for FP8-only, or 2 if you
/// plan to dispatch both FP8 and BF16 (combine is typically BF16).
///
/// Memory characteristics by backend:
///   DeepEP:  O(E*B*P*elem_size) -- expert-indexed, larger
///   NCCL-EP: O(N*B*P*elem_size + B*K*P*elem_size) -- rank-indexed, ~14x smaller
size_t get_nvl_buffer_size_hint(Backend backend,
                                int hidden_dim, int world_size,
                                int num_experts, int top_k, int max_tokens,
                                int max_elem_size = 2);
size_t get_rdma_buffer_size_hint(Backend backend,
                                 int hidden_dim, int world_size,
                                 int num_experts, int top_k, int max_tokens,
                                 int max_elem_size = 2);

/// DeepEP-specific: precise RDMA buffer hint for low-latency mode.
/// Returns 0 for NCCL-EP backend.
size_t get_low_latency_rdma_size_hint(Backend backend,
                                      int max_tokens, int hidden_dim,
                                      int world_size, int num_experts,
                                      int max_elem_size = 2);

/// Query optimal kernel configuration for a given setup.
struct KernelConfig {
  int    recommended_num_sms;      // 0 = auto (NCCL-EP persistent kernel)
  size_t nvl_buffer_hint;
  size_t rdma_buffer_hint;
};
KernelConfig get_dispatch_config(Backend backend, int world_size,
                                 int hidden_dim, int num_experts);
KernelConfig get_combine_config(Backend backend, int world_size,
                                int hidden_dim, int num_experts);

/// Query active configuration of an existing group.
EpGroupConfig get_group_config(const EpGroup* group);

}  // namespace ep
}  // namespace flashinfer
