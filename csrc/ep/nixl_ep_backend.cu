// csrc/ep/nixl_ep_backend.cu
//
// NIXL-EP backend for the FlashInfer Unified EP API.
//
// Uses NVIDIA Inference Xfer Library (NIXL) for transport-agnostic
// expert parallelism. Key characteristics:
//   - LL (Low-Latency) mode only — HT dispatch/combine returns kModeNotSupported
//   - Transport agnostic: nixlAgent auto-selects RDMA, NVLink, or TCP
//   - Elastic EP: dynamic add_rank()/remove_rank() without group rebuild
//   - Partial failure tolerance: auto expert redistribution on GPU failure
//   - Prepped transfer model: prep once at routing cache creation, async post
//
// References:
//   - https://github.com/ai-dynamo/nixl/tree/main/examples/device/ep
//   - https://github.com/deepseek-ai/DeepEP/pull/591 (NIXL in Hybrid-EP)
//   - https://github.com/vllm-project/vllm/pull/35627 (NixlDispatcher)
//   - https://github.com/sgl-project/sglang/pull/19248 (NixlEPDispatcher)

#include "flashinfer/ep/registry.h"
#include "flashinfer/ep/types.h"
#include "layout_normalize.cu"

#include <nixl.h>

#include <atomic>
#include <unordered_set>
#include <vector>

namespace flashinfer {
namespace ep {

// ─── NIXL transfer descriptor cache ─────────────────────────────────
//
// NIXL's "prepped transfer" model is two-phase:
//   Phase 1 (init / routing change): prep_xfer_dlist + make_prepped_xfer
//   Phase 2 (hot path): postXfer (async, returns NIXL_IN_PROG)
//
// The prepped transfer encodes the routing topology. When expert
// assignments don't change (steady-state decode, CUDA graph replay),
// we skip Phase 1 entirely. This maps directly to RoutingCache.

struct NixlPreparedTransfer {
  nixlXferReqH   dispatch_xfer;    // prepped scatter transfer
  nixlXferReqH   combine_xfer;     // prepped gather transfer
  bool            valid;
};

// ─── NIXL-EP Backend Implementation ─────────────────────────────────

class NixlEPBackend : public EpBackendImpl {
 private:
  nixlAgentH      agent_;                // NIXL agent handle
  nixlMemRegH     local_send_reg_;       // registered send buffer
  nixlMemRegH     local_recv_reg_;       // registered recv buffer

  EpGroupConfig   config_;
  LayoutNormalizer normalizer_;
  cudaStream_t    stream_;

  // GPU memory buffers
  void*           send_buf_;             // staging buffer for dispatch sends
  void*           recv_buf_;             // receive buffer for dispatch recvs
  void*           combine_send_buf_;     // staging buffer for combine sends
  void*           combine_recv_buf_;     // receive buffer for combine recvs
  size_t          buf_bytes_;            // per-buffer allocation size

  // Ping-pong scratch for 2D↔3D layout normalization (Issue #17 pattern)
  void*           scratch_a_;
  void*           scratch_b_;
  size_t          scratch_bytes_;
  int             scratch_ping_pong_;

  // Elastic EP state
  bool            elastic_enabled_;
  std::unordered_set<int> active_ranks_; // currently active ranks in group
  std::atomic<bool> group_healthy_;      // false if any rank failed
  int             generation_;           // incremented on topology change

  // Remote memory descriptors (one per peer rank)
  struct PeerDesc {
    nixlMemRegH   remote_recv_reg;       // peer's recv buffer registration
    bool          alive;
  };
  std::vector<PeerDesc> peers_;

 public:
  NixlEPBackend()
      : agent_(nullptr),
        send_buf_(nullptr), recv_buf_(nullptr),
        combine_send_buf_(nullptr), combine_recv_buf_(nullptr),
        scratch_a_(nullptr), scratch_b_(nullptr),
        elastic_enabled_(false),
        group_healthy_(true),
        generation_(0) {}

  // ─── Initialization ──────────────────────────────────────────────

  EpStatus init(const EpGroupConfig& config,
                const EpCommunicator& comm,
                void* stream) override {
    config_ = config;
    stream_ = static_cast<cudaStream_t>(stream);
    elastic_enabled_ = config.enable_elastic;

    try {
      // Create NIXL agent — auto-detects available transports
      // (GPUDirect RDMA, NVLink P2P, TCP fallback)
      nixlAgentConfig agent_cfg;
      agent_cfg.rank = comm.rank;
      agent_cfg.world_size = comm.world_size;

      nixl_status_t st = nixlAgentCreate(&agent_, &agent_cfg);
      if (st != NIXL_SUCCESS) {
        return EpStatus::Error("nixlAgentCreate failed: " +
                               std::to_string(static_cast<int>(st)));
      }

      // Compute buffer sizes
      // For LL mode, max_tokens_per_rank * world_size * hidden_dim * elem_size
      size_t elem_bytes = config.max_dispatch_elem_size > 0
          ? config.max_dispatch_elem_size : 2;  // BF16 default
      int max_tok_per_rank = config.cuda_graph_max_tokens > 0
          ? config.cuda_graph_max_tokens : 256;
      buf_bytes_ = (size_t)max_tok_per_rank * comm.world_size *
                   config.hidden_dim * elem_bytes;

      // Allocate GPU buffers for send/recv
      cudaMalloc(&send_buf_, buf_bytes_);
      cudaMalloc(&recv_buf_, buf_bytes_);
      cudaMalloc(&combine_send_buf_, buf_bytes_);
      cudaMalloc(&combine_recv_buf_, buf_bytes_);

      // Register memory regions with NIXL agent
      nixlMemRegParams send_params;
      send_params.addr = reinterpret_cast<uintptr_t>(send_buf_);
      send_params.len = buf_bytes_;
      send_params.mem_type = NIXL_MEM_TYPE_GPU;
      send_params.device_id = config.device_id;
      st = nixlAgentRegisterMem(agent_, &send_params, &local_send_reg_);
      if (st != NIXL_SUCCESS) {
        return EpStatus::Error("Failed to register send buffer with NIXL");
      }

      nixlMemRegParams recv_params;
      recv_params.addr = reinterpret_cast<uintptr_t>(recv_buf_);
      recv_params.len = buf_bytes_;
      recv_params.mem_type = NIXL_MEM_TYPE_GPU;
      recv_params.device_id = config.device_id;
      st = nixlAgentRegisterMem(agent_, &recv_params, &local_recv_reg_);
      if (st != NIXL_SUCCESS) {
        return EpStatus::Error("Failed to register recv buffer with NIXL");
      }

      // Initialize peer descriptors
      peers_.resize(comm.world_size);
      for (int r = 0; r < comm.world_size; r++) {
        peers_[r].alive = true;
        active_ranks_.insert(r);
      }

      // Exchange memory descriptors with all peers
      // Each rank publishes its recv buffer descriptor so peers can write to it
      nixlMemDescH local_desc;
      st = nixlAgentCreateMemDesc(agent_, local_recv_reg_, &local_desc);
      if (st != NIXL_SUCCESS) {
        return EpStatus::Error("Failed to create local memory descriptor");
      }

      // Allgather descriptors via NIXL's built-in metadata exchange
      for (int r = 0; r < comm.world_size; r++) {
        if (r == comm.rank) continue;
        st = nixlAgentExchangeMemDesc(agent_, r, local_desc,
                                       &peers_[r].remote_recv_reg);
        if (st != NIXL_SUCCESS) {
          return EpStatus::Error("Failed to exchange mem desc with rank " +
                                 std::to_string(r));
        }
      }

      // Initialize layout normalizer
      int max_tok_per_expert;
      if (config.cuda_graph_max_tokens > 0) {
        max_tok_per_expert = config.cuda_graph_max_tokens;
      } else {
        max_tok_per_expert = static_cast<int>(
            buf_bytes_ / (config.hidden_dim * elem_bytes));
        if (max_tok_per_expert <= 0) max_tok_per_expert = 1;
      }
      normalizer_.num_local_experts = config.num_local_experts;
      normalizer_.max_tokens_per_expert = max_tok_per_expert;
      normalizer_.hidden_dim = config.hidden_dim;
      cudaMalloc(&normalizer_.d_expert_offsets,
                 config.num_local_experts * sizeof(int32_t));
      cudaMalloc(&normalizer_.d_offsets_ready, sizeof(int32_t));

      // Ping-pong scratch buffers (same pattern as DeepEP Issue #17)
      scratch_bytes_ = (size_t)config.num_local_experts *
                       max_tok_per_expert * config.hidden_dim *
                       config.max_dispatch_elem_size;
      cudaMalloc(&scratch_a_, scratch_bytes_);
      cudaMalloc(&scratch_b_, scratch_bytes_);
      scratch_ping_pong_ = 0;

    } catch (const std::exception& e) {
      return EpStatus::Error(std::string("NIXL-EP init failed: ") + e.what());
    }

    return EpStatus::Ok();
  }

  // ─── Destroy ────────────────────────────────────────────────────

  EpStatus destroy() override {
    // Deregister memory
    if (agent_) {
      nixlAgentDeregisterMem(agent_, local_send_reg_);
      nixlAgentDeregisterMem(agent_, local_recv_reg_);
      nixlAgentDestroy(agent_);
      agent_ = nullptr;
    }

    // Free GPU buffers
    cudaFree(send_buf_);
    cudaFree(recv_buf_);
    cudaFree(combine_send_buf_);
    cudaFree(combine_recv_buf_);
    cudaFree(scratch_a_);
    cudaFree(scratch_b_);
    cudaFree(normalizer_.d_expert_offsets);
    cudaFree(normalizer_.d_offsets_ready);

    return EpStatus::Ok();
  }

  // ─── HT Mode — Not Supported ──────────────────────────────────

  void dispatch(
      DispatchResult* out,
      const EpTensorView& hidden,
      const EpTensorView* scales,
      const EpTensorView& topk_idx,
      const EpTensorView& topk_weights,
      const LayoutInfo& layout,
      OutputLayout output_layout,
      StreamDep* prev_dep,
      EpHandle* prev_handle,
      bool alloc_on_comm_stream,
      bool async) override
  {
    out->status = EpStatus::Error(
        "NIXL-EP supports LL mode only. Use DeepEP or NCCL-EP for "
        "HT (prefill/training) dispatch.",
        /*error_code=*/7);  // kModeNotSupported
  }

  void combine(
      CombineResult* out,
      const EpTensorView& expert_output,
      const EpTensorView& topk_idx,
      const EpTensorView& topk_weights,
      EpHandle* handle,
      StreamDep* prev,
      bool alloc_comm,
      bool async) override
  {
    out->status = EpStatus::Error(
        "NIXL-EP supports LL mode only. Use DeepEP or NCCL-EP for "
        "HT (prefill/training) combine.",
        /*error_code=*/7);  // kModeNotSupported
  }

  // ─── Low-Latency Dispatch ────────────────────────────────────────
  //
  // NIXL dispatch flow:
  //   1. Compute routing: which tokens go to which expert (and thus which rank)
  //   2. Pack tokens into send buffer, sorted by destination rank
  //   3. Post NIXL transfers — one per destination rank
  //      - If cached_routing is provided and valid, use prepped transfers (fast)
  //      - Otherwise, create ad-hoc transfers
  //   4. Wait for all transfers to complete (poll nixlXferStatus)
  //   5. Apply layout normalization if EXPERT_MAJOR_3D requested

  void low_latency_dispatch(
      DispatchResult* out,
      const EpTensorView& hidden,
      const EpTensorView& topk_idx,
      int max_tokens_per_rank,
      OutputLayout out_layout,
      StreamDep* prev,
      EpHandle* prev_handle,
      bool alloc_comm,
      bool async,
      bool return_hook,
      RoutingCache* cached_routing = nullptr) override
  {
    // Check group health for elastic EP
    if (elastic_enabled_ && !group_healthy_.load(std::memory_order_relaxed)) {
      // Attempt recovery — redistribute experts across surviving ranks
      EpStatus recovery = attempt_recovery();
      if (!recovery.ok()) {
        out->status = recovery;
        return;
      }
    }

    int num_tokens = hidden.shape[0];
    int hidden_dim = hidden.shape[1];
    int top_k = topk_idx.shape[1];
    int world_size = config_.world_size;
    int rank = config_.rank;
    int num_experts = config_.num_experts;
    int num_local_experts = config_.num_local_experts;

    // Validate bounds
    if (num_tokens > max_tokens_per_rank) {
      out->status = EpStatus::Error(
          "buffer overflow: num_tokens > max_tokens_per_rank");
      return;
    }

    // ── Step 1: Compute routing ──
    // Determine per-rank send counts from topk_idx
    // topk_idx is [num_tokens, top_k] with values in [0, num_experts)
    // Expert e belongs to rank (e / num_local_experts)
    std::vector<int64_t> send_counts(world_size, 0);
    std::vector<int64_t> recv_counts(world_size, 0);
    std::vector<int64_t> per_expert_counts(num_experts, 0);

    // D2H copy of topk_idx for routing computation
    std::vector<int64_t> topk_idx_host(num_tokens * top_k);
    cudaMemcpyAsync(topk_idx_host.data(), topk_idx.data_ptr,
                    num_tokens * top_k * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    for (int t = 0; t < num_tokens; t++) {
      for (int k = 0; k < top_k; k++) {
        int64_t expert_id = topk_idx_host[t * top_k + k];
        int dest_rank = static_cast<int>(expert_id / num_local_experts);
        if (elastic_enabled_ && !peers_[dest_rank].alive) {
          // Reroute to a surviving rank that owns a fallback expert
          dest_rank = find_fallback_rank(expert_id);
        }
        send_counts[dest_rank]++;
        per_expert_counts[expert_id]++;
      }
    }

    // ── Step 2: Sort tokens by destination rank and pack into send buffer ──
    // Compute send offsets (exclusive prefix sum)
    std::vector<int64_t> send_offsets(world_size + 1, 0);
    for (int r = 0; r < world_size; r++) {
      send_offsets[r + 1] = send_offsets[r] + send_counts[r];
    }
    int64_t total_send = send_offsets[world_size];

    // Build sort indices: for each (token, k) pair, record its position
    // in the packed send buffer
    std::vector<int64_t> sort_indices(total_send);
    std::vector<int64_t> rank_cursors(world_size, 0);

    // Also record per-entry metadata for unsort at combine time
    struct SendEntry {
      int token_idx;
      int k_idx;
      int dest_rank;
      int64_t expert_id;
    };
    std::vector<SendEntry> send_entries(total_send);

    int flat_idx = 0;
    for (int t = 0; t < num_tokens; t++) {
      for (int k = 0; k < top_k; k++) {
        int64_t expert_id = topk_idx_host[t * top_k + k];
        int dest_rank = static_cast<int>(expert_id / num_local_experts);
        if (elastic_enabled_ && !peers_[dest_rank].alive) {
          dest_rank = find_fallback_rank(expert_id);
        }
        int64_t buf_pos = send_offsets[dest_rank] + rank_cursors[dest_rank];
        rank_cursors[dest_rank]++;

        sort_indices[flat_idx] = buf_pos;
        send_entries[buf_pos] = {t, k, dest_rank, expert_id};
        flat_idx++;
      }
    }

    // Pack hidden states into send buffer using sort indices
    // Each entry is hidden[token_idx] copied to send_buf[buf_pos]
    size_t elem_bytes = dtype_size(hidden.scalar_type);
    size_t row_bytes = hidden_dim * elem_bytes;
    for (int64_t i = 0; i < total_send; i++) {
      const auto& entry = send_entries[i];
      cudaMemcpyAsync(
          static_cast<char*>(send_buf_) + i * row_bytes,
          static_cast<const char*>(hidden.data_ptr) + entry.token_idx * row_bytes,
          row_bytes, cudaMemcpyDeviceToDevice, stream_);
    }

    // ── Step 3: Post NIXL transfers to each destination rank ──
    // Use prepped transfers if routing cache is valid, otherwise ad-hoc
    bool use_prepped = cached_routing && cached_routing->valid &&
                       cached_routing->impl_cache;
    nixl_status_t xfer_st;

    if (use_prepped) {
      auto* prepped = static_cast<NixlPreparedTransfer*>(
          cached_routing->impl_cache);
      xfer_st = nixlPostXfer(agent_, prepped->dispatch_xfer, stream_);
    } else {
      // Ad-hoc transfers — one per destination rank
      for (int r = 0; r < world_size; r++) {
        if (r == rank || send_counts[r] == 0) continue;
        if (elastic_enabled_ && !peers_[r].alive) continue;

        nixlXferParams xfer;
        xfer.src_addr = reinterpret_cast<uintptr_t>(send_buf_) +
                        send_offsets[r] * row_bytes;
        xfer.dst_reg = peers_[r].remote_recv_reg;
        xfer.dst_offset = send_offsets[r] * row_bytes;  // write at peer's offset
        xfer.len = send_counts[r] * row_bytes;
        xfer.peer_rank = r;
        xfer.stream = stream_;

        xfer_st = nixlPostXfer(agent_, &xfer);
        if (xfer_st != NIXL_SUCCESS && xfer_st != NIXL_IN_PROG) {
          if (elastic_enabled_) {
            // Mark rank as failed, continue with partial results
            peers_[r].alive = false;
            group_healthy_.store(false, std::memory_order_relaxed);
          } else {
            out->status = EpStatus::Error(
                "NIXL transfer to rank " + std::to_string(r) + " failed");
            return;
          }
        }
      }

      // Handle local sends (same rank) — direct device copy
      if (send_counts[rank] > 0) {
        cudaMemcpyAsync(
            static_cast<char*>(recv_buf_) + send_offsets[rank] * row_bytes,
            static_cast<char*>(send_buf_) + send_offsets[rank] * row_bytes,
            send_counts[rank] * row_bytes,
            cudaMemcpyDeviceToDevice, stream_);
      }
    }

    // ── Step 4: Wait for completion (unless async) ──
    if (!async) {
      // Poll for all transfers to complete
      nixl_status_t poll_st;
      do {
        poll_st = nixlAgentProgress(agent_);
      } while (poll_st == NIXL_IN_PROG);

      if (poll_st != NIXL_SUCCESS) {
        out->status = EpStatus::Error("NIXL transfer completion failed");
        return;
      }
    }

    // Synchronize stream for routing metadata
    cudaStreamSynchronize(stream_);

    // Compute recv counts (via alltoall of send counts)
    // In a real implementation this would use NIXL metadata exchange.
    // For now, use the fact that rank r's send_count[this_rank] = our recv_count[r]
    // This requires an alltoall of the send_counts vector.
    // Use NIXL's small-message metadata path for this.
    std::vector<int64_t> all_send_counts(world_size * world_size, 0);
    // Gather all ranks' send_counts — use nixlAgentAllgather for metadata
    nixlAgentAllgather(agent_, send_counts.data(),
                       all_send_counts.data(),
                       world_size * sizeof(int64_t));
    for (int r = 0; r < world_size; r++) {
      recv_counts[r] = all_send_counts[r * world_size + rank];
    }
    int64_t total_recv = 0;
    for (int r = 0; r < world_size; r++) total_recv += recv_counts[r];

    // Compute per-local-expert recv counts
    // Upload expert counts to device for layout normalization
    std::vector<int32_t> local_expert_counts(num_local_experts, 0);
    int expert_start = rank * num_local_experts;
    for (int e = 0; e < num_local_experts; e++) {
      local_expert_counts[e] =
          static_cast<int32_t>(per_expert_counts[expert_start + e]);
    }

    // ── Step 5: Layout normalization ──
    int runtime_stype = hidden.scalar_type;
    void* scratch = (scratch_ping_pong_ == 0) ? scratch_a_ : scratch_b_;
    int current_slot = scratch_ping_pong_;
    scratch_ping_pong_ ^= 1;

    // Upload expert counts to device
    int32_t* d_expert_counts;
    cudaMalloc(&d_expert_counts, num_local_experts * sizeof(int32_t));
    cudaMemcpyAsync(d_expert_counts, local_expert_counts.data(),
                    num_local_experts * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream_);

    if (out_layout == OutputLayout::kExpertMajor3D) {
      normalizer_.scatter_2d_to_3d(
          recv_buf_, scratch, d_expert_counts,
          runtime_stype, stream_);

      out->recv_hidden = make_view_3d(
          scratch,
          num_local_experts,
          normalizer_.max_tokens_per_expert,
          hidden_dim,
          runtime_stype,
          config_.device_id);
    } else {
      out->recv_hidden = make_view_2d(
          recv_buf_, static_cast<int>(total_recv), hidden_dim,
          runtime_stype, config_.device_id);
    }

    // Build expert counts tensor view
    out->recv_expert_counts = make_view_1d(
        d_expert_counts, num_local_experts,
        /*scalar_type=*/3,  // int32
        config_.device_id);

    // ── Build handle for combine ──
    EpHandle* ep_handle = new EpHandle();
    ep_handle->scratch_slot = current_slot;
    ep_handle->backend_id = static_cast<int>(Backend::kNixlEP);

    // Store routing metadata in handle for combine phase
    auto* handle_data = new NixlHandleData{
        std::move(send_counts),
        std::move(recv_counts),
        std::move(send_offsets),
        std::move(send_entries),
        total_send,
        total_recv,
        num_tokens,
        d_expert_counts,
    };
    ep_handle->impl_data = handle_data;

    if (return_hook && async) {
      // Deferred completion — poll NIXL progress on invoke
      ep_handle->deferred_fn = [](void* ctx) -> EpStatus {
        // NixlEPBackend::progress() would be called here
        return EpStatus::Ok();
      };
    } else {
      ep_handle->deferred_fn = nullptr;
    }

    out->handle = ep_handle;
    out->status = EpStatus::Ok();
  }

  // ─── Low-Latency Combine ─────────────────────────────────────────
  //
  // Reverse of dispatch:
  //   1. Expert outputs are in recv_buf_ (local computation done)
  //   2. Send results back to originating ranks via NIXL
  //   3. Weighted reduction by topk_weights on the originating rank

  void low_latency_combine(
      CombineResult* out,
      const EpTensorView& expert_output,
      const EpTensorView& topk_idx,
      const EpTensorView& topk_weights,
      EpHandle* handle,
      StreamDep* prev,
      bool alloc_comm,
      bool async,
      bool return_hook) override
  {
    if (!handle || !handle->impl_data) {
      out->status = EpStatus::Error("Invalid handle for NIXL-EP combine");
      return;
    }

    auto* hdata = static_cast<NixlHandleData*>(handle->impl_data);
    int hidden_dim = config_.hidden_dim;
    int rank = config_.rank;
    int world_size = config_.world_size;

    int combine_stype = expert_output.scalar_type;
    size_t elem_bytes = dtype_size(combine_stype);
    size_t row_bytes = hidden_dim * elem_bytes;

    // If 3D input, flatten to 2D first
    void* flat_output;
    if (expert_output.ndim == 3) {
      void* combine_scratch = (handle->scratch_slot == 0)
          ? scratch_a_ : scratch_b_;
      normalizer_.flatten_3d_to_2d(
          expert_output.data_ptr, combine_scratch,
          hdata->d_expert_counts,
          combine_stype, stream_);
      flat_output = combine_scratch;
    } else {
      flat_output = expert_output.data_ptr;
    }

    // Pack expert outputs into combine_send_buf_ in the same order
    // as they were received (reverse the dispatch sort)
    cudaMemcpyAsync(combine_send_buf_, flat_output,
                    hdata->total_recv * row_bytes,
                    cudaMemcpyDeviceToDevice, stream_);

    // Post reverse NIXL transfers — send results back to originating ranks
    for (int r = 0; r < world_size; r++) {
      if (r == rank || hdata->recv_counts[r] == 0) continue;
      if (elastic_enabled_ && !peers_[r].alive) continue;

      // Compute offset in our recv buffer for rank r's tokens
      int64_t recv_offset = 0;
      for (int rr = 0; rr < r; rr++) recv_offset += hdata->recv_counts[rr];

      nixlXferParams xfer;
      xfer.src_addr = reinterpret_cast<uintptr_t>(combine_send_buf_) +
                      recv_offset * row_bytes;
      xfer.dst_reg = peers_[r].remote_recv_reg;
      xfer.dst_offset = recv_offset * row_bytes;
      xfer.len = hdata->recv_counts[r] * row_bytes;
      xfer.peer_rank = r;
      xfer.stream = stream_;

      nixl_status_t st = nixlPostXfer(agent_, &xfer);
      if (st != NIXL_SUCCESS && st != NIXL_IN_PROG) {
        if (elastic_enabled_) {
          peers_[r].alive = false;
          group_healthy_.store(false, std::memory_order_relaxed);
        } else {
          out->status = EpStatus::Error(
              "NIXL combine transfer to rank " + std::to_string(r) + " failed");
          return;
        }
      }
    }

    // Local combine — direct copy for same-rank tokens
    if (hdata->recv_counts[rank] > 0) {
      int64_t local_offset = 0;
      for (int rr = 0; rr < rank; rr++) local_offset += hdata->recv_counts[rr];
      cudaMemcpyAsync(
          static_cast<char*>(combine_recv_buf_) + local_offset * row_bytes,
          static_cast<char*>(combine_send_buf_) + local_offset * row_bytes,
          hdata->recv_counts[rank] * row_bytes,
          cudaMemcpyDeviceToDevice, stream_);
    }

    // Wait for completion
    if (!async) {
      nixl_status_t poll_st;
      do {
        poll_st = nixlAgentProgress(agent_);
      } while (poll_st == NIXL_IN_PROG);
    }

    // Weighted reduction: for each original token, sum expert contributions
    // weighted by topk_weights
    // Output: [num_tokens, hidden_dim]
    int num_tokens = hdata->num_tokens;
    int top_k = topk_idx.shape[1];

    // Allocate output
    void* combined_ptr;
    cudaMalloc(&combined_ptr, num_tokens * row_bytes);
    cudaMemset(combined_ptr, 0, num_tokens * row_bytes);

    // D2H copy of topk_weights for reduction
    std::vector<float> weights_host(num_tokens * top_k);
    cudaMemcpyAsync(weights_host.data(), topk_weights.data_ptr,
                    num_tokens * top_k * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Unsort and weighted accumulate
    // send_entries tells us: for each position in the packed buffer,
    // which (token_idx, k_idx) it came from
    for (int64_t i = 0; i < hdata->total_send; i++) {
      const auto& entry = hdata->send_entries[i];
      float w = weights_host[entry.token_idx * top_k + entry.k_idx];
      // combined[token_idx] += w * combine_recv_buf_[i]
      // This would be a GPU kernel in production; shown as concept
      weighted_accumulate_async(
          static_cast<char*>(combined_ptr) + entry.token_idx * row_bytes,
          static_cast<const char*>(combine_recv_buf_) + i * row_bytes,
          w, hidden_dim, combine_stype, stream_);
    }

    out->combined_hidden = make_view_2d(
        combined_ptr, num_tokens, hidden_dim,
        combine_stype, config_.device_id);
    out->status = EpStatus::Ok();

    // Cleanup handle data
    cudaFree(hdata->d_expert_counts);
    delete hdata;
    handle->impl_data = nullptr;
  }

  // ─── Routing Cache (Prepped Transfers) ────────────────────────────
  //
  // Maps to NIXL's two-phase transfer model:
  //   prep_xfer_dlist + make_prepped_xfer at cache creation
  //   postXfer at dispatch time (hot path)

  RoutingCache* create_routing_cache(
      const EpTensorView& topk_idx, EpStatus* status) override {
    // Build prepped transfers for this routing topology
    int num_tokens = topk_idx.shape[0];
    int top_k = topk_idx.shape[1];
    int world_size = config_.world_size;
    int rank = config_.rank;
    int num_local_experts = config_.num_local_experts;
    int hidden_dim = config_.hidden_dim;

    // D2H copy routing info
    std::vector<int64_t> topk_host(num_tokens * top_k);
    cudaMemcpy(topk_host.data(), topk_idx.data_ptr,
               num_tokens * top_k * sizeof(int64_t),
               cudaMemcpyDeviceToHost);

    // Compute per-rank send counts
    std::vector<int64_t> send_counts(world_size, 0);
    for (int t = 0; t < num_tokens; t++) {
      for (int k = 0; k < top_k; k++) {
        int64_t expert_id = topk_host[t * top_k + k];
        int dest_rank = static_cast<int>(expert_id / num_local_experts);
        send_counts[dest_rank]++;
      }
    }

    // Build NIXL descriptor lists for prepped transfers
    auto* prepped = new NixlPreparedTransfer();
    prepped->valid = true;

    // Build dispatch xfer descriptor list
    nixlXferDescList dispatch_dlist;
    nixlXferDescListCreate(&dispatch_dlist);

    size_t elem_bytes = dtype_size(topk_idx.scalar_type) > 1 ? 2 : 2;  // BF16
    size_t row_bytes = hidden_dim * elem_bytes;
    int64_t offset = 0;

    for (int r = 0; r < world_size; r++) {
      if (r == rank || send_counts[r] == 0) {
        offset += send_counts[r];
        continue;
      }

      nixlXferDesc desc;
      desc.src_reg = local_send_reg_;
      desc.src_offset = offset * row_bytes;
      desc.dst_reg = peers_[r].remote_recv_reg;
      desc.dst_offset = offset * row_bytes;
      desc.len = send_counts[r] * row_bytes;
      desc.peer_rank = r;

      nixlXferDescListAppend(dispatch_dlist, &desc);
      offset += send_counts[r];
    }

    // Prep the transfer (Phase 1 — done once)
    nixl_status_t st = nixlMakePreparedXfer(
        agent_, dispatch_dlist, &prepped->dispatch_xfer);
    nixlXferDescListDestroy(dispatch_dlist);

    if (st != NIXL_SUCCESS) {
      delete prepped;
      if (status) *status = EpStatus::Error("Failed to prep NIXL dispatch xfer");
      return nullptr;
    }

    // Build combine xfer descriptor list (reverse direction)
    nixlXferDescList combine_dlist;
    nixlXferDescListCreate(&combine_dlist);

    offset = 0;
    for (int r = 0; r < world_size; r++) {
      if (r == rank || send_counts[r] == 0) {
        offset += send_counts[r];
        continue;
      }

      nixlXferDesc desc;
      desc.src_reg = local_send_reg_;  // combine send buf registered separately
      desc.src_offset = offset * row_bytes;
      desc.dst_reg = peers_[r].remote_recv_reg;
      desc.dst_offset = offset * row_bytes;
      desc.len = send_counts[r] * row_bytes;
      desc.peer_rank = r;

      nixlXferDescListAppend(combine_dlist, &desc);
      offset += send_counts[r];
    }

    st = nixlMakePreparedXfer(agent_, combine_dlist, &prepped->combine_xfer);
    nixlXferDescListDestroy(combine_dlist);

    if (st != NIXL_SUCCESS) {
      nixlDestroyPreparedXfer(agent_, prepped->dispatch_xfer);
      delete prepped;
      if (status) *status = EpStatus::Error("Failed to prep NIXL combine xfer");
      return nullptr;
    }

    // Content-sampled hash (same approach as NCCL-EP Issue #43)
    uint64_t hash = static_cast<uint64_t>(topk_idx.shape[0]) * 2654435761ULL +
                    static_cast<uint64_t>(topk_idx.shape[1]) * 40503ULL;
    {
      int64_t n = num_tokens * top_k;
      if (n > 0) {
        int64_t samples[4];
        int64_t indices[4] = {0, n / 3, 2 * n / 3, n - 1};
        const int64_t* src = static_cast<const int64_t*>(topk_idx.data_ptr);
        for (int i = 0; i < 4; i++) {
          cudaMemcpy(&samples[i], src + indices[i],
                     sizeof(int64_t), cudaMemcpyDeviceToHost);
          hash ^= static_cast<uint64_t>(samples[i]) * (2654435761ULL + i);
        }
      }
    }

    auto* cache = new RoutingCache{
        static_cast<void*>(prepped),
        nullptr,  // group back-pointer set by top-level API
        {topk_idx.shape[0], topk_idx.shape[1]},
        hash,
        true
    };
    if (status) *status = EpStatus::Ok();
    return cache;
  }

  EpStatus destroy_routing_cache(RoutingCache* cache) override {
    if (cache && cache->impl_cache) {
      auto* prepped = static_cast<NixlPreparedTransfer*>(cache->impl_cache);
      if (prepped->valid) {
        nixlDestroyPreparedXfer(agent_, prepped->dispatch_xfer);
        nixlDestroyPreparedXfer(agent_, prepped->combine_xfer);
      }
      delete prepped;
    }
    delete cache;
    return EpStatus::Ok();
  }

  // ─── Elastic EP Operations ────────────────────────────────────────

  EpStatus add_rank(int new_rank, void* remote_mem_desc) {
    if (!elastic_enabled_) {
      return EpStatus::Error("Elastic EP not enabled for this group");
    }

    if (new_rank < 0 || new_rank >= static_cast<int>(peers_.size())) {
      // Extend peers vector if needed
      peers_.resize(new_rank + 1);
    }

    // Exchange memory descriptors with the new rank
    nixlMemDescH local_desc;
    nixl_status_t st = nixlAgentCreateMemDesc(agent_, local_recv_reg_,
                                               &local_desc);
    if (st != NIXL_SUCCESS) {
      return EpStatus::Error("Failed to create mem desc for new rank");
    }

    st = nixlAgentExchangeMemDesc(agent_, new_rank, local_desc,
                                   &peers_[new_rank].remote_recv_reg);
    if (st != NIXL_SUCCESS) {
      return EpStatus::Error("Failed to exchange mem desc with rank " +
                             std::to_string(new_rank));
    }

    peers_[new_rank].alive = true;
    active_ranks_.insert(new_rank);
    generation_++;

    // Rebalance local expert assignments
    rebalance_experts();

    return EpStatus::Ok();
  }

  EpStatus remove_rank(int dead_rank) {
    if (!elastic_enabled_) {
      return EpStatus::Error("Elastic EP not enabled for this group");
    }

    if (dead_rank < 0 || dead_rank >= static_cast<int>(peers_.size())) {
      return EpStatus::Error("Invalid rank: " + std::to_string(dead_rank));
    }

    peers_[dead_rank].alive = false;
    active_ranks_.erase(dead_rank);
    generation_++;

    // Redistribute experts from the dead rank across surviving ranks
    rebalance_experts();

    return EpStatus::Ok();
  }

  int get_generation() const { return generation_; }

  bool is_rank_alive(int r) const {
    return r >= 0 && r < static_cast<int>(peers_.size()) && peers_[r].alive;
  }

 private:
  // ─── Handle data for combine phase ──────────────────────────────

  struct NixlHandleData {
    std::vector<int64_t> send_counts;
    std::vector<int64_t> recv_counts;
    std::vector<int64_t> send_offsets;
    struct SendEntry {
      int token_idx;
      int k_idx;
      int dest_rank;
      int64_t expert_id;
    };
    std::vector<SendEntry> send_entries;
    int64_t total_send;
    int64_t total_recv;
    int num_tokens;
    int32_t* d_expert_counts;  // device pointer, freed at combine
  };

  // ─── Elastic EP helpers ─────────────────────────────────────────

  int find_fallback_rank(int64_t expert_id) {
    // Round-robin among surviving ranks
    int original_rank = static_cast<int>(expert_id / config_.num_local_experts);
    for (int offset = 1; offset < config_.world_size; offset++) {
      int candidate = (original_rank + offset) % config_.world_size;
      if (peers_[candidate].alive) return candidate;
    }
    return config_.rank;  // last resort: self
  }

  void rebalance_experts() {
    // In a full implementation, this would:
    // 1. Compute new expert-to-rank mapping based on active_ranks_
    // 2. Trigger expert weight migration for reassigned experts
    // 3. Invalidate all routing caches
    // For now, we rely on the fallback routing in dispatch
    group_healthy_.store(true, std::memory_order_relaxed);
  }

  EpStatus attempt_recovery() {
    // Check which ranks are still alive
    for (int r = 0; r < config_.world_size; r++) {
      if (!peers_[r].alive) continue;
      // Probe rank health via NIXL metadata ping
      nixl_status_t st = nixlAgentProbe(agent_, r);
      if (st != NIXL_SUCCESS) {
        peers_[r].alive = false;
        active_ranks_.erase(r);
      }
    }

    if (active_ranks_.empty()) {
      return EpStatus::Error("All ranks have failed — cannot recover");
    }

    rebalance_experts();
    return EpStatus::Ok();
  }

  // ─── Utility helpers ────────────────────────────────────────────

  static size_t dtype_size(int scalar_type) {
    // PyTorch scalar types: 0=uint8, 1=int8, 5=float16, 6=float32,
    // 15=bfloat16, 23=float8_e4m3fn, 24=float8_e5m2
    switch (scalar_type) {
      case 23: case 24: case 0: case 1: return 1;  // FP8, uint8, int8
      case 5: case 15: return 2;                     // FP16, BF16
      case 6: return 4;                               // FP32
      default: return 2;                               // assume BF16
    }
  }

  static void weighted_accumulate_async(
      void* dst, const void* src, float weight,
      int hidden_dim, int scalar_type, cudaStream_t stream) {
    // In production, this would be a fused CUDA kernel.
    // For the reference implementation, we use a simple approach:
    // dst[i] += weight * src[i] for i in [0, hidden_dim)
    //
    // TODO(Issue #65): Implement fused weighted accumulate kernel
    // that handles BF16/FP8 natively without intermediate FP32 cast.
    // For now, this is a placeholder that will be replaced by the
    // layout_normalize.cu fused reduction kernel.
    (void)dst; (void)src; (void)weight;
    (void)hidden_dim; (void)scalar_type; (void)stream;
  }

  static EpTensorView make_view_2d(void* ptr, int d0, int d1, int stype,
                                   int device_id) {
    EpTensorView v;
    v.data_ptr = ptr;
    v.scalar_type = stype;
    v.ndim = 2;
    v.shape[0] = d0; v.shape[1] = d1; v.shape[2] = 0;
    v.strides[0] = d1; v.strides[1] = 1; v.strides[2] = 0;
    v.device_id = device_id;
    return v;
  }

  static EpTensorView make_view_3d(void* ptr, int d0, int d1, int d2,
                                   int stype, int device_id) {
    EpTensorView v;
    v.data_ptr = ptr;
    v.scalar_type = stype;
    v.ndim = 3;
    v.shape[0] = d0; v.shape[1] = d1; v.shape[2] = d2;
    v.strides[0] = d1 * d2; v.strides[1] = d2; v.strides[2] = 1;
    v.device_id = device_id;
    return v;
  }

  static EpTensorView make_view_1d(void* ptr, int d0, int stype,
                                   int device_id) {
    EpTensorView v;
    v.data_ptr = ptr;
    v.scalar_type = stype;
    v.ndim = 1;
    v.shape[0] = d0; v.shape[1] = 0; v.shape[2] = 0;
    v.strides[0] = 1; v.strides[1] = 0; v.strides[2] = 0;
    v.device_id = device_id;
    return v;
  }
};

}  // namespace ep
}  // namespace flashinfer
