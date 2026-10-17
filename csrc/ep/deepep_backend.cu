// csrc/ep/deepep_backend.cu

#include "flashinfer/ep/registry.h"
#include "flashinfer/ep/types.h"
#include "layout_normalize.cu"

// DeepEP's native Python/C++ API
#include <deep_ep/buffer.h>

namespace flashinfer {
namespace ep {

class DeepEPBackend : public EpBackendImpl {
 private:
  deep_ep::Buffer* buffer_;           // single buffer, both modes
  bool ll_active_;                    // Issue #12: tracks current mode
  int  graph_capture_stype_;          // Issue #19: dtype recorded during graph capture (-1 = not in graph)
  EpGroupConfig config_;
  LayoutNormalizer normalizer_;
  cudaStream_t stream_;

  // Issue #17: Ping-pong scratch buffers for 3D output.
  // When double buffering (previous_handle != nullptr), two micro-batches
  // are live simultaneously. A single scratch buffer would cause data
  // corruption. We alternate between scratch_3d_a_ and scratch_3d_b_.
  void* scratch_3d_a_;
  void* scratch_3d_b_;
  size_t scratch_3d_bytes_;
  int scratch_ping_pong_;            // toggles 0/1 per dispatch

 public:
  EpStatus init(const EpGroupConfig& config,
                const EpCommunicator& comm,
                void* stream) override {
    config_ = config;
    stream_ = static_cast<cudaStream_t>(stream);

    // Compute buffer sizes to cover both HT and LL
    size_t nvl = config.nvl_buffer_bytes;
    size_t rdma = config.rdma_buffer_bytes;

    try {
      // Issue #13: Auto-set num_qps_per_rank to num_local_experts if unset.
      // DeepEP LL kernel requires exactly one QP per local expert.
      int qps = config.num_qps_per_rank;
      if (qps == 0) {
        qps = config.num_local_experts;
      } else if (qps != config.num_local_experts) {
        return EpStatus::Error(
            "DeepEP: num_qps_per_rank (" + std::to_string(qps) +
            ") must equal num_local_experts (" +
            std::to_string(config.num_local_experts) +
            ") for LL mode. Set to 0 for auto.");
      }

      // Issue #12: DeepEP Buffer MUST be created with low_latency_mode=True.
      // NVSHMEM cannot be re-initialized in the same process, so the buffer
      // must be initialized for LL upfront. When HT dispatch is needed,
      // call clean_low_latency_buffer() first to release LL-specific state
      // (double send/recv/signal buffers), then proceed with HT operations.
      // See: https://github.com/deepseek-ai/DeepEP/issues/76
      //      https://github.com/deepseek-ai/DeepEP/issues/221
      buffer_ = new deep_ep::Buffer(
          comm.rank, comm.world_size,
          nvl, rdma,
          /*low_latency_mode=*/true,
          qps);
      ll_active_ = true;

      if (config.num_sms > 0) {
        deep_ep::Buffer::set_num_sms(config.num_sms);
      }

      // Initialize layout normalizer
      // Issue #37: max_tok_per_expert must be derived from actual buffer
      // sizing, not hardcoded. The RDMA buffer is sized for a worst-case
      // distribution: all tokens land on one expert. So the safe upper
      // bound is (rdma_buffer_bytes / (hidden_dim * max_dispatch_elem_size)).
      // For CUDA graph mode, use the explicit cap. Otherwise, derive from
      // buffer capacity to avoid silent overflow on small models.
      int max_tok_per_expert;
      if (config.cuda_graph_max_tokens > 0) {
        max_tok_per_expert = config.cuda_graph_max_tokens;
      } else {
        // Issue #47: Derive from allocated buffer. Worst case is ALL dispatched
        // tokens landing on a SINGLE expert (extreme routing skew). The safe
        // upper bound is total buffer capacity / per-token size — do NOT
        // divide by num_local_experts, which would undercount by that factor
        // and cause false buffer overflow errors on skewed workloads.
        size_t elem_bytes = config.max_dispatch_elem_size > 0
            ? config.max_dispatch_elem_size : 2;
        max_tok_per_expert = static_cast<int>(
            rdma / (config.hidden_dim * elem_bytes));
        if (max_tok_per_expert <= 0) max_tok_per_expert = 1;
      }
      normalizer_.num_local_experts = config.num_local_experts;
      normalizer_.max_tokens_per_expert = max_tok_per_expert;
      normalizer_.hidden_dim = config.hidden_dim;

      // Allocate prefix sum scratch + fused sync flag (Issue #27)
      cudaMalloc(&normalizer_.d_expert_offsets,
                 config.num_local_experts * sizeof(int32_t));
      cudaMalloc(&normalizer_.d_offsets_ready, sizeof(int32_t));

      // Issue #11: Allocate 3D scratch using max_dispatch_elem_size
      // (worst-case element size) so the scratch can handle any dtype
      // dispatched at runtime (FP8 or BF16).
      // Issue #17: Allocate TWO scratch buffers for ping-pong double buffering.
      scratch_3d_bytes_ = (size_t)config.num_local_experts *
                          max_tok_per_expert * config.hidden_dim *
                          config.max_dispatch_elem_size;
      cudaMalloc(&scratch_3d_a_, scratch_3d_bytes_);
      cudaMalloc(&scratch_3d_b_, scratch_3d_bytes_);
      scratch_ping_pong_ = 0;
      graph_capture_stype_ = -1;  // Issue #19: no graph captured yet

    } catch (const std::exception& e) {
      return EpStatus::Error(std::string("DeepEP init failed: ") + e.what());
    }
    return EpStatus::Ok();
  }

  EpStatus destroy() override {
    cudaFree(normalizer_.d_expert_offsets);
    cudaFree(normalizer_.d_offsets_ready);  // Issue #27: fused sync flag
    cudaFree(scratch_3d_a_);
    cudaFree(scratch_3d_b_);  // Issue #17: ping-pong pair
    delete buffer_;
    return EpStatus::Ok();
  }

  // ─── Issue #12: Mode switching helpers ───────────────────────────
  //
  // DeepEP Buffer is created with low_latency_mode=True (NVSHMEM requires
  // this to be set at init time and cannot be re-initialized). When
  // switching to HT mode, call clean_low_latency_buffer() to release
  // LL-specific double send/recv/signal buffers. When switching back
  // to LL, call setup_low_latency_buffer() to re-allocate them.
  //
  // The switch is lazy — only triggered when the mode actually changes.

  void ensure_ll_mode() {
    if (!ll_active_) {
      buffer_->setup_low_latency_buffer();
      ll_active_ = true;
    }
  }

  void ensure_ht_mode() {
    if (ll_active_) {
      buffer_->clean_low_latency_buffer();
      ll_active_ = false;
    }
  }

  // ─── Low-Latency Dispatch ────────────────────────────────────────

  // Issue #24: Output via pointer — zero-copy across virtual dispatch
  // Issue #39: cached_routing accepted but ignored (DeepEP routing is implicit)
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
    (void)cached_routing;  // Issue #39: DeepEP ignores — routing is implicit in NVSHMEM puts
    ensure_ll_mode();  // Issue #12: lazy mode switch

    // Issue #19: Graph dtype guard — prevent silent corruption on replay
    if (config_.cuda_graph_max_tokens > 0) {
      if (graph_capture_stype_ == -1) {
        graph_capture_stype_ = hidden.scalar_type;  // record on first call
      } else if (graph_capture_stype_ != hidden.scalar_type) {
        out->status = EpStatus::Error("graph dtype mismatch");
        return;
      }
    }

    // Validate buffer bounds (Issue #7, #31)
    int num_tokens = hidden.shape[0];
    if (num_tokens > max_tokens_per_rank) {
      out->status = EpStatus::Error("buffer overflow: num_tokens > max_tokens_per_rank");
      return;
    }

    // Call DeepEP's native low_latency_dispatch
    auto [recv_hidden, recv_expert_count, handle, event, hook_fn] =
        buffer_->low_latency_dispatch(
            /*hidden_states=*/to_torch(hidden),
            /*topk_idx=*/to_torch(topk_idx),
            /*num_max_dispatch_tokens_per_rank=*/max_tokens_per_rank,
            /*num_experts=*/config_.num_experts,
            /*async_finish=*/async,
            /*return_recv_hook=*/return_hook);

    // Issue #31: Validate max_tokens_per_expert before scatter
    int runtime_stype = hidden.scalar_type;
    void* scratch = (scratch_ping_pong_ == 0) ? scratch_3d_a_ : scratch_3d_b_;
    scratch_ping_pong_ ^= 1;

    if (out_layout == OutputLayout::kExpertMajor3D) {
        // Issue #31: Check that no expert received more tokens than scratch fits
        // (async check via kernel; for debug builds, add host-side validation)
        normalizer_.scatter_2d_to_3d(
            recv_hidden.data_ptr(), scratch,
            static_cast<const int32_t*>(recv_expert_count.data_ptr()),
            runtime_stype, stream_);

        out->recv_hidden = make_view_3d(
            scratch,
            config_.num_local_experts,
            normalizer_.max_tokens_per_expert,
            config_.hidden_dim,
            runtime_stype,
            config_.device_id);  // Issue #50: pass actual device_id
    } else {
        out->recv_hidden = from_torch(recv_hidden);
    }

    out->recv_expert_counts = from_torch(recv_expert_count);

    // Issue #23 + #33: Store deferred recv in handle fully inline (no heap alloc).
    // Issue #34: Save scratch slot so combine reads the correct buffer.
    EpHandle* ep_handle = wrap_handle(handle);
    ep_handle->scratch_slot = scratch_ping_pong_ ^ 1;  // slot used by THIS dispatch (already toggled above)
    if (return_hook && hook_fn) {
        // Issue #33 + #48: Store hook_fn inline in deferred_ctx_storage.
        //
        // Issue #48: DeepEP's hook_fn type varies by toolchain.
        //   - If it's a raw function pointer or small lambda (≤ 16 bytes): fits inline.
        //   - If it's std::function<void()> on libstdc++ (~32 bytes): does NOT fit.
        //
        // Strategy: compile-time check. If HookFnT fits in kDeferCtxSize (16 bytes),
        // store inline (zero alloc). If it exceeds kDeferCtxSize, fall back to a
        // per-thread reusable slot (thread_local avoids heap alloc on the hot path).
        using HookFnT = decltype(hook_fn);
        if constexpr (sizeof(HookFnT) <= EpHandle::kDeferCtxSize) {
            // Fast path: inline storage, no heap alloc
            ep_handle->deferred_ctx_as<HookFnT>() = hook_fn;
            ep_handle->deferred_fn = [](void* ctx) -> EpStatus {
                auto* fn = reinterpret_cast<HookFnT*>(ctx);
                try { (*fn)(); return EpStatus::Ok(); }
                catch (const std::exception& e) { return EpStatus::Error(e.what()); }
            };
        } else {
            // Fallback: thread_local slot avoids per-dispatch heap alloc.
            // Safe because EP dispatch is single-threaded per GPU.
            thread_local HookFnT tl_hook;
            tl_hook = std::move(hook_fn);
            ep_handle->deferred_fn = [](void* /*ctx*/) -> EpStatus {
                try { tl_hook(); return EpStatus::Ok(); }
                catch (const std::exception& e) { return EpStatus::Error(e.what()); }
            };
        }
    } else {
        ep_handle->deferred_fn = nullptr;
    }
    out->handle = ep_handle;
    out->status = EpStatus::Ok();
  }

  // ─── Low-Latency Combine ─────────────────────────────────────────

  // Issue #24: Output via pointer
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
    int combine_stype = expert_output.scalar_type;
    torch::Tensor expert_out_2d;
    if (expert_output.ndim == 3) {
        // Issue #34: Read scratch slot from handle, not from shared toggle.
        // In chunked prefill (LL dispatch → HT scatter → LL combine), the
        // shared scratch_ping_pong_ may have advanced. The handle records
        // which slot was used by the paired dispatch.
        void* combine_scratch = (handle->scratch_slot == 0)
            ? scratch_3d_a_ : scratch_3d_b_;
        normalizer_.flatten_3d_to_2d(
            expert_output.data_ptr, combine_scratch,
            get_handle_expert_counts(handle),
            combine_stype, stream_);
        expert_out_2d = torch_from_flat(combine_scratch,
            get_handle_total_recv(handle), config_.hidden_dim,
            combine_stype);
    } else {
        expert_out_2d = to_torch(expert_output);
    }

    auto [combined, event, hook_fn] =
        buffer_->low_latency_combine(
            /*hidden_states=*/expert_out_2d,
            /*topk_idx=*/to_torch(topk_idx),
            /*topk_weights=*/to_torch(topk_weights),
            /*handle=*/unwrap_handle(handle),
            /*async_finish=*/async,
            /*return_recv_hook=*/return_hook);

    out->combined_hidden = from_torch(combined);
    out->status = EpStatus::Ok();
  }

  // ─── HT Dispatch (similar structure, omitted for brevity) ────────
  // Key difference: no layout normalization needed in HT mode —
  // both backends produce 2D natively. 3D scatter is applied if requested.
  // Issue #12: HT dispatch calls ensure_ht_mode() which triggers
  // clean_low_latency_buffer() if currently in LL mode.

  // ... dispatch() and combine() implementations follow same pattern,
  //     with ensure_ht_mode() at the top of dispatch() and combine() ...

 private:
  // Conversion helpers between EpTensorView and torch::Tensor
  static torch::Tensor to_torch(const EpTensorView& v) {
    auto opts = torch::TensorOptions()
        .dtype(scalar_type_to_torch(v.scalar_type))
        .device(torch::kCUDA, v.device_id);
    std::vector<int64_t> shape(v.shape, v.shape + v.ndim);
    return torch::from_blob(v.data_ptr, shape, opts);
  }

  static EpTensorView from_torch(const torch::Tensor& t) {
    EpTensorView v;
    v.data_ptr = t.data_ptr();
    v.scalar_type = static_cast<int>(t.scalar_type());
    v.ndim = t.dim();
    for (int i = 0; i < t.dim(); i++) {
      v.shape[i] = t.size(i);
      v.strides[i] = t.stride(i);
    }
    v.device_id = t.device().index();
    return v;
  }

  static EpTensorView make_view_3d(void* ptr, int d0, int d1, int d2, int stype,
                                   int device_id) {
    EpTensorView v;
    v.data_ptr = ptr;
    v.scalar_type = stype;
    v.ndim = 3;
    v.shape[0] = d0; v.shape[1] = d1; v.shape[2] = d2;
    v.strides[0] = d1 * d2; v.strides[1] = d2; v.strides[2] = 1;
    v.device_id = device_id;  // Issue #50: was hardcoded to 0
    return v;
  }
};

}  // namespace ep
}  // namespace flashinfer
