// csrc/ep/nccl_ep_backend.cu

#include "flashinfer/ep/registry.h"
#include "flashinfer/ep/types.h"
#include "layout_normalize.cu"

// NCCL-EP C API
#include <nccl_ep.h>

namespace flashinfer {
namespace ep {

class NcclEPBackend : public EpBackendImpl {
 private:
  // Two internal groups: LL and HT (Issue #2)
  // Share the same ncclComm_t but have different algorithm configs
  ncclEpGroup_t ll_group_;
  ncclEpGroup_t ht_group_;

  EpGroupConfig config_;
  LayoutNormalizer normalizer_;
  cudaStream_t stream_;

  // Issue #42: Ping-pong scratch buffers for 2D output.
  // Same rationale as DeepEP's Issue #17: when double buffering
  // (previous_handle != nullptr), two micro-batches are live
  // simultaneously. A single scratch buffer would cause the combine
  // of micro-batch N to overwrite the dispatch result of micro-batch N+1
  // (or vice versa in _forward_chunked). Two buffers, toggled per
  // dispatch, prevent this corruption.
  void* scratch_2d_a_;
  void* scratch_2d_b_;
  size_t scratch_2d_bytes_;
  int scratch_ping_pong_;

 public:
  EpStatus init(const EpGroupConfig& config,
                const EpCommunicator& comm,
                void* stream) override {
    config_ = config;
    stream_ = static_cast<cudaStream_t>(stream);
    ncclComm_t nccl_comm = static_cast<ncclComm_t>(comm.nccl_comm);

    if (!nccl_comm) {
      return EpStatus::Error("NCCL-EP requires a valid ncclComm_t");
    }

    // Create LL group
    ncclEpConfig_t ll_cfg;
    ll_cfg.algo = NCCL_EP_ALGO_LOW_LATENCY;
    ll_cfg.num_experts = config.num_experts;
    ll_cfg.hidden_dim = config.hidden_dim;
    ll_cfg.top_k = config.top_k;
    // Buffer sizes — NCCL-EP manages internally but we can hint
    ncclResult_t res = ncclEpCreateGroup(&ll_group_, nccl_comm, &ll_cfg);
    if (res != ncclSuccess) {
      return EpStatus::Error("Failed to create NCCL-EP LL group");
    }

    // Create HT group
    ncclEpConfig_t ht_cfg;
    ht_cfg.algo = NCCL_EP_ALGO_HIGH_THROUGHPUT;
    ht_cfg.num_experts = config.num_experts;
    ht_cfg.hidden_dim = config.hidden_dim;
    ht_cfg.top_k = config.top_k;
    res = ncclEpCreateGroup(&ht_group_, nccl_comm, &ht_cfg);
    if (res != ncclSuccess) {
      ncclEpGroupDestroy(ll_group_);
      return EpStatus::Error("Failed to create NCCL-EP HT group");
    }

    // Initialize normalizer (for 3D->2D flatten)
    // Issue #37 + #47: Derive max_tok from buffer sizing, not hardcoded.
    // Issue #47: Worst case is all tokens on one expert — do NOT divide
    // by num_local_experts (which underestimates by that factor).
    int max_tok;
    if (config.cuda_graph_max_tokens > 0) {
        max_tok = config.cuda_graph_max_tokens;
    } else {
        size_t elem_bytes = config.max_dispatch_elem_size > 0
            ? config.max_dispatch_elem_size : 2;
        max_tok = static_cast<int>(
            config.rdma_buffer_bytes /
            (config.hidden_dim * elem_bytes));
        if (max_tok <= 0) max_tok = 1;
    }
    normalizer_.num_local_experts = config.num_local_experts;
    normalizer_.max_tokens_per_expert = max_tok;
    normalizer_.hidden_dim = config.hidden_dim;
    cudaMalloc(&normalizer_.d_expert_offsets,
               config.num_local_experts * sizeof(int32_t));
    cudaMalloc(&normalizer_.d_offsets_ready, sizeof(int32_t));  // Issue #27

    // Issue #11: 2D scratch sized with max_dispatch_elem_size (worst-case)
    // so it can handle any dtype dispatched at runtime.
    // Issue #42: Allocate TWO scratch buffers for ping-pong double buffering,
    // mirroring the DeepEP pattern (Issue #17). Prevents data corruption
    // when two micro-batches are live simultaneously (double buffering)
    // or when _forward_chunked interleaves LL dispatch and HT combine.
    scratch_2d_bytes_ = (size_t)max_tok * comm.world_size *
                        config.hidden_dim * config.max_dispatch_elem_size;
    cudaMalloc(&scratch_2d_a_, scratch_2d_bytes_);
    cudaMalloc(&scratch_2d_b_, scratch_2d_bytes_);
    scratch_ping_pong_ = 0;

    return EpStatus::Ok();
  }

  EpStatus destroy() override {
    ncclEpGroupDestroy(ll_group_);
    ncclEpGroupDestroy(ht_group_);
    cudaFree(normalizer_.d_expert_offsets);
    cudaFree(normalizer_.d_offsets_ready);  // Issue #27
    cudaFree(scratch_2d_a_);               // Issue #42: ping-pong pair
    cudaFree(scratch_2d_b_);
    return EpStatus::Ok();
  }

  // ─── Low-Latency Dispatch ────────────────────────────────────────

  // Issue #24: Output via pointer
  // Issue #28 + #39: ncclEpCreateHandle per-dispatch cost eliminated via cache.
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
    // Validate (Issue #7)
    int num_tokens = hidden.shape[0];
    if (num_tokens > max_tokens_per_rank) {
      out->status = EpStatus::Error("buffer overflow: num_tokens > max_tokens_per_rank");
      return;
    }

    // Issue #39: Use cached routing handle if provided and valid.
    // This skips ncclEpCreateHandle() (~1-2 μs) when the routing
    // topology is unchanged (steady-state decode, CUDA graph replay).
    ncclEpHandle_t handle;
    bool handle_is_cached = false;
    if (cached_routing && cached_routing->valid && cached_routing->impl_cache) {
        handle = *static_cast<ncclEpHandle_t*>(cached_routing->impl_cache);
        handle_is_cached = true;
    } else {
        // Issue #28: Per-dispatch metadata setup (~1-2 μs).
        ncclEpCreateHandle(&handle, ll_group_,
                           to_nccl_tensor(topk_idx, NCCL_EP_TENSOR_TAG_TOPK_IDX));
    }

    ncclNDTensor_t recv_tensor;
    ncclEpDispatch(ll_group_, handle,
                   to_nccl_tensor(hidden, NCCL_EP_TENSOR_TAG_TOKENS),
                   &recv_tensor,
                   return_hook ? 1 : 0,
                   stream_);

    ncclNDTensor_t expert_counts_tensor;
    ncclEpGetExpertCounts(handle, &expert_counts_tensor);

    // Issue #42: Select ping-pong scratch buffer for this dispatch.
    // Toggle BEFORE use so dispatch and combine of different micro-batches
    // never alias the same scratch buffer.
    int current_slot = scratch_ping_pong_;
    scratch_ping_pong_ ^= 1;
    void* scratch = (current_slot == 0) ? scratch_2d_a_ : scratch_2d_b_;

    int runtime_stype = hidden.scalar_type;
    if (out_layout == OutputLayout::kFlat2D) {
        normalizer_.flatten_3d_to_2d(
            recv_tensor.data_ptr, scratch,
            static_cast<const int32_t*>(expert_counts_tensor.data_ptr),
            runtime_stype, stream_);

        int total_recv = 0;
        if (config_.cuda_graph_max_tokens > 0) {
            total_recv = max_tokens_per_rank * config_.top_k;
        } else {
            total_recv = ncclEpHandleGetNumRecvTokens(handle);
        }

        out->recv_hidden = make_view_2d(
            scratch, total_recv, config_.hidden_dim, runtime_stype);
    } else {
        out->recv_hidden = from_nccl_tensor(recv_tensor);
    }

    out->recv_expert_counts = from_nccl_tensor(expert_counts_tensor);

    // Issue #23 + #33: DeferredRecv stored fully inline in handle (no heap alloc).
    EpHandle* ep_handle = wrap_nccl_handle(handle);
    // Issue #42: Save scratch slot so combine reads the correct buffer.
    // Same pattern as DeepEP Issue #34 — prevents wrong-buffer read
    // when _forward_chunked interleaves LL dispatch and HT combine.
    ep_handle->scratch_slot = current_slot;
    if (return_hook) {
        // Issue #33: NcclDeferCtx stored inline in deferred_ctx_storage.
        // {ncclEpHandle_t, cudaStream_t} = 16 bytes — fits exactly.
        struct NcclDeferCtx { ncclEpHandle_t h; cudaStream_t s; };
        static_assert(sizeof(NcclDeferCtx) <= EpHandle::kDeferCtxSize,
                      "NcclDeferCtx exceeds inline context size");
        ep_handle->deferred_ctx_as<NcclDeferCtx>() = {handle, stream_};
        ep_handle->deferred_fn = [](void* c) -> EpStatus {
            auto* ctx = reinterpret_cast<NcclDeferCtx*>(c);
            ncclResult_t res = ncclEpComplete(ctx->h, ctx->s);
            return (res == ncclSuccess) ? EpStatus::Ok()
                                        : EpStatus::Error("ncclEpComplete failed");
        };
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
    ncclEpHandle_t h = unwrap_nccl_handle(handle);

    int combine_stype = expert_output.scalar_type;
    ncclNDTensor_t combine_input;
    if (expert_output.ndim == 2) {
        // Issue #42: Read scratch slot from handle (same pattern as DeepEP #34).
        // In _forward_chunked, the shared scratch_ping_pong_ may have advanced.
        // The handle records which slot was used by the paired dispatch.
        void* combine_scratch = (handle->scratch_slot == 0)
            ? scratch_2d_a_ : scratch_2d_b_;
        normalizer_.scatter_2d_to_3d(
            expert_output.data_ptr, combine_scratch,
            get_nccl_handle_expert_counts(h),
            combine_stype, stream_);
        combine_input = make_nccl_tensor_3d(combine_scratch, config_);
    } else {
        combine_input = to_nccl_tensor(expert_output, NCCL_EP_TENSOR_TAG_TOKENS);
    }

    ncclNDTensor_t combined_tensor;
    ncclEpCombine(ll_group_, h,
                  combine_input,
                  to_nccl_tensor(topk_weights, NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS),
                  &combined_tensor,
                  return_hook ? 1 : 0,
                  stream_);

    out->combined_hidden = from_nccl_tensor(combined_tensor);
    out->status = EpStatus::Ok();
  }

  // ─── Issue #39: Routing Cache ─────────────────────────────────────

  RoutingCache* create_routing_cache(
      const EpTensorView& topk_idx, EpStatus* status) override {
    // Build the ncclEpHandle_t once and cache it.
    auto* cached_handle = new ncclEpHandle_t;
    ncclResult_t res = ncclEpCreateHandle(
        cached_handle, ll_group_,
        to_nccl_tensor(topk_idx, NCCL_EP_TENSOR_TAG_TOPK_IDX));
    if (res != ncclSuccess) {
        delete cached_handle;
        if (status) *status = EpStatus::Error("ncclEpCreateHandle failed in cache creation");
        return nullptr;
    }

    // Issue #43: Compute a topology hash that includes content samples,
    // not just shape. Shape-only hash (v5.1) was constant for an entire
    // decode session (same batch_size * 31 + top_k every step), causing
    // false cache hits when expert assignments change.
    //
    // Strategy: hash shape + 4 sampled elements from the device tensor.
    // The D2H copy of 4 int64s (~32 bytes) costs < 0.5 μs on PCIe and
    // is negligible compared to the ~1-2 μs ncclEpCreateHandle it saves.
    // We also mix in a per-group monotonic generation counter so that
    // caches from different create_routing_cache() calls never collide.
    uint64_t hash = static_cast<uint64_t>(topk_idx.shape[0]) * 2654435761ULL +
                    static_cast<uint64_t>(topk_idx.shape[1]) * 40503ULL;
    {
        // Sample 4 elements: first, last, 1/3, 2/3
        int64_t n = topk_idx.shape[0] * topk_idx.shape[1];
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

    // Issue #43A: Store the shape so is_routing_cache_valid() can do a
    // cheap shape-only check on the hot path without any D2H copy.
    auto* cache = new RoutingCache{
        static_cast<void*>(cached_handle),
        nullptr,  // group back-pointer set by top-level API
        {topk_idx.shape[0], topk_idx.shape[1]},  // cached_shape
        hash,     // content-sampled hash (computed ONCE here, never recomputed)
        true      // valid
    };
    if (status) *status = EpStatus::Ok();
    return cache;
  }

  EpStatus destroy_routing_cache(RoutingCache* cache) override {
    if (cache && cache->impl_cache) {
        // ncclEpHandle_t does not have a separate destroy — it's a
        // lightweight routing descriptor. Just free our allocation.
        delete static_cast<ncclEpHandle_t*>(cache->impl_cache);
    }
    delete cache;
    return EpStatus::Ok();
  }

  // ─── HT ops use ht_group_ instead of ll_group_ ──────────────────
  // Same structure, dispatch to ht_group_ with HT algorithm.
  // Both backends produce 2D natively in HT mode.

  // ... dispatch() and combine() follow same pattern with ht_group_ ...

 private:
  // NCCL tensor conversion helpers
  static ncclNDTensor_t to_nccl_tensor(const EpTensorView& v, int tag) {
    ncclNDTensor_t t;
    t.data_ptr = v.data_ptr;
    t.ndim = v.ndim;
    for (int i = 0; i < v.ndim; i++) {
      t.shape[i] = v.shape[i];
      t.strides[i] = v.strides[i];
    }
    t.dtype = scalar_type_to_nccl(v.scalar_type);
    t.tag = tag;
    return t;
  }

  static EpTensorView from_nccl_tensor(const ncclNDTensor_t& t) {
    EpTensorView v;
    v.data_ptr = t.data_ptr;
    v.ndim = t.ndim;
    for (int i = 0; i < t.ndim; i++) {
      v.shape[i] = t.shape[i];
      v.strides[i] = t.strides[i];
    }
    v.scalar_type = nccl_to_scalar_type(t.dtype);
    return v;
  }

  static EpTensorView make_view_2d(void* ptr, int d0, int d1, int stype) {
    EpTensorView v;
    v.data_ptr = ptr;
    v.scalar_type = stype;
    v.ndim = 2;
    v.shape[0] = d0; v.shape[1] = d1; v.shape[2] = 0;
    v.strides[0] = d1; v.strides[1] = 1; v.strides[2] = 0;
    return v;
  }
};

}  // namespace ep
}  // namespace flashinfer
