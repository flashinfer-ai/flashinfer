// csrc/ep/registry.cpp

#include "flashinfer/ep/registry.h"
#include <unordered_map>
#include <mutex>

namespace flashinfer {
namespace ep {

static std::unordered_map<int, std::unique_ptr<EpBackendImpl>> g_backends;
static std::once_flag g_init_flag;
static std::mutex g_registry_mutex;  // Issue #36: protects g_backends

static void register_builtin_backends() {
    // Called under std::call_once — no mutex needed here.
    g_backends[static_cast<int>(Backend::kDeepEP)] =
        std::make_unique<DeepEPBackend>();
    g_backends[static_cast<int>(Backend::kNcclEP)] =
        std::make_unique<NcclEPBackend>();
}

// Issue #36: register_backend() can be called from multiple threads
// (e.g., Python import from different threads, or test harnesses).
// The mutex prevents data races on the g_backends map.
void register_backend(Backend id, std::unique_ptr<EpBackendImpl> impl) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_backends[static_cast<int>(id)] = std::move(impl);
}

EpBackendImpl* get_backend(Backend id) {
    std::call_once(g_init_flag, register_builtin_backends);
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    auto it = g_backends.find(static_cast<int>(id));
    return (it != g_backends.end()) ? it->second.get() : nullptr;
}

// ─── Top-level API functions that delegate to backends ─────────────

EpGroup* create_group(const EpGroupConfig& config,
                      const EpCommunicator& comm,
                      void* stream,
                      EpStatus* status) {
    auto* impl = get_backend(config.backend);
    if (!impl) {
        if (status) *status = EpStatus::Error("Unknown backend");
        return nullptr;
    }

    EpStatus s = impl->init(config, comm, stream);
    if (!s.ok()) {  // Issue #45: was `s.ok` (member access) — must be `s.ok()` (method call)
        if (status) *status = s;
        return nullptr;
    }

    // EpGroup wraps the impl pointer + config
    auto* group = new EpGroup();
    group->impl = impl;
    group->config = config;
    group->comm = comm;
    group->stream = stream;

    if (status) *status = EpStatus::Ok();
    return group;
}

// Issue #24: All top-level functions use output-pointer pattern
void dispatch(EpGroup* group,
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
              bool async) {
    group->impl->dispatch(out,
        hidden, scales, topk_idx, topk_weights, layout,
        output_layout, prev_dep, prev_handle,
        alloc_on_comm_stream, async);
}

void low_latency_dispatch(EpGroup* group,
                          DispatchResult* out,
                          const EpTensorView& hidden,
                          const EpTensorView& topk_idx,
                          int max_tokens_per_rank,
                          OutputLayout output_layout,
                          StreamDep* prev_dep,
                          EpHandle* prev_handle,
                          bool alloc_on_comm_stream,
                          bool async,
                          bool return_recv_hook,
                          RoutingCache* cached_routing) {  // Issue #39
    group->impl->low_latency_dispatch(out,
        hidden, topk_idx, max_tokens_per_rank, output_layout,
        prev_dep, prev_handle,
        alloc_on_comm_stream, async, return_recv_hook,
        cached_routing);
}

// Issue #39: Routing cache lifecycle delegated to backend
RoutingCache* create_routing_cache(EpGroup* group,
                                   const EpTensorView& topk_idx,
                                   EpStatus* status) {
    auto* cache = group->impl->create_routing_cache(topk_idx, status);
    if (cache) cache->group = group;
    return cache;
}

EpStatus destroy_routing_cache(RoutingCache* cache) {
    if (!cache || !cache->group) {
        delete cache;
        return EpStatus::Ok();
    }
    auto* group = static_cast<EpGroup*>(cache->group);
    return group->impl->destroy_routing_cache(cache);
}

bool is_routing_cache_valid(const RoutingCache* cache,
                            const EpTensorView& topk_idx) {
    if (!cache || !cache->valid) return false;
    // Issue #43A: HOT-PATH check — shape-only, zero D2H copies.
    //
    // The v6 implementation (Issue #43) recomputed the content-sampled hash
    // here, requiring 4 blocking cudaMemcpy D2H calls (~0.5 μs) on every
    // decode step. This directly undid the ~1-2 μs savings that RoutingCache
    // was designed to provide.
    //
    // Fix: Compare only the cached shape (two int64 compares, ~0 μs).
    // The content-sampled hash is computed ONCE at create_routing_cache()
    // time and stored in the cache, but is NOT recomputed here.
    //
    // This means: if batch size or top_k changes, the cache is invalidated
    // (shape mismatch). If expert assignments change with the same shape,
    // the caller must explicitly destroy and recreate the cache. This is
    // the correct contract for the two primary use cases:
    //   - CUDA graph replay: routing is identical every step (shape + content).
    //   - Eager decode: batch size changes when sequences complete (shape catch).
    return cache->cached_shape[0] == topk_idx.shape[0] &&
           cache->cached_shape[1] == topk_idx.shape[1];
}

// ... combine, low_latency_combine, destroy_group follow same pattern ...

}  // namespace ep
}  // namespace flashinfer
