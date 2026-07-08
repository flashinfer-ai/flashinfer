/*
 * Copyright (c) 2026 by samhuang@nvidia.com (Samuel En-Ming Huang).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// GPU-side Distribution-Aware DAKNNv2 selector for MoE tile dispatch.
//
// After autotuning or bundle loading, DAKNNv2 exemplars are uploaded to the
// GPU. During CUDA graph capture, a lightweight selector kernel chooses the
// nearest exemplar and drives a SWITCH conditional node entirely on-device.

#include <cuda_runtime.h>
#include <flashinfer/exception.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "flashinfer/trtllm/fused_moe/da_heuristic.cuh"
#include "tvm_ffi_utils.h"

// Convenience macro: throw on CUDA error.
#define DA_CUDA_CHECK(expr)                                                                     \
  do {                                                                                          \
    cudaError_t _e = (expr);                                                                    \
    FLASHINFER_CHECK(_e == cudaSuccess, "CUDA error in DA heuristic: ", cudaGetErrorString(_e), \
                     " (", static_cast<int>(_e), ") at ", __FILE__, ":", __LINE__);             \
  } while (0)

namespace flashinfer {

using da_heuristic::DAKnnParams;
using da_heuristic::kMaxExemplars;
using da_heuristic::kMaxKnnExperts;
using da_heuristic::kMaxTiles;

// ---------------------------------------------------------------------------
// k-NN state populated by da_upload_knn_exemplars(...).
// ---------------------------------------------------------------------------
struct DAKnnState {
  DAKnnParams* d_params = nullptr;
  int32_t* d_selected_tile_idx = nullptr;
  int32_t* d_selected_tile_n = nullptr;
  int32_t* d_counts = nullptr;
  unsigned int* d_block_done = nullptr;
  DAKnnParams h_params{};
  int top_k = 0;

  ~DAKnnState() {
    if (d_params) cudaFree(d_params);
    if (d_selected_tile_idx) cudaFree(d_selected_tile_idx);
    if (d_selected_tile_n) cudaFree(d_selected_tile_n);
    if (d_counts) cudaFree(d_counts);
    if (d_block_done) cudaFree(d_block_done);
  }
};

constexpr int64_t kDefaultSelectorHandle = 0;

struct DAKnnStateKey {
  int64_t selector_handle;
  int num_tokens_bucket;

  bool operator==(const DAKnnStateKey& other) const {
    return selector_handle == other.selector_handle && num_tokens_bucket == other.num_tokens_bucket;
  }
};

struct DAKnnStateKeyHash {
  size_t operator()(const DAKnnStateKey& key) const {
    size_t seed = std::hash<int64_t>{}(key.selector_handle);
    auto combine = [&seed](int value) {
      seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };
    combine(key.num_tokens_bucket);
    return seed;
  }
};

static std::unordered_map<DAKnnStateKey, std::unique_ptr<DAKnnState>, DAKnnStateKeyHash>
    g_da_knn_states;

static DAKnnStateKey make_state_key(int64_t selector_handle, int num_tokens_bucket) {
  return {selector_handle, num_tokens_bucket};
}

static bool validate_knn_state_topology(const DAKnnState& state, int64_t top_k,
                                        int64_t num_local_experts, int64_t local_expert_offset) {
  return state.top_k == top_k && state.h_params.num_local_experts == num_local_experts &&
         state.h_params.local_expert_offset == local_expert_offset;
}

static bool da_verbose_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("FLASHINFER_DA_VERBOSE");
    return v && std::strcmp(v, "1") == 0;
  }();
  return enabled;
}

// ---------------------------------------------------------------------------
// Upload k-NN exemplars for one (bucket, top_k, nle, leo) cell. Each call
// replaces any previously uploaded exemplars for the same key.
//
//   exemplar_norm_flat: float32, shape [num_exemplars * num_local_experts],
//       sorted-descending L2-normalized per-expert counts (row-major).
//   exemplar_body_idx:     int64, shape [num_exemplars], SWITCH body index.
//   exemplar_tile_shapes:  int64, shape [num_exemplars], tile_n per exemplar.
//   exemplar_kernel_ids:   int64, shape [num_exemplars], kernel config ID.
// Caller responsibility: rows already sorted descending and L2-normalized.
// ---------------------------------------------------------------------------
void da_upload_knn_exemplars_with_handle(int64_t selector_handle,
                                         tvm::ffi::Tensor exemplar_norm_flat,
                                         tvm::ffi::Array<int64_t> exemplar_body_idx,
                                         tvm::ffi::Array<int64_t> exemplar_tile_shapes,
                                         tvm::ffi::Array<int64_t> exemplar_kernel_ids,
                                         tvm::ffi::Array<int64_t> tile_sizes,
                                         int64_t num_local_experts, int64_t local_expert_offset,
                                         int64_t top_k, int64_t num_tokens_bucket) {
  int n_exemplars = static_cast<int>(exemplar_body_idx.size());
  TVM_FFI_ICHECK(static_cast<int>(exemplar_tile_shapes.size()) == n_exemplars)
      << "exemplar_tile_shapes size mismatch";
  TVM_FFI_ICHECK(static_cast<int>(exemplar_kernel_ids.size()) == n_exemplars)
      << "exemplar_kernel_ids size mismatch";
  TVM_FFI_ICHECK(n_exemplars > 0 && n_exemplars <= kMaxExemplars)
      << "num_exemplars=" << n_exemplars << " out of range [1, " << kMaxExemplars << "]";
  TVM_FFI_ICHECK(num_local_experts > 0 && num_local_experts <= kMaxKnnExperts)
      << "num_local_experts=" << num_local_experts << " > kMaxKnnExperts=" << kMaxKnnExperts;
  TVM_FFI_ICHECK(exemplar_norm_flat.numel() ==
                 static_cast<int64_t>(n_exemplars) * num_local_experts)
      << "exemplar_norm_flat size mismatch: got " << exemplar_norm_flat.numel() << ", expected "
      << static_cast<int64_t>(n_exemplars) * num_local_experts;
  TVM_FFI_ICHECK(encode_dlpack_dtype(exemplar_norm_flat.dtype()) == float32_code)
      << "exemplar_norm_flat must be float32";
  if (exemplar_norm_flat.device().device_type == kDLCUDA) {
    int current_device = -1;
    DA_CUDA_CHECK(cudaGetDevice(&current_device));
    TVM_FFI_ICHECK(current_device == exemplar_norm_flat.device().device_id)
        << "exemplar_norm_flat is on cuda:" << exemplar_norm_flat.device().device_id
        << " but the active upload device is cuda:" << current_device;
  }

  DAKnnStateKey key = make_state_key(selector_handle, static_cast<int>(num_tokens_bucket));

  auto& state = g_da_knn_states[key];
  if (!state) {
    state = std::make_unique<DAKnnState>();
    DA_CUDA_CHECK(cudaMalloc(&state->d_params, sizeof(DAKnnParams)));
    DA_CUDA_CHECK(cudaMalloc(&state->d_selected_tile_idx, sizeof(int32_t)));
    DA_CUDA_CHECK(cudaMalloc(&state->d_selected_tile_n, sizeof(int32_t)));
    DA_CUDA_CHECK(cudaMalloc(&state->d_counts, sizeof(int32_t) * kMaxKnnExperts));
  } else if (!state->d_counts) {
    DA_CUDA_CHECK(cudaMalloc(&state->d_counts, sizeof(int32_t) * kMaxKnnExperts));
  }
  if (!state->d_block_done) {
    DA_CUDA_CHECK(cudaMalloc(&state->d_block_done, sizeof(unsigned int)));
  }
  // Inline CUDA-graph capture cannot allocate or memset. Keep a persistent
  // counts buffer in the uploaded selector state; split-select re-zeros the
  // entries after each replay, and this initializes the first replay.
  DA_CUDA_CHECK(cudaMemset(state->d_counts, 0, sizeof(int32_t) * kMaxKnnExperts));
  // atomicInc(d_block_done, gridDim.x-1) auto-wraps to 0, self-resetting
  // for the next replay. Only needs to be zeroed once at allocation.
  DA_CUDA_CHECK(cudaMemset(state->d_block_done, 0, sizeof(unsigned int)));

  // Build host-side params buffer, then push to device.
  DAKnnParams h_params{};
  h_params.num_exemplars = n_exemplars;
  h_params.num_tiles = std::min(static_cast<int>(tile_sizes.size()), kMaxTiles);
  for (int i = 0; i < h_params.num_tiles; i++) {
    h_params.tile_sizes[i] = static_cast<int>(tile_sizes[i]);
  }
  h_params.num_local_experts = static_cast<int>(num_local_experts);
  h_params.local_expert_offset = static_cast<int>(local_expert_offset);
  for (int e = 0; e < n_exemplars; e++) {
    TVM_FFI_ICHECK(exemplar_body_idx[e] >= 0 && exemplar_body_idx[e] < kMaxExemplars)
        << "exemplar_body_idx[" << e << "]=" << exemplar_body_idx[e] << " out of range [0, "
        << kMaxExemplars << ")";
    h_params.best_body_idx[e] = static_cast<int>(exemplar_body_idx[e]);
    h_params.exemplar_tile_shape[e] = static_cast<int>(exemplar_tile_shapes[e]);
    h_params.exemplar_kernel_id[e] = static_cast<int>(exemplar_kernel_ids[e]);
  }
  // Copy exemplar rows from caller-provided tensor (host or device) into
  // the host params struct. The struct layout is [kMaxExemplars][kMaxKnnExperts],
  // so we need to scatter from a packed [num_exemplars][num_local_experts]
  // input into a strided slot.
  std::vector<float> src_flat(n_exemplars * num_local_experts);
  cudaMemcpyKind kind = (exemplar_norm_flat.device().device_type == kDLCUDA)
                            ? cudaMemcpyDeviceToHost
                            : cudaMemcpyHostToHost;
  DA_CUDA_CHECK(cudaMemcpy(src_flat.data(), exemplar_norm_flat.data_ptr(),
                           src_flat.size() * sizeof(float), kind));
  for (int e = 0; e < n_exemplars; e++) {
    for (int i = 0; i < static_cast<int>(num_local_experts); i++) {
      h_params.exemplar_norm[e * kMaxKnnExperts + i] = src_flat[e * num_local_experts + i];
    }
    // Pad columns past num_local_experts with zeros so dot products ignore
    // them; the kernel reads up to num_local_experts only, but better safe.
    for (int i = num_local_experts; i < kMaxKnnExperts; i++) {
      h_params.exemplar_norm[e * kMaxKnnExperts + i] = 0.0f;
    }
  }

  DA_CUDA_CHECK(
      cudaMemcpy(state->d_params, &h_params, sizeof(DAKnnParams), cudaMemcpyHostToDevice));
  state->h_params = h_params;
  state->top_k = static_cast<int>(top_k);
}

void da_upload_knn_exemplars(tvm::ffi::Tensor exemplar_norm_flat,
                             tvm::ffi::Array<int64_t> exemplar_body_idx,
                             tvm::ffi::Array<int64_t> exemplar_tile_shapes,
                             tvm::ffi::Array<int64_t> exemplar_kernel_ids,
                             tvm::ffi::Array<int64_t> tile_sizes, int64_t num_local_experts,
                             int64_t local_expert_offset, int64_t top_k,
                             int64_t num_tokens_bucket) {
  da_upload_knn_exemplars_with_handle(kDefaultSelectorHandle, exemplar_norm_flat, exemplar_body_idx,
                                      exemplar_tile_shapes, exemplar_kernel_ids, tile_sizes,
                                      num_local_experts, local_expert_offset, top_k,
                                      num_tokens_bucket);
}

// ---------------------------------------------------------------------------
// CUDA graph with SWITCH conditional node for N-way tile dispatch.
// ---------------------------------------------------------------------------

// Decision kernel arguments — stable storage required because
// cudaGraphAddKernelNode captures pointers to each parameter, and those
// pointers must remain valid for the lifetime of the graph.
struct DecisionKernelArgs {
  const int32_t* topk_ids_ptr;
  int num_elements;
  DAKnnParams* d_params;
  int32_t* d_selected_tile_idx;
  int32_t* d_selected_tile_n;
  cudaGraphConditionalHandle switch_handle;
};

struct HistogramCountsKernelArgs {
  const int32_t* topk_ids_ptr;
  int num_elements;
  DAKnnParams* d_params;
  int32_t* counts;
};

struct SelectFromCountsKernelArgs {
  // Non-const: the kernel re-zeros entries as it consumes them (see
  // da_knn_compact_counts_for_sort_and_zero) so the next replay's histogram
  // pass sees a zero buffer without a separate zero kernel in the graph.
  int32_t* counts;
  int count_upper_bound;
  DAKnnParams* d_params;
  int32_t* d_selected_tile_idx;
  int32_t* d_selected_tile_n;
  cudaGraphConditionalHandle switch_handle;
};

struct SelectFromGlobalCountsKernelArgs {
  const int32_t* counts;
  int count_upper_bound;
  DAKnnParams* d_params;
  int32_t* d_selected_tile_idx;
  int32_t* d_selected_tile_n;
  cudaGraphConditionalHandle switch_handle;
};

struct FusedHistDecisionKernelArgs {
  const int32_t* topk_ids_ptr;
  int num_elements;
  DAKnnParams* d_params;
  int32_t* counts;
  unsigned int* d_block_done;
  int32_t* d_selected_tile_idx;
  int32_t* d_selected_tile_n;
  cudaGraphConditionalHandle switch_handle;
};

static int choose_knn_histogram_blocks(int num_elements) {
  constexpr int kItemsPerBlock =
      da_heuristic::kKnnSplitHistogramBlockThreads * da_heuristic::kKnnSplitHistogramItemsPerThread;
  int blocks = (num_elements + kItemsPerBlock - 1) / kItemsPerBlock;
  return std::max(1, std::min(blocks, da_heuristic::kKnnSplitHistogramMaxBlocks));
}

static bool split_knn_pdl_enabled() {
  const char* v = std::getenv("FLASHINFER_DA_KNN_SPLIT_PDL");
  return !(v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
                 std::strcmp(v, "False") == 0));
}

// PDL between the upstream routing kernel (e.g. routingIndicesClusterKernel)
// and the DA decision kernel. Same default-on policy as split_knn_pdl_enabled
// — disabling is for A/B comparison only.
static bool routing_to_decision_pdl_enabled() {
  const char* v = std::getenv("FLASHINFER_DA_ROUTING_PDL");
  return !(v && (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 ||
                 std::strcmp(v, "False") == 0));
}

// When enabled, routing tails wire directly to the SWITCH node instead of
// the histogram/decision node.  This places routing and DA-decision on
// independent graph branches so the graph executor can overlap their
// execution on different SMs — true concurrency, not just PDL launch
// overlap.  Both branches must complete before the SWITCH fires.
static bool routing_concurrent_enabled() {
  const char* v = std::getenv("FLASHINFER_DA_ROUTING_CONCURRENT");
  return (v &&
          (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "True") == 0));
}

static int split_knn_min_elements() {
  const char* v = std::getenv("FLASHINFER_DA_KNN_SPLIT_MIN_ELEMENTS");
  if (!v || !v[0]) return da_heuristic::kKnnSplitHistogramMinElements;
  char* end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if (end == v || parsed <= 0) return da_heuristic::kKnnSplitHistogramMinElements;
  return static_cast<int>(std::min<long>(parsed, std::numeric_limits<int>::max()));
}

// Fused histogram + k-NN decision: single multi-block kernel replaces the
// histogram + select two-node graph path. Uses atomicInc last-block pattern
// for intra-kernel synchronization. Off by default until benchmarked.
static bool fused_knn_enabled() {
  static const bool enabled = []() {
    const char* v = std::getenv("FLASHINFER_DA_FUSED");
    return v && v[0] != '\0' && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "False") != 0;
  }();
  return enabled;
}

// Real inline-switch API.
//
// Designed to be called by the Python MoE wrapper when it detects that the
// caller's stream is in capture mode (e.g. inside vLLM's torch.cuda.graph()
// region). Flow:
//
//   ctx_id = da_inline_switch_begin(topk_ids, bucket/top_k..., tile_sizes,
//                                     side_stream_handle)
//   for i in range(num_bodies):
//       da_inline_body_begin_capture(ctx_id, i)    # side_stream starts capturing
//       <Python launches real MoE kernels on side_stream with tactic for tile i>
//       da_inline_body_end_capture(ctx_id)          # side_stream ends capturing
//   da_inline_switch_end(ctx_id)                     # adds SWITCH to outer capture
//   da_inline_destroy(ctx_id)
//
// Allocator routing must be set up on the Python side via
// torch._C._cuda_beginAllocateToPool so that MoE internal workspace
// allocations (workspace_fc1/fc2 etc.) go to a pool that outlives the
// outer graph. The pool handle is managed on the Python side; C++ here
// only handles graph bookkeeping.
// ---------------------------------------------------------------------------
struct DAInlineContext {
  cudaGraph_t outer_graph = nullptr;  // vLLM's captured graph
  cudaGraphNode_t decision_node{};
  cudaGraphNode_t switch_node{};
  cudaGraphConditionalHandle switch_handle{};
  std::vector<cudaGraph_t> body_graphs;
  std::vector<cudaGraphNode_t> root_deps;
  cudaStream_t side_stream = nullptr;            // caller-supplied body capture stream
  cudaStream_t active_routing_stream = nullptr;  // caller-supplied routing branch stream
  cudaStream_t outer_stream = nullptr;           // vLLM's capture stream
  cudaEvent_t routing_start_event = nullptr;
  cudaEvent_t routing_done_event = nullptr;
  int num_bodies = 0;
  int active_body = -1;  // index of body currently capturing
  bool routing_capture_active = false;
  bool switch_added = false;
  bool direct_mode = false;
  std::unique_ptr<DecisionKernelArgs> decision_kargs;
  std::unique_ptr<HistogramCountsKernelArgs> histogram_counts_kargs;
  std::unique_ptr<SelectFromCountsKernelArgs> select_from_counts_kargs;
  std::unique_ptr<SelectFromGlobalCountsKernelArgs> select_from_global_counts_kargs;
  std::unique_ptr<FusedHistDecisionKernelArgs> fused_hist_decision_kargs;

  ~DAInlineContext() {
    if (routing_start_event) cudaEventDestroy(routing_start_event);
    if (routing_done_event) cudaEventDestroy(routing_done_event);
  }
};

static std::unordered_map<int64_t, std::unique_ptr<DAInlineContext>> g_inline_ctxs;
static int64_t g_next_inline_ctx = 1;

static int64_t da_inline_switch_begin_impl(TensorView topk_ids_tv,
                                           const TensorView* expert_counts_tv,
                                           int64_t selector_handle, int64_t num_tokens_bucket,
                                           int64_t top_k, int64_t num_local_experts,
                                           int64_t local_expert_offset,
                                           tvm::ffi::Array<int64_t> tile_sizes_arr,
                                           int64_t side_stream_handle) {
  cudaStream_t outer_stream = get_stream(topk_ids_tv.device());

  // Fetch the outer in-progress capture graph + current tail dependencies.
  cudaStreamCaptureStatus status;
  cudaGraph_t outer_graph;
  const cudaGraphNode_t* cur_deps = nullptr;
  const cudaGraphEdgeData* cur_edge_data = nullptr;
  size_t n_cur_deps = 0;
  DA_CUDA_CHECK(cudaStreamGetCaptureInfo(outer_stream, &status, nullptr, &outer_graph, &cur_deps,
                                         &cur_edge_data, &n_cur_deps));
  FLASHINFER_CHECK(status == cudaStreamCaptureStatusActive,
                   "da_inline_switch_begin requires the caller stream to be capturing");

  // Look up uploaded DAKNNv2 state for this selector context and token bucket.
  DAKnnStateKey state_key = make_state_key(selector_handle, static_cast<int>(num_tokens_bucket));
  auto knn_state_it = g_da_knn_states.find(state_key);
  FLASHINFER_CHECK(knn_state_it != g_da_knn_states.end(),
                   "No DAKNNv2 selector state uploaded for bucket=", num_tokens_bucket,
                   " top_k=", top_k, " num_local_experts=", num_local_experts,
                   " (call da_upload_knn_exemplars first).");
  DAKnnState* knn_state = knn_state_it->second.get();
  FLASHINFER_CHECK(
      validate_knn_state_topology(*knn_state, top_k, num_local_experts, local_expert_offset),
      "DAKNNv2 selector topology does not match the uploaded state.");
  const bool use_global_counts = expert_counts_tv != nullptr;
  if (expert_counts_tv != nullptr) {
    FLASHINFER_CHECK(expert_counts_tv->dtype() == dl_int32, "expert_counts must have dtype int32.");
    FLASHINFER_CHECK(expert_counts_tv->ndim() == 1, "expert_counts must be a 1D tensor.");
    FLASHINFER_CHECK(
        expert_counts_tv->size(0) >= static_cast<int64_t>(local_expert_offset + num_local_experts),
        "expert_counts is too small for the local expert range.");
  }

  auto ctx = std::make_unique<DAInlineContext>();
  ctx->outer_graph = outer_graph;
  ctx->outer_stream = outer_stream;
  ctx->side_stream = reinterpret_cast<cudaStream_t>(side_stream_handle);
  int knn_num_exemplars_inline = knn_state->h_params.num_exemplars;
  int knn_num_bodies_inline = 0;
  for (int e = 0; e < knn_num_exemplars_inline; ++e) {
    knn_num_bodies_inline =
        std::max(knn_num_bodies_inline, knn_state->h_params.best_body_idx[e] + 1);
  }
  if (n_cur_deps > 0) {
    ctx->root_deps.assign(cur_deps, cur_deps + n_cur_deps);
  }

  int static_body_idx = -1;
  if (knn_num_exemplars_inline == 1) {
    static_body_idx = knn_state->h_params.best_body_idx[0];
  } else if (knn_num_exemplars_inline > 1) {
    bool all_same = true;
    for (int e = 1; e < knn_num_exemplars_inline; ++e) {
      if (knn_state->h_params.exemplar_tile_shape[e] !=
              knn_state->h_params.exemplar_tile_shape[0] ||
          knn_state->h_params.exemplar_kernel_id[e] != knn_state->h_params.exemplar_kernel_id[0]) {
        all_same = false;
        break;
      }
    }
    if (all_same) {
      static_body_idx = knn_state->h_params.best_body_idx[0];
    }
  }
  ctx->num_bodies = knn_num_bodies_inline;
  if (da_verbose_enabled()) {
    std::fprintf(stderr,
                 "[DA C++ switch_begin] bucket=%d top_k=%d nle=%d leo=%d "
                 "num_tiles=%d knn_exemplars=%d num_bodies=%d static_body_idx=%d",
                 static_cast<int>(num_tokens_bucket), static_cast<int>(top_k),
                 static_cast<int>(num_local_experts), static_cast<int>(local_expert_offset),
                 static_cast<int>(tile_sizes_arr.size()), knn_state->h_params.num_exemplars,
                 ctx->num_bodies, static_body_idx);
    for (int e = 0; e < knn_state->h_params.num_exemplars; ++e) {
      std::fprintf(stderr, " ex[%d]=(tile=%d,kid=%d,body=%d)", e,
                   knn_state->h_params.exemplar_tile_shape[e],
                   knn_state->h_params.exemplar_kernel_id[e], knn_state->h_params.best_body_idx[e]);
    }
    std::fprintf(stderr, "\n");
  }
  const bool use_static_switch = static_body_idx >= 0 && static_body_idx < ctx->num_bodies;

  // Direct mode: single body, no SWITCH node, no decision kernel.
  // Subsequent body(0) captures on the outer stream; zero overhead vs NoDA.
  if (use_static_switch && ctx->num_bodies == 1) {
    ctx->direct_mode = true;
    int64_t ctx_id = g_next_inline_ctx++;
    g_inline_ctxs[ctx_id] = std::move(ctx);
    return ctx_id;
  }

  // Conditional handle must be owned by the outer graph (that's the graph the
  // SWITCH will live in). Per CUDA docs, a handle is graph-scoped.
  DA_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &ctx->switch_handle, outer_graph,
      static_cast<unsigned int>(use_static_switch ? static_body_idx : 0),
      cudaGraphCondAssignDefault));

  // Decision kernel node — reads topk_ids + uploaded selector state, sets SWITCH.
  int num_elements = 1;
  for (int i = 0; i < topk_ids_tv.ndim(); ++i)
    num_elements *= static_cast<int>(topk_ids_tv.size(i));
  int knn_num_exemplars = knn_state->h_params.num_exemplars;
  const bool use_split_knn_histogram =
      knn_num_exemplars > 1 && num_elements >= split_knn_min_elements();
  const bool use_split_knn_pdl = use_split_knn_histogram && split_knn_pdl_enabled();

  if (!use_static_switch) {
    if (use_global_counts) {
      ctx->select_from_global_counts_kargs = std::make_unique<SelectFromGlobalCountsKernelArgs>();
      auto* select_ka = ctx->select_from_global_counts_kargs.get();
      select_ka->counts = static_cast<const int32_t*>(expert_counts_tv->data_ptr());
      select_ka->count_upper_bound = num_elements;
      select_ka->d_params = knn_state->d_params;
      select_ka->d_selected_tile_idx = knn_state->d_selected_tile_idx;
      select_ka->d_selected_tile_n = knn_state->d_selected_tile_n;
      select_ka->switch_handle = ctx->switch_handle;
      void* select_args[] = {&select_ka->counts,
                             &select_ka->count_upper_bound,
                             &select_ka->d_params,
                             &select_ka->d_selected_tile_idx,
                             &select_ka->d_selected_tile_n,
                             &select_ka->switch_handle};

      cudaKernelNodeParams select_kp{};
      select_kp.func =
          reinterpret_cast<void*>(da_heuristic::da_knn_select_tile_from_global_counts_graph_kernel);
      select_kp.gridDim = {1, 1, 1};
      select_kp.blockDim = {da_heuristic::kKnnSelectorBlockThreads, 1, 1};
      select_kp.kernelParams = select_args;
      DA_CUDA_CHECK(cudaGraphAddKernelNode(&ctx->decision_node, outer_graph, cur_deps, n_cur_deps,
                                           &select_kp));
    } else if (use_split_knn_histogram && fused_knn_enabled()) {
      // Fused histogram + decision: single multi-block kernel with atomicInc
      // last-block pattern. Uses knn_state buffers allocated at upload time.
      const bool use_fused_pdl =
          n_cur_deps > 0 && cur_deps != nullptr && routing_to_decision_pdl_enabled();
      ctx->fused_hist_decision_kargs = std::make_unique<FusedHistDecisionKernelArgs>();
      auto* fused_ka = ctx->fused_hist_decision_kargs.get();
      fused_ka->topk_ids_ptr = static_cast<const int32_t*>(topk_ids_tv.data_ptr());
      fused_ka->num_elements = num_elements;
      fused_ka->d_params = knn_state->d_params;
      fused_ka->counts = knn_state->d_counts;
      fused_ka->d_block_done = knn_state->d_block_done;
      fused_ka->d_selected_tile_idx = knn_state->d_selected_tile_idx;
      fused_ka->d_selected_tile_n = knn_state->d_selected_tile_n;
      fused_ka->switch_handle = ctx->switch_handle;
      void* fused_args[] = {&fused_ka->topk_ids_ptr,      &fused_ka->num_elements,
                            &fused_ka->d_params,          &fused_ka->counts,
                            &fused_ka->d_block_done,      &fused_ka->d_selected_tile_idx,
                            &fused_ka->d_selected_tile_n, &fused_ka->switch_handle};

      cudaKernelNodeParams fused_kp{};
      fused_kp.func =
          use_fused_pdl ? reinterpret_cast<void*>(da_heuristic::da_fused_hist_decision_graph_kernel<
                                                  true, da_heuristic::KnnDecisionPolicy>)
                        : reinterpret_cast<void*>(da_heuristic::da_fused_hist_decision_graph_kernel<
                                                  false, da_heuristic::KnnDecisionPolicy>);
      fused_kp.gridDim = {static_cast<unsigned int>(choose_knn_histogram_blocks(num_elements)), 1,
                          1};
      fused_kp.blockDim = {da_heuristic::kKnnSplitHistogramBlockThreads, 1, 1};
      fused_kp.kernelParams = fused_args;

      if (use_fused_pdl) {
        DA_CUDA_CHECK(
            cudaGraphAddKernelNode(&ctx->decision_node, outer_graph, nullptr, 0, &fused_kp));
        cudaGraphEdgeData pdl_edge{};
        pdl_edge.from_port = cudaGraphKernelNodePortLaunchCompletion;
        pdl_edge.type = cudaGraphDependencyTypeProgrammatic;
        std::vector<cudaGraphEdgeData> edges(n_cur_deps, pdl_edge);
        std::vector<cudaGraphNode_t> dst(n_cur_deps, ctx->decision_node);
        DA_CUDA_CHECK(
            cudaGraphAddDependencies(outer_graph, cur_deps, dst.data(), edges.data(), n_cur_deps));
      } else {
        DA_CUDA_CHECK(cudaGraphAddKernelNode(&ctx->decision_node, outer_graph, cur_deps, n_cur_deps,
                                             &fused_kp));
      }
    } else if (use_split_knn_histogram) {
      // Split histogram + select: two kernel nodes with optional PDL overlap.
      ctx->histogram_counts_kargs = std::make_unique<HistogramCountsKernelArgs>();
      auto* hist_ka = ctx->histogram_counts_kargs.get();
      hist_ka->topk_ids_ptr = static_cast<const int32_t*>(topk_ids_tv.data_ptr());
      hist_ka->num_elements = num_elements;
      hist_ka->d_params = knn_state->d_params;
      hist_ka->counts = knn_state->d_counts;
      void* hist_args[] = {&hist_ka->topk_ids_ptr, &hist_ka->num_elements, &hist_ka->d_params,
                           &hist_ka->counts};
      cudaKernelNodeParams hist_kp{};
      hist_kp.func =
          use_split_knn_pdl
              ? reinterpret_cast<void*>(da_heuristic::da_knn_histogram_counts_graph_kernel<true>)
              : reinterpret_cast<void*>(da_heuristic::da_knn_histogram_counts_graph_kernel<false>);
      hist_kp.gridDim = {static_cast<unsigned int>(choose_knn_histogram_blocks(num_elements)), 1,
                         1};
      hist_kp.blockDim = {da_heuristic::kKnnSplitHistogramBlockThreads, 1, 1};
      hist_kp.kernelParams = hist_args;
      cudaGraphNode_t histogram_node = nullptr;
      DA_CUDA_CHECK(
          cudaGraphAddKernelNode(&histogram_node, outer_graph, cur_deps, n_cur_deps, &hist_kp));

      ctx->select_from_counts_kargs = std::make_unique<SelectFromCountsKernelArgs>();
      auto* select_ka = ctx->select_from_counts_kargs.get();
      select_ka->counts = knn_state->d_counts;
      select_ka->count_upper_bound = num_elements;
      select_ka->d_params = knn_state->d_params;
      select_ka->d_selected_tile_idx = knn_state->d_selected_tile_idx;
      select_ka->d_selected_tile_n = knn_state->d_selected_tile_n;
      select_ka->switch_handle = ctx->switch_handle;
      void* select_args[] = {&select_ka->counts,
                             &select_ka->count_upper_bound,
                             &select_ka->d_params,
                             &select_ka->d_selected_tile_idx,
                             &select_ka->d_selected_tile_n,
                             &select_ka->switch_handle};
      cudaKernelNodeParams select_kp{};
      select_kp.func = use_split_knn_pdl
                           ? reinterpret_cast<void*>(
                                 da_heuristic::da_knn_select_tile_from_counts_graph_kernel<true>)
                           : reinterpret_cast<void*>(
                                 da_heuristic::da_knn_select_tile_from_counts_graph_kernel<false>);
      select_kp.gridDim = {1, 1, 1};
      select_kp.blockDim = {da_heuristic::kKnnSelectorBlockThreads, 1, 1};
      select_kp.kernelParams = select_args;

      if (use_split_knn_pdl) {
        DA_CUDA_CHECK(
            cudaGraphAddKernelNode(&ctx->decision_node, outer_graph, nullptr, 0, &select_kp));
        cudaGraphEdgeData pdl_edge{};
        pdl_edge.from_port = cudaGraphKernelNodePortLaunchCompletion;
        pdl_edge.type = cudaGraphDependencyTypeProgrammatic;
        DA_CUDA_CHECK(cudaGraphAddDependencies(outer_graph, &histogram_node, &ctx->decision_node,
                                               &pdl_edge, 1));
      } else {
        DA_CUDA_CHECK(cudaGraphAddKernelNode(&ctx->decision_node, outer_graph, &histogram_node, 1,
                                             &select_kp));
      }
    } else {
      ctx->decision_kargs = std::make_unique<DecisionKernelArgs>();
      auto* ka = ctx->decision_kargs.get();
      ka->topk_ids_ptr = static_cast<const int32_t*>(topk_ids_tv.data_ptr());
      ka->num_elements = num_elements;
      ka->d_params = knn_state->d_params;
      ka->d_selected_tile_idx = knn_state->d_selected_tile_idx;
      ka->d_selected_tile_n = knn_state->d_selected_tile_n;
      ka->switch_handle = ctx->switch_handle;
      void* args[] = {&ka->topk_ids_ptr,        &ka->num_elements,      &ka->d_params,
                      &ka->d_selected_tile_idx, &ka->d_selected_tile_n, &ka->switch_handle};
      cudaKernelNodeParams kp{};
      kp.func = reinterpret_cast<void*>(da_heuristic::da_knn_select_tile_graph_kernel);
      kp.gridDim = {1, 1, 1};
      kp.blockDim = {da_heuristic::kKnnSelectorBlockThreads, 1, 1};
      kp.sharedMemBytes = 0;
      kp.kernelParams = args;
      DA_CUDA_CHECK(
          cudaGraphAddKernelNode(&ctx->decision_node, outer_graph, cur_deps, n_cur_deps, &kp));
    }
  }

  // SWITCH conditional node — depends on decision, has N body subgraphs.
  {
    cudaGraphNodeParams sp{};
    sp.type = cudaGraphNodeTypeConditional;
    sp.conditional.handle = ctx->switch_handle;
    sp.conditional.type = cudaGraphCondTypeSwitch;
    sp.conditional.size = static_cast<unsigned int>(ctx->num_bodies);
    if (use_static_switch) {
      DA_CUDA_CHECK(
          cudaGraphAddNode(&ctx->switch_node, outer_graph, cur_deps, nullptr, n_cur_deps, &sp));
    } else {
      DA_CUDA_CHECK(
          cudaGraphAddNode(&ctx->switch_node, outer_graph, &ctx->decision_node, nullptr, 1, &sp));
    }
    ctx->body_graphs.resize(ctx->num_bodies);
    for (int i = 0; i < ctx->num_bodies; ++i) ctx->body_graphs[i] = sp.conditional.phGraph_out[i];
  }

  int64_t ctx_id = g_next_inline_ctx++;
  g_inline_ctxs[ctx_id] = std::move(ctx);
  return ctx_id;
}

int64_t da_inline_switch_begin(TensorView topk_ids_tv, int64_t num_tokens_bucket, int64_t top_k,
                               int64_t num_local_experts, int64_t local_expert_offset,
                               tvm::ffi::Array<int64_t> tile_sizes_arr,
                               int64_t side_stream_handle) {
  return da_inline_switch_begin_impl(topk_ids_tv, nullptr, kDefaultSelectorHandle,
                                     num_tokens_bucket, top_k, num_local_experts,
                                     local_expert_offset, tile_sizes_arr, side_stream_handle);
}

int64_t da_inline_switch_begin_with_handle(int64_t selector_handle, TensorView topk_ids_tv,
                                           int64_t num_tokens_bucket, int64_t top_k,
                                           int64_t num_local_experts, int64_t local_expert_offset,
                                           tvm::ffi::Array<int64_t> tile_sizes_arr,
                                           int64_t side_stream_handle) {
  return da_inline_switch_begin_impl(topk_ids_tv, nullptr, selector_handle, num_tokens_bucket,
                                     top_k, num_local_experts, local_expert_offset, tile_sizes_arr,
                                     side_stream_handle);
}

int64_t da_inline_switch_begin_from_counts(TensorView topk_ids_tv, TensorView expert_counts_tv,
                                           int64_t num_tokens_bucket, int64_t top_k,
                                           int64_t num_local_experts, int64_t local_expert_offset,
                                           tvm::ffi::Array<int64_t> tile_sizes_arr,
                                           int64_t side_stream_handle) {
  return da_inline_switch_begin_impl(topk_ids_tv, &expert_counts_tv, kDefaultSelectorHandle,
                                     num_tokens_bucket, top_k, num_local_experts,
                                     local_expert_offset, tile_sizes_arr, side_stream_handle);
}

int64_t da_inline_switch_begin_from_counts_with_handle(
    int64_t selector_handle, TensorView topk_ids_tv, TensorView expert_counts_tv,
    int64_t num_tokens_bucket, int64_t top_k, int64_t num_local_experts,
    int64_t local_expert_offset, tvm::ffi::Array<int64_t> tile_sizes_arr,
    int64_t side_stream_handle) {
  return da_inline_switch_begin_impl(topk_ids_tv, &expert_counts_tv, selector_handle,
                                     num_tokens_bucket, top_k, num_local_experts,
                                     local_expert_offset, tile_sizes_arr, side_stream_handle);
}

int64_t da_inline_get_num_bodies(int64_t ctx_id) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  return static_cast<int64_t>(it->second->num_bodies);
}

void da_inline_routing_begin_capture(int64_t ctx_id, int64_t stream_handle) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  auto& ctx = it->second;
  if (ctx->direct_mode) return;
  FLASHINFER_CHECK(ctx->active_body == -1,
                   "cannot capture routing while a conditional body is capturing");
  FLASHINFER_CHECK(!ctx->routing_capture_active, "routing capture is already active");

  if (!ctx->routing_start_event) {
    DA_CUDA_CHECK(cudaEventCreateWithFlags(&ctx->routing_start_event, cudaEventDisableTiming));
  }
  if (!ctx->routing_done_event) {
    DA_CUDA_CHECK(cudaEventCreateWithFlags(&ctx->routing_done_event, cudaEventDisableTiming));
  }

  ctx->active_routing_stream = reinterpret_cast<cudaStream_t>(stream_handle);
  DA_CUDA_CHECK(cudaEventRecord(ctx->routing_start_event, ctx->outer_stream));
  DA_CUDA_CHECK(cudaStreamWaitEvent(ctx->active_routing_stream, ctx->routing_start_event, 0));
  ctx->routing_capture_active = true;
}

void da_inline_routing_end_capture(int64_t ctx_id) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  auto& ctx = it->second;
  if (ctx->direct_mode) return;
  FLASHINFER_CHECK(ctx->routing_capture_active, "no routing capture is active");
  FLASHINFER_CHECK(ctx->active_routing_stream != nullptr, "routing stream is not set");

  DA_CUDA_CHECK(cudaEventRecord(ctx->routing_done_event, ctx->active_routing_stream));

  cudaStreamCaptureStatus status;
  cudaGraph_t captured_graph = nullptr;
  const cudaGraphNode_t* tail_deps = nullptr;
  const cudaGraphEdgeData* tail_edge_data = nullptr;
  size_t n_tail_deps = 0;
  DA_CUDA_CHECK(cudaStreamGetCaptureInfo(ctx->active_routing_stream, &status, nullptr,
                                         &captured_graph, &tail_deps, &tail_edge_data,
                                         &n_tail_deps));
  FLASHINFER_CHECK(status == cudaStreamCaptureStatusActive,
                   "routing side stream is not actively capturing");
  FLASHINFER_CHECK(captured_graph == ctx->outer_graph,
                   "routing side stream is capturing into the wrong graph");

  std::vector<cudaGraphNode_t> routing_tails;
  if (n_tail_deps > 0) {
    routing_tails.assign(tail_deps, tail_deps + n_tail_deps);
  }
  if (!routing_tails.empty()) {
    std::vector<cudaGraphNode_t> switch_deps(routing_tails.size(), ctx->switch_node);
    DA_CUDA_CHECK(cudaGraphAddDependencies(ctx->outer_graph, routing_tails.data(),
                                           switch_deps.data(), nullptr, routing_tails.size()));
  }

  DA_CUDA_CHECK(cudaStreamWaitEvent(ctx->outer_stream, ctx->routing_done_event, 0));
  ctx->routing_capture_active = false;
  ctx->active_routing_stream = nullptr;
}

void da_inline_body_begin_capture(int64_t ctx_id, int64_t body_index) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  auto& ctx = it->second;
  if (ctx->direct_mode) return;
  FLASHINFER_CHECK(body_index >= 0 && body_index < ctx->num_bodies,
                   "body_index out of range: ", body_index);
  FLASHINFER_CHECK(ctx->active_body == -1, "a body capture is already in progress; end it first");
  FLASHINFER_CHECK(!ctx->routing_capture_active,
                   "cannot capture a conditional body while routing capture is active");
  DA_CUDA_CHECK(cudaStreamBeginCaptureToGraph(ctx->side_stream, ctx->body_graphs[body_index],
                                              nullptr, nullptr, 0,
                                              cudaStreamCaptureModeThreadLocal));
  ctx->active_body = static_cast<int>(body_index);
}

void da_inline_body_end_capture(int64_t ctx_id) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  auto& ctx = it->second;
  if (ctx->direct_mode) return;
  FLASHINFER_CHECK(ctx->active_body != -1, "no body capture in progress");
  cudaGraph_t _unused = nullptr;
  DA_CUDA_CHECK(cudaStreamEndCapture(ctx->side_stream, &_unused));
  ctx->active_body = -1;
}

void da_inline_switch_end(int64_t ctx_id) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  auto& ctx = it->second;
  if (ctx->direct_mode) return;
  FLASHINFER_CHECK(ctx->active_body == -1,
                   "da_inline_switch_end called while a body capture is in progress");
  FLASHINFER_CHECK(!ctx->routing_capture_active,
                   "da_inline_switch_end called while routing capture is in progress");
  FLASHINFER_CHECK(!ctx->switch_added, "da_inline_switch_end already called");
  // Rewire the outer capture stream so its subsequent ops wait for SWITCH.
  DA_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      ctx->outer_stream, &ctx->switch_node, /*dependencyData=*/nullptr,
      /*numDependencies=*/1, cudaStreamSetCaptureDependencies));
  ctx->switch_added = true;
}

int64_t da_inline_is_direct_mode(int64_t ctx_id) {
  auto it = g_inline_ctxs.find(ctx_id);
  FLASHINFER_CHECK(it != g_inline_ctxs.end(), "Unknown inline ctx: ", ctx_id);
  return it->second->direct_mode ? 1 : 0;
}

void da_inline_destroy(int64_t ctx_id) { g_inline_ctxs.erase(ctx_id); }

int64_t da_get_static_knn_tile_with_handle(int64_t selector_handle, int64_t num_tokens_bucket,
                                           int64_t top_k, int64_t num_local_experts,
                                           int64_t local_expert_offset,
                                           tvm::ffi::Array<int64_t> tile_sizes_arr) {
  DAKnnStateKey key = make_state_key(selector_handle, static_cast<int>(num_tokens_bucket));
  auto it = g_da_knn_states.find(key);
  if (it == g_da_knn_states.end()) return -1;
  if (!validate_knn_state_topology(*it->second, top_k, num_local_experts, local_expert_offset)) {
    return -1;
  }
  DAKnnParams const& params = it->second->h_params;

  if (params.num_exemplars != 1) return -1;
  int const tile_n = params.exemplar_tile_shape[0];
  for (int64_t i = 0; i < tile_sizes_arr.size(); ++i) {
    if (static_cast<int64_t>(tile_n) == tile_sizes_arr[i]) {
      return static_cast<int64_t>(tile_n);
    }
  }
  return -1;
}

int64_t da_get_static_knn_tile(int64_t num_tokens_bucket, int64_t top_k, int64_t num_local_experts,
                               int64_t local_expert_offset,
                               tvm::ffi::Array<int64_t> tile_sizes_arr) {
  return da_get_static_knn_tile_with_handle(kDefaultSelectorHandle, num_tokens_bucket, top_k,
                                            num_local_experts, local_expert_offset, tile_sizes_arr);
}

void da_destroy_knn_selector(int64_t selector_handle) {
  for (auto it = g_da_knn_states.begin(); it != g_da_knn_states.end();) {
    if (it->first.selector_handle == selector_handle) {
      it = g_da_knn_states.erase(it);
    } else {
      ++it;
    }
  }
}

// ---------------------------------------------------------------------------
}  // namespace flashinfer

// ---------------------------------------------------------------------------
// TVM-FFI registrations.
// ---------------------------------------------------------------------------
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_upload_knn_exemplars, flashinfer::da_upload_knn_exemplars);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_upload_knn_exemplars_with_handle,
                              flashinfer::da_upload_knn_exemplars_with_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_switch_begin, flashinfer::da_inline_switch_begin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_switch_begin_with_handle,
                              flashinfer::da_inline_switch_begin_with_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_get_num_bodies, flashinfer::da_inline_get_num_bodies);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_is_direct_mode, flashinfer::da_inline_is_direct_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_switch_begin_from_counts,
                              flashinfer::da_inline_switch_begin_from_counts);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_switch_begin_from_counts_with_handle,
                              flashinfer::da_inline_switch_begin_from_counts_with_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_routing_begin_capture,
                              flashinfer::da_inline_routing_begin_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_routing_end_capture,
                              flashinfer::da_inline_routing_end_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_body_begin_capture,
                              flashinfer::da_inline_body_begin_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_body_end_capture, flashinfer::da_inline_body_end_capture);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_switch_end, flashinfer::da_inline_switch_end);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_inline_destroy, flashinfer::da_inline_destroy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_get_static_knn_tile, flashinfer::da_get_static_knn_tile);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_get_static_knn_tile_with_handle,
                              flashinfer::da_get_static_knn_tile_with_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(da_destroy_knn_selector, flashinfer::da_destroy_knn_selector);
