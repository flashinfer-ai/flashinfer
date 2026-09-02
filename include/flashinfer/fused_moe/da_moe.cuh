/*
 * Copyright (c) 2026 by FlashInfer team.
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

#ifndef FLASHINFER_FUSED_MOE_DA_MOE_CUH_
#define FLASHINFER_FUSED_MOE_DA_MOE_CUH_

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "flashinfer/fused_moe/da_config.cuh"

namespace flashinfer::da_moe {

/** Captured graph and frontier dependencies observed at one injection point. */
struct ActiveCaptureContext {
  // CUDA runtime status observed at the graph-injection point.
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  // CUDA-assigned generation identifier for the active outer capture.
  unsigned long long capture_id = 0;
  // Non-owning handle to the graph currently being captured.
  cudaGraph_t graph = nullptr;
  // Current outer-graph frontier inherited by both independent DA roots.
  std::vector<cudaGraphNode_t> dependencies;
};

/** Runtime topology retained by a concrete DA graph injector for inspection. */
struct GraphTopology {
  // CUDA stream-capture identifier associated with this inspection.
  // FFI[0]; sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.capture_id.
  unsigned long long capture_id = 0;
  // Number of nodes in the outer captured graph after DA injection.
  // FFI[1]; sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.outer_node_count.
  size_t outer_node_count = 0;
  // Number of dependency edges in the outer captured graph.
  // FFI[2]; sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.outer_edge_count.
  size_t outer_edge_count = 0;
  // Number of conditional SWITCH nodes injected into the outer graph.
  // FFI[3]; sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.conditional_node_count.
  size_t conditional_node_count = 0;
  // Number of predecessor dependencies inherited by the selector node.
  // FFI[5]; sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.selector_dependency_count.
  size_t selector_dependency_count = 0;
  // Number of predecessor dependencies inherited by the pre-body-work node.
  // FFI[6]; sync with
  // flashinfer/fused_moe/da_moe.py:DAGraphTopology.parallel_work_dependency_count.
  size_t parallel_work_dependency_count = 0;
  // Whether runtime inspection proves selector and preamble root independence.
  // FFI[7]; sync with
  // flashinfer/fused_moe/da_moe.py:DAGraphTopology.is_selector_preamble_parallelizable.
  bool is_selector_preamble_parallelizable = false;
  // Whether this invocation is transitively ordered after the prior workspace-lane user.
  // FFI[8]; sync with
  // flashinfer/fused_moe/da_moe.py:DAGraphTopology.is_workspace_lane_serialized.
  bool is_workspace_lane_serialized = false;
  // Number of serial workspace-lane invocations represented by this inspection.
  // FFI[9]; sync with
  // flashinfer/fused_moe/da_moe.py:DAGraphTopology.workspace_lane_invocation_count.
  size_t workspace_lane_invocation_count = 0;
  // Node count for each conditional body graph in body-index order.
  // FFI[10:]; sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.body_node_counts.
  std::vector<size_t> body_node_counts;
  // Internal terminal node passed back to the lane owner for the next serial invocation.
  cudaGraphNode_t conditional_node = nullptr;
};

/** Query capture state with the CUDA 12.x or CUDA 13.x ABI. */
inline cudaError_t GetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* status,
                                  unsigned long long* capture_id, cudaGraph_t* graph,
                                  const cudaGraphNode_t** dependencies, size_t* num_dependencies) {
#if CUDART_VERSION >= 13000
  return cudaStreamGetCaptureInfo(stream, status, capture_id, graph, dependencies, nullptr,
                                  num_dependencies);
#else
  return cudaStreamGetCaptureInfo_v3(stream, status, capture_id, graph, dependencies, nullptr,
                                     num_dependencies);
#endif
}

/** Add a generic graph node with the CUDA 12.x or CUDA 13.x ABI. */
inline cudaError_t AddGraphNode(cudaGraphNode_t* node, cudaGraph_t graph,
                                const cudaGraphNode_t* dependencies, size_t num_dependencies,
                                cudaGraphNodeParams* params) {
#if CUDART_VERSION >= 13000
  return cudaGraphAddNode(node, graph, dependencies, nullptr, num_dependencies, params);
#else
  return cudaGraphAddNode_v2(node, graph, dependencies, nullptr, num_dependencies, params);
#endif
}

/** Own one typed kernel launch without imposing a common argument ABI. */
template <typename Kernel, typename... KernelArgs>
class TypedKernelLaunch {
 public:
  /** Retain a kernel, launch configuration, and its backend-specific arguments. */
  TypedKernelLaunch(Kernel kernel, dim3 grid, dim3 block, size_t shared_memory_bytes,
                    KernelArgs... arguments)
      : kernel_(kernel),
        grid_(grid),
        block_(block),
        shared_memory_bytes_(shared_memory_bytes),
        arguments_(std::move(arguments)...) {}

  /** Launch the retained kernel and typed arguments on a CUDA stream. */
  cudaError_t Launch(cudaStream_t stream) {
    return WithArgumentPointers([&](void** argument_pointers) {
      return cudaLaunchKernel(reinterpret_cast<const void*>(kernel_), grid_, block_,
                              argument_pointers, shared_memory_bytes_, stream);
    });
  }

  /** Add the retained kernel and typed arguments to a CUDA graph. */
  cudaError_t AddToGraph(cudaGraphNode_t* node, cudaGraph_t graph,
                         const cudaGraphNode_t* dependencies, size_t num_dependencies) {
    return WithArgumentPointers([&](void** argument_pointers) {
      cudaKernelNodeParams params{};
      params.func = reinterpret_cast<void*>(kernel_);
      params.gridDim = grid_;
      params.blockDim = block_;
      params.sharedMemBytes = shared_memory_bytes_;
      params.kernelParams = argument_pointers;
      return cudaGraphAddKernelNode(node, graph, dependencies, num_dependencies, &params);
    });
  }

 private:
  /** Expose mutable pointers to retained argument values for one CUDA API call. */
  template <typename Operation>
  cudaError_t WithArgumentPointers(Operation&& operation) {
    return std::apply(
        [&](auto&... arguments) {
          void* argument_pointers[] = {static_cast<void*>(&arguments)...};
          return operation(argument_pointers);
        },
        arguments_);
  }

  // Concrete CUDA kernel function owned by this launch record.
  Kernel kernel_;
  // Grid dimensions used for eager launch or graph-node construction.
  dim3 grid_;
  // Thread-block dimensions used for eager launch or graph-node construction.
  dim3 block_;
  // Dynamic shared-memory bytes requested by the concrete kernel.
  size_t shared_memory_bytes_;
  // Exact typed argument values for the concrete backend kernel signature.
  std::tuple<KernelArgs...> arguments_;
};

/** Create a typed eager/graph launch record from an arbitrary kernel ABI. */
template <typename Kernel, typename... KernelArgs>
auto MakeTypedKernelLaunch(Kernel kernel, dim3 grid, dim3 block, size_t shared_memory_bytes,
                           KernelArgs... arguments) {
  return TypedKernelLaunch<Kernel, KernelArgs...>(kernel, grid, block, shared_memory_bytes,
                                                  std::move(arguments)...);
}

/** Replace stream-capture dependencies with the CUDA 12.x or CUDA 13.x ABI. */
inline cudaError_t SetCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* dependencies,
                                          size_t num_dependencies) {
#if CUDART_VERSION >= 13000
  return cudaStreamUpdateCaptureDependencies(stream, dependencies, nullptr, num_dependencies,
                                             cudaStreamSetCaptureDependencies);
#else
  return cudaStreamUpdateCaptureDependencies_v2(stream, dependencies, nullptr, num_dependencies,
                                                cudaStreamSetCaptureDependencies);
#endif
}

/** Count graph edges with the CUDA 12.x or CUDA 13.x ABI. */
inline cudaError_t GetGraphEdgeCount(cudaGraph_t graph, size_t* edge_count) {
#if CUDART_VERSION >= 13000
  return cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, edge_count);
#else
  return cudaGraphGetEdges(graph, nullptr, nullptr, edge_count);
#endif
}

/** Read the concrete predecessor nodes attached to one CUDA Graph node. */
inline cudaError_t GetGraphNodeDependenciesView(cudaGraphNode_t node, cudaGraphNode_t* dependencies,
                                                size_t* num_dependencies) {
#if CUDART_VERSION >= 13000
  return cudaGraphNodeGetDependencies(node, dependencies, nullptr, num_dependencies);
#else
  return cudaGraphNodeGetDependencies(node, dependencies, num_dependencies);
#endif
}

/** Copy the concrete predecessor nodes attached to one CUDA Graph node. */
inline cudaError_t GetGraphNodeDependencies(cudaGraphNode_t node,
                                            std::vector<cudaGraphNode_t>* dependencies) {
  size_t num_dependencies = 0;
  cudaError_t status = GetGraphNodeDependenciesView(node, nullptr, &num_dependencies);
  if (status != cudaSuccess) {
    return status;
  }
  dependencies->resize(num_dependencies);
  if (num_dependencies == 0) {
    return cudaSuccess;
  }
  return GetGraphNodeDependenciesView(node, dependencies->data(), &num_dependencies);
}

/** Return whether two dependency lists contain the same concrete graph nodes. */
inline bool HaveSameGraphDependencies(const std::vector<cudaGraphNode_t>& first,
                                      const std::vector<cudaGraphNode_t>& second) {
  return first.size() == second.size() &&
         std::is_permutation(first.begin(), first.end(), second.begin());
}

/** Return whether one graph node is equal to or transitively depends on an ancestor. */
inline cudaError_t GraphNodeDependsOn(cudaGraphNode_t node, cudaGraphNode_t ancestor,
                                      bool* depends_on) {
  if (node == ancestor) {
    *depends_on = true;
    return cudaSuccess;
  }
  std::vector<cudaGraphNode_t> frontier{node};
  std::vector<cudaGraphNode_t> visited;
  while (!frontier.empty()) {
    cudaGraphNode_t current = frontier.back();
    frontier.pop_back();
    if (std::find(visited.begin(), visited.end(), current) != visited.end()) {
      continue;
    }
    visited.push_back(current);
    std::vector<cudaGraphNode_t> dependencies;
    cudaError_t status = GetGraphNodeDependencies(current, &dependencies);
    if (status != cudaSuccess) {
      return status;
    }
    if (std::find(dependencies.begin(), dependencies.end(), ancestor) != dependencies.end()) {
      *depends_on = true;
      return cudaSuccess;
    }
    frontier.insert(frontier.end(), dependencies.begin(), dependencies.end());
  }
  *depends_on = false;
  return cudaSuccess;
}

/** Prove that an active capture frontier is safe to reuse one serial workspace lane. */
inline cudaError_t ValidateWorkspaceLaneSequence(const ActiveCaptureContext& context,
                                                 unsigned long long expected_capture_id,
                                                 cudaGraphNode_t previous_conditional_node,
                                                 bool* is_serialized) {
  // A lane's first invocation has no predecessor token; later invocations require both fields.
  if (expected_capture_id == 0 && previous_conditional_node == nullptr) {
    *is_serialized = true;
    return cudaSuccess;
  }
  if (expected_capture_id == 0 || previous_conditional_node == nullptr ||
      context.capture_id != expected_capture_id) {
    *is_serialized = false;
    return cudaSuccess;
  }

  // Any current frontier node transitively descending from the previous conditional establishes
  // the happens-before path inherited by both sibling roots of the next invocation.
  for (cudaGraphNode_t dependency : context.dependencies) {
    bool depends_on = false;
    cudaError_t status = GraphNodeDependsOn(dependency, previous_conditional_node, &depends_on);
    if (status != cudaSuccess) {
      return status;
    }
    if (depends_on) {
      *is_serialized = true;
      return cudaSuccess;
    }
  }
  *is_serialized = false;
  return cudaSuccess;
}

/** Snapshot the active stream capture graph and its current dependency frontier. */
inline cudaError_t GetActiveCaptureContext(cudaStream_t stream, ActiveCaptureContext* context) {
  const cudaGraphNode_t* dependency_view = nullptr;
  size_t num_dependencies = 0;
  cudaError_t status = GetCaptureInfo(stream, &context->status, &context->capture_id,
                                      &context->graph, &dependency_view, &num_dependencies);
  if (status != cudaSuccess) {
    return status;
  }
  context->dependencies.clear();
  if (num_dependencies != 0) {
    context->dependencies.assign(dependency_view, dependency_view + num_dependencies);
  }
  return cudaSuccess;
}

/**
 * Add pre-body work before its sibling selector, then join both at a SWITCH.
 *
 * The body builder receives each CUDA-owned child graph and its stable body index.
 * It must populate the complete body, including any layout-finalization kernel.
 */
template <typename BodyGraphBuilder>
cudaError_t AddDASwitchToCapture(const ActiveCaptureContext& context, cudaStream_t stream,
                                 cudaGraphConditionalHandle conditional_handle,
                                 cudaKernelNodeParams* parallel_work_params,
                                 cudaKernelNodeParams* selector_params, int num_bodies,
                                 unsigned long long expected_capture_id,
                                 cudaGraphNode_t previous_conditional_node,
                                 BodyGraphBuilder&& body_builder, GraphTopology* topology) {
  // Reject cross-capture or unordered lane reuse before adding any node to the outer graph.
  cudaError_t status =
      ValidateWorkspaceLaneSequence(context, expected_capture_id, previous_conditional_node,
                                    &topology->is_workspace_lane_serialized);
  if (status != cudaSuccess || !topology->is_workspace_lane_serialized) {
    topology->capture_id = context.capture_id;
    return status;
  }

  // Give the longer pre-body work the first independent root slot so it can
  // begin before and overlap the selector without creating a serial edge.
  cudaGraphNode_t parallel_work_node = nullptr;
  status = cudaGraphAddKernelNode(&parallel_work_node, context.graph, context.dependencies.data(),
                                  context.dependencies.size(), parallel_work_params);
  if (status != cudaSuccess) {
    return status;
  }

  cudaGraphNode_t selector_node = nullptr;
  status = cudaGraphAddKernelNode(&selector_node, context.graph, context.dependencies.data(),
                                  context.dependencies.size(), selector_params);
  if (status != cudaSuccess) {
    return status;
  }

  cudaGraphNode_t switch_dependencies[] = {parallel_work_node, selector_node};
  cudaGraphNodeParams conditional_params{};
  conditional_params.type = cudaGraphNodeTypeConditional;
  conditional_params.conditional.handle = conditional_handle;
  conditional_params.conditional.type = cudaGraphCondTypeSwitch;
  conditional_params.conditional.size = num_bodies;
  cudaGraphNode_t conditional_node = nullptr;
  status =
      AddGraphNode(&conditional_node, context.graph, switch_dependencies, 2, &conditional_params);
  if (status != cudaSuccess) {
    return status;
  }

  std::vector<cudaGraph_t> body_graphs;
  body_graphs.reserve(num_bodies);
  for (int body_index = 0; body_index < num_bodies; ++body_index) {
    cudaGraph_t body_graph = conditional_params.conditional.phGraph_out[body_index];
    status = body_builder(body_graph, body_index);
    if (status != cudaSuccess) {
      return status;
    }
    body_graphs.push_back(body_graph);
  }

  status = SetCaptureDependencies(stream, &conditional_node, 1);
  if (status != cudaSuccess) {
    return status;
  }

  std::vector<cudaGraphNode_t> parallel_work_dependencies;
  status = GetGraphNodeDependencies(parallel_work_node, &parallel_work_dependencies);
  if (status != cudaSuccess) {
    return status;
  }
  std::vector<cudaGraphNode_t> selector_dependencies;
  status = GetGraphNodeDependencies(selector_node, &selector_dependencies);
  if (status != cudaSuccess) {
    return status;
  }

  topology->capture_id = context.capture_id;
  topology->conditional_node_count = 1;
  topology->workspace_lane_invocation_count = 1;
  topology->conditional_node = conditional_node;
  topology->selector_dependency_count = selector_dependencies.size();
  topology->parallel_work_dependency_count = parallel_work_dependencies.size();
  topology->is_selector_preamble_parallelizable =
      HaveSameGraphDependencies(selector_dependencies, context.dependencies) &&
      HaveSameGraphDependencies(parallel_work_dependencies, context.dependencies) &&
      std::find(selector_dependencies.begin(), selector_dependencies.end(), parallel_work_node) ==
          selector_dependencies.end() &&
      std::find(parallel_work_dependencies.begin(), parallel_work_dependencies.end(),
                selector_node) == parallel_work_dependencies.end();
  status = cudaGraphGetNodes(context.graph, nullptr, &topology->outer_node_count);
  if (status != cudaSuccess) {
    return status;
  }
  status = GetGraphEdgeCount(context.graph, &topology->outer_edge_count);
  if (status != cudaSuccess) {
    return status;
  }
  topology->body_node_counts.clear();
  topology->body_node_counts.reserve(body_graphs.size());
  for (cudaGraph_t body_graph : body_graphs) {
    size_t body_node_count = 0;
    status = cudaGraphGetNodes(body_graph, nullptr, &body_node_count);
    if (status != cudaSuccess) {
      return status;
    }
    topology->body_node_counts.push_back(body_node_count);
  }
  return cudaSuccess;
}

constexpr int kDASelectorBlockThreads = 256;
constexpr int kDASelectorHistogramItemsPerThread = 8;

/** Return the smallest power of two greater than or equal to a positive integer. */
__device__ __forceinline__ unsigned int DANextPowerOfTwo(unsigned int value) {
  if (value == 0) {
    return 1;
  }
  if ((value & (value - 1)) == 0) {
    return value;
  }
  return 1U << (32 - __clz(value - 1));
}

/** Sort up to one block of compacted counts in descending order with bitonic exchange. */
__device__ __forceinline__ void DASortCountsRegisterBitonicDescending(int* counts,
                                                                      int sort_length) {
  int value = threadIdx.x < sort_length ? counts[threadIdx.x] : INT_MIN;
  for (int sequence_length = 2; sequence_length <= sort_length; sequence_length <<= 1) {
    for (int stride = sequence_length >> 1; stride > 0; stride >>= 1) {
      int partner;
      if (stride >= 32) {
        counts[threadIdx.x] = value;
        __syncthreads();
        partner = counts[threadIdx.x ^ stride];
        __syncthreads();
      } else {
        partner = __shfl_xor_sync(0xFFFFFFFF, value, stride);
      }
      const bool first_sequence_half = (threadIdx.x & sequence_length) == 0;
      const bool lower_exchange_lane = (threadIdx.x & stride) == 0;
      const bool keep_maximum = first_sequence_half == lower_exchange_lane;
      value = keep_maximum ? max(value, partner) : min(value, partner);
    }
  }
  counts[threadIdx.x] = threadIdx.x < sort_length ? value : 0;
  __syncthreads();
}

/** Sort at most one warp of compacted counts in descending order. */
__device__ __forceinline__ void DASortCountsWarpBitonicDescending(int* counts, int sort_length) {
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    // Grow sorted bitonic subsequences while every lane owns a strided set of values.
    for (int sequence_length = 2; sequence_length <= sort_length; sequence_length <<= 1) {
      for (int stride = sequence_length >> 1; stride > 0; stride >>= 1) {
        __syncwarp();
        // Exchange each pair exactly once and orient it for the current bitonic half.
        for (int index = lane; index < sort_length; index += 32) {
          const int partner_index = index ^ stride;
          if (partner_index > index) {
            const bool first_sequence_half = (index & sequence_length) == 0;
            const int first = counts[index];
            const int second = counts[partner_index];
            const bool should_swap = first_sequence_half ? first < second : first > second;
            if (should_swap) {
              counts[index] = second;
              counts[partner_index] = first;
            }
          }
        }
      }
    }
  }
  __syncthreads();
}

/** Sort a two-item-per-thread count vector in descending order with CUB radix sort. */
template <int ItemsPerThread>
__device__ __forceinline__ void DASortCountsCubDescending(int* counts, int num_experts,
                                                          int64_t count_upper_bound) {
  using BlockSort = cub::BlockRadixSort<int, kDASelectorBlockThreads, ItemsPerThread>;
  __shared__ typename BlockSort::TempStorage sort_storage;

  int items[ItemsPerThread];
#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = threadIdx.x * ItemsPerThread + item;
    items[item] = index < num_experts ? counts[index] : 0;
  }

  int end_bit = 32;
  if (count_upper_bound > 0 && count_upper_bound <= INT_MAX) {
    const unsigned int upper_bound = static_cast<unsigned int>(count_upper_bound);
    end_bit = 32 - __clz(upper_bound);
  }
  BlockSort(sort_storage).SortDescendingBlockedToStriped(items, 0, end_bit);

#pragma unroll
  for (int item = 0; item < ItemsPerThread; ++item) {
    const int index = threadIdx.x + item * kDASelectorBlockThreads;
    if (index < num_experts) {
      counts[index] = items[item];
    }
  }
  __syncthreads();
}

/** Compact nonzero histogram bins and return the number of values that require sorting. */
template <typename CountType>
__device__ __forceinline__ int DACompactCountsForSort(int* counts, const CountType* raw_counts,
                                                      int num_experts) {
  if (num_experts <= kDASelectorBlockThreads) {
    __shared__ int warp_nonzero[8];
    __shared__ int warp_minimum[8];
    __shared__ int warp_maximum[8];
    __shared__ int total_nonzero;
    __shared__ int minimum_count;
    __shared__ int maximum_count;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int value = threadIdx.x < num_experts ? static_cast<int>(raw_counts[threadIdx.x]) : 0;
    const bool active = threadIdx.x < num_experts && value > 0;
    const unsigned int mask = __ballot_sync(0xFFFFFFFF, active);
    const int rank = __popc(mask & ((1U << lane) - 1U));
    const int nonzero = __popc(mask);

    int local_minimum = active ? value : INT_MAX;
    int local_maximum = active ? value : 0;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_minimum = min(local_minimum, __shfl_down_sync(0xFFFFFFFF, local_minimum, offset));
      local_maximum = max(local_maximum, __shfl_down_sync(0xFFFFFFFF, local_maximum, offset));
    }
    if (lane == 0) {
      warp_nonzero[warp] = nonzero;
      warp_minimum[warp] = local_minimum;
      warp_maximum[warp] = local_maximum;
    }
    __syncthreads();

    int prefix = 0;
#pragma unroll
    for (int previous_warp = 0; previous_warp < 8; ++previous_warp) {
      if (previous_warp < warp) {
        prefix += warp_nonzero[previous_warp];
      }
    }
    if (active) {
      counts[prefix + rank] = value;
    }
    if (threadIdx.x == 0) {
      int count = 0;
      int minimum = INT_MAX;
      int maximum = 0;
#pragma unroll
      for (int candidate_warp = 0; candidate_warp < 8; ++candidate_warp) {
        count += warp_nonzero[candidate_warp];
        minimum = min(minimum, warp_minimum[candidate_warp]);
        maximum = max(maximum, warp_maximum[candidate_warp]);
      }
      total_nonzero = count;
      minimum_count = minimum;
      maximum_count = maximum;
    }
    __syncthreads();

    for (int index = threadIdx.x + total_nonzero; index < num_experts; index += blockDim.x) {
      counts[index] = 0;
    }
    __syncthreads();
    if (total_nonzero <= 1 || minimum_count == maximum_count) {
      return 1;
    }
    return total_nonzero;
  }

  __shared__ int total_nonzero;
  __shared__ int minimum_count;
  __shared__ int maximum_count;
  if (threadIdx.x == 0) {
    total_nonzero = 0;
    minimum_count = INT_MAX;
    maximum_count = 0;
  }
  __syncthreads();

  for (int index = threadIdx.x; index < num_experts; index += blockDim.x) {
    const int value = static_cast<int>(raw_counts[index]);
    if (value > 0) {
      const int destination = atomicAdd(&total_nonzero, 1);
      counts[destination] = value;
      atomicMin(&minimum_count, value);
      atomicMax(&maximum_count, value);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x + total_nonzero; index < num_experts; index += blockDim.x) {
    counts[index] = 0;
  }
  __syncthreads();
  if (total_nonzero <= 1 || minimum_count == maximum_count) {
    return 1;
  }
  return total_nonzero;
}

/** Compute one unnormalized cosine-ranking dot product per exemplar warp. */
template <int MaxExemplars>
__device__ __forceinline__ void DAComputeSimilarities(const int* counts,
                                                      const float* exemplar_spectra,
                                                      float* similarities, int num_experts,
                                                      int num_selector_exemplars) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (warp < num_selector_exemplars && warp < MaxExemplars) {
    const float* exemplar = exemplar_spectra + warp * num_experts;
    float partial = 0.0F;
    for (int index = lane; index < num_experts; index += kDASelectorBlockThreads) {
#pragma unroll
      for (int offset = 0; offset < kDASelectorBlockThreads; offset += 32) {
        const int element = index + offset;
        if (element < num_experts) {
          partial += static_cast<float>(counts[element]) * exemplar[element];
        }
      }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    }
    if (lane == 0) {
      similarities[warp] = partial;
    }
  }
}

/** Sort the live spectrum, choose its nearest exemplar, and activate the mapped body. */
template <int MaxExperts, int MaxExemplars>
__device__ __forceinline__ void DAFinishSelection(int* counts, float* similarities,
                                                  const float* exemplar_spectra,
                                                  const int32_t* exemplar_body_indices,
                                                  int num_experts, int num_selector_exemplars,
                                                  int64_t assignment_numel, int sort_items,
                                                  cudaGraphConditionalHandle conditional_handle,
                                                  int32_t* selected_body) {
  int sort_length = static_cast<int>(DANextPowerOfTwo(static_cast<unsigned int>(sort_items)));
  sort_length = min(sort_length, MaxExperts);
  for (int index = threadIdx.x + num_experts; index < sort_length; index += blockDim.x) {
    counts[index] = 0;
  }
  __syncthreads();

  if (sort_length > 1) {
    if (sort_length <= 32) {
      DASortCountsWarpBitonicDescending(counts, sort_length);
    } else if (sort_length <= kDASelectorBlockThreads) {
      DASortCountsRegisterBitonicDescending(counts, sort_length);
    } else {
      static_assert(MaxExperts <= 2 * kDASelectorBlockThreads,
                    "DA selector supports at most two radix-sort items per thread");
      DASortCountsCubDescending<2>(counts, num_experts, assignment_numel);
    }
  }

  DAComputeSimilarities<MaxExemplars>(counts, exemplar_spectra, similarities, num_experts,
                                      num_selector_exemplars);
  __syncthreads();
  if (threadIdx.x == 0) {
    int nearest_exemplar = 0;
    float nearest_similarity = similarities[0];
    for (int exemplar = 1; exemplar < num_selector_exemplars; ++exemplar) {
      if (similarities[exemplar] > nearest_similarity) {
        nearest_similarity = similarities[exemplar];
        nearest_exemplar = exemplar;
      }
    }
    const int body_index = exemplar_body_indices[nearest_exemplar];
    selected_body[0] = body_index;
    cudaGraphSetConditional(conditional_handle, static_cast<unsigned int>(body_index));
  }
}

/** Select the nearest uploaded load spectrum from packed or unpacked routing entries. */
template <int MaxExperts, int MaxExemplars, bool PackedRoutingEntries = false,
          typename RoutingEntry = int32_t>
__global__ void DASelectorKernel(const RoutingEntry* routing_entries, int64_t assignment_numel,
                                 int num_experts, const float* exemplar_spectra,
                                 const int32_t* exemplar_body_indices, int num_selector_exemplars,
                                 cudaGraphConditionalHandle conditional_handle,
                                 int32_t* selected_body) {
  static_assert(MaxExperts <= 2 * kDASelectorBlockThreads,
                "DA selector supports at most two expert bins per thread");
  static_assert(MaxExperts < USHRT_MAX,
                "DA selector expert and sentinel bins must fit in unsigned short");
  static_assert(MaxExemplars <= kDASelectorBlockThreads / 32,
                "DA selector assigns exactly one warp to every exemplar");
  static_assert(!PackedRoutingEntries || std::is_same_v<RoutingEntry, int32_t>,
                "packed routing entries require int32 storage");
  if (num_selector_exemplars == 1) {
    if (threadIdx.x == 0) {
      const int body_index = exemplar_body_indices[0];
      selected_body[0] = body_index;
      cudaGraphSetConditional(conditional_handle, static_cast<unsigned int>(body_index));
    }
    return;
  }

  __shared__ int counts[MaxExperts];
  __shared__ unsigned int histogram[MaxExperts + 1];
  __shared__ float similarities[MaxExemplars];
  using BlockHistogram = cub::BlockHistogram<unsigned short, kDASelectorBlockThreads,
                                             kDASelectorHistogramItemsPerThread, MaxExperts + 1,
                                             cub::BLOCK_HISTO_ATOMIC>;
  __shared__ typename BlockHistogram::TempStorage histogram_storage;

  BlockHistogram(histogram_storage).InitHistogram(histogram);
  __syncthreads();
  for (int64_t base = 0; base < assignment_numel;
       base += blockDim.x * kDASelectorHistogramItemsPerThread) {
    unsigned short items[kDASelectorHistogramItemsPerThread];
#pragma unroll
    for (int item = 0; item < kDASelectorHistogramItemsPerThread; ++item) {
      const int64_t index = base + threadIdx.x + item * blockDim.x;
      int expert = -1;
      if (index < assignment_numel) {
        expert = routing_entries[index];
        if constexpr (PackedRoutingEntries) {
          expert >>= 16;
        }
      }
      items[item] =
          static_cast<unsigned short>(expert >= 0 && expert < num_experts ? expert : MaxExperts);
    }
    BlockHistogram(histogram_storage).Composite(items, histogram);
    __syncthreads();
  }

  const int sort_items = DACompactCountsForSort(counts, histogram, num_experts);
  DAFinishSelection<MaxExperts, MaxExemplars>(
      counts, similarities, exemplar_spectra, exemplar_body_indices, num_experts,
      num_selector_exemplars, assignment_numel, sort_items, conditional_handle, selected_body);
}

}  // namespace flashinfer::da_moe

#endif  // FLASHINFER_FUSED_MOE_DA_MOE_CUH_
