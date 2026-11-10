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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "flashinfer/fused_moe/da_moe.cuh"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

namespace flashinfer::da_moe::testing {

constexpr int kThreads = 256;

using tvm::ffi::Array;

using GraphInspectionRecord = GraphTopology;

/** Captured public-weight and lane-workspace pointers for one mock layer. */
struct MockWorkspaceBindingInspection {
  // Per-layer routing-weight pointer captured by this invocation's body ABI.
  int64_t expert_weights = 0;
  // Lane-owned activation pointer captured by this invocation's body ABI.
  int64_t activation_workspace = 0;
  // Lane-owned intermediate pointer captured by this invocation's finalize chain.
  int64_t intermediate_workspace = 0;
};

std::mutex g_graph_inspection_mutex;
GraphInspectionRecord g_graph_inspection;
std::vector<MockWorkspaceBindingInspection> g_workspace_binding_inspections;

/** Validate and narrow a runtime expert count against compiled mock capacity. */
int ValidateNumExperts(int64_t num_experts) {
  if (num_experts <= 0 || num_experts > kDAMaxExperts) {
    TVM_FFI_THROW(ValueError) << "num_experts must be in [1, " << kDAMaxExperts << "], received "
                              << num_experts;
  }
  return static_cast<int>(num_experts);
}

/** Validate and narrow an exemplar count against immutable selector capacity. */
int ValidateNumSelectorExemplars(int64_t num_selector_exemplars) {
  if (num_selector_exemplars <= 0 || num_selector_exemplars > kDAMaxExemplars) {
    TVM_FFI_THROW(ValueError) << "num_selector_exemplars must be in [1, " << kDAMaxExemplars
                              << "], received " << num_selector_exemplars;
  }
  return static_cast<int>(num_selector_exemplars);
}

/** Validate a unique-body count against immutable plan capacity. */
int ValidateNumBodies(int64_t num_bodies) {
  if (num_bodies <= 0 || num_bodies > kDAMaxBodies) {
    TVM_FFI_THROW(ValueError) << "num_bodies must be in [1, " << kDAMaxBodies << "], received "
                              << num_bodies;
  }
  return static_cast<int>(num_bodies);
}

/** Convert a supported scalar to float for canonical mock arithmetic. */
template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
  return static_cast<float>(value);
}

/** Convert a half scalar to float for canonical mock arithmetic. */
template <>
__device__ __forceinline__ float ToFloat<nv_half>(nv_half value) {
  return __half2float(value);
}

/** Convert a bfloat16 scalar to float for canonical mock arithmetic. */
template <>
__device__ __forceinline__ float ToFloat<nv_bfloat16>(nv_bfloat16 value) {
  return __bfloat162float(value);
}

/** Convert a float result back to a supported scalar. */
template <typename T>
__device__ __forceinline__ T FromFloat(float value) {
  return static_cast<T>(value);
}

/** Convert a float result back to half. */
template <>
__device__ __forceinline__ nv_half FromFloat<nv_half>(float value) {
  return __float2half_rn(value);
}

/** Convert a float result back to bfloat16. */
template <>
__device__ __forceinline__ nv_bfloat16 FromFloat<nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

/** Read the device global timer in nanoseconds. */
__device__ __forceinline__ uint64_t GlobalTimer() {
  uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

/** Keep one thread resident for a controlled amount of observable kernel time. */
__device__ __forceinline__ void DelayNanoseconds(uint64_t duration) {
  const uint64_t start = GlobalTimer();
  while (GlobalTimer() - start < duration) {
  }
}

/** Compute a distribution-sensitive delay for one mock tactic. */
template <int Tactic>
__device__ __forceinline__ uint64_t TacticDelayNanoseconds(int max_load, int num_assignments) {
  const float concentration =
      num_assignments == 0 ? 0.0f : static_cast<float>(max_load) / num_assignments;
  float delay_us;
  if constexpr (Tactic == 0) {
    delay_us = 8.0f + 100.0f * concentration;
  } else if constexpr (Tactic == 1) {
    delay_us = 88.0f - 72.0f * concentration;
  } else {
    delay_us = 42.0f;
  }
  return static_cast<uint64_t>(delay_us * 1000.0f);
}

/** Prepare mock body-specific routing metadata before the conditional SWITCH node. */
__global__ void MockPreBodyWorkKernel(const int32_t* expert_ids, int64_t assignment_numel,
                                      int num_experts, const int32_t* body_tile_ns, int num_bodies,
                                      int32_t* routing_metadata) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts) {
    return;
  }
  int count = 0;
  for (int64_t index = 0; index < assignment_numel; ++index) {
    count += expert_ids[index] == expert;
  }
  for (int body = 0; body < num_bodies; ++body) {
    const int tile_n = body_tile_ns[body];
    routing_metadata[body * num_experts + expert] = (count + tile_n - 1) / tile_n;
  }
}

/** Shape and expert-domain values shared within one dtype-owned body family. */
struct MockMoEShape {
  // Number of scalar elements processed by the body and finalize kernels.
  int64_t hidden_numel;
  // Number of routed token-to-expert assignments in the invocation.
  int64_t assignment_numel;
  // Runtime expert extent bounded by kDAMaxExperts.
  int num_experts;
};

/** Optional DA metadata slot consumed by a captured mock body. */
struct MockRoutingSlot {
  // Body-major routing metadata prepared before the conditional node.
  const int32_t* routing_metadata;
  // Deduplicated body index selecting one metadata row.
  int body_index;
};

/** Execute common arithmetic behind dtype-owned body kernels with unrelated ABIs. */
template <typename InputT, typename WeightT, typename OutputT, int Tactic>
__device__ __forceinline__ void ExecuteMockMoEBody(const InputT* hidden_states,
                                                   const int32_t* expert_ids,
                                                   const WeightT* expert_weights,
                                                   OutputT* workspace, int32_t* body_trace,
                                                   MockMoEShape shape, MockRoutingSlot routing,
                                                   int32_t* expert_loads) {
  for (int expert = threadIdx.x; expert < shape.num_experts; expert += blockDim.x) {
    expert_loads[expert] = 0;
  }
  __syncthreads();

  for (int64_t index = threadIdx.x; index < shape.assignment_numel; index += blockDim.x) {
    const int expert = expert_ids[index];
    if (expert >= 0 && expert < shape.num_experts) {
      atomicAdd(&expert_loads[expert], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int max_load = 0;
    for (int expert = 0; expert < shape.num_experts; ++expert) {
      max_load = max(max_load, expert_loads[expert]);
    }
    body_trace[0] = Tactic;
    const int first_expert = expert_ids[0];
    if (routing.routing_metadata != nullptr && first_expert >= 0 &&
        first_expert < shape.num_experts &&
        routing.routing_metadata[routing.body_index * shape.num_experts + first_expert] < 0) {
      body_trace[0] = -Tactic - 2;
    }
    DelayNanoseconds(TacticDelayNanoseconds<Tactic>(max_load, shape.assignment_numel));
  }

  constexpr int kLayoutShift = Tactic;
  for (int64_t index = threadIdx.x; index < shape.hidden_numel; index += blockDim.x) {
    const int64_t destination = (index + kLayoutShift) % shape.hidden_numel;
    const float value =
        ToFloat(hidden_states[index]) + ToFloat(expert_weights[index % shape.assignment_numel]);
    workspace[destination] = FromFloat<OutputT>(value);
  }
}

/** Execute one BF16 body through its split, ten-argument backend ABI. */
template <int Tactic>
__global__ void Bf16MockMoEBodyKernel(const nv_bfloat16* hidden_states, const int32_t* expert_ids,
                                      const nv_bfloat16* expert_weights, nv_bfloat16* workspace,
                                      int32_t* body_trace, int64_t hidden_numel,
                                      int64_t assignment_numel, int num_experts,
                                      const int32_t* routing_metadata, int body_index) {
  __shared__ int32_t expert_loads[kDAMaxExperts];
  MockMoEShape shape{hidden_numel, assignment_numel, num_experts};
  MockRoutingSlot routing{routing_metadata, body_index};
  ExecuteMockMoEBody<nv_bfloat16, nv_bfloat16, nv_bfloat16, Tactic>(
      hidden_states, expert_ids, expert_weights, workspace, body_trace, shape, routing,
      expert_loads);
}

/** Exact packed argument record owned only by the FP8 mock body ABI. */
struct Fp8MockMoEBodyArguments {
  // FP8 hidden-state input consumed by the FP8 body.
  const __nv_fp8_e4m3* hidden_states;
  // Live routing assignments consumed by the FP8 body.
  const int32_t* expert_ids;
  // Live BF16 routing weights consumed by the FP8 body.
  const nv_bfloat16* expert_weights;
  // BF16 intermediate layout written by the FP8 body.
  nv_bfloat16* workspace;
  // Device-visible tactic observation written by the FP8 body.
  int32_t* body_trace;
};

/** Execute one FP8 body through its packed, three-argument backend ABI. */
template <int Tactic>
__global__ void Fp8MockMoEBodyKernel(Fp8MockMoEBodyArguments arguments, MockMoEShape shape,
                                     MockRoutingSlot routing) {
  __shared__ int32_t expert_loads[kDAMaxExperts];
  ExecuteMockMoEBody<__nv_fp8_e4m3, nv_bfloat16, nv_bfloat16, Tactic>(
      arguments.hidden_states, arguments.expert_ids, arguments.expert_weights, arguments.workspace,
      arguments.body_trace, shape, routing, expert_loads);
}

/** Canonicalize a tactic-specific mock body layout into the public output tensor. */
template <typename T>
__global__ void MockMoEIntermediateKernel(const T* activation_workspace, T* intermediate_workspace,
                                          int64_t hidden_numel) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_numel;
       index += blockDim.x * gridDim.x) {
    intermediate_workspace[index] = activation_workspace[index];
  }
}

/** Canonicalize a tactic-specific mock intermediate layout into the public output tensor. */
template <typename T>
__global__ void MockMoEFinalizeKernel(const T* intermediate_workspace, T* output,
                                      int64_t hidden_numel, int tactic) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < hidden_numel;
       index += blockDim.x * gridDim.x) {
    const int64_t source = (index + tactic) % hidden_numel;
    output[index] = intermediate_workspace[source];
  }
}

/** Bind the BF16 body's exact argument ABI behind the generic runner. */
class Bf16MockMoEBodyAdapter {
 public:
  /** Bind the exact BF16 eager and graph-stable invocation arguments. */
  Bf16MockMoEBodyAdapter(const nv_bfloat16* hidden_states, const int32_t* expert_ids,
                         const nv_bfloat16* expert_weights, nv_bfloat16* output,
                         nv_bfloat16* workspace, int32_t* body_trace, MockMoEShape shape)
      : hidden_states_(hidden_states),
        expert_ids_(expert_ids),
        expert_weights_(expert_weights),
        output_(output),
        workspace_(workspace),
        body_trace_(body_trace),
        shape_(shape) {}

  /** Visit one BF16 tactic while retaining its exact ten-argument ABI. */
  template <typename Visitor>
  cudaError_t VisitBodyLaunch(int tactic, MockRoutingSlot routing, Visitor&& visitor) const {
    switch (tactic) {
      case 0:
        return VisitTactic<0>(routing, std::forward<Visitor>(visitor));
      case 1:
        return VisitTactic<1>(routing, std::forward<Visitor>(visitor));
      case 2:
        return VisitTactic<2>(routing, std::forward<Visitor>(visitor));
      default:
        TVM_FFI_THROW(ValueError) << "Unsupported BF16 mock MoE tactic: " << tactic;
    }
  }

  /** Return live expert IDs used by DA selector and pre-body work. */
  const int32_t* expert_ids() const { return expert_ids_; }

  /** Return the BF16 intermediate layout destination. */
  nv_bfloat16* workspace() const { return workspace_; }

  /** Return the BF16 canonical output destination. */
  nv_bfloat16* output() const { return output_; }

  /** Return immutable runtime shape values used by every body adapter. */
  const MockMoEShape& shape() const { return shape_; }

 private:
  /** Bind one BF16 tactic to its exact ten kernel arguments once. */
  template <int Tactic, typename Visitor>
  cudaError_t VisitTactic(MockRoutingSlot routing, Visitor&& visitor) const {
    return visitor(MakeTypedKernelLaunch(
        Bf16MockMoEBodyKernel<Tactic>, dim3(1), dim3(kThreads), 0, hidden_states_, expert_ids_,
        expert_weights_, workspace_, body_trace_, shape_.hidden_numel, shape_.assignment_numel,
        shape_.num_experts, routing.routing_metadata, routing.body_index));
  }

  // BF16 hidden-state input owned by the BF16 runner.
  const nv_bfloat16* hidden_states_;
  // Live routing assignments owned by the BF16 runner.
  const int32_t* expert_ids_;
  // BF16 routing weights owned by the BF16 runner.
  const nv_bfloat16* expert_weights_;
  // Canonical BF16 output owned by the BF16 runner.
  nv_bfloat16* output_;
  // BF16 body workspace owned by the BF16 runner.
  nv_bfloat16* workspace_;
  // Device-visible tactic observation owned by the BF16 runner.
  int32_t* body_trace_;
  // Immutable BF16 invocation shape and expert domain.
  MockMoEShape shape_;
};

/** Bind the FP8 body's exact argument ABI behind the generic runner. */
class Fp8MockMoEBodyAdapter {
 public:
  /** Bind the packed FP8 body arguments and separate canonical output. */
  Fp8MockMoEBodyAdapter(Fp8MockMoEBodyArguments body_arguments, nv_bfloat16* output,
                        MockMoEShape shape)
      : body_arguments_(body_arguments), output_(output), shape_(shape) {}

  /** Visit one FP8 tactic while retaining its exact three-argument ABI. */
  template <typename Visitor>
  cudaError_t VisitBodyLaunch(int tactic, MockRoutingSlot routing, Visitor&& visitor) const {
    switch (tactic) {
      case 0:
        return VisitTactic<0>(routing, std::forward<Visitor>(visitor));
      case 1:
        return VisitTactic<1>(routing, std::forward<Visitor>(visitor));
      case 2:
        return VisitTactic<2>(routing, std::forward<Visitor>(visitor));
      default:
        TVM_FFI_THROW(ValueError) << "Unsupported FP8 mock MoE tactic: " << tactic;
    }
  }

  /** Return live expert IDs used by DA selector and pre-body work. */
  const int32_t* expert_ids() const { return body_arguments_.expert_ids; }

  /** Return the BF16 intermediate layout destination used by the FP8 body. */
  nv_bfloat16* workspace() const { return body_arguments_.workspace; }

  /** Return the BF16 canonical output destination. */
  nv_bfloat16* output() const { return output_; }

  /** Return immutable runtime shape values used by the FP8 body. */
  const MockMoEShape& shape() const { return shape_; }

 private:
  /** Bind one FP8 tactic to its exact three packed kernel arguments once. */
  template <int Tactic, typename Visitor>
  cudaError_t VisitTactic(MockRoutingSlot routing, Visitor&& visitor) const {
    return visitor(MakeTypedKernelLaunch(Fp8MockMoEBodyKernel<Tactic>, dim3(1), dim3(kThreads), 0,
                                         body_arguments_, shape_, routing));
  }

  // Packed arguments required only by the FP8 body kernel ABI.
  Fp8MockMoEBodyArguments body_arguments_;
  // Canonical BF16 output used only after the FP8 body completes.
  nv_bfloat16* output_;
  // Immutable FP8 invocation shape and expert domain.
  MockMoEShape shape_;
};

using MockMoEBodyAdapter = std::variant<Bf16MockMoEBodyAdapter, Fp8MockMoEBodyAdapter>;

/** Dtype-agnostic mock of the TRTLLM MoERunner operation boundary. */
class MockMoERunner {
 public:
  /** Configure the generic runner with one exact-ABI BF16 adapter. */
  MockMoERunner(Bf16MockMoEBodyAdapter adapter, nv_bfloat16* intermediate_workspace)
      : body_adapter_(std::move(adapter)), intermediate_workspace_(intermediate_workspace) {}

  /** Configure the generic runner with one exact-ABI FP8 adapter. */
  MockMoERunner(Fp8MockMoEBodyAdapter adapter, nv_bfloat16* intermediate_workspace)
      : body_adapter_(std::move(adapter)), intermediate_workspace_(intermediate_workspace) {}

  /** Run one complete fixed-tactic MoE invocation on a CUDA stream. */
  void Run(int tactic, cudaStream_t stream) const {
    VisitBodyLaunch(tactic, MockRoutingSlot{nullptr, 0},
                    [&](auto launch) { return launch.Launch(stream); });
    const MockMoEShape& runtime_shape = shape();
    const int blocks = static_cast<int>((runtime_shape.hidden_numel + kThreads - 1) / kThreads);
    auto intermediate =
        MakeTypedKernelLaunch(MockMoEIntermediateKernel<nv_bfloat16>, dim3(blocks), dim3(kThreads),
                              0, workspace(), intermediate_workspace_, runtime_shape.hidden_numel);
    intermediate.Launch(stream);
    auto finalize = MakeTypedKernelLaunch(MockMoEFinalizeKernel<nv_bfloat16>, dim3(blocks),
                                          dim3(kThreads), 0, intermediate_workspace_, output(),
                                          runtime_shape.hidden_numel, tactic);
    finalize.Launch(stream);
  }

  /** Add one from-metadata body and finalize pair to a conditional child graph. */
  cudaError_t AddBodyGraph(cudaGraph_t graph, int tactic, const int32_t* routing_metadata,
                           int body_index) const {
    cudaGraphNode_t body_node = nullptr;
    cudaError_t status = VisitBodyLaunch(
        tactic, MockRoutingSlot{routing_metadata, body_index},
        [&](auto launch) { return launch.AddToGraph(&body_node, graph, nullptr, 0); });
    if (status != cudaSuccess) {
      return status;
    }
    const MockMoEShape& runtime_shape = shape();
    const int blocks = static_cast<int>((runtime_shape.hidden_numel + kThreads - 1) / kThreads);
    auto intermediate =
        MakeTypedKernelLaunch(MockMoEIntermediateKernel<nv_bfloat16>, dim3(blocks), dim3(kThreads),
                              0, workspace(), intermediate_workspace_, runtime_shape.hidden_numel);
    cudaGraphNode_t intermediate_node = nullptr;
    status = intermediate.AddToGraph(&intermediate_node, graph, &body_node, 1);
    if (status != cudaSuccess) {
      return status;
    }
    auto finalize = MakeTypedKernelLaunch(MockMoEFinalizeKernel<nv_bfloat16>, dim3(blocks),
                                          dim3(kThreads), 0, intermediate_workspace_, output(),
                                          runtime_shape.hidden_numel, tactic);
    cudaGraphNode_t finalize_node = nullptr;
    return finalize.AddToGraph(&finalize_node, graph, &intermediate_node, 1);
  }

  /** Return live expert IDs used by DA selector and pre-body work. */
  const int32_t* expert_ids() const {
    return std::visit([](const auto& adapter) { return adapter.expert_ids(); }, body_adapter_);
  }

  /** Return the BF16 intermediate layout destination. */
  nv_bfloat16* workspace() const {
    return std::visit([](const auto& adapter) { return adapter.workspace(); }, body_adapter_);
  }

  /** Return the BF16 canonical output destination. */
  nv_bfloat16* output() const {
    return std::visit([](const auto& adapter) { return adapter.output(); }, body_adapter_);
  }

  /** Return immutable runtime shape values used by the selected adapter. */
  const MockMoEShape& shape() const {
    return std::visit([](const auto& adapter) -> const MockMoEShape& { return adapter.shape(); },
                      body_adapter_);
  }

 private:
  /** Visit the configured exact-ABI adapter without exposing its dtype to DA. */
  template <typename Visitor>
  cudaError_t VisitBodyLaunch(int tactic, MockRoutingSlot routing, Visitor&& visitor) const {
    return std::visit(
        [&](const auto& adapter) {
          return adapter.VisitBodyLaunch(tactic, routing, std::forward<Visitor>(visitor));
        },
        body_adapter_);
  }

  // Exact-ABI body adapter selected once from the operation dtype.
  MockMoEBodyAdapter body_adapter_;
  // BF16 post-FC2 intermediate shared by every body behind this runner.
  nv_bfloat16* intermediate_workspace_;
};

/** Construct one generic runner by selecting its private adapter from dtype. */
MockMoERunner MakeMockMoERunner(TensorView hidden_states, TensorView expert_ids,
                                TensorView expert_weights, TensorView output,
                                TensorView activation_workspace, TensorView intermediate_workspace,
                                TensorView body_trace, int runtime_num_experts) {
  // Freeze shape values once so both exact dtype adapters see an identical operation domain.
  const MockMoEShape shape{hidden_states.numel(), expert_ids.numel(), runtime_num_experts};
  // Select one concrete body ABI without requiring unrelated dtype families to share arguments.
  switch (encode_dlpack_dtype(hidden_states.dtype())) {
    case bfloat16_code:
      return MockMoERunner(
          Bf16MockMoEBodyAdapter(static_cast<const nv_bfloat16*>(hidden_states.data_ptr()),
                                 static_cast<const int32_t*>(expert_ids.data_ptr()),
                                 static_cast<const nv_bfloat16*>(expert_weights.data_ptr()),
                                 static_cast<nv_bfloat16*>(output.data_ptr()),
                                 static_cast<nv_bfloat16*>(activation_workspace.data_ptr()),
                                 static_cast<int32_t*>(body_trace.data_ptr()), shape),
          static_cast<nv_bfloat16*>(intermediate_workspace.data_ptr()));
    case float8_e4m3fn_code:
      return MockMoERunner(
          Fp8MockMoEBodyAdapter(
              Fp8MockMoEBodyArguments{static_cast<const __nv_fp8_e4m3*>(hidden_states.data_ptr()),
                                      static_cast<const int32_t*>(expert_ids.data_ptr()),
                                      static_cast<const nv_bfloat16*>(expert_weights.data_ptr()),
                                      static_cast<nv_bfloat16*>(activation_workspace.data_ptr()),
                                      static_cast<int32_t*>(body_trace.data_ptr())},
              static_cast<nv_bfloat16*>(output.data_ptr()), shape),
          static_cast<nv_bfloat16*>(intermediate_workspace.data_ptr()));
    default:
      TVM_FFI_THROW(TypeError) << "MockMoERunner does not support the hidden-state dtype";
  }
}

/** Run one complete mock MoE invocation using the configured operation dtype. */
void RunMockMoE(TensorView hidden_states, TensorView expert_ids, TensorView expert_weights,
                TensorView output, TensorView activation_workspace,
                TensorView intermediate_workspace, TensorView body_trace, int64_t num_experts,
                int64_t tactic) {
  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  const cudaStream_t stream = get_stream(hidden_states.device());
  MakeMockMoERunner(hidden_states, expert_ids, expert_weights, output, activation_workspace,
                    intermediate_workspace, body_trace, ValidateNumExperts(num_experts))
      .Run(static_cast<int>(tactic), stream);
}

/** Distribution-aware graph layer composed around one standalone MockMoERunner. */
class MockDAMoERunner {
 public:
  /** Bind DA plan tensors while retaining the standalone runner by value. */
  MockDAMoERunner(MockMoERunner moe_runner, const float* exemplar_spectra,
                  const int32_t* exemplar_body_indices, const int32_t* body_tile_ns_device,
                  int32_t* selected_body, int32_t* routing_metadata)
      : moe_runner_(std::move(moe_runner)),
        exemplar_spectra_(exemplar_spectra),
        exemplar_body_indices_(exemplar_body_indices),
        body_tile_ns_device_(body_tile_ns_device),
        selected_body_(selected_body),
        routing_metadata_(routing_metadata) {}

  /** Inject selector, pre-body work, and configured bodies into an outer capture. */
  int64_t Capture(int num_selector_exemplars, const std::vector<int>& body_tactics,
                  unsigned long long expected_capture_id, cudaGraphNode_t previous_conditional_node,
                  cudaStream_t stream) const {
    // Resolve the outer graph and refuse to hide graph creation inside an eager invocation.
    ActiveCaptureContext context{};
    cudaError_t status = GetActiveCaptureContext(stream, &context);
    if (status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "Failed to inspect active DA capture: "
                                  << cudaGetErrorString(status);
    }
    if (context.status != cudaStreamCaptureStatusActive || context.graph == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "DA graph injection requires an active stream capture";
    }
    bool is_workspace_lane_serialized = false;
    status = ValidateWorkspaceLaneSequence(context, expected_capture_id, previous_conditional_node,
                                           &is_workspace_lane_serialized);
    if (status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "Failed to inspect DA workspace-lane ordering: "
                                  << cudaGetErrorString(status);
    }
    if (!is_workspace_lane_serialized) {
      return 0;
    }

    cudaGraphConditionalHandle conditional_handle = 0;
    status = cudaGraphConditionalHandleCreate(&conditional_handle, context.graph, 0,
                                              cudaGraphCondAssignDefault);
    if (status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "Failed to create DA conditional handle: "
                                  << cudaGetErrorString(status);
    }

    // Retain concrete argument storage while constructing two independent outer-graph roots.
    const MockMoEShape& shape = moe_runner_.shape();
    const int32_t* expert_ids = moe_runner_.expert_ids();
    const float* exemplar_spectra = exemplar_spectra_;
    const int32_t* exemplar_body_indices = exemplar_body_indices_;
    const int32_t* body_tile_ns_device = body_tile_ns_device_;
    int32_t* selected_body = selected_body_;
    int32_t* routing_metadata = routing_metadata_;
    int64_t assignment_numel = shape.assignment_numel;
    int num_experts = shape.num_experts;
    cudaKernelNodeParams selector_params{};
    selector_params.func =
        reinterpret_cast<void*>(DASelectorKernel<kDAMaxExperts, kDAMaxExemplars>);
    selector_params.gridDim = dim3(1);
    selector_params.blockDim = dim3(kDASelectorBlockThreads);
    void* selector_arguments[] = {
        &expert_ids,         &assignment_numel,      &num_experts,
        &exemplar_spectra,   &exemplar_body_indices, &num_selector_exemplars,
        &conditional_handle, &selected_body};
    selector_params.kernelParams = selector_arguments;

    int num_bodies = static_cast<int>(body_tactics.size());
    cudaKernelNodeParams parallel_params{};
    parallel_params.func = reinterpret_cast<void*>(MockPreBodyWorkKernel);
    parallel_params.gridDim = dim3((num_experts + kThreads - 1) / kThreads);
    parallel_params.blockDim = dim3(kThreads);
    void* parallel_arguments[] = {&expert_ids,          &assignment_numel, &num_experts,
                                  &body_tile_ns_device, &num_bodies,       &routing_metadata};
    parallel_params.kernelParams = parallel_arguments;

    // Compose every exact-ABI mock body behind the shared selector/preamble join point.
    GraphTopology topology;
    status = AddDASwitchToCapture(
        context, stream, conditional_handle, &parallel_params, &selector_params, num_bodies,
        expected_capture_id, previous_conditional_node,
        [&](cudaGraph_t body_graph, int body_index) {
          return moe_runner_.AddBodyGraph(body_graph, body_tactics[body_index], routing_metadata_,
                                          body_index);
        },
        &topology);
    if (status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "Failed to inject DA conditional graph: "
                                  << cudaGetErrorString(status);
    }
    std::lock_guard<std::mutex> lock(g_graph_inspection_mutex);
    if (g_graph_inspection.capture_id == topology.capture_id) {
      topology.conditional_node_count += g_graph_inspection.conditional_node_count;
      topology.workspace_lane_invocation_count +=
          g_graph_inspection.workspace_lane_invocation_count;
      topology.is_workspace_lane_serialized =
          topology.is_workspace_lane_serialized && g_graph_inspection.is_workspace_lane_serialized;
    }
    g_graph_inspection = std::move(topology);
    return reinterpret_cast<int64_t>(g_graph_inspection.conditional_node);
  }

 private:
  // Standalone fixed-body runner used to populate every conditional body.
  MockMoERunner moe_runner_;
  // Fixed-capacity normalized exemplar spectra consumed by the selector.
  const float* exemplar_spectra_;
  // Exemplar-to-deduplicated-body mapping consumed by the selector.
  const int32_t* exemplar_body_indices_;
  // Per-body routing tile values consumed by pre-body work.
  const int32_t* body_tile_ns_device_;
  // Device scalar written by the selector with the chosen body index.
  int32_t* selected_body_;
  // Body-major input-dependent routing metadata produced before the SWITCH.
  int32_t* routing_metadata_;
};

/** Inject one dtype-agnostic DA runner into an active outer capture. */
int64_t CaptureMockDAMoE(TensorView hidden_states, TensorView expert_ids, TensorView expert_weights,
                         TensorView output, TensorView activation_workspace,
                         TensorView intermediate_workspace, TensorView body_trace,
                         TensorView exemplar_spectra, TensorView exemplar_body_indices,
                         TensorView body_tile_ns_device, TensorView selected_body,
                         TensorView routing_metadata, int64_t num_experts,
                         int64_t num_selector_exemplars, Array<int64_t> body_tactic_ids,
                         int64_t expected_capture_id, int64_t previous_conditional_node_handle) {
  // Narrow public plan dimensions before building any native runner or graph state.
  const int runtime_num_selector_exemplars = ValidateNumSelectorExemplars(num_selector_exemplars);
  const int runtime_num_bodies = ValidateNumBodies(body_tactic_ids.size());
  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  const cudaStream_t stream = get_stream(hidden_states.device());
  std::vector<int> body_tactics;
  body_tactics.reserve(runtime_num_bodies);
  for (int64_t tactic : body_tactic_ids) {
    body_tactics.push_back(static_cast<int>(tactic));
  }
  if (body_tactics.size() < 2) {
    TVM_FFI_THROW(ValueError) << "A DA SWITCH requires at least two unique bodies";
  }

  // Layer DA composition around the same standalone dtype-agnostic runner used by eager mode.
  MockDAMoERunner da_runner(
      MakeMockMoERunner(hidden_states, expert_ids, expert_weights, output, activation_workspace,
                        intermediate_workspace, body_trace, ValidateNumExperts(num_experts)),
      static_cast<const float*>(exemplar_spectra.data_ptr()),
      static_cast<const int32_t*>(exemplar_body_indices.data_ptr()),
      static_cast<const int32_t*>(body_tile_ns_device.data_ptr()),
      static_cast<int32_t*>(selected_body.data_ptr()),
      static_cast<int32_t*>(routing_metadata.data_ptr()));
  int64_t conditional_node_handle = da_runner.Capture(
      runtime_num_selector_exemplars, body_tactics,
      static_cast<unsigned long long>(expected_capture_id),
      reinterpret_cast<cudaGraphNode_t>(previous_conditional_node_handle), stream);
  if (conditional_node_handle != 0) {
    std::lock_guard<std::mutex> lock(g_graph_inspection_mutex);
    g_workspace_binding_inspections.push_back(
        {reinterpret_cast<int64_t>(expert_weights.data_ptr()),
         reinterpret_cast<int64_t>(activation_workspace.data_ptr()),
         reinterpret_cast<int64_t>(intermediate_workspace.data_ptr())});
  }
  return conditional_node_handle;
}

/** Clear retained inspection handles before a new outer capture realization. */
void ResetGraphInspection() {
  std::lock_guard<std::mutex> lock(g_graph_inspection_mutex);
  g_graph_inspection = GraphInspectionRecord{};
  g_workspace_binding_inspections.clear();
}

/** Inspect actual nodes, edges, and conditional child graphs from the latest injection. */
Array<int64_t> InspectLastGraph() {
  std::lock_guard<std::mutex> lock(g_graph_inspection_mutex);
  if (g_graph_inspection.capture_id == 0) {
    return Array<int64_t>();
  }
  // Encode the fixed topology header before its variable child-node-count tail.
  // Sync with flashinfer/fused_moe/da_moe.py:DAGraphTopology.from_native.
  std::vector<int64_t> values = {
      static_cast<int64_t>(g_graph_inspection.capture_id),
      static_cast<int64_t>(g_graph_inspection.outer_node_count),
      static_cast<int64_t>(g_graph_inspection.outer_edge_count),
      static_cast<int64_t>(g_graph_inspection.conditional_node_count),
      static_cast<int64_t>(g_graph_inspection.body_node_counts.size()),
      static_cast<int64_t>(g_graph_inspection.selector_dependency_count),
      static_cast<int64_t>(g_graph_inspection.parallel_work_dependency_count),
      static_cast<int64_t>(g_graph_inspection.is_selector_preamble_parallelizable),
      static_cast<int64_t>(g_graph_inspection.is_workspace_lane_serialized),
      static_cast<int64_t>(g_graph_inspection.workspace_lane_invocation_count)};
  // Append FFI[10:] child sizes after the fixed topology header.
  for (size_t body_node_count : g_graph_inspection.body_node_counts) {
    values.push_back(static_cast<int64_t>(body_node_count));
  }
  return Array<int64_t>(values);
}

/** Inspect captured layer weights and lane-owned workspace pointers in invocation order. */
Array<int64_t> InspectLastWorkspaceBindings() {
  std::lock_guard<std::mutex> lock(g_graph_inspection_mutex);
  std::vector<int64_t> values;
  values.reserve(1 + 3 * g_workspace_binding_inspections.size());
  values.push_back(static_cast<int64_t>(g_workspace_binding_inspections.size()));
  for (auto const& inspection : g_workspace_binding_inspections) {
    values.push_back(inspection.expert_weights);
    values.push_back(inspection.activation_workspace);
    values.push_back(inspection.intermediate_workspace);
  }
  return Array<int64_t>(values);
}

}  // namespace flashinfer::da_moe::testing

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_mock_moe, flashinfer::da_moe::testing::RunMockMoE);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(capture_mock_da_moe, flashinfer::da_moe::testing::CaptureMockDAMoE);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(reset_graph_inspection,
                              flashinfer::da_moe::testing::ResetGraphInspection);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(inspect_last_graph, flashinfer::da_moe::testing::InspectLastGraph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(inspect_last_workspace_bindings,
                              flashinfer::da_moe::testing::InspectLastWorkspaceBindings);
