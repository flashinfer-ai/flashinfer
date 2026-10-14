/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cstdio>
#include <cstdlib>

#include "flashinfer/fused_moe/da_moe.cuh"

// Compiled and executed by test_da_graph_dependencies.py.
__global__ void DependencyProbe() {}

__global__ void WriteWorkspace(int* workspace) { *workspace = 7; }

__global__ void TriggerSuccessor() { cudaTriggerProgrammaticLaunchCompletion(); }

__global__ void ReadWorkspace(int* workspace, int* observed) { *observed = *workspace; }

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

void Require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  Check(cudaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
  cudaGraph_t graph;
  Check(cudaGraphCreate(&graph, 0));
  int* values;
  Check(cudaMalloc(&values, 3 * sizeof(int)));

  // The real lane predecessor is a conditional, not a kernel with a PDL output port.
  cudaGraphConditionalHandle handle;
  Check(cudaGraphConditionalHandleCreate(&handle, graph, 0, cudaGraphCondAssignDefault));
  cudaGraphNodeParams conditional_params{};
  conditional_params.type = cudaGraphNodeTypeConditional;
  conditional_params.conditional.handle = handle;
  conditional_params.conditional.type = cudaGraphCondTypeSwitch;
  conditional_params.conditional.size = 1;
  cudaGraphNode_t conditional;
  Check(flashinfer::da_moe::AddGraphNode(&conditional, graph, nullptr, 0, &conditional_params));
  auto write =
      flashinfer::da_moe::MakeTypedKernelLaunch(WriteWorkspace, dim3(1), dim3(1), 0, values);
  cudaGraphNode_t body;
  Check(write.AddToGraph(&body, conditional_params.conditional.phGraph_out[0], nullptr, 0));

  cudaKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(DependencyProbe);
  params.gridDim = dim3(1);
  params.blockDim = dim3(1);
  cudaGraphNode_t ancestor, trigger_only;
  auto trigger = flashinfer::da_moe::MakeTypedKernelLaunch(TriggerSuccessor, dim3(1), dim3(1), 0);
  Check(trigger.AddToGraph(&ancestor, graph, &conditional, 1));
  Check(cudaGraphAddKernelNode(&trigger_only, graph, nullptr, 0, &params));
  cudaGraphEdgeData edge{};
  edge.type = cudaGraphDependencyTypeProgrammatic;
  edge.from_port = cudaGraphKernelNodePortProgrammatic;
#if CUDART_VERSION >= 13000
  Check(cudaGraphAddDependencies(graph, &ancestor, &trigger_only, &edge, 1));
#else
  Check(cudaGraphAddDependencies_v2(graph, &ancestor, &trigger_only, &edge, 1));
#endif
  cudaGraphNode_t predecessor;
  size_t predecessor_count = 1;
#if CUDART_VERSION >= 13000
  cudaError_t lossy =
      cudaGraphNodeGetDependencies(trigger_only, &predecessor, nullptr, &predecessor_count);
#else
  cudaError_t lossy =
      cudaGraphNodeGetDependencies_v2(trigger_only, &predecessor, nullptr, &predecessor_count);
#endif
  Require(lossy == cudaErrorLossyQuery, "metadata-free query must reproduce the failure");
  std::vector<cudaGraphNode_t> dependencies;
  Check(flashinfer::da_moe::GetGraphNodeDependencies(trigger_only, &dependencies));
  Require(dependencies.size() == 1 && dependencies[0] == ancestor,
          "metadata-aware query must preserve a programmatic predecessor");
  bool depends_on = true;
  Check(flashinfer::da_moe::GraphNodeDependsOn(trigger_only, conditional, &depends_on));
  Require(depends_on, "PDL descendants still depend on the completed conditional");
  flashinfer::da_moe::ActiveCaptureContext context{};
  context.capture_id = 42;
  context.graph = graph;
  context.dependencies = {trigger_only};
  size_t edges_before, edges_after;
  Check(flashinfer::da_moe::GetGraphEdgeCount(graph, &edges_before));
  Check(flashinfer::da_moe::ValidateWorkspaceLaneSequence(context, 42, conditional, &depends_on));
  Check(flashinfer::da_moe::GetGraphEdgeCount(graph, &edges_after));
  Require(depends_on && context.dependencies == std::vector<cudaGraphNode_t>{trigger_only} &&
              edges_before == edges_after,
          "conditional ancestry must admit reuse without adding completion edges");

  // Both next-invocation consumers must observe the completed body's writes on replay.
  cudaGraphNode_t readers[2];
  for (int i = 0; i < 2; ++i) {
    auto read = flashinfer::da_moe::MakeTypedKernelLaunch(ReadWorkspace, dim3(1), dim3(1), 0,
                                                          values, values + i + 1);
    Check(read.AddToGraph(&readers[i], graph, context.dependencies.data(),
                          context.dependencies.size()));
  }
  cudaGraphNode_t unrelated;
  Check(cudaGraphAddKernelNode(&unrelated, graph, nullptr, 0, &params));
  context.dependencies = {unrelated};
  Check(flashinfer::da_moe::ValidateWorkspaceLaneSequence(context, 42, conditional, &depends_on));
  Require(!depends_on && context.dependencies.size() == 1,
          "unordered fork must remain rejected without frontier mutation");
  context.dependencies = {trigger_only};
  Check(flashinfer::da_moe::ValidateWorkspaceLaneSequence(context, 43, conditional, &depends_on));
  Require(!depends_on && context.dependencies.size() == 1,
          "cross-generation lane must remain rejected");
  auto after = context;
  after.dependencies = {readers[0], trigger_only};
  cudaGraphNode_t new_node = nullptr;
  Check(flashinfer::da_moe::GetNewCaptureFrontierNode(context, after, &new_node));
  Require(new_node == readers[0],
          "new root must be selected independently of retained frontier ordering");
  after.dependencies = {trigger_only};
  Require(flashinfer::da_moe::GetNewCaptureFrontierNode(context, after, &new_node) ==
              cudaErrorInvalidValue,
          "missing new roots must be rejected");
  after.dependencies = {readers[0], readers[1], trigger_only};
  Require(flashinfer::da_moe::GetNewCaptureFrontierNode(context, after, &new_node) ==
              cudaErrorInvalidValue,
          "ambiguous new roots must be rejected");
  after.dependencies = {readers[0]};
  after.capture_id++;
  Require(flashinfer::da_moe::GetNewCaptureFrontierNode(context, after, &new_node) ==
              cudaErrorInvalidValue,
          "cross-capture root identification must be rejected");

  cudaGraphExec_t executable;
  Check(cudaGraphInstantiate(&executable, graph, 0));
  for (int replay = 0; replay < 10; ++replay) {
    Check(cudaMemset(values, 0, 3 * sizeof(int)));
    Check(cudaGraphLaunch(executable, nullptr));
    int observed[3];
    Check(cudaMemcpy(observed, values, sizeof(observed), cudaMemcpyDeviceToHost));
    Require(observed[1] == 7 && observed[2] == 7,
            "both workspace consumers must observe conditional completion");
  }
  Check(cudaGraphExecDestroy(executable));
  Check(cudaGraphDestroy(graph));
  Check(cudaFree(values));
  std::puts("PASS: metadata query, conditional ancestry replay, and retained frontier roots");
}
