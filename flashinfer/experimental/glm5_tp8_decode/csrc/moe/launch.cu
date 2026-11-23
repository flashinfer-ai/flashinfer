/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include <stdexcept>
#include <string>

#include "internal.cuh"

namespace flashinfer::glm5_tp8 {
namespace {

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

struct DeviceGuard {
  int previous;
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&previous), "cudaGetDevice");
    if (device != previous) check(cudaSetDevice(device), "cudaSetDevice");
  }
  ~DeviceGuard() { (void)cudaSetDevice(previous); }
};

bool aligned(const void* pointer, uintptr_t alignment = 16) {
  return pointer && (reinterpret_cast<uintptr_t>(pointer) % alignment == 0);
}

}  // namespace

KernelResources initialize_moe(int q_len, int device) {
  if (q_len != 1 && q_len != 4) throw std::invalid_argument("MoE q_len must be 1 or 4");
  DeviceGuard guard(device);
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
  if (properties.major != 10 || properties.minor != 3) {
    throw std::runtime_error("GLM-5 TP8 MoE requires a B300 (compute capability 10.3)");
  }
  const void* kernel = q_len == 1 ? moe_detail::q1_kernel() : moe_detail::q4_kernel();
  const int block = q_len == 1 ? 384 : 320;
  const int shared = q_len == 1 ? 214016 : 193536;
  check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared),
        "cudaFuncSetAttribute(MaxDynamicSharedMemorySize)");
  cudaFuncAttributes attributes{};
  check(cudaFuncGetAttributes(&attributes, kernel), "cudaFuncGetAttributes");
  int active = 0;
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, block, shared),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
  if (active < 1 || active * properties.multiProcessorCount < 148) {
    throw std::runtime_error("GLM-5 TP8 MoE requires all 148 CTAs to be resident");
  }
  return {148,
          block,
          shared,
          attributes.numRegs,
          static_cast<int>(attributes.sharedSizeBytes),
          static_cast<int>(attributes.localSizeBytes),
          active};
}

cudaError_t launch_moe(const MoeArguments& args, cudaStream_t stream, bool enable_pdl,
                       bool cooperative) {
  if ((args.q_len != 1 && args.q_len != 4) || args.rank < 0 || args.rank >= 8 ||
      !aligned(args.state, 256) || !aligned(args.x) || !aligned(args.gamma) ||
      !aligned(args.router) || !aligned(args.bias) || !aligned(args.gate_up) ||
      !aligned(args.down) || !aligned(args.down_scale) || !aligned(args.scores) ||
      !aligned(args.mid) || !aligned(args.x_norm) || !aligned(args.out) ||
      !aligned(args.probabilities) || !aligned(args.indices)) {
    return cudaErrorInvalidValue;
  }
  for (auto peer : args.peers)
    if (!aligned(peer)) return cudaErrorInvalidValue;
  const uintptr_t x = reinterpret_cast<uintptr_t>(args.x);
  const uintptr_t out = reinterpret_cast<uintptr_t>(args.out);
  const size_t bytes = static_cast<size_t>(args.q_len) * 6144 * sizeof(uint16_t);
  if (out < x + bytes && x < out + bytes) return cudaErrorInvalidValue;
  if (args.q_len == 1) {
    if (!aligned(args.context, 8)) return cudaErrorInvalidValue;
    return moe_detail::launch_q1(args, stream, enable_pdl, cooperative);
  }
  if (!aligned(args.shared_down_scratch)) return cudaErrorInvalidValue;
  return moe_detail::launch_q4(args, stream, enable_pdl, cooperative);
}

}  // namespace flashinfer::glm5_tp8
