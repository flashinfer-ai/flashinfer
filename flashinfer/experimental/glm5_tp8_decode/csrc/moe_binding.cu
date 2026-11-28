/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include "binding_utils.h"
#include "moe_api.h"

namespace flashinfer::glm5_tp8::binding {

Array<int64_t> moe_initialize(int64_t q_len, int64_t rank, int64_t device) {
  TVM_FFI_ICHECK(rank >= 0 && rank < 8);
  return resources(initialize_moe(static_cast<int>(q_len), static_cast<int>(device)));
}

// Private tensor order is shared with backend.py; public callers use named weights/plans.
void moe_run(Array<Tensor> t, Array<int64_t> peers, int64_t rank, bool pdl, bool cooperative) {
  TVM_FFI_ICHECK_EQ(t.size(), 16) << "invalid MoE tensor bundle";
  TVM_FFI_ICHECK(rank >= 0 && rank < 8);
  CHECK_DIM(2, t[0]);
  const int64_t q = t[0].size(0);
  TVM_FFI_ICHECK(q == 1 || q == 4) << "q_len must be 1 or 4";
  TVM_FFI_ICHECK_EQ(t[0].size(1), 6144);
  CHECK_DIM(2, t[10]);
  TVM_FFI_ICHECK_EQ(t[10].size(0), q);
  TVM_FFI_ICHECK_EQ(t[10].size(1), 6144);
  const auto device = t[0].device();
  check_tensor(t[0], dl_bfloat16, device, q * 6144, "x");
  check_tensor(t[1], dl_float32, device, 6144, "gamma");
  check_tensor(t[2], dl_bfloat16, device, 256 * 6144, "router");
  check_tensor(t[3], dl_float32, device, 256, "bias");
  check_tensor(t[4], dl_uint8, device, int64_t(257) * 16 * 6 * 32896, "gate_up");
  check_tensor(t[5], dl_uint8, device, int64_t(257) * 6144 * 256, "down");
  check_tensor(t[6], dl_float32, device, 257 * 1024 * 2, "down_scale");
  check_tensor(t[7], dl_float32, device, 128 * 32, "scores");
  check_tensor(t[8], dl_bfloat16, device, q * 9 * 256, "mid");
  check_tensor(t[9], dl_bfloat16, device, q * 6144, "x_norm");
  check_tensor(t[10], dl_bfloat16, device, q * 6144, "out");
  check_tensor(t[11], dl_float32, device, q * 8, "probabilities");
  check_tensor(t[12], dl_int32, device, q * 8, "indices");
  check_tensor(t[13], dl_uint8, device, q == 1 ? 1024 : 0x500, "state");
  check_tensor(t[14], dl_float32, device, q == 4 ? 4 * 6144 : 0, "shared_down_scratch");
  check_tensor(t[15], dl_int64, device, q == 1 ? 80 : 0, "context");
  check_peers(peers);
  MoeArguments a{};
  a.rank = static_cast<int>(rank);
  a.q_len = static_cast<int>(q);
  for (int r = 0; r < 8; ++r) a.peers[r] = reinterpret_cast<uint8_t*>(peers[r]);
  a.x = pointer<uint16_t>(t[0]);
  a.gamma = pointer<float>(t[1]);
  a.router = pointer<uint16_t>(t[2]);
  a.bias = pointer<float>(t[3]);
  a.gate_up = pointer<uint8_t>(t[4]);
  a.down = pointer<uint8_t>(t[5]);
  a.down_scale = pointer<float>(t[6]);
  a.scores = pointer<float>(t[7]);
  a.mid = pointer<uint16_t>(t[8]);
  a.x_norm = pointer<uint16_t>(t[9]);
  a.out = pointer<uint16_t>(t[10]);
  a.probabilities = pointer<float>(t[11]);
  a.indices = pointer<int32_t>(t[12]);
  a.state = pointer<uint8_t>(t[13]);
  a.shared_down_scratch = pointer<float>(t[14]);
  a.context = pointer<uint64_t>(t[15]);
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  auto status = launch_moe(a, get_stream(device), pdl, cooperative);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "GLM-5 TP8 MoE launch: " << cudaGetErrorString(status);
}
}  // namespace flashinfer::glm5_tp8::binding

TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize, flashinfer::glm5_tp8::binding::moe_initialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::glm5_tp8::binding::moe_run);
