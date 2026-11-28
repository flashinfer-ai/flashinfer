/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#include "attention_api.h"
#include "binding_utils.h"

namespace flashinfer::glm5_tp8::binding {

Array<int64_t> attention_initialize(int64_t q_len, int64_t rank, int64_t device) {
  return resources(initialize_attention(static_cast<int>(q_len), static_cast<int>(rank),
                                        static_cast<int>(device)));
}

void attention_encode_tmap(TensorView base, int64_t box_rows, int64_t box_cols, TensorView out) {
  CHECK_INPUT_AND_TYPE(base, dl_bfloat16);
  CHECK_DIM(2, base);
  CHECK_CPU(out);
  CHECK_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(out.dtype()), uint8_code);
  TVM_FFI_ICHECK_GE(out.numel(), 128);
  tvm::ffi::CUDADeviceGuard guard(base.device().device_id);
  auto status =
      encode_attention_tmap(base.data_ptr(), base.size(0), base.size(1), static_cast<int>(box_rows),
                            static_cast<int>(box_cols), out.data_ptr());
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << "GLM-5 tensor map: " << cudaGetErrorString(status);
}

// Always: [x,out,context,q4_epoch,q1_state]. Attention ranks additionally pass the
// named MLA tensors in backend.py's order; rank0 uses its prepared descriptor.
void attention_run(Array<Tensor> t, Array<int64_t> arenas, Array<int64_t> records, int64_t rank,
                   bool shared, bool pdl, bool cooperative) {
  TVM_FFI_ICHECK(rank >= 0 && rank < 8);
  TVM_FFI_ICHECK_EQ(t.size(), rank == 0 ? 5 : 27);
  CHECK_DIM(2, t[0]);
  const int64_t q = t[0].size(0);
  TVM_FFI_ICHECK(q == 1 || q == 4);
  TVM_FFI_ICHECK_EQ(t[0].size(1), 6144);
  const auto device = t[0].device();
  check_tensor(t[0], dl_bfloat16, device, q * 6144, "x");
  check_tensor(t[1], dl_bfloat16, device, q * 6144, "out");
  check_tensor(t[2], dl_int64, device, (rank == 0 || q == 1) ? 128 : 0, "context");
  check_tensor(t[3], dl_int32, device, 148, "q4 epoch");
  check_tensor(t[4], dl_uint8, device, 1024, "q1 state");
  check_peers(arenas);
  check_peers(records);
  AttentionArguments a{};
  a.rank = static_cast<int>(rank);
  a.q_len = static_cast<int>(q);
  a.shared_indexer = shared;
  a.context = pointer<uint64_t>(t[2]);
  a.epoch_ctr = pointer<uint32_t>(t[3]);
  a.q1_state = pointer<uint8_t>(t[4]);
  for (int r = 0; r < 8; ++r) {
    a.peer_arenas[r] = reinterpret_cast<uint8_t*>(arenas[r]);
    a.peer_records[r] = reinterpret_cast<uint8_t*>(records[r]);
  }
  if (rank != 0) {
    auto& m = a.mla;
    m.x = pointer<uint16_t>(t[0]);
    m.out = pointer<uint16_t>(t[1]);
    check_tensor(t[5], dl_float32, device, 6144, "x_gamma");
    check_tensor(t[6], dl_float32, device, 2048, "q_gamma");
    check_tensor(t[7], dl_float32, device, 512, "kv_gamma");
    check_tensor(t[8], dl_float32, device, 64, "rope");
    const int64_t weight_bytes[] = {82LL * 6 * 32896, 80LL * 2 * 32896, 80LL * 12416, 80LL * 16512,
                                    128LL * 5 * 24704};
    for (int i = 0; i < 5; ++i)
      check_tensor(t[9 + i], dl_uint8, device, weight_bytes[i], "packed MLA weight");
    check_tensor(t[14], dl_bfloat16, device, 512, "kv_cache");
    check_tensor(t[15], dl_bfloat16, device, 64, "pe_cache");
    CHECK_DIM(2, t[14]);
    CHECK_DIM(2, t[15]);
    CHECK_DIM(2, t[8]);
    TVM_FFI_ICHECK_EQ(t[8].size(1), 64);
    TVM_FFI_ICHECK_GE(t[8].size(0), t[14].size(0))
        << "RoPE table must cover the cache row capacity";
    TVM_FFI_ICHECK_EQ(t[14].size(0), t[15].size(0));
    TVM_FFI_ICHECK_EQ(t[14].size(1), 512);
    TVM_FFI_ICHECK_EQ(t[15].size(1), 64);
    TVM_FFI_ICHECK_LE(t[14].size(0), INT32_MAX);
    check_tensor(t[16], dl_int32, device, 4 * 2048, "selected_ids");
    check_tensor(t[17], dl_int32, device, 4, "selected_lengths");
    check_tensor(t[18], dl_int32, device, 1, "cur_pos", 4);
    const int64_t scratch_sizes[] = {4 * 2048,     4 * 512,      4 * 10 * 192, 4 * 10 * 64,
                                     4 * 10 * 512, 4 * 10 * 512, 4 * 10 * 256};
    for (int i = 0; i < 7; ++i)
      check_tensor(t[19 + i], dl_bfloat16, device, scratch_sizes[i], "MLA scratch");
    check_tensor(t[26], dl_int32, device, 32768, "LL receive buffer");
    m.x_gamma = pointer<float>(t[5]);
    m.q_gamma = pointer<float>(t[6]);
    m.kv_gamma = pointer<float>(t[7]);
    m.rope = pointer<float>(t[8]);
    m.qkv_a_proj = pointer<uint8_t>(t[9]);
    m.q_b_proj = pointer<uint8_t>(t[10]);
    m.w_uk = pointer<uint8_t>(t[11]);
    m.w_uv = pointer<uint8_t>(t[12]);
    m.o_proj = pointer<uint8_t>(t[13]);
    m.kv_cache = pointer<uint16_t>(t[14]);
    m.pe_cache = pointer<uint16_t>(t[15]);
    m.cache_rows = static_cast<int>(t[14].size(0));
    m.selected_ids = pointer<int32_t>(t[16]);
    m.selected_lengths = pointer<int32_t>(t[17]);
    m.cur_pos = pointer<int32_t>(t[18]);
    m.post_q = pointer<uint16_t>(t[19]);
    m.post_kv = pointer<uint16_t>(t[20]);
    m.post_qnd = pointer<uint16_t>(t[21]);
    m.post_qpe = pointer<uint16_t>(t[22]);
    m.post_qnope = pointer<uint16_t>(t[23]);
    m.post_o = pointer<uint16_t>(t[24]);
    m.post_uv = pointer<uint16_t>(t[25]);
    m.ll_recv = pointer<uint8_t>(t[26]);
    m.ll_tag = 1;
  }
  tvm::ffi::CUDADeviceGuard guard(device.device_id);
  auto status = launch_attention(a, get_stream(device), pdl, cooperative);
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "GLM-5 TP8 attention launch: " << cudaGetErrorString(status);
}
}  // namespace flashinfer::glm5_tp8::binding

TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize, flashinfer::glm5_tp8::binding::attention_initialize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(encode_tmap, flashinfer::glm5_tp8::binding::attention_encode_tmap);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::glm5_tp8::binding::attention_run);
