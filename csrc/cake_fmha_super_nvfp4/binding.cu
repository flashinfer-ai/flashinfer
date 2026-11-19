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
// FFI entry points of the Nemotron 3.5 Super NVFP4-KV attention kernels (sm_100a).
//
// Each launch returns an empty string after enqueueing its kernels, or the reason the kernel does
// not serve the call, having launched nothing. The *_reason entry points return the same reason
// without launching: route selection asks them first. A CUDA failure is raised as an error.
#include <cuda_runtime.h>
#include <tvm/ffi/container/variant.h>

#include <cmath>
#include <cstdint>
#include <string>

#include "flashinfer/math.cuh"
#include "fp4_mul.cuh"
#include "host.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::Variant;

namespace {

namespace super_nvfp4 = flashinfer::super_nvfp4;

constexpr int64_t kDecodeMtp = 0;
constexpr int64_t kDecodeBase = 1;

bool sameDtype(DLDataType a, DLDataType b) {
  return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

// Returns "" when `t` is a dense [rows, 32, 128] E4M3 tensor.
std::string tokenTensorReason(TensorView const& t, char const* name) {
  if (!sameDtype(t.dtype(), dl_float8_e4m3fn)) return std::string(name) + " dtype is not E4M3";
  if (t.ndim() != 3 || t.size(1) != super_nvfp4::kNumHeadsQ || t.size(2) != super_nvfp4::kHeadDim)
    return std::string(name) + " is not [tokens, 32, 128]";
  if (t.stride(2) != 1 || t.stride(1) != super_nvfp4::kHeadDim ||
      t.stride(0) != super_nvfp4::kNumHeadsQ * super_nvfp4::kHeadDim)
    return std::string(name) + " is not packed";
  return "";
}

// Returns "" when `t` is [pages, 2, 128, inner] with dense tokens (HND layout).
std::string pageTensorReason(TensorView const& t, DLDataType dtype, int64_t inner,
                             char const* name) {
  if (!sameDtype(t.dtype(), dtype)) return std::string(name) + " has the wrong dtype";
  if (t.ndim() != 4 || t.size(1) != super_nvfp4::kNumHeadsKv ||
      t.size(2) != super_nvfp4::kPageSize || t.size(3) != inner)
    return std::string(name) + " is not [pages, 2, 128, " + std::to_string(inner) + "]";
  if (t.stride(3) != 1 || t.stride(2) != inner) return std::string(name) + " tokens are not dense";
  return "";
}

bool onDevice(DLDevice device, DLDevice expected) {
  return device.device_type == kDLCUDA && device.device_id == expected.device_id;
}

std::string indexTensorReason(TensorView const& t, int64_t elements, char const* name) {
  if (!sameDtype(t.dtype(), dl_int32)) return std::string(name) + " is not int32";
  if (!t.IsContiguous()) return std::string(name) + " is not contiguous";
  if (t.numel() < elements) return std::string(name) + " is too short";
  return "";
}

// Reads the call from the tensors. Returns the first reason the geometry is not the one the
// kernels serve, or "".
std::string readCall(super_nvfp4::Call& c, TensorView const& out, TensorView const& query,
                     TensorView const& k_cache, TensorView const& v_cache, TensorView const& k_sf,
                     TensorView const& v_sf, TensorView const& workspace,
                     TensorView const& block_tables, TensorView const& seq_lens,
                     Optional<TensorView> const& cum_seq_lens_q, int64_t batch_size,
                     int64_t max_q_len, int64_t max_kv_len,
                     Variant<double, ffi::Tensor> const& bmm1_scale,
                     Variant<double, ffi::Tensor> const& bmm2_scale, bool enable_pdl) {
  char const* const otherDevice = "tensors are not on one CUDA device";
  for (auto const* t :
       {&out, &query, &k_cache, &v_cache, &k_sf, &v_sf, &workspace, &block_tables, &seq_lens}) {
    if (!onDevice(t->device(), query.device())) return otherDevice;
  }
  if (cum_seq_lens_q.has_value() && !onDevice(cum_seq_lens_q.value().device(), query.device()))
    return otherDevice;
  for (auto const* scale : {&bmm1_scale, &bmm2_scale}) {
    if (auto tensor = scale->as<ffi::Tensor>()) {
      if (!onDevice(tensor.value().device(), query.device())) return otherDevice;
      if (!sameDtype(tensor.value().dtype(), dl_float32)) return "scale tensor is not float32";
    }
  }
  std::string reason = tokenTensorReason(query, "query");
  if (reason.empty()) reason = tokenTensorReason(out, "out");
  if (reason.empty() && out.size(0) != query.size(0)) reason = "out and query row counts differ";
  if (reason.empty())
    reason = pageTensorReason(k_cache, dl_uint8, super_nvfp4::kHeadDim / 2, "k_cache");
  if (reason.empty())
    reason = pageTensorReason(v_cache, dl_uint8, super_nvfp4::kHeadDim / 2, "v_cache");
  if (reason.empty())
    reason = pageTensorReason(k_sf, dl_float8_e4m3fn, super_nvfp4::kHeadDim / 16, "k_cache_sf");
  if (reason.empty())
    reason = pageTensorReason(v_sf, dl_float8_e4m3fn, super_nvfp4::kHeadDim / 16, "v_cache_sf");
  if (reason.empty() && (k_cache.size(0) != v_cache.size(0) || k_cache.size(0) != k_sf.size(0) ||
                         k_cache.size(0) != v_sf.size(0)))
    reason = "K, V and scale page counts differ";
  if (reason.empty() && (batch_size < 1 || batch_size > INT32_MAX)) reason = "batch out of range";
  if (reason.empty() && (max_q_len < 1 || max_q_len > INT32_MAX)) reason = "max q_len out of range";
  if (reason.empty() && (block_tables.ndim() != 2 || block_tables.size(0) != batch_size))
    reason = "block_tables is not [batch, max_pages]";
  if (reason.empty()) reason = indexTensorReason(block_tables, batch_size, "block_tables");
  if (reason.empty()) reason = indexTensorReason(seq_lens, batch_size, "seq_lens");
  if (reason.empty() && cum_seq_lens_q.has_value())
    reason = indexTensorReason(cum_seq_lens_q.value(), batch_size + 1, "cum_seq_lens_q");
  // The kernels write their scratch as one contiguous range from the workspace pointer.
  if (reason.empty() && !workspace.IsContiguous()) reason = "workspace is not contiguous";
  if (!reason.empty()) return reason;

  c.q = query.data_ptr();
  c.k = k_cache.data_ptr();
  c.v = v_cache.data_ptr();
  c.kSf = k_sf.data_ptr();
  c.vSf = v_sf.data_ptr();
  c.out = out.data_ptr();
  c.workspace = workspace.data_ptr();
  c.workspaceBytes = workspace.numel() * get_element_size(workspace);
  c.pageTable = static_cast<int32_t const*>(block_tables.data_ptr());
  c.seqLensKv = static_cast<int32_t const*>(seq_lens.data_ptr());
  if (cum_seq_lens_q.has_value())
    c.cumSeqLensQ = static_cast<int32_t const*>(cum_seq_lens_q.value().data_ptr());
  c.batch = static_cast<int>(batch_size);
  c.maxQ = static_cast<int>(max_q_len);
  c.sumQ = query.size(0);
  c.maxKv = max_kv_len;
  c.maxPagesPerSeq = block_tables.size(1);
  c.poolPages = k_cache.size(0);
  c.kStrideHead = k_cache.stride(1);
  c.kStridePage = k_cache.stride(0);
  c.vStrideHead = v_cache.stride(1);
  c.vStridePage = v_cache.stride(0);
  c.kSfStrideHead = k_sf.stride(1);
  c.kSfStridePage = k_sf.stride(0);
  c.vSfStrideHead = v_sf.stride(1);
  c.vSfStridePage = v_sf.stride(0);
  c.enablePdl = enable_pdl;

  // A tensor scale overrides the scalar one; the bmm1 tensor already holds the log2 scale.
  if (auto tensor = bmm1_scale.as<ffi::Tensor>()) {
    c.scaleSoftmaxLog2Ptr = static_cast<float const*>(tensor.value().data_ptr());
  } else if (auto value = bmm1_scale.as<double>()) {
    // The kernels mask with -FLT_MAX before scaling, so the scale applied in float must keep a
    // masked score far below any visible one (and every product finite); trtllm-gen serves
    // other scales. Real scales are many orders of magnitude inside this range.
    float const scaleLog2 = static_cast<float>(value.value() * flashinfer::math::log2e_d);
    if (!(scaleLog2 >= 1e-30f && scaleLog2 <= 1e30f)) return "bmm1_scale outside [1e-30, 1e30]";
    c.scaleSoftmaxLog2 = scaleLog2;
  } else {
    return "bmm1_scale is neither a float nor a tensor";
  }
  if (auto tensor = bmm2_scale.as<ffi::Tensor>()) {
    c.outputScalePtr = static_cast<float const*>(tensor.value().data_ptr());
  } else if (auto value = bmm2_scale.as<double>()) {
    c.outputScale = static_cast<float>(value.value());
  } else {
    return "bmm2_scale is neither a float nor a tensor";
  }
  return "";
}

super_nvfp4::Device currentDevice() {
  int device = 0;
  super_nvfp4::Device dev;
  TVM_FFI_ICHECK(cudaGetDevice(&device) == cudaSuccess);
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&dev.major, cudaDevAttrComputeCapabilityMajor, device) ==
                 cudaSuccess);
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&dev.minor, cudaDevAttrComputeCapabilityMinor, device) ==
                 cudaSuccess);
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&dev.smCount, cudaDevAttrMultiProcessorCount, device) ==
                 cudaSuccess);
  return dev;
}

// Reads the call into `call` and plans `kernel` 0 ("mtp") or 1 ("base") on the current device.
// Returns "" when the kernel serves the call. Without cum_seq_lens_q every request has max_q_len
// query tokens.
std::string planDecode(int64_t kernel, super_nvfp4::Call& call, super_nvfp4::DecodePlan& plan,
                       TensorView out, TensorView query, TensorView k_cache, TensorView v_cache,
                       TensorView k_cache_sf, TensorView v_cache_sf, TensorView workspace_buffer,
                       TensorView block_tables, TensorView seq_lens, int64_t batch_size,
                       int64_t max_q_len, int64_t max_kv_len,
                       Variant<double, ffi::Tensor> const& bmm1_scale,
                       Variant<double, ffi::Tensor> const& bmm2_scale, bool enable_pdl,
                       Optional<TensorView> const& cum_seq_lens_q) {
  TVM_FFI_ICHECK(kernel == kDecodeMtp || kernel == kDecodeBase) << "unknown decode kernel";
  std::string const reason =
      readCall(call, out, query, k_cache, v_cache, k_cache_sf, v_cache_sf, workspace_buffer,
               block_tables, seq_lens, cum_seq_lens_q, batch_size, max_q_len, max_kv_len,
               bmm1_scale, bmm2_scale, enable_pdl);
  if (!reason.empty()) return reason;
  super_nvfp4::Device const dev = currentDevice();
  plan = kernel == kDecodeMtp ? super_nvfp4::planDecodeMtp(call, dev)
                              : super_nvfp4::planDecodeBase(call, dev);
  return plan.reason;
}

std::string planPrefill(super_nvfp4::Call& call, super_nvfp4::Device& dev, TensorView out,
                        TensorView query, TensorView k_cache, TensorView v_cache,
                        TensorView k_cache_sf, TensorView v_cache_sf, TensorView workspace_buffer,
                        TensorView block_tables, TensorView seq_lens, TensorView cum_seq_lens_q,
                        int64_t batch_size, int64_t max_q_len, int64_t max_kv_len,
                        Variant<double, ffi::Tensor> const& bmm1_scale,
                        Variant<double, ffi::Tensor> const& bmm2_scale) {
  std::string const reason =
      readCall(call, out, query, k_cache, v_cache, k_cache_sf, v_cache_sf, workspace_buffer,
               block_tables, seq_lens, Optional<TensorView>(cum_seq_lens_q), batch_size, max_q_len,
               max_kv_len, bmm1_scale, bmm2_scale, /*enable_pdl=*/false);
  if (!reason.empty()) return reason;
  dev = currentDevice();
  return super_nvfp4::prefillReason(call, dev);
}

}  // namespace

// Why decode `kernel` would decline the call, or "" when it serves it. Launches nothing.
String super_nvfp4_decode_reason(int64_t kernel, TensorView out, TensorView query,
                                 TensorView k_cache, TensorView v_cache, TensorView k_cache_sf,
                                 TensorView v_cache_sf, TensorView workspace_buffer,
                                 TensorView block_tables, TensorView seq_lens, int64_t batch_size,
                                 int64_t max_q_len, int64_t max_kv_len,
                                 Variant<double, ffi::Tensor> bmm1_scale,
                                 Variant<double, ffi::Tensor> bmm2_scale, bool enable_pdl,
                                 Optional<TensorView> cum_seq_lens_q) {
  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  super_nvfp4::Call call;
  super_nvfp4::DecodePlan plan;
  return String(planDecode(kernel, call, plan, out, query, k_cache, v_cache, k_cache_sf, v_cache_sf,
                           workspace_buffer, block_tables, seq_lens, batch_size, max_q_len,
                           max_kv_len, bmm1_scale, bmm2_scale, enable_pdl, cum_seq_lens_q));
}

// Decode with `kernel`; returns "" once enqueued, or why it declined.
String super_nvfp4_decode(int64_t kernel, TensorView out, TensorView query, TensorView k_cache,
                          TensorView v_cache, TensorView k_cache_sf, TensorView v_cache_sf,
                          TensorView workspace_buffer, TensorView block_tables, TensorView seq_lens,
                          int64_t batch_size, int64_t max_q_len, int64_t max_kv_len,
                          Variant<double, ffi::Tensor> bmm1_scale,
                          Variant<double, ffi::Tensor> bmm2_scale, bool enable_pdl,
                          Optional<TensorView> cum_seq_lens_q) {
  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  super_nvfp4::Call call;
  super_nvfp4::DecodePlan plan;
  std::string const reason =
      planDecode(kernel, call, plan, out, query, k_cache, v_cache, k_cache_sf, v_cache_sf,
                 workspace_buffer, block_tables, seq_lens, batch_size, max_q_len, max_kv_len,
                 bmm1_scale, bmm2_scale, enable_pdl, cum_seq_lens_q);
  if (!reason.empty()) return String(reason);
  void* const stream = get_stream(query.device());
  if (kernel == kDecodeMtp) {
    super_nvfp4::launchDecodeMtp(call, plan, stream);
  } else {
    super_nvfp4::launchDecodeBase(call, plan, stream);
  }
  return String("");
}

// Why prefill would decline the call, or "" when it serves it. Launches nothing. Request i owns
// query rows cum_seq_lens_q[i] to cum_seq_lens_q[i + 1] and attends to its first seq_lens[i] KV
// tokens, which already include those rows.
String super_nvfp4_prefill_reason(TensorView out, TensorView query, TensorView k_cache,
                                  TensorView v_cache, TensorView k_cache_sf, TensorView v_cache_sf,
                                  TensorView workspace_buffer, TensorView block_tables,
                                  TensorView seq_lens, TensorView cum_seq_lens_q,
                                  int64_t batch_size, int64_t max_q_len, int64_t max_kv_len,
                                  Variant<double, ffi::Tensor> bmm1_scale,
                                  Variant<double, ffi::Tensor> bmm2_scale) {
  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  super_nvfp4::Call call;
  super_nvfp4::Device dev;
  return String(planPrefill(call, dev, out, query, k_cache, v_cache, k_cache_sf, v_cache_sf,
                            workspace_buffer, block_tables, seq_lens, cum_seq_lens_q, batch_size,
                            max_q_len, max_kv_len, bmm1_scale, bmm2_scale));
}

// Prefill; returns "" once enqueued, or why it declined.
String super_nvfp4_prefill(TensorView out, TensorView query, TensorView k_cache, TensorView v_cache,
                           TensorView k_cache_sf, TensorView v_cache_sf,
                           TensorView workspace_buffer, TensorView block_tables,
                           TensorView seq_lens, TensorView cum_seq_lens_q, int64_t batch_size,
                           int64_t max_q_len, int64_t max_kv_len,
                           Variant<double, ffi::Tensor> bmm1_scale,
                           Variant<double, ffi::Tensor> bmm2_scale) {
  ffi::CUDADeviceGuard device_guard(query.device().device_id);
  super_nvfp4::Call call;
  super_nvfp4::Device dev;
  std::string const reason =
      planPrefill(call, dev, out, query, k_cache, v_cache, k_cache_sf, v_cache_sf, workspace_buffer,
                  block_tables, seq_lens, cum_seq_lens_q, batch_size, max_q_len, max_kv_len,
                  bmm1_scale, bmm2_scale);
  if (!reason.empty()) return String(reason);
  super_nvfp4::launchPrefill(call, dev, get_stream(query.device()));
  return String("");
}

// Whether the decode kernels were built with the fused NVFP4 dequantization instruction, which
// the CUDA 13.4 toolkit introduced (fp4_mul.cuh). Every translation unit of the module is built
// by one compiler.
bool super_nvfp4_fused_fp4_mul() { return SUPER_NVFP4_FUSED_FP4_MUL != 0; }

TVM_FFI_DLL_EXPORT_TYPED_FUNC(super_nvfp4_decode_reason, super_nvfp4_decode_reason);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(super_nvfp4_decode, super_nvfp4_decode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(super_nvfp4_prefill_reason, super_nvfp4_prefill_reason);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(super_nvfp4_prefill, super_nvfp4_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(super_nvfp4_fused_fp4_mul, super_nvfp4_fused_fp4_mul);
