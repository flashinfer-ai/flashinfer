// Copyright (c) 2026 by FlashInfer team.
// Licensed under the Apache License, Version 2.0.
// Fair fallback baseline: one launch converts both complete allocated KV pools.
// Per-tensor scale rounding matches upstream MLA's BF16 packed multiply.
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <flashinfer/vec_dtypes.cuh>

#include "tvm_ffi_utils.h"

__global__ void fused_pool_dequant(const __nv_fp8_e4m3* c, const __nv_fp8_e4m3* p,
                                   __nv_bfloat16* co, __nv_bfloat16* po, int64_t c_units,
                                   int64_t total_units, __nv_bfloat16 cs, __nv_bfloat16 ps) {
  int64_t u = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (u >= total_units) return;
  const bool is_c = u < c_units;
  const int64_t off = (is_c ? u : u - c_units) * 16;
  const auto* src = (is_c ? c : p) + off;
  auto* dst = (is_c ? co : po) + off;
  uint4 raw = *reinterpret_cast<const uint4*>(src);
  alignas(16) __nv_bfloat16 converted[16];
  flashinfer::vec_cast<__nv_bfloat16, __nv_fp8_e4m3>::cast<16>(
      converted, reinterpret_cast<const __nv_fp8_e4m3*>(&raw));
  const __nv_bfloat16 s = is_c ? cs : ps;
  const __nv_bfloat162 pair{s, s};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    auto* x = reinterpret_cast<__nv_bfloat162*>(converted);
    x[i] = __hmul2(x[i], pair);
  }
  reinterpret_cast<uint4*>(dst)[0] = reinterpret_cast<uint4*>(converted)[0];
  reinterpret_cast<uint4*>(dst)[1] = reinterpret_cast<uint4*>(converted)[1];
}

void Dequant(TensorView c, TensorView p, TensorView co, TensorView po, double cs, double ps) {
  CHECK_INPUT_TYPE(c, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(p, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(co, dl_bfloat16);
  CHECK_INPUT_TYPE(po, dl_bfloat16);
  TVM_FFI_ICHECK(c.IsContiguous() && p.IsContiguous() && co.IsContiguous() && po.IsContiguous());
  TVM_FFI_ICHECK(c.numel() == co.numel() && p.numel() == po.numel());
  TVM_FFI_ICHECK(c.numel() % 16 == 0 && p.numel() % 16 == 0);
  ffi::CUDADeviceGuard guard(c.device().device_id);
  const int64_t cu = c.numel() / 16, total = cu + p.numel() / 16;
  if (!total) return;
  fused_pool_dequant<<<(total + 255) / 256, 256, 0, get_stream(c.device())>>>(
      static_cast<const __nv_fp8_e4m3*>(c.data_ptr()),
      static_cast<const __nv_fp8_e4m3*>(p.data_ptr()), static_cast<__nv_bfloat16*>(co.data_ptr()),
      static_cast<__nv_bfloat16*>(po.data_ptr()), cu, total, __float2bfloat16(float(cs)),
      __float2bfloat16(float(ps)));
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess) << cudaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dequant, Dequant);
