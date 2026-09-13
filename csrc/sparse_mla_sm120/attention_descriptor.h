// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <climits>
#include <utility>

#include <cuda_bf16.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/extra/module.h>

#include <flashinfer/attention/sparse_mla_sm120/execution/attention_plan.h>

#include "../tvm_ffi_utils.h"

namespace flashinfer::sparse_mla_sm120::execution {

inline AttentionMetadata unpack_metadata(const ffi::Array<int64_t> &d, size_t i = 0) {
  TVM_FFI_ICHECK_EQ(d.size() - i, 18) << "attention metadata ABI mismatch";
  for (size_t j = i; j < d.size(); ++j) {
    TVM_FFI_ICHECK_GE(d[j], 0) << "negative attention metadata";
    const auto field = j - i;
    if (field != 7 && field != 8 && field != 10 && field != 11 && field != 12)
      TVM_FFI_ICHECK_LE(d[j], INT_MAX) << "attention metadata integer range";
    if (field >= 13 && field <= 16) TVM_FFI_ICHECK_LE(d[j], 1);
  }
  return {int(d[i]),         int(d[i + 1]),   int(d[i + 2]),     int(d[i + 3]),
          int(d[i + 4]),     int(d[i + 5]),   int(d[i + 6]),     size_t(d[i + 7]),
          size_t(d[i + 8]),  int(d[i + 9]),   size_t(d[i + 10]), size_t(d[i + 11]),
          size_t(d[i + 12]), bool(d[i + 13]), bool(d[i + 14]),   bool(d[i + 15]),
          bool(d[i + 16]),   int(d[i + 17])};
}

inline ffi::Array<int64_t> pack_metadata(const AttentionMetadata &m) {
  return {m.model,
          m.tokens,
          m.heads,
          m.topk,
          m.extra_topk,
          m.page_size,
          m.extra_page_size,
          int64_t(m.page_stride_bytes),
          int64_t(m.extra_page_stride_bytes),
          m.row_stride_bytes,
          int64_t(m.indices_stride),
          int64_t(m.extra_indices_stride),
          int64_t(m.lse_stride),
          m.has_lengths,
          m.has_extra_lengths,
          m.has_sink,
          m.extra_fp4,
          m.variant};
}

class AttentionPlanObj : public ffi::ModuleObj {
 public:
  explicit AttentionPlanObj(ExecutionPlan value) : plan(std::move(value)) {}
  const char *kind() const final { return "sparse_mla_execution_plan"; }
  const ExecutionPlan plan;

  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final {
    if (name == "workspace")
      return ffi::Function::FromTyped([this]() {
        auto requirement = [](ffi::Array<int64_t> shape, const char *dtype, size_t bytes,
                              int alignment) -> ffi::Array<ffi::Any> {
          return {shape, ffi::String(dtype), int64_t(bytes), alignment};
        };
        return ffi::Array<ffi::Array<ffi::Any>>{
            requirement({int64_t(plan.partial_bytes / sizeof(__nv_bfloat16))}, "bfloat16",
                        plan.partial_bytes, plan.alignment),
            requirement({int64_t(plan.lse_bytes / sizeof(float))}, "float32", plan.lse_bytes,
                        plan.alignment),
            requirement({plan.metadata.tokens, plan.metadata.heads}, "float32",
                        size_t(plan.metadata.tokens) * plan.metadata.heads * sizeof(float),
                        alignof(float))};
      });
    if (name == "inspect")
      return ffi::Function::FromTyped([this]() {
        const char *numeric[] = {"fp8", "hybrid", "bf16", "nvfp4"};
        const char *implementation[] = {
            "ordinary", "mixed", "full_bf16", "dsv4_nvfp4_decode", "dsv4_nvfp4_grouped_decode",
            "sg",       "mg",    "fulltile",  "swapab",             "dsv4_nvfp4_prefill"};
        const char *merge[] = {"direct", "merge2", "general", "stage1"};
        return ffi::Map<ffi::String, ffi::Any>{
            {"numeric_route", ffi::String(numeric[int(plan.numeric)])},
            {"implementation", ffi::String(implementation[int(plan.implementation)])},
            {"merge", ffi::String(merge[int(plan.merge)])},
            {"cpb", plan.cpb},
            {"active_splits", plan.active_splits},
            {"scratch_split_stride", plan.scratch_split_stride},
            {"scratch_heads", plan.scratch_heads},
            {"partial_bytes", int64_t(plan.partial_bytes)},
            {"lse_bytes", int64_t(plan.lse_bytes)},
            {"block_threads", plan.block_threads},
            {"shared_bytes", int64_t(plan.shared_bytes)},
            {"variant", plan.metadata.variant},
            {"tokens", plan.metadata.tokens},
            {"row_stride_bytes", plan.metadata.row_stride_bytes}};
      });
    return ffi::Function(nullptr);
  }
};

inline ffi::Module pack_plan(ExecutionPlan plan) {
  return ffi::Module(ffi::make_object<AttentionPlanObj>(std::move(plan)));
}

inline const ExecutionPlan &unpack_plan(const ffi::Module &carrier, bool is_dsv4_nvfp4) {
  const auto *object = dynamic_cast<const AttentionPlanObj *>(carrier.get());
  TVM_FFI_ICHECK(object != nullptr) << "execution plan module mismatch";
  TVM_FFI_ICHECK((object->plan.numeric == NumericRoute::NVFP4) == is_dsv4_nvfp4)
      << "execution plan route/module mismatch";
  return object->plan;
}

}  // namespace flashinfer::sparse_mla_sm120::execution
