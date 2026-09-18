// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
// Standalone routing -> exported cuDNN Frost FC1/SwiGLU -> exported cuDNN Frost FC2 -> finalize.
// No CUTLASS runner, headers, workspace, or callbacks participate in this path.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <limits>

#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::ffi::Optional;

namespace {
size_t align128(size_t n) { return (n + 127) / 128 * 128; }
void checked(cudaError_t err) { TVM_FFI_ICHECK_EQ(err, cudaSuccess) << cudaGetErrorString(err); }

__global__ void histogram(const int32_t* ids, int32_t* counts, int rows, int experts) {
  for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < rows;
       r += int64_t(blockDim.x) * gridDim.x) {
    int e = ids[r];
    // Keep dummy rows in expert zero so every expanded row has storage. Invalid
    // ids are masked in gather/finalize, never used as memory addresses.
    atomicAdd(counts + (e >= 0 && e < experts ? e : 0), 1);
  }
}

__global__ void prefix(const int32_t* counts, int32_t* offsets, int32_t* cursors, int experts,
                       float* scale) {
  int start = 0;
  for (int e = 0; e < experts; ++e) {
    offsets[e] = cursors[e] = start;
    start += counts[e];
  }
  *scale = 1.f;
}

__global__ void gather(const __nv_bfloat16* x, const int32_t* ids, int32_t* cursors,
                       int32_t* mapping, __nv_bfloat16* grouped, int rows, int hidden, int topk,
                       int experts) {
  __shared__ int dest;
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    if (threadIdx.x == 0) {
      dest = atomicAdd(cursors + (valid ? e : 0), 1);
      mapping[r] = dest;
    }
    __syncthreads();
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 8);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 8);
    for (int h = threadIdx.x; h < hidden / 8; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    __syncthreads();
  }
}

__global__ void finalize(const __nv_bfloat16* grouped, const int32_t* ids, const int32_t* mapping,
                         const float* scores, __nv_bfloat16* out, int tokens, int hidden, int topk,
                         int experts) {
  union Pack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  // One CTA per token: no per-element division, 128-bit data loads/stores,
  // and deterministic FP32 top-k reduction before a single BF16 conversion.
  for (int64_t t = blockIdx.x; t < tokens; t += gridDim.x) {
    for (int h = threadIdx.x; h < hidden / 8; h += blockDim.x) {
      float sum[8] = {};
      for (int k = 0; k < topk; ++k) {
        int64_t r = t * topk + k;
        if (ids[r] >= 0 && ids[r] < experts) {
          Pack value;
          value.words =
              reinterpret_cast<const int4*>(grouped)[int64_t(mapping[r]) * (hidden / 8) + h];
          float score = scores[r];
#pragma unroll
          for (int v = 0; v < 8; ++v) sum[v] += __bfloat162float(value.values[v]) * score;
        }
      }
      Pack result;
#pragma unroll
      for (int v = 0; v < 8; ++v) result.values[v] = __float2bfloat16(sum[v]);
      reinterpret_cast<int4*>(out)[t * (hidden / 8) + h] = result.words;
    }
  }
}

void tensor(TensorView t, DLDevice device, DLDataType dtype, std::initializer_list<int64_t> shape,
            size_t alignment = 16) {
  TVM_FFI_ICHECK(t.device().device_type == kDLCUDA && t.device().device_id == device.device_id);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(t.dtype()), encode_dlpack_dtype(dtype));
  TVM_FFI_ICHECK(t.IsContiguous());
  TVM_FFI_ICHECK_EQ(t.ndim(), shape.size());
  int dim = 0;
  for (auto size : shape) TVM_FFI_ICHECK_EQ(t.size(dim++), size);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment, 0);
}

class CudnnFrostMoePlan final : public tvm::ffi::ModuleObj {
 public:
  CudnnFrostMoePlan(Function fc1, Function fc2, int64_t tokens, int64_t hidden,
                    int64_t intermediate, int64_t experts, int64_t topk, int device,
                    size_t scratch1, size_t scratch2, bool scale_first, bool swap1, bool swap2)
      : fc1_(std::move(fc1)),
        fc2_(std::move(fc2)),
        t_(tokens),
        h_(hidden),
        i_(intermediate),
        e_(experts),
        k_(topk),
        s_(tokens * topk),
        device_{kDLCUDA, device},
        scratch1_(scratch1),
        scratch2_(scratch2),
        scale_first_(scale_first),
        swap1_(swap1),
        swap2_(swap2),
        problem1_{s_, i_,          h_, e_, e_,          h_, 1, s_ * h_, h_,
                  1,  2 * i_ * h_, h_, 1,  2 * i_ * h_, i_, 1, s_ * i_},
        problem2_{s_, h_, i_, e_, e_, i_, 1, s_ * i_, i_, 1, h_ * i_, h_, 1, s_ * h_} {
    if (swap1_)
      problem1_ = Array<int64_t>{i_, s_,          h_, e_, e_,      h_, 1,  2 * i_ * h_, h_,
                                 1,  2 * i_ * h_, h_, 1,  s_ * h_, 1,  i_, s_ * i_};
    if (swap2_)
      problem2_ =
          Array<int64_t>{h_, s_, i_, e_, e_, i_, 1, h_ * i_, i_, 1, s_ * i_, 1, h_, s_ * h_};
    size_t pos = 0;
    auto reserve = [&](size_t bytes) {
      size_t start = pos;
      pos += align128(bytes);
      return start;
    };
    x_pos_ = reserve(s_ * h_ * 2);
    mid_pos_ = reserve(s_ * i_ * 2);
    // FC1 has finished consuming grouped tokens before FC2 writes its output.
    y_pos_ = x_pos_;
    counts_pos_ = reserve(e_ * 4);
    offsets_pos_ = reserve(e_ * 4);
    cursors_pos_ = reserve(e_ * 4);
    mapping_pos_ = reserve(s_ * 4);
    scale_pos_ = reserve(4);
    scratch_pos_ = reserve(std::max(scratch1_, scratch2_));
    workspace_size_ = pos;
  }

  const char* kind() const final { return "cudnn_frost_bf16_moe_plan"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "workspace_size")
      return Function::FromTyped([this]() { return int64_t(workspace_size_); });
    if (name == "run")
      return Function::FromTyped(
          [this](TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
                 TensorView w2,
                 TensorView workspace) { run(out, x, ids, scores, w1, w2, workspace); });
    return Function(nullptr);
  }

 private:
  void run(TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
           TensorView w2, TensorView workspace) const {
    tensor(out, device_, dl_bfloat16, {t_, h_});
    tensor(x, device_, dl_bfloat16, {t_, h_});
    tensor(ids, device_, dl_int32, {t_, k_}, 4);
    tensor(scores, device_, dl_float32, {t_, k_}, 4);
    tensor(w1, device_, dl_bfloat16, {e_, 2 * i_, h_});
    tensor(w2, device_, dl_bfloat16, {e_, h_, i_});
    tensor(workspace, device_, dl_uint8, {workspace.numel()}, 128);
    TVM_FFI_ICHECK_GE(workspace.numel(), workspace_size_);
    ffi::CUDADeviceGuard guard(device_.device_id);
    auto stream = get_stream(device_);
    auto base = static_cast<char*>(workspace.data_ptr());
    auto gx = reinterpret_cast<__nv_bfloat16*>(base + x_pos_);
    auto mid = reinterpret_cast<__nv_bfloat16*>(base + mid_pos_);
    auto gy = reinterpret_cast<__nv_bfloat16*>(base + y_pos_);
    auto counts = reinterpret_cast<int32_t*>(base + counts_pos_);
    auto offsets = reinterpret_cast<int32_t*>(base + offsets_pos_);
    auto cursors = reinterpret_cast<int32_t*>(base + cursors_pos_);
    auto mapping = reinterpret_cast<int32_t*>(base + mapping_pos_);
    auto scale = reinterpret_cast<float*>(base + scale_pos_);
    auto scratch = reinterpret_cast<int64_t*>(base + scratch_pos_);
    auto expert_ids = static_cast<int32_t*>(ids.data_ptr());
    checked(cudaMemsetAsync(counts, 0, e_ * 4, stream));
    histogram<<<std::min<int64_t>((s_ + 255) / 256, 1024), 256, 0, stream>>>(expert_ids, counts, s_,
                                                                             e_);
    prefix<<<1, 1, 0, stream>>>(counts, offsets, cursors, e_, scale);
    gather<<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data_ptr()), expert_ids, cursors, mapping, gx, s_, h_, k_,
        e_);
    checked(cudaGetLastError());

    int64_t xshape[]{s_, h_, 1}, mshape[]{s_, i_, 1};
    int64_t xstride[]{h_, 1, s_ * h_}, mstride[]{i_, 1, s_ * i_};
    int64_t w1shape[]{i_, h_, e_}, w1stride[]{h_, 1, 2 * i_ * h_};
    int64_t w2shape[]{h_, i_, e_}, w2stride[]{i_, 1, h_ * i_};
    int64_t eshape[]{e_}, dshape[]{int64_t(scratch1_ / 8)}, unit[]{1};
    int64_t scale_shape[]{1, 1, 1}, scale_stride[]{1, 1, 1};
    DLTensor tx{gx, device_, 3, dl_bfloat16, xshape, xstride, 0};
    DLTensor tm{mid, device_, 3, dl_bfloat16, mshape, mstride, 0};
    DLTensor ty{gy, device_, 3, dl_bfloat16, xshape, xstride, 0};
    DLTensor up{w1.data_ptr(), device_, 3, dl_bfloat16, w1shape, w1stride, 0};
    DLTensor gate = up;
    gate.data = static_cast<__nv_bfloat16*>(w1.data_ptr()) + i_ * h_;
    DLTensor down{w2.data_ptr(), device_, 3, dl_bfloat16, w2shape, w2stride, 0};
    DLTensor first{offsets, device_, 1, dl_int32, eshape, unit, 0};
    DLTensor desc{scratch, device_, 1, dl_int64, dshape, unit, 0};
    DLTensor factor{scale, device_, 3, dl_float32, scale_shape, scale_stride, 0};
    int64_t mshape_sw[]{i_, s_, 1}, mstride_sw[]{1, i_, s_ * i_};
    int64_t yshape_sw[]{h_, s_, 1}, ystride_sw[]{1, h_, s_ * h_};
    DLTensor tm_sw{mid, device_, 3, dl_bfloat16, mshape_sw, mstride_sw, 0};
    DLTensor ty_sw{gy, device_, 3, dl_bfloat16, yshape_sw, ystride_sw, 0};
    auto fc1_out = swap1_ ? &tm_sw : &tm;
    // Persistent scheduler state must be reset on every invocation, including replay.
    checked(cudaMemsetAsync(scratch, 0, scratch1_, stream));
    fc1_(problem1_, TensorView(&first), TensorView(&desc), TensorView(swap1_ ? &gate : &tx),
         TensorView(swap1_ ? &up : &gate), TensorView(swap1_ ? &tx : &up),
         TensorView(scale_first_ ? &factor : fc1_out), TensorView(scale_first_ ? fc1_out : &factor),
         static_cast<void*>(stream));
    checked(cudaMemsetAsync(scratch, 0, scratch2_, stream));
    dshape[0] = scratch2_ / 8;
    fc2_(problem2_, TensorView(&first), TensorView(&desc), TensorView(swap2_ ? &down : &tm),
         TensorView(swap2_ ? &tm : &down), TensorView(swap2_ ? &ty_sw : &ty),
         static_cast<void*>(stream));
    finalize<<<std::min<int64_t>(t_, 4096), 128, 0, stream>>>(
        gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_, k_, e_);
    checked(cudaGetLastError());
  }

  Function fc1_, fc2_;
  int64_t t_, h_, i_, e_, k_, s_;
  DLDevice device_;
  size_t scratch1_, scratch2_;
  bool scale_first_, swap1_, swap2_;
  Array<int64_t> problem1_, problem2_;
  size_t x_pos_, mid_pos_, y_pos_, counts_pos_, offsets_pos_, cursors_pos_, mapping_pos_,
      scale_pos_, scratch_pos_, workspace_size_;
};

Module make_plan(Function fc1, Function fc2, int64_t tokens, int64_t hidden, int64_t intermediate,
                 int64_t experts, int64_t topk, int64_t device, int64_t scratch1, int64_t scratch2,
                 bool scale_first, bool swap1, bool swap2) {
  // Limit the private ABI to practical int32-indexed dimensions and avoid
  // overflow in the workspace/launch descriptors even for malformed callers.
  constexpr int64_t max_dim = 1 << 20;
  TVM_FFI_ICHECK(tokens > 0 && tokens <= max_dim && hidden > 0 && hidden <= max_dim &&
                 intermediate > 0 && intermediate <= max_dim && experts > 0 && experts <= 1024 &&
                 topk > 0 && topk <= experts);
  TVM_FFI_ICHECK_LE(tokens * topk, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_EQ(hidden % 8, 0) << "cuDNN Frost MoE routing uses 128-bit BF16 packs";
  TVM_FFI_ICHECK(scratch1 >= 0 && scratch1 % 128 == 0 && scratch2 >= 0 && scratch2 % 128 == 0);
  return Module(tvm::ffi::make_object<CudnnFrostMoePlan>(
      std::move(fc1), std::move(fc2), tokens, hidden, intermediate, experts, topk, device, scratch1,
      scratch2, scale_first, swap1, swap2));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_plan, make_plan);
