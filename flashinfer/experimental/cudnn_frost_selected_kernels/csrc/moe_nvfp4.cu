// Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
// Routing, block-scale packing, frozen Frost GEMMs, and weighted finalization.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <array>
#include <limits>

#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Function;
using tvm::ffi::Module;
using tvm::ffi::Optional;

namespace {
constexpr DLDataType dl_fp4{kDLFloat4_e2m1fn, 4, 2};
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

__device__ int64_t sf_index(int row, int col, int columns) {
  return int64_t(row / 128) * 128 * columns + (col / 4) * 512 + (row % 32) * 16 +
         ((row % 128) / 32) * 4 + col % 4;
}

__global__ void prefix(const int32_t* counts, int32_t* offsets, int32_t* cursors,
                       int32_t* sf_offsets, int experts, float* scale) {
  int start = 0, sf_start = 0;
  for (int e = 0; e < experts; ++e) {
    offsets[e] = cursors[e] = start;
    sf_offsets[e] = sf_start;
    start += counts[e];
    sf_start += (counts[e] + 127) / 128 * 128;
  }
  scale[0] = 1.f;
  scale[1] = 4.f;
  scale[2] = 25.f;
}

__global__ void gather(const uint8_t* x, const uint8_t* input_sf, const int32_t* ids,
                       const int32_t* offsets, const int32_t* sf_offsets, int32_t* cursors,
                       int32_t* mapping, int32_t* row_experts, uint8_t* grouped, uint8_t* sf,
                       int rows, int hidden, int topk, int experts, bool swizzled) {
  __shared__ int dest;
  for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
    int e = ids[r];
    bool valid = e >= 0 && e < experts;
    e = valid ? e : 0;
    if (threadIdx.x == 0) {
      dest = atomicAdd(cursors + e, 1);
      mapping[r] = dest;
      row_experts[dest] = e;
    }
    __syncthreads();
    auto source = reinterpret_cast<const int4*>(x) + (r / topk) * (hidden / 32);
    auto target = reinterpret_cast<int4*>(grouped) + int64_t(dest) * (hidden / 32);
    for (int h = threadIdx.x; h < hidden / 32; h += blockDim.x)
      target[h] = valid ? source[h] : make_int4(0, 0, 0, 0);
    int cols = hidden / 16;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
      int64_t src = swizzled ? sf_index(r / topk, col, cols) : (r / topk) * cols + col;
      int64_t dst = int64_t(sf_offsets[e]) * cols + sf_index(dest - offsets[e], col, cols);
      sf[dst] = valid ? input_sf[src] : 0;
    }
    __syncthreads();
  }
}

__global__ void requantize(const __nv_bfloat16* input, const int32_t* row_experts,
                           const int32_t* offsets, const int32_t* sf_offsets, uint8_t* output,
                           uint8_t* scales, const float* global_scale, int rows, int width) {
  union InputPack {
    int4 words;
    __nv_bfloat16 values[8];
  };
  union OutputPack {
    uint32_t words;
    uint8_t values[4];
  };
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    int e = row_experts[row];
    for (int col = threadIdx.x * 8; col < width; col += blockDim.x * 8) {
      InputPack in;
      in.words = reinterpret_cast<const int4*>(input)[(row * width + col) / 8];
      float values[8], maximum = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        values[j] = __bfloat162float(in.values[j]);
        maximum = fmaxf(maximum, fabsf(values[j]));
      }
      // Two adjacent lanes own one 16-element NVFP4 block. FC1 already
      // rounded its activated output to BF16, matching CUTLASS's quantizer.
      auto mask = __activemask();
      maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1));
      __nv_fp8_e4m3 sf(maximum * (1.f / 6.f) * global_scale[0]);
      float scale = static_cast<float>(sf);
      float inverse = maximum == 0.f ? 0.f : global_scale[0] / scale;
      if (threadIdx.x % 2 == 0) {
        int cols = width / 16;
        scales[int64_t(sf_offsets[e]) * cols + sf_index(row - offsets[e], col / 16, cols)] = sf.__x;
      }
      OutputPack out;
#pragma unroll
      for (int j = 0; j < 4; ++j)
        out.values[j] = __nv_cvt_float2_to_fp4x2(
            make_float2(values[2 * j] * inverse, values[2 * j + 1] * inverse), __NV_E2M1,
            cudaRoundNearest);
      reinterpret_cast<uint32_t*>(output)[(row * width + col) / 8] = out.words;
    }
  }
}

class CudnnFrostNvfp4MoePlan final : public tvm::ffi::ModuleObj {
 public:
  CudnnFrostNvfp4MoePlan(Function fc1, Function fc2, int64_t tokens, int64_t hidden,
                         int64_t intermediate, int64_t experts, int64_t topk, int device,
                         size_t scratch1, size_t scratch2, bool gated, Array<int64_t> tail1,
                         Array<int64_t> tail2, bool swap1, bool swap2, bool swizzled)
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
        gated_(gated),
        tail1_(std::move(tail1)),
        tail2_(std::move(tail2)),
        swap1_(swap1),
        swap2_(swap2),
        swizzled_(swizzled) {
    int64_t active = std::min(s_, e_);
    sf_rows_ = 128 * (active + (s_ - active) / 128);
    size_t pos = 0;
    auto reserve = [&](size_t bytes) {
      size_t start = pos;
      pos += align128(bytes);
      return start;
    };
    // FC2 can overwrite the grouped input after FC1 has consumed it.
    x_pos_ = reserve(s_ * h_ * 2);
    mid_pos_ = reserve(s_ * i_ * 2);
    qmid_pos_ = reserve(s_ * i_ / 2);
    sf1_pos_ = reserve(sf_rows_ * h_ / 16);
    sf2_pos_ = reserve(sf_rows_ * i_ / 16);
    counts_pos_ = reserve(e_ * 4);
    offsets_pos_ = reserve(e_ * 4);
    sf_offsets_pos_ = reserve(e_ * 4);
    cursors_pos_ = reserve(e_ * 4);
    mapping_pos_ = reserve(s_ * 4);
    row_experts_pos_ = reserve(s_ * 4);
    scale_pos_ = reserve(3 * sizeof(float));
    scratch_pos_ = reserve(std::max(scratch1_, scratch2_));
    workspace_size_ = pos;
    auto problem = [&](int64_t n, int64_t k, bool swap, bool gated) {
      Array<int64_t> shape{swap ? n : s_, swap ? s_ : n, k, e_, e_};
      auto token = [&]() {
        shape.push_back(k / 2);
        shape.push_back(1);
        shape.push_back(s_ * k / 2);
      };
      auto weight = [&]() {
        shape.push_back(k / 2);
        shape.push_back(1);
        shape.push_back((gated ? 2 : 1) * n * k / 2);
      };
      if (!swap) token();
      weight();
      if (gated) weight();
      if (swap) token();
      shape.push_back(swap ? 1 : n);
      shape.push_back(swap ? n : 1);
      shape.push_back(s_ * n);
      return shape;
    };
    problem1_ = problem(i_, h_, swap1_, gated_);
    problem2_ = problem(h_, i_, swap2_, false);
  }

  const char* kind() const final { return "cudnn_frost_nvfp4_moe_plan"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "workspace_size")
      return Function::FromTyped([this]() { return int64_t(workspace_size_); });
    if (name == "stage_layout")
      return Function::FromTyped([this]() {
        return Array<int64_t>{int64_t(x_pos_),   int64_t(mid_pos_), int64_t(qmid_pos_),
                              int64_t(sf1_pos_), int64_t(sf2_pos_), int64_t(offsets_pos_),
                              sf_rows_};
      });
    if (name == "run" || name == "prepare_stages") {
      bool stages = name == "prepare_stages";
      return Function::FromTyped(
          [this, stages](TensorView out, TensorView x, TensorView ids, TensorView scores,
                         TensorView w1, TensorView w2, TensorView sf1, TensorView sf2,
                         TensorView xsf, TensorView global1, TensorView alpha1, TensorView global2,
                         TensorView alpha2, TensorView workspace) {
            run(out, x, ids, scores, w1, w2, sf1, sf2, xsf, global1, alpha1, global2, alpha2,
                workspace, stages);
          });
    }
    return Function(nullptr);
  }

 private:
  void run(TensorView out, TensorView x, TensorView ids, TensorView scores, TensorView w1,
           TensorView w2, TensorView sf1, TensorView sf2, TensorView xsf, TensorView global1,
           TensorView alpha1, TensorView global2, TensorView alpha2, TensorView workspace,
           bool stages) const {
    tensor(out, device_, dl_bfloat16, {t_, h_});
    tensor(x, device_, dl_uint8, {t_, h_ / 2});
    tensor(ids, device_, dl_int32, {t_, k_}, 4);
    tensor(scores, device_, dl_float32, {t_, k_}, 4);
    tensor(w1, device_, dl_uint8, {e_, (gated_ ? 2 : 1) * i_, h_ / 2});
    tensor(w2, device_, dl_uint8, {e_, h_, i_ / 2});
    tensor(sf1, device_, dl_uint8, {(gated_ ? 2 : 1), e_, i_, h_ / 16});
    tensor(sf2, device_, dl_uint8, {e_, h_, i_ / 16});
    tensor(global1, device_, dl_float32, {1}, 4);
    tensor(global2, device_, dl_float32, {1}, 4);
    tensor(alpha1, device_, dl_float32, {e_}, 4);
    tensor(alpha2, device_, dl_float32, {e_}, 4);
    if (swizzled_)
      tensor(xsf, device_, dl_uint8, {(t_ + 127) / 128 * 128 * h_ / 16});
    else
      tensor(xsf, device_, dl_uint8, {t_, h_ / 16});
    tensor(workspace, device_, dl_uint8, {workspace.numel()}, 128);
    TVM_FFI_ICHECK_GE(workspace.numel(), workspace_size_);
    ffi::CUDADeviceGuard guard(device_.device_id);
    auto stream = get_stream(device_);
    auto base = static_cast<char*>(workspace.data_ptr());
    auto gx = reinterpret_cast<uint8_t*>(base + x_pos_);
    auto mid = reinterpret_cast<__nv_bfloat16*>(base + mid_pos_);
    auto qm = reinterpret_cast<uint8_t*>(base + qmid_pos_);
    auto gy = reinterpret_cast<__nv_bfloat16*>(base + x_pos_);
    auto sfx = reinterpret_cast<uint8_t*>(base + sf1_pos_);
    auto sfm = reinterpret_cast<uint8_t*>(base + sf2_pos_);
    auto counts = reinterpret_cast<int32_t*>(base + counts_pos_);
    auto offsets = reinterpret_cast<int32_t*>(base + offsets_pos_);
    auto sf_offsets = reinterpret_cast<int32_t*>(base + sf_offsets_pos_);
    auto cursors = reinterpret_cast<int32_t*>(base + cursors_pos_);
    auto mapping = reinterpret_cast<int32_t*>(base + mapping_pos_);
    auto row_experts = reinterpret_cast<int32_t*>(base + row_experts_pos_);
    auto scale = reinterpret_cast<float*>(base + scale_pos_);
    auto scratch = reinterpret_cast<int64_t*>(base + scratch_pos_);
    auto expert_ids = static_cast<int32_t*>(ids.data_ptr());
    checked(cudaMemsetAsync(counts, 0, e_ * 4, stream));
    histogram<<<std::min<int64_t>((s_ + 255) / 256, 1024), 256, 0, stream>>>(expert_ids, counts, s_,
                                                                             e_);
    prefix<<<1, 1, 0, stream>>>(counts, offsets, cursors, sf_offsets, e_, scale);
    gather<<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
        static_cast<uint8_t*>(x.data_ptr()), static_cast<uint8_t*>(xsf.data_ptr()), expert_ids,
        offsets, sf_offsets, cursors, mapping, row_experts, gx, sfx, s_, h_, k_, e_, swizzled_);
    checked(cudaGetLastError());

    int64_t xshape[]{s_, h_ / 2, 1}, qshape[]{s_, i_ / 2, 1};
    int64_t xstride[]{h_ / 2, 1, s_ * h_ / 2}, qstride[]{i_ / 2, 1, s_ * i_ / 2};
    int64_t mshape[]{s_, i_, 1}, mstride[]{i_, 1, s_ * i_};
    int64_t yshape[]{s_, h_, 1}, ystride[]{h_, 1, s_ * h_};
    int64_t w1shape[]{i_, h_ / 2, e_}, w1stride[]{h_ / 2, 1, (gated_ ? 2 : 1) * i_ * h_ / 2};
    int64_t w2shape[]{h_, i_ / 2, e_}, w2stride[]{i_ / 2, 1, h_ * i_ / 2};
    int64_t sf1shape[]{i_ * h_ / 16, 1, e_}, sf1stride[]{1, 1, i_ * h_ / 16};
    int64_t sf2shape[]{h_ * i_ / 16, 1, e_}, sf2stride[]{1, 1, h_ * i_ / 16};
    int64_t sfxshape[]{sf_rows_ * h_ / 16, 1, 1}, sfxstride[]{1, 1, 1};
    int64_t sfmshape[]{sf_rows_ * i_ / 16, 1, 1};
    int64_t eshape[]{e_}, dshape[]{int64_t(scratch1_ / 8)}, unit[]{1};
    int64_t scalar_shape[]{1, 1, 1}, scalar_stride[]{1, 1, 1};
    DLTensor tx{gx, device_, 3, dl_fp4, xshape, xstride, 0};
    DLTensor tm{mid, device_, 3, dl_bfloat16, mshape, mstride, 0};
    DLTensor tqm{qm, device_, 3, dl_fp4, qshape, qstride, 0};
    DLTensor ty{gy, device_, 3, dl_bfloat16, yshape, ystride, 0};
    DLTensor up{w1.data_ptr(), device_, 3, dl_fp4, w1shape, w1stride, 0};
    DLTensor gate = up;
    gate.data = static_cast<uint8_t*>(w1.data_ptr()) + (gated_ ? i_ * h_ / 2 : 0);
    DLTensor down{w2.data_ptr(), device_, 3, dl_fp4, w2shape, w2stride, 0};
    DLTensor sf_up{sf1.data_ptr(), device_, 3, dl_float8_e4m3fn, sf1shape, sf1stride, 0};
    DLTensor sf_gate = sf_up;
    sf_gate.data = static_cast<uint8_t*>(sf1.data_ptr()) + (gated_ ? e_ * i_ * h_ / 16 : 0);
    DLTensor sf_down{sf2.data_ptr(), device_, 3, dl_float8_e4m3fn, sf2shape, sf2stride, 0};
    DLTensor sf_x{sfx, device_, 3, dl_float8_e4m3fn, sfxshape, sfxstride, 0};
    DLTensor sf_mid{sfm, device_, 3, dl_float8_e4m3fn, sfmshape, sfxstride, 0};
    DLTensor first{offsets, device_, 1, dl_int32, eshape, unit, 0};
    DLTensor desc{scratch, device_, 1, dl_int64, dshape, unit, 0};
    DLTensor factors[3];
    for (int j = 0; j < 3; ++j)
      factors[j] = DLTensor{scale + j, device_, 3, dl_float32, scalar_shape, scalar_stride, 0};
    int64_t group_shape[]{e_, 1, 1};
    DLTensor alpha_first{alpha1.data_ptr(), device_, 3, dl_float32, group_shape, scalar_stride, 0};
    DLTensor alpha_second{alpha2.data_ptr(), device_, 3, dl_float32, group_shape, scalar_stride, 0};
    int64_t mshape_sw[]{i_, s_, 1}, mstride_sw[]{1, i_, s_ * i_};
    int64_t yshape_sw[]{h_, s_, 1}, ystride_sw[]{1, h_, s_ * h_};
    DLTensor tm_sw{mid, device_, 3, dl_bfloat16, mshape_sw, mstride_sw, 0};
    DLTensor ty_sw{gy, device_, 3, dl_bfloat16, yshape_sw, ystride_sw, 0};
    checked(cudaMemsetAsync(scratch, 0, scratch1_, stream));
    // Own the TensorView descriptors until the borrowed AnyView arguments return.
    std::array<TensorView, 14> tensors{TensorView(&first),
                                       TensorView(&desc),
                                       TensorView(swap1_ ? &gate : &tx),
                                       TensorView(swap1_ ? (gated_ ? &up : &tx) : &gate),
                                       TensorView(swap1_ ? &tx : &up),
                                       TensorView(swap1_ ? &sf_gate : &sf_x),
                                       TensorView(swap1_ ? (gated_ ? &sf_up : &sf_x) : &sf_gate),
                                       TensorView(swap1_ ? &sf_x : &sf_up),
                                       TensorView(swap1_ ? &tm_sw : &tm),
                                       TensorView(&factors[0]),
                                       TensorView(&factors[1]),
                                       TensorView(&factors[2]),
                                       TensorView(&alpha_first),
                                       TensorView(&alpha_first)};
    tvm::ffi::AnyView args[17];
    int argc = 0;
    args[argc++] = problem1_;
    args[argc++] = tensors[0];
    args[argc++] = tensors[1];
    for (int j = 0; j < (gated_ ? 3 : 2); ++j) args[argc++] = tensors[j + 2];
    for (int j = 0; j < (gated_ ? 3 : 2); ++j) args[argc++] = tensors[j + 5];
    for (auto slot : tail1_) args[argc++] = tensors[slot + 8];
    args[argc++] = static_cast<void*>(stream);
    tvm::ffi::Any result;
    fc1_.CallPacked(args, argc, &result);
    requantize<<<std::min<int64_t>(s_, 4096), 128, 0, stream>>>(
        mid, row_experts, offsets, sf_offsets, qm, sfm, static_cast<float*>(global2.data_ptr()), s_,
        i_);
    checked(cudaGetLastError());
    if (stages) return;
    checked(cudaMemsetAsync(scratch, 0, scratch2_, stream));
    dshape[0] = scratch2_ / 8;
    std::array<TensorView, 8> second{TensorView(&first),
                                     TensorView(&desc),
                                     TensorView(swap2_ ? &down : &tqm),
                                     TensorView(swap2_ ? &tqm : &down),
                                     TensorView(swap2_ ? &sf_down : &sf_mid),
                                     TensorView(swap2_ ? &sf_mid : &sf_down),
                                     TensorView(swap2_ ? &ty_sw : &ty),
                                     TensorView(&alpha_second)};
    argc = 0;
    args[argc++] = problem2_;
    for (int j = 0; j < 6; ++j) args[argc++] = second[j];
    for (auto slot : tail2_) args[argc++] = second[slot == 0 ? 6 : 7];
    args[argc++] = static_cast<void*>(stream);
    fc2_.CallPacked(args, argc, &result);
    finalize<<<std::min<int64_t>(t_, 4096), 128, 0, stream>>>(
        gy, expert_ids, mapping, static_cast<float*>(scores.data_ptr()),
        static_cast<__nv_bfloat16*>(out.data_ptr()), t_, h_, k_, e_);
    checked(cudaGetLastError());
  }

  Function fc1_, fc2_;
  int64_t t_, h_, i_, e_, k_, s_, sf_rows_;
  DLDevice device_;
  size_t scratch1_, scratch2_;
  bool gated_;
  Array<int64_t> tail1_, tail2_;
  bool swap1_, swap2_, swizzled_;
  Array<int64_t> problem1_, problem2_;
  size_t x_pos_, mid_pos_, qmid_pos_, sf1_pos_, sf2_pos_, counts_pos_, offsets_pos_,
      sf_offsets_pos_, cursors_pos_, mapping_pos_, row_experts_pos_, scale_pos_, scratch_pos_,
      workspace_size_;
};

Module make_plan(Function fc1, Function fc2, int64_t tokens, int64_t hidden, int64_t intermediate,
                 int64_t experts, int64_t topk, int64_t device, int64_t scratch1, int64_t scratch2,
                 bool gated, Array<int64_t> tail1, Array<int64_t> tail2, bool swap1, bool swap2,
                 bool swizzled) {
  constexpr int64_t max_dim = 1 << 20;
  TVM_FFI_ICHECK(tokens > 0 && tokens <= max_dim && hidden > 0 && hidden <= max_dim &&
                 intermediate > 0 && intermediate <= max_dim && experts > 0 && experts <= 1024 &&
                 topk > 0 && topk <= experts);
  TVM_FFI_ICHECK_LT(tokens * topk, std::numeric_limits<int32_t>::max() - 128 * experts);
  TVM_FFI_ICHECK_EQ(hidden % 128, 0);
  TVM_FFI_ICHECK_EQ(intermediate % 128, 0);
  TVM_FFI_ICHECK(scratch1 > 0 && scratch1 % 128 == 0 && scratch2 > 0 && scratch2 % 128 == 0);
  int mask = 0;
  for (auto slot : tail1) {
    TVM_FFI_ICHECK(slot >= 0 && slot <= 5);
    TVM_FFI_ICHECK_EQ(mask & (1 << slot), 0);
    mask |= 1 << slot;
  }
  int required = 3 | (1 << 4) | (gated ? (1 << 5) : 0);
  TVM_FFI_ICHECK(mask == required || mask == (required | 12));
  TVM_FFI_ICHECK(tail2.size() == 2 &&
                 ((tail2[0] == 0 && tail2[1] == 4) || (tail2[0] == 4 && tail2[1] == 0)));
  return Module(tvm::ffi::make_object<CudnnFrostNvfp4MoePlan>(
      std::move(fc1), std::move(fc2), tokens, hidden, intermediate, experts, topk, device, scratch1,
      scratch2, gated, std::move(tail1), std::move(tail2), swap1, swap2, swizzled));
}
}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(make_plan, make_plan);
