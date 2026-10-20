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

// TVM-FFI binding of the fused mHC post -> pre (+ input RMSNorm) kernels: per-device
// preparation, operand validation (declines instead of throwing), regime selection,
// TMA tensor-map caching and the launch on the caller's current stream.

#include <tvm/ffi/container/array.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "mhc_fused_post_pre.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace mhc_fpp {
namespace {

// Status codes of mhc_fused_post_pre_run and mhc_fused_post_pre_check (for the latter
// kLaunched means "would launch"). Anything but kLaunched means nothing was enqueued
// and the caller keeps its own implementation. The order of the checks is semantics ->
// shape/stride -> device, so every operand check runs on host tensors too.
enum Status : int64_t {
  kLaunched = 0,
  kBadDtype = 1,
  kBadShape = 2,
  kNotContiguous = 3,
  kMisaligned = 4,
  kOverlap = 5,
  kBadDevice = 6,
  kNotPrepared = 7,
  kNoKernelForShape = 8,
};

constexpr int kMaxDevices = 64;

struct DeviceState {
  std::atomic<bool> ready{false};
  ClusterLimits limits{0, 0, 0, 0, 0};
};
std::array<DeviceState, kMaxDevices> g_devices;
std::mutex g_prepare_mutex;

using EncodeTiledFn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                                   const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                                   const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
std::atomic<EncodeTiledFn> g_encode_tiled{nullptr};

std::atomic<int64_t> g_tensor_map_hits{0};
std::atomic<int64_t> g_tensor_map_misses{0};
std::array<std::atomic<int64_t>, kNumVariants> g_launches{};

// Box of `box_rows` tokens x one CTA's hidden slice over a dense bf16 [M, H] (rank 2) or
// [M, 4, H] (rank 3) tensor; rows past num_tokens are zero-filled on load, clipped on store.
void encode_tensor_map(CUtensorMap* map, const void* ptr, int rank, int num_tokens, int box_rows) {
  cuuint64_t dims[3];
  cuuint64_t strides[2];
  cuuint32_t box[3];
  cuuint32_t elem_strides[3] = {1, 1, 1};
  if (rank == 2) {
    dims[0] = kHidden;
    dims[1] = static_cast<cuuint64_t>(num_tokens);
    strides[0] = static_cast<cuuint64_t>(kHidden) * 2;
    box[0] = kSliceWidth;
    box[1] = static_cast<cuuint32_t>(box_rows);
  } else {
    dims[0] = kHidden;
    dims[1] = kStreams;
    dims[2] = static_cast<cuuint64_t>(num_tokens);
    strides[0] = static_cast<cuuint64_t>(kHidden) * 2;
    strides[1] = static_cast<cuuint64_t>(kStreams) * kHidden * 2;
    box[0] = kSliceWidth;
    box[1] = kStreams;
    box[2] = static_cast<cuuint32_t>(box_rows);
  }
  const CUresult res = g_encode_tiled.load(std::memory_order_acquire)(
      map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, const_cast<void*>(ptr), dims, strides, box,
      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "mhc_fused_post_pre: cuTensorMapEncodeTiled failed with CUresult " << res;
}

// Per-thread direct-mapped cache of encoded tensor maps. A map is a pure function of
// (address, rank, num_tokens, box rows): H and the dense strides are fixed and checked per
// call, so a hit is byte-identical to a fresh encode. Nothing is owned: a stale entry for
// freed memory is never dereferenced on the host and is only reused for the same address
// and geometry, where it is exact again. Captured CUDA graphs hold their own copy of the
// launch parameters, so replays never read this cache.
struct TensorMapSlot {
  CUtensorMap map;
  uint64_t address;
  int32_t num_tokens;
  int16_t box_rows;
  int8_t rank;
  int8_t device;
  bool valid;
};
constexpr int kTensorMapSlots = 64;
struct TensorMapCache {
  TensorMapSlot slots[kTensorMapSlots];
};
thread_local std::unique_ptr<TensorMapCache> tls_tensor_maps;

CUtensorMap cached_tensor_map(const void* ptr, int rank, int num_tokens, int box_rows, int device) {
  if (!tls_tensor_maps) tls_tensor_maps = std::make_unique<TensorMapCache>();
  const uint64_t address = reinterpret_cast<uint64_t>(ptr);
  uint64_t h = address ^ (static_cast<uint64_t>(num_tokens) * 0x9E3779B97F4A7C15ull) ^
               (static_cast<uint64_t>(box_rows) << 3) ^ (static_cast<uint64_t>(rank) << 9) ^
               (static_cast<uint64_t>(device) << 11);
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  TensorMapSlot& slot = tls_tensor_maps->slots[h & (kTensorMapSlots - 1)];
  if (slot.valid && slot.address == address && slot.num_tokens == num_tokens &&
      slot.box_rows == box_rows && slot.rank == rank && slot.device == device) {
    g_tensor_map_hits.fetch_add(1, std::memory_order_relaxed);
    return slot.map;
  }
  slot.valid = false;
  encode_tensor_map(&slot.map, ptr, rank, num_tokens, box_rows);
  slot.address = address;
  slot.num_tokens = num_tokens;
  slot.box_rows = static_cast<int16_t>(box_rows);
  slot.rank = static_cast<int8_t>(rank);
  slot.device = static_cast<int8_t>(device);
  slot.valid = true;
  g_tensor_map_misses.fetch_add(1, std::memory_order_relaxed);
  return slot.map;
}

void check_cuda(cudaError_t err, const char* what) {
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "mhc_fused_post_pre: " << what << " failed: " << cudaGetErrorString(err);
}

bool same_shape(const TensorView& t, std::initializer_list<int64_t> shape) {
  if (t.ndim() != static_cast<int32_t>(shape.size())) return false;
  int32_t i = 0;
  for (int64_t s : shape) {
    if (t.size(i++) != s) return false;
  }
  return true;
}

uintptr_t address_of(const TensorView& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) + static_cast<uintptr_t>(t.byte_offset());
}

int64_t num_bytes(const TensorView& t) { return t.numel() * get_element_size(t); }

bool overlaps(const TensorView& a, const TensorView& b) {
  const uintptr_t a0 = address_of(a), b0 = address_of(b);
  return a0 < b0 + static_cast<uintptr_t>(num_bytes(b)) &&
         b0 < a0 + static_cast<uintptr_t>(num_bytes(a));
}

// Operand checks shared by the launch and the inputs-only query. With `outputs` the checks
// run in the launch's order (dtype -> shape -> layout -> aliasing -> device) and return its
// status codes; without them only the inputs are checked.
int64_t check_operands(const TensorView& x, const TensorView& residual,
                       const TensorView& post_layer_mix, const TensorView& comb_res_mix,
                       const TensorView& fn, const TensorView& hc_scale, const TensorView& hc_base,
                       const TensorView& norm_weight,
                       const std::array<const TensorView*, 4>* outputs) {
  const TensorView* inputs[] = {&x,  &residual, &post_layer_mix, &comb_res_mix,
                                &fn, &hc_scale, &hc_base,        &norm_weight};
  // Semantics: dtypes.
  for (const TensorView* t : {&x, &residual, &norm_weight}) {
    if (t->dtype() != dl_bfloat16) return kBadDtype;
  }
  for (const TensorView* t : {&post_layer_mix, &comb_res_mix, &fn, &hc_scale, &hc_base}) {
    if (t->dtype() != dl_float32) return kBadDtype;
  }
  if (outputs != nullptr) {
    const auto& out = *outputs;  // residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur
    if (out[0]->dtype() != dl_bfloat16 || out[3]->dtype() != dl_bfloat16 ||
        out[1]->dtype() != dl_float32 || out[2]->dtype() != dl_float32) {
      return kBadDtype;
    }
  }
  // Shapes: the kernels are compiled for H = 4096, four streams and 24 mixes, and derive
  // every stride from these shapes.
  if (x.ndim() != 2) return kBadShape;
  const int64_t m = x.size(0);
  if (m <= 0 || m > (int64_t{1} << 24)) return kBadShape;
  if (!same_shape(x, {m, kHidden}) || !same_shape(residual, {m, kStreams, kHidden}) ||
      !same_shape(post_layer_mix, {m, kStreams}) ||
      !same_shape(comb_res_mix, {m, kStreams, kStreams}) ||
      !same_shape(fn, {kMixWidth, kStreams * kHidden}) || !same_shape(hc_scale, {3}) ||
      !same_shape(hc_base, {kMixWidth}) || !same_shape(norm_weight, {kHidden})) {
    return kBadShape;
  }
  if (outputs != nullptr) {
    const auto& out = *outputs;
    if (!same_shape(*out[0], {m, kStreams, kHidden}) || !same_shape(*out[1], {m, kStreams}) ||
        !same_shape(*out[2], {m, kStreams, kStreams}) || !same_shape(*out[3], {m, kHidden})) {
      return kBadShape;
    }
  }
  // Layout: dense row-major; TMA / bulk copies need 16-byte addresses, the fn slice loads
  // are 32-byte vectors.
  for (const TensorView* t : inputs) {
    if (!t->IsContiguous()) return kNotContiguous;
    if (address_of(*t) % 16 != 0) return kMisaligned;
  }
  if (outputs != nullptr) {
    for (const TensorView* t : *outputs) {
      if (!t->IsContiguous()) return kNotContiguous;
      if (address_of(*t) % 16 != 0) return kMisaligned;
    }
  }
  if (address_of(fn) % 32 != 0) return kMisaligned;
  // Outputs are written while inputs are still being read: no aliasing.
  if (outputs != nullptr) {
    const auto& out = *outputs;
    for (int i = 0; i < 4; ++i) {
      for (const TensorView* t : inputs) {
        if (overlaps(*out[i], *t)) return kOverlap;
      }
      for (int j = i + 1; j < 4; ++j) {
        if (overlaps(*out[i], *out[j])) return kOverlap;
      }
    }
  }
  // Device last.
  const DLDevice device = x.device();
  if (device.device_type != kDLCUDA || device.device_id < 0 || device.device_id >= kMaxDevices) {
    return kBadDevice;
  }
  for (const TensorView* t : inputs) {
    if (t->device().device_type != kDLCUDA || t->device().device_id != device.device_id) {
      return kBadDevice;
    }
  }
  if (outputs != nullptr) {
    for (const TensorView* t : *outputs) {
      if (t->device().device_type != kDLCUDA || t->device().device_id != device.device_id) {
        return kBadDevice;
      }
    }
  }
  return kLaunched;
}

// Readiness of `device_id` and the regime for `num_tokens` (operands already checked).
int64_t check_plan(int device_id, int64_t num_tokens, LaunchPlan* plan) {
  const DeviceState& state = g_devices[device_id];
  if (!state.ready.load(std::memory_order_acquire)) return kNotPrepared;
  *plan = plan_launch(static_cast<int>(num_tokens), state.limits);
  if (plan->variant == Variant::kUnsupported) return kNoKernelForShape;
  return kLaunched;
}

}  // namespace

// Sets the kernels' launch attributes and measures their co-resident cluster capacity on
// `device_id`; idempotent. Must run outside CUDA graph capture before the first launch on
// that device. Throws on any CUDA error.
void mhc_fused_post_pre_prepare(int64_t device_id) {
  TVM_FFI_ICHECK(device_id >= 0 && device_id < kMaxDevices)
      << "mhc_fused_post_pre: device index out of range: " << device_id;
  DeviceState& state = g_devices[device_id];
  if (state.ready.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(g_prepare_mutex);
  if (state.ready.load(std::memory_order_acquire)) return;
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  if (g_encode_tiled.load(std::memory_order_acquire) == nullptr) {
    void* fn = nullptr;
    cudaDriverEntryPointQueryResult query;
    check_cuda(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000,
                                                cudaEnableDefault, &query),
               "cudaGetDriverEntryPointByVersion(cuTensorMapEncodeTiled)");
    TVM_FFI_ICHECK(query == cudaDriverEntryPointSuccess && fn != nullptr)
        << "mhc_fused_post_pre: cuTensorMapEncodeTiled is not available";
    g_encode_tiled.store(reinterpret_cast<EncodeTiledFn>(fn), std::memory_order_release);
  }
  ClusterLimits limits{0, 0, 0, 0, 0};
  check_cuda(prepare_decode_stream(&limits.decode, &limits.stream), "decode/stream prepare");
  check_cuda(prepare_short_stream(&limits.short_stream), "short-stream prepare");
  check_cuda(prepare_crossover(&limits.crossover), "crossover prepare");
  check_cuda(prepare_long_stream(&limits.long_stream), "long-stream prepare");
  state.limits = limits;
  state.ready.store(true, std::memory_order_release);
}

// [prepared, decode, stream, short_stream, crossover, long_stream] cluster capacities.
ffi::Array<int64_t> mhc_fused_post_pre_limits(int64_t device_id) {
  TVM_FFI_ICHECK(device_id >= 0 && device_id < kMaxDevices)
      << "mhc_fused_post_pre: device index out of range: " << device_id;
  const DeviceState& state = g_devices[device_id];
  if (!state.ready.load(std::memory_order_acquire)) return ffi::Array<int64_t>{0, 0, 0, 0, 0, 0};
  const ClusterLimits& l = state.limits;
  return ffi::Array<int64_t>{1, l.decode, l.stream, l.short_stream, l.crossover, l.long_stream};
}

// Pure regime selection (no CUDA calls): [variant, tile_tokens, num_tiles, num_clusters].
ffi::Array<int64_t> mhc_fused_post_pre_plan(int64_t num_tokens, int64_t decode, int64_t stream,
                                            int64_t short_stream, int64_t crossover,
                                            int64_t long_stream) {
  const ClusterLimits limits{static_cast<int>(decode), static_cast<int>(stream),
                             static_cast<int>(short_stream), static_cast<int>(crossover),
                             static_cast<int>(long_stream)};
  const LaunchPlan plan = plan_launch(static_cast<int>(num_tokens), limits);
  return ffi::Array<int64_t>{static_cast<int64_t>(plan.variant), plan.tile_tokens, plan.num_tiles,
                             plan.num_clusters};
}

// [tensor-map cache hits, misses, launches per Variant...] (process-wide).
ffi::Array<int64_t> mhc_fused_post_pre_counters() {
  ffi::Array<int64_t> out{g_tensor_map_hits.load(std::memory_order_relaxed),
                          g_tensor_map_misses.load(std::memory_order_relaxed)};
  for (const auto& n : g_launches) out.push_back(n.load(std::memory_order_relaxed));
  return out;
}

int64_t mhc_fused_post_pre_run(TensorView x, TensorView residual, TensorView post_layer_mix,
                               TensorView comb_res_mix, TensorView fn, TensorView hc_scale,
                               TensorView hc_base, TensorView norm_weight, TensorView residual_cur,
                               TensorView post_mix_cur, TensorView comb_mix_cur,
                               TensorView layer_input_cur, bool dry_run) {
  const std::array<const TensorView*, 4> outputs = {&residual_cur, &post_mix_cur, &comb_mix_cur,
                                                    &layer_input_cur};
  const int64_t operands = check_operands(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                                          hc_base, norm_weight, &outputs);
  if (operands != kLaunched) return operands;
  const int64_t m = x.size(0);
  const DLDevice device = x.device();
  LaunchPlan plan{Variant::kUnsupported, 0, 0, 0};
  const int64_t planned = check_plan(device.device_id, m, &plan);
  if (planned != kLaunched) return planned;
  if (dry_run) return kLaunched;

  // Tensor-map encoding needs the tensors' device context current.
  ffi::CUDADeviceGuard device_guard(device.device_id);
  TensorMaps maps;
  maps.x = cached_tensor_map(reinterpret_cast<const void*>(address_of(x)), 2, static_cast<int>(m),
                             plan.tile_tokens, device.device_id);
  maps.residual = cached_tensor_map(reinterpret_cast<const void*>(address_of(residual)), 3,
                                    static_cast<int>(m), plan.tile_tokens, device.device_id);
  maps.residual_cur = cached_tensor_map(reinterpret_cast<const void*>(address_of(residual_cur)), 3,
                                        static_cast<int>(m), plan.tile_tokens, device.device_id);

  LaunchArgs args;
  args.x = reinterpret_cast<const void*>(address_of(x));
  args.residual = reinterpret_cast<const void*>(address_of(residual));
  args.post_layer_mix = reinterpret_cast<const float*>(address_of(post_layer_mix));
  args.comb_res_mix = reinterpret_cast<const float*>(address_of(comb_res_mix));
  args.fn = reinterpret_cast<const float*>(address_of(fn));
  args.hc_scale = reinterpret_cast<const float*>(address_of(hc_scale));
  args.hc_base = reinterpret_cast<const float*>(address_of(hc_base));
  args.norm_weight = reinterpret_cast<const void*>(address_of(norm_weight));
  args.residual_cur = reinterpret_cast<void*>(address_of(residual_cur));
  args.post_mix_cur = reinterpret_cast<float*>(address_of(post_mix_cur));
  args.comb_mix_cur = reinterpret_cast<float*>(address_of(comb_mix_cur));
  args.layer_input_cur = reinterpret_cast<void*>(address_of(layer_input_cur));
  args.num_tokens = static_cast<int>(m);
  args.tile_tokens = plan.tile_tokens;
  args.num_tiles = plan.num_tiles;

  const cudaStream_t stream = get_stream(device);
  cudaError_t err = cudaSuccess;
  switch (plan.variant) {
    case Variant::kDecodeNarrow:
    case Variant::kDecodeWide:
      err = launch_decode(args, plan.num_clusters, maps, stream);
      break;
    case Variant::kStream:
    case Variant::kStreamLongRegs:
      err = launch_stream(args, plan.num_clusters, plan.variant == Variant::kStreamLongRegs, maps,
                          stream);
      break;
    case Variant::kShortStream:
      err = launch_short_stream(args, plan.num_clusters, maps, stream);
      break;
    case Variant::kCrossover:
      err = launch_crossover(args, plan.num_clusters, maps, stream);
      break;
    case Variant::kLongStream:
      err = launch_long_stream(args, plan.num_clusters, maps, stream);
      break;
    default:
      return kNoKernelForShape;
  }
  check_cuda(err, "kernel launch");
  g_launches[static_cast<int>(plan.variant)].fetch_add(1, std::memory_order_relaxed);
  return kLaunched;
}

// Inputs-only form of mhc_fused_post_pre_run(..., dry_run=true) for a call that lets the op
// allocate its outputs: freshly allocated outputs always pass the output checks, so the
// verdict depends on the inputs alone and nothing has to be allocated to obtain it.
int64_t mhc_fused_post_pre_check(TensorView x, TensorView residual, TensorView post_layer_mix,
                                 TensorView comb_res_mix, TensorView fn, TensorView hc_scale,
                                 TensorView hc_base, TensorView norm_weight) {
  const int64_t operands = check_operands(x, residual, post_layer_mix, comb_res_mix, fn, hc_scale,
                                          hc_base, norm_weight, nullptr);
  if (operands != kLaunched) return operands;
  LaunchPlan plan{Variant::kUnsupported, 0, 0, 0};
  return check_plan(x.device().device_id, x.size(0), &plan);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_fused_post_pre_prepare, mhc_fused_post_pre_prepare);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_fused_post_pre_limits, mhc_fused_post_pre_limits);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_fused_post_pre_plan, mhc_fused_post_pre_plan);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_fused_post_pre_counters, mhc_fused_post_pre_counters);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_fused_post_pre_run, mhc_fused_post_pre_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_fused_post_pre_check, mhc_fused_post_pre_check);

}  // namespace mhc_fpp
}  // namespace flashinfer
