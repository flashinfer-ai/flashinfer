// TVM-FFI binding/launch TU for the specialized sm120 fp8 block-scale MoE kernels.
// The framework-independent device code lives in
// include/flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/prefill.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/prefill.cuh>

namespace ffi = tvm::ffi;

namespace {  // reopened; the device code is in the header

CUtensorMap make_u8_map(const uint8_t* base, uint64_t width, uint64_t height) {
  alignas(64) CUtensorMap map{};
  const cuuint64_t dims[2] = {width, height};
  const cuuint64_t strides[1] = {width};
  const cuuint32_t box[2] = {128, 128};
  const cuuint32_t elem_stride[2] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(base), dims, strides, box,
      elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed with status " << static_cast<int>(status);
  return map;
}

CUtensorMap make_u8_map64(const uint8_t* base, uint64_t width, uint64_t height) {
  alignas(64) CUtensorMap map{};
  const cuuint64_t dims[2] = {width, height};
  const cuuint64_t strides[1] = {width};
  const cuuint32_t box[2] = {64, 128};
  const cuuint32_t elem_stride[2] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(base), dims, strides, box,
      elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed with status " << static_cast<int>(status);
  return map;
}

CUtensorMap make_u8_map64x256(const uint8_t* base, uint64_t width, uint64_t height,
                              bool promote_256b) {
  alignas(64) CUtensorMap map{};
  const cuuint64_t dims[2] = {width, height};
  const cuuint64_t strides[1] = {width};
  const cuuint32_t box[2] = {64, 256};
  const cuuint32_t elem_stride[2] = {1, 1};
  const CUresult status = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<uint8_t*>(base), dims, strides, box,
      elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
      promote_256b ? CU_TENSOR_MAP_L2_PROMOTION_L2_256B : CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed with status " << static_cast<int>(status);
  return map;
}

}  // namespace

// Scratch is the caller-provided per-call workspace: the Python dispatch
// layer owns one buffer per (device, stream) for eager calls and one per
// CUDA-graph capture, so independently executing calls and independently
// captured graphs never share live scratch (the module-global per-device
// cache this replaced let two streams overwrite each other's FC1
// intermediates). Every input-dependent byte a kernel consumes is written on
// the same call (the preparation kernels zero their own counters), so
// recycled workspace memory is safe without a bulk clear.
static constexpr int kMaxDevices = 64;
// Reserved counter-region prefix shared by all paths (layout contract in
// decode_kernel.cu): this path's data begins past it.
static constexpr size_t kWorkspaceDataOffset = 2048;

size_t PrefillWorkspaceBytes(int tokens) {
  const int routes = tokens * TOPK;
  const size_t xq_bytes = static_cast<size_t>(tokens) * H;
  const size_t xs_bytes = static_cast<size_t>(tokens) * 16 * sizeof(float);
  const size_t zq_bytes = static_cast<size_t>(routes) * I;
  const size_t zs_bytes = static_cast<size_t>(routes) * 4 * sizeof(float);
  const size_t zbf_bytes = static_cast<size_t>(routes) * I * sizeof(__nv_bfloat16);
  if (tokens <= 64) {
    const size_t zf_bytes = static_cast<size_t>(routes) * I * sizeof(float);
    return kWorkspaceDataOffset + xq_bytes + xs_bytes + zf_bytes + zq_bytes + zs_bytes + 8;
  }
  const size_t accum_out_bytes = static_cast<size_t>(tokens) * H * sizeof(float);
  const bool use_m48 = tokens >= 256 && tokens <= 1024;
  const bool use_m96 = tokens >= 8192;
  const int tile_m = use_m48 ? PREFILL_M48 : PREFILL_M;
  const int max_tiles = (routes + tile_m - 1) / tile_m + 256;
  const int max_tiles96 = use_m96 ? (routes + PREFILL_M96 - 1) / PREFILL_M96 + 256 : 0;
  const size_t metadata_bytes = 256 * sizeof(int) + 257 * sizeof(int) + 256 * sizeof(int) +
                                static_cast<size_t>(routes) * sizeof(int) +
                                static_cast<size_t>(max_tiles) * 3 * sizeof(int) + 2 * sizeof(int) +
                                static_cast<size_t>(max_tiles96) * 3 * sizeof(int) +
                                (use_m96 ? sizeof(int) : 0) + 256;
  return kWorkspaceDataOffset + xq_bytes + xs_bytes + zbf_bytes + accum_out_bytes + metadata_bytes +
         static_cast<size_t>(routes) * sizeof(float) + 256;
}

void RunOurs(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
             ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
             ffi::TensorView topk_weights, ffi::TensorView out, void* workspace,
             size_t workspace_bytes) {
  const int tokens = static_cast<int>(hidden_states.size(0));
  const int routes = tokens * TOPK;
  const int dev = hidden_states.device().device_id;
  TVM_FFI_ICHECK(dev >= 0 && dev < kMaxDevices) << "device index out of range: " << dev;
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, dev));
  auto* xp = static_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
  auto* w13p = static_cast<const uint8_t*>(w13.data_ptr());
  auto* w13sp = static_cast<const float*>(w13_scale.data_ptr());
  auto* w2p = static_cast<const uint8_t*>(w2.data_ptr());
  auto* w2sp = static_cast<const float*>(w2_scale.data_ptr());
  auto* idp = static_cast<const int32_t*>(topk_ids.data_ptr());
  auto* rwp = static_cast<const float*>(topk_weights.data_ptr());
  auto* outp = static_cast<__nv_bfloat16*>(out.data_ptr());

  const size_t xq_bytes = static_cast<size_t>(tokens) * H;
  const size_t xs_bytes = static_cast<size_t>(tokens) * 16 * sizeof(float);
  const size_t zq_bytes = static_cast<size_t>(routes) * I;
  const size_t zs_bytes = static_cast<size_t>(routes) * 4 * sizeof(float);
  const size_t zbf_bytes = static_cast<size_t>(routes) * I * sizeof(__nv_bfloat16);

  TVM_FFI_ICHECK(workspace != nullptr && workspace_bytes >= PrefillWorkspaceBytes(tokens))
      << "prefill workspace too small: " << workspace_bytes << " < "
      << PrefillWorkspaceBytes(tokens);
  if (tokens <= 64) {
    const size_t zf_bytes = static_cast<size_t>(routes) * I * sizeof(float);
    uint8_t* scratch = static_cast<uint8_t*>(workspace) + kWorkspaceDataOffset;
    uint8_t* xq = scratch;
    float* xs = reinterpret_cast<float*>(xq + xq_bytes);
    float* zfloat = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(xs) + xs_bytes);
    uint8_t* zq = reinterpret_cast<uint8_t*>(zfloat) + zf_bytes;
    float* zs = reinterpret_cast<float*>(zq + zq_bytes);
    int* barriers = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(zs) + zs_bytes);
    const int blocks = tokens == 1 ? 175 : (tokens <= 4 ? 128 : (tokens <= 16 ? 160 : 175));
    const int threads = 256;
    int token_arg = tokens;
    void* args[] = {&xp, &w13p, &w13sp,  &w2p, &w2sp, &idp,      &rwp,      &outp,
                    &xq, &xs,   &zfloat, &zq,  &zs,   &barriers, &token_arg};
    const cudaError_t st = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(fused_decode_kernel), dim3(blocks), dim3(threads), args, 0, stream);
    TVM_FFI_ICHECK(st == cudaSuccess)
        << "fused decode cooperative launch failed: " << cudaGetErrorString(st);
    return;
  }

  const CUtensorMap w13_map = make_u8_map(w13p, H, 256ull * (2 * I));
  const CUtensorMap w2_map = make_u8_map64x256(w2p, I, 256ull * H, tokens >= 4096);
  void* w2_map_arg = const_cast<CUtensorMap*>(&w2_map);

  const size_t accum_out_bytes = static_cast<size_t>(tokens) * H * sizeof(float);
  const bool use_m48 = tokens >= 256 && tokens <= 1024;
  const bool use_m96 = tokens >= 8192;
  const int tile_m = use_m48 ? PREFILL_M48 : PREFILL_M;
  const int max_tiles = (routes + tile_m - 1) / tile_m + 256;
  const int max_tiles96 = use_m96 ? (routes + PREFILL_M96 - 1) / PREFILL_M96 + 256 : 0;
  uint8_t* scratch = static_cast<uint8_t*>(workspace) + kWorkspaceDataOffset;
  uint8_t* ptr = scratch;
  uint8_t* xq = ptr;
  ptr += xq_bytes;
  float* xs = reinterpret_cast<float*>(ptr);
  ptr += xs_bytes;
  __nv_bfloat16* zbf = reinterpret_cast<__nv_bfloat16*>(ptr);
  ptr += zbf_bytes;
  float* accum_out = reinterpret_cast<float*>(ptr);
  ptr += accum_out_bytes;
  uintptr_t aligned = (reinterpret_cast<uintptr_t>(ptr) + 255u) & ~uintptr_t(255u);
  ptr = reinterpret_cast<uint8_t*>(aligned);
  int* counts = reinterpret_cast<int*>(ptr);
  ptr += 256 * sizeof(int);
  int* starts = reinterpret_cast<int*>(ptr);
  ptr += 257 * sizeof(int);
  int* cursors = reinterpret_cast<int*>(ptr);
  ptr += 256 * sizeof(int);
  int* sorted_routes = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(routes) * sizeof(int);
  int* tile_expert = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles) * sizeof(int);
  int* tile_base = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles) * sizeof(int);
  int* tile_valid = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles) * sizeof(int);
  int* total_tiles = reinterpret_cast<int*>(ptr);
  ptr += sizeof(int);
  int* fc2_counter = reinterpret_cast<int*>(ptr);
  ptr += sizeof(int);
  int* tile96_expert = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles96) * sizeof(int);
  int* tile96_base = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles96) * sizeof(int);
  int* tile96_valid = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles96) * sizeof(int);
  int* total_tiles96 = use_m96 ? reinterpret_cast<int*>(ptr) : nullptr;
  ptr += use_m96 ? sizeof(int) : 0;
  float* rw_safe =
      reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(ptr) + 255u) & ~uintptr_t(255u));
  const float* rwp_in = rwp;
  if (tokens < 256) {
    int sanitize_blocks = (routes + 255) / 256;
    if (sanitize_blocks > 256) sanitize_blocks = 256;
    sanitize_routing_weights_kernel<<<sanitize_blocks, 256, 0, stream>>>(idp, rwp_in, rw_safe,
                                                                         routes);
    rwp = rw_safe;
  }

  int token_arg = tokens;
  if (tokens >= 256) {
    if (tokens == 256) {
      void* hist_fused_args[] = {&xp,         &idp,           &xq,          &xs,
                                 &counts,     &rwp_in,        &rw_safe,     &starts,
                                 &cursors,    &sorted_routes, &tile_expert, &tile_base,
                                 &tile_valid, &total_tiles,   &fc2_counter, &token_arg};
      const cudaError_t st =
          cudaLaunchCooperativeKernel(reinterpret_cast<void*>(prepare_hist_quant_m48_fused_kernel),
                                      dim3(96), dim3(1024), hist_fused_args, 0, stream);
      TVM_FFI_ICHECK(st == cudaSuccess)
          << "prefill preparation launch failed: " << cudaGetErrorString(st);
    } else {
      void* hist_args[] = {&xp, &idp, &xq, &xs, &counts, &rwp_in, &rw_safe, &token_arg};
      const cudaError_t st =
          cudaLaunchCooperativeKernel(reinterpret_cast<void*>(prepare_hist_quant_kernel), dim3(188),
                                      dim3(1024), hist_args, 0, stream);
      // A failed preparation must never continue into consumers that read the
      // routing metadata it was meant to produce.
      TVM_FFI_ICHECK(st == cudaSuccess)
          << "prefill preparation launch failed: " << cudaGetErrorString(st);
    }
    rwp = rw_safe;
    void* route_args[] = {
        &idp,       &counts,     &starts,      &cursors,     &sorted_routes,           &tile_expert,
        &tile_base, &tile_valid, &total_tiles, &fc2_counter, const_cast<int*>(&routes)};
    if (use_m96) {
      void* route96_args[] = {&idp,
                              &counts,
                              &starts,
                              &cursors,
                              &sorted_routes,
                              &tile96_expert,
                              &tile96_base,
                              &tile96_valid,
                              &total_tiles96,
                              &fc2_counter,
                              const_cast<int*>(&routes)};
      const cudaError_t st = cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(prepare_prefix_scatter_kernel<PREFILL_M96>), dim3(94), dim3(256),
          route96_args, 0, stream);
      TVM_FFI_ICHECK(st == cudaSuccess)
          << "prefill route-scatter launch failed: " << cudaGetErrorString(st);
      prepare_prefix_kernel<<<1, 256, 0, stream>>>(counts, starts, cursors, tile_expert, tile_base,
                                                   tile_valid, total_tiles, fc2_counter);
    } else if (use_m48) {
      if (tokens != 256) {
        // Eight warps per CTA scatter 256 routes. Use exactly one wave over
        // the route list instead of sending idle CTAs through the grid barrier.
        const int route_blocks = tokens / 32;
        const cudaError_t st = cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(prepare_prefix_scatter_kernel<PREFILL_M48>), dim3(route_blocks),
            dim3(256), route_args, 0, stream);
        TVM_FFI_ICHECK(st == cudaSuccess)
            << "prefill route-scatter launch failed: " << cudaGetErrorString(st);
      }
    } else {
      const cudaError_t st = cudaLaunchCooperativeKernel(
          reinterpret_cast<void*>(prepare_prefix_scatter_kernel<PREFILL_M>), dim3(94), dim3(256),
          route_args, 0, stream);
      TVM_FFI_ICHECK(st == cudaSuccess)
          << "prefill route-scatter launch failed: " << cudaGetErrorString(st);
    }
  } else {
    void* prep_args[] = {&xp,          &idp,       &xq,         &xs,
                         &counts,      &starts,    &cursors,    &sorted_routes,
                         &tile_expert, &tile_base, &tile_valid, &total_tiles,
                         &fc2_counter, &accum_out, &token_arg};
    const cudaError_t st =
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(prepare_prefill_kernel), dim3(188),
                                    dim3(1024), prep_args, 0, stream);
    TVM_FFI_ICHECK(st == cudaSuccess)
        << "prefill preparation launch failed: " << cudaGetErrorString(st);
  }

  if (use_m96) {
    // The shared-memory opt-in applies to the launching device only, so track
    // it per device (the same holds for the M48 and M64 guards below).
    static bool m96_attrs_set[kMaxDevices] = {};
    if (!m96_attrs_set[dev]) {
      cudaFuncSetAttribute(grouped_fc1_kernel<PREFILL_M96>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_M96_SMEM);
      cudaFuncSetAttribute(grouped_fc2_bf16_kernel<PREFILL_M, BF16_FC2_THREADS>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_SMEM);
      m96_attrs_set[dev] = true;
    }
    grouped_fc1_kernel<PREFILL_M96><<<PREFILL_BLOCKS, PREFILL_THREADS, PREFILL_M96_SMEM, stream>>>(
        xq, xs, w13_map, w13sp, sorted_routes, tile96_expert, tile96_base, tile96_valid,
        total_tiles96, zbf, accum_out, tokens);
    void* fc2_args[] = {&zbf,         w2_map_arg, &w2sp,       &rwp,         &sorted_routes,
                        &tile_expert, &tile_base, &tile_valid, &total_tiles, &fc2_counter,
                        &accum_out,   &outp,      &token_arg};
    const cudaError_t st = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(grouped_fc2_bf16_kernel<PREFILL_M, BF16_FC2_THREADS>),
        dim3(PREFILL_BLOCKS), dim3(BF16_FC2_THREADS), fc2_args, BF16_FC2_SMEM, stream);
    TVM_FFI_ICHECK(st == cudaSuccess) << "prefill FC2 launch failed: " << cudaGetErrorString(st);
  } else if (use_m48) {
    static bool m48_attrs_set[kMaxDevices] = {};
    if (!m48_attrs_set[dev]) {
      cudaFuncSetAttribute(grouped_fc1_kernel<PREFILL_M48>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_M48_SMEM);
      cudaFuncSetAttribute(grouped_fc2_bf16_kernel<PREFILL_M48, BF16_FC2_M48_THREADS>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_M48_SMEM);
      m48_attrs_set[dev] = true;
    }
    const int m48_blocks = tokens == 256 ? 175 : PREFILL_BLOCKS;
    grouped_fc1_kernel<PREFILL_M48><<<PREFILL_BLOCKS, PREFILL_THREADS, PREFILL_M48_SMEM, stream>>>(
        xq, xs, w13_map, w13sp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles, zbf,
        accum_out, tokens);
    void* fc2_args[] = {&zbf,         w2_map_arg, &w2sp,       &rwp,         &sorted_routes,
                        &tile_expert, &tile_base, &tile_valid, &total_tiles, &fc2_counter,
                        &accum_out,   &outp,      &token_arg};
    const cudaError_t st = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(grouped_fc2_bf16_kernel<PREFILL_M48, BF16_FC2_M48_THREADS>),
        dim3(m48_blocks), dim3(BF16_FC2_M48_THREADS), fc2_args, BF16_FC2_M48_SMEM, stream);
    TVM_FFI_ICHECK(st == cudaSuccess) << "prefill FC2 launch failed: " << cudaGetErrorString(st);
  } else {
    static bool m64_attrs_set[kMaxDevices] = {};
    if (!m64_attrs_set[dev]) {
      cudaFuncSetAttribute(grouped_fc1_kernel<PREFILL_M>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_SMEM);
      cudaFuncSetAttribute(grouped_fc2_bf16_kernel<PREFILL_M, BF16_FC2_THREADS>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_SMEM);
      m64_attrs_set[dev] = true;
    }
    grouped_fc1_kernel<PREFILL_M><<<PREFILL_BLOCKS, PREFILL_THREADS, PREFILL_SMEM, stream>>>(
        xq, xs, w13_map, w13sp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles, zbf,
        accum_out, tokens);
    void* fc2_args[] = {&zbf,         w2_map_arg, &w2sp,       &rwp,         &sorted_routes,
                        &tile_expert, &tile_base, &tile_valid, &total_tiles, &fc2_counter,
                        &accum_out,   &outp,      &token_arg};
    const cudaError_t st = cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(grouped_fc2_bf16_kernel<PREFILL_M, BF16_FC2_THREADS>),
        dim3(PREFILL_BLOCKS), dim3(BF16_FC2_THREADS), fc2_args, BF16_FC2_SMEM, stream);
    TVM_FFI_ICHECK(st == cudaSuccess) << "prefill FC2 launch failed: " << cudaGetErrorString(st);
  }
  const cudaError_t tail_status = cudaGetLastError();
  TVM_FFI_ICHECK(tail_status == cudaSuccess)
      << "prefill kernel launch failed: " << cudaGetErrorString(tail_status);
}
