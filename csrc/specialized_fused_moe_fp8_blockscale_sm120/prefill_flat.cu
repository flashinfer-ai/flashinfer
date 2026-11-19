// TVM-FFI binding/launch TU for the specialized sm120 fp8 block-scale MoE kernels.
// The framework-independent device code lives in
// include/flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/prefill_flat.cuh;
// this TU keeps the host-side launch glue and dispatch.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/fused_moe/specialized_fused_moe_fp8_blockscale_sm120/prefill_flat.cuh>

namespace flat16384 {  // reopened; the device code is in the header
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
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(status == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed with status " << static_cast<int>(status);
  return map;
}

}  // namespace

// Scratch is the caller-provided per-call workspace: the Python dispatch
// layer owns one buffer per (device, stream) for eager calls and one per
// CUDA-graph capture, so independently executing calls and independently
// captured graphs never share live scratch. Every input-dependent byte a
// kernel consumes is written on the same call (counts is memset
// stream-ordered below; the other counters are zeroed in-kernel), so
// recycled workspace memory is safe without a bulk clear.
static constexpr int kMaxDevices = 64;
// Reserved counter-region prefix shared by all paths (layout contract in
// decode_kernel.cu): this path's data begins past it.
static constexpr size_t kWorkspaceDataOffset = 2048;

size_t WorkspaceBytes(int tokens) {
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
  const int fc1_m = (tokens == 256 || tokens == 1024)
                        ? PREFILL_M48
                        : (tokens >= 8192 ? PREFILL_FC1_BIG_M : PREFILL_M);
  const int max_tiles = (routes + fc1_m - 1) / fc1_m + 256;
  const int max_tiles2 = tokens >= 8192 ? (routes + PREFILL_M - 1) / PREFILL_M + 256 : 1;
  const size_t metadata_bytes =
      256 * sizeof(int) + 257 * sizeof(int) + 256 * sizeof(int) +
      static_cast<size_t>(routes) * sizeof(int) + static_cast<size_t>(max_tiles) * 3 * sizeof(int) +
      sizeof(int) + static_cast<size_t>(max_tiles2) * 3 * sizeof(int) + sizeof(int) + 256;
  return kWorkspaceDataOffset + xq_bytes + xs_bytes + zbf_bytes + accum_out_bytes + metadata_bytes +
         static_cast<size_t>(routes) * sizeof(float) + 256;
}

void RunOurs(ffi::TensorView hidden_states, ffi::TensorView w13, ffi::TensorView w13_scale,
             ffi::TensorView w2, ffi::TensorView w2_scale, ffi::TensorView topk_ids,
             ffi::TensorView topk_weights, ffi::TensorView out, void* workspace,
             size_t workspace_bytes) {
  const int64_t T = hidden_states.size(0);
  TVM_FFI_ICHECK(T == 16384) << "RunOursFlat requires T=16384, got " << T;
  const int tokens = static_cast<int>(T);
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

  TVM_FFI_ICHECK(workspace != nullptr && workspace_bytes >= WorkspaceBytes(tokens))
      << "prefill workspace too small: " << workspace_bytes << " < " << WorkspaceBytes(tokens);
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
  const CUtensorMap w2_map = make_u8_map64(w2p, I, 256ull * H);

  const size_t accum_out_bytes = static_cast<size_t>(tokens) * H * sizeof(float);
  const int fc1_m = (tokens == 256 || tokens == 1024)
                        ? PREFILL_M48
                        : (tokens >= 8192 ? PREFILL_FC1_BIG_M : PREFILL_M);
  const int max_tiles = (routes + fc1_m - 1) / fc1_m + 256;
  const int max_tiles2 = tokens >= 8192 ? (routes + PREFILL_M - 1) / PREFILL_M + 256 : 1;
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
  int* tile2_expert = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles2) * sizeof(int);
  int* tile2_base = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles2) * sizeof(int);
  int* tile2_valid = reinterpret_cast<int*>(ptr);
  ptr += static_cast<size_t>(max_tiles2) * sizeof(int);
  int* total_tiles2 = reinterpret_cast<int*>(ptr);
  ptr += sizeof(int);
  float* rw_safe =
      reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(ptr) + 255u) & ~uintptr_t(255u));
  const float* rwp_in = rwp;

  int token_arg = tokens;
  void* prep_args[] = {
      &xp,           &idp,           &xq,          &xs,           &counts,     &starts,
      &cursors,      &sorted_routes, &tile_expert, &tile_base,    &tile_valid, &total_tiles,
      &tile2_expert, &tile2_base,    &tile2_valid, &total_tiles2, &accum_out,  &token_arg};
  if (tokens >= 8192) {
    cudaMemsetAsync(counts, 0, 256 * sizeof(int), stream);
    prepare_count_quant_kernel<<<376, 512, 0, stream>>>(xp, idp, xq, xs, counts, rwp_in, rw_safe,
                                                        tokens);
    rwp = rw_safe;
    prepare_tiles_kernel<<<1, 256, 0, stream>>>(counts, starts, cursors, tile_expert, tile_base,
                                                tile_valid, total_tiles, tile2_expert, tile2_base,
                                                tile2_valid, total_tiles2);
    prepare_scatter_kernel<<<256, 512, 0, stream>>>(idp, cursors, sorted_routes, routes);
  } else {
    const cudaError_t st =
        cudaLaunchCooperativeKernel(reinterpret_cast<void*>(prepare_prefill_kernel), dim3(188),
                                    dim3(1024), prep_args, 0, stream);
    // A failed preparation must never continue into consumers that read the
    // routing metadata it was meant to produce.
    TVM_FFI_ICHECK(st == cudaSuccess)
        << "prefill preparation launch failed: " << cudaGetErrorString(st);
  }

  // The shared-memory opt-in applies to the launching device only.
  static bool attrs_set[kMaxDevices] = {};
  if (!attrs_set[dev]) {
    cudaFuncSetAttribute(grouped_fc1_kernel<PREFILL_M>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         PREFILL_SMEM);
    cudaFuncSetAttribute(grouped_fc1_kernel<PREFILL_M48>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_FC1_M48_SMEM);
    cudaFuncSetAttribute(grouped_fc1_kernel<PREFILL_FC1_BIG_M>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_FC1_BIG_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         BF16_FC2_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         BF16_FC2_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<16, PREFILL_M48, BF16_FC2_M48_THREADS>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_M48_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<16, PREFILL_M48, BF16_FC2_M48_THREADS, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_M48_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<16, PREFILL_M, BF16_FC2_THREADS, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<8, PREFILL_M, BF16_FC2_THREADS, true, 8>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_SMEM);
    cudaFuncSetAttribute(grouped_fc2_bf16_kernel<8, PREFILL_M, BF16_FC2_THREADS, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, BF16_FC2_SMEM);
    attrs_set[dev] = true;
  }
  if (tokens >= 8192) {
    grouped_fc1_kernel<PREFILL_FC1_BIG_M>
        <<<PREFILL_BLOCKS, PREFILL_THREADS, PREFILL_FC1_BIG_SMEM, stream>>>(
            xq, xs, w13_map, w13sp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles,
            zbf, accum_out, tokens);
  } else if (tokens == 256 || tokens == 1024) {
    grouped_fc1_kernel<PREFILL_M48>
        <<<PREFILL_BLOCKS, PREFILL_THREADS, PREFILL_FC1_M48_SMEM, stream>>>(
            xq, xs, w13_map, w13sp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles,
            zbf, accum_out, tokens);
  } else {
    grouped_fc1_kernel<PREFILL_M><<<PREFILL_BLOCKS, PREFILL_THREADS, PREFILL_SMEM, stream>>>(
        xq, xs, w13_map, w13sp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles, zbf,
        accum_out, tokens);
  }
  if (tokens == 256) {
    grouped_fc2_bf16_kernel<16, PREFILL_M48, BF16_FC2_M48_THREADS>
        <<<PREFILL_BLOCKS, BF16_FC2_M48_THREADS, BF16_FC2_M48_SMEM, stream>>>(
            zbf, w2_map, w2sp, rwp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles,
            accum_out);
  } else if (tokens == 1024) {
    grouped_fc2_bf16_kernel<16, PREFILL_M48, BF16_FC2_M48_THREADS, true>
        <<<PREFILL_BLOCKS, BF16_FC2_M48_THREADS, BF16_FC2_M48_SMEM, stream>>>(
            zbf, w2_map, w2sp, rwp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles,
            accum_out);
  } else if (tokens >= 16384) {
    // Eight warp columns: each B fragment feeds two A fragments, so the
    // ldmatrix count per MMA drops from 1.25 to 1.0 on the kernel whose
    // highest-utilised memory path is L1/shared.
    grouped_fc2_bf16_kernel<8, PREFILL_M, BF16_FC2_THREADS, true, 8>
        <<<PREFILL_BLOCKS, BF16_FC2_THREADS, BF16_FC2_SMEM, stream>>>(
            zbf, w2_map, w2sp, rwp, sorted_routes, tile2_expert, tile2_base, tile2_valid,
            total_tiles2, accum_out);
  } else if (tokens >= 8192) {
    grouped_fc2_bf16_kernel<16><<<PREFILL_BLOCKS, BF16_FC2_THREADS, BF16_FC2_SMEM, stream>>>(
        zbf, w2_map, w2sp, rwp, sorted_routes, tile2_expert, tile2_base, tile2_valid, total_tiles2,
        accum_out);
  } else {
    grouped_fc2_bf16_kernel<16, PREFILL_M, BF16_FC2_THREADS, true>
        <<<PREFILL_BLOCKS, BF16_FC2_THREADS, BF16_FC2_SMEM, stream>>>(
            zbf, w2_map, w2sp, rwp, sorted_routes, tile_expert, tile_base, tile_valid, total_tiles,
            accum_out);
  }
  const int cast_blocks = (tokens * (H / 4) + 767) / 768;
  cast_output_kernel<<<cast_blocks, 768, 0, stream>>>(accum_out, outp, tokens);
  const cudaError_t tail_status = cudaGetLastError();
  TVM_FFI_ICHECK(tail_status == cudaSuccess)
      << "prefill kernel launch failed: " << cudaGetErrorString(tail_status);
}

}  // namespace flat16384

void RunOursFlat(tvm::ffi::TensorView hidden_states, tvm::ffi::TensorView w13,
                 tvm::ffi::TensorView w13_scale, tvm::ffi::TensorView w2,
                 tvm::ffi::TensorView w2_scale, tvm::ffi::TensorView topk_ids,
                 tvm::ffi::TensorView topk_weights, tvm::ffi::TensorView out, void* workspace,
                 size_t workspace_bytes) {
  flat16384::RunOurs(hidden_states, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights, out,
                     workspace, workspace_bytes);
}

size_t PrefillFlatWorkspaceBytes(int tokens) { return flat16384::WorkspaceBytes(tokens); }
