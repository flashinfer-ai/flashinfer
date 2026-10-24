// Experimental local flat-bin decode binding. Apache-2.0.
#include <flashinfer/attention/hpc_flat_bin.cuh>

#include "tvm_ffi_utils.h"
using namespace flashinfer;

void HPCFlatAssign(TensorView indptr, TensorView tail, TensorView lengths, TensorView state_ptr,
                   TensorView cta_ptr, TensorView tasks, TensorView requests, TensorView meta,
                   TensorView raw, int64_t heads, int64_t min_tiles, int64_t capacity,
                   int64_t phase) {
  ffi::CUDADeviceGuard guard(indptr.device().device_id);
  auto stream = get_stream(indptr.device());
  int B = tail.size(0), H = heads, P = cta_ptr.size(0) - 1;
  int* rp = static_cast<int32_t*>(raw.data_ptr());
  int* lp = static_cast<int32_t*>(lengths.data_ptr());
  // Original kernel has cross-CTA waits: all planner CTAs must fit concurrently.
  // Checked at allocation time by Python via max_planner_ctas(), not timed here.
  if (phase == 0 || phase == 1)
    HPCPrepare<<<(std::max(B, ((P + 11) / 12) * 12) + 127) / 128, 128, 0, stream>>>(
        static_cast<int32_t*>(indptr.data_ptr()), static_cast<int32_t*>(tail.data_ptr()), B, H, P,
        capacity, lp, rp);
  if (phase == 0 || phase == 2)
    hpc_original::assign_attention_decode_task_kernel<2048, 128>
        <<<P, 128, 0, stream>>>(rp, lp, B, H, 1, true, min_tiles * 128, P, P);
  if (phase == 0 || phase == 3) {
    HPCScan<<<1, 128, 0, stream>>>(rp, B, H, P, static_cast<int32_t*>(state_ptr.data_ptr()),
                                   static_cast<int32_t*>(cta_ptr.data_ptr()),
                                   static_cast<int32_t*>(meta.data_ptr()));
    HPCAdapt<<<P, 128, 0, stream>>>(
        rp, static_cast<int32_t*>(state_ptr.data_ptr()), static_cast<int32_t*>(cta_ptr.data_ptr()),
        H, static_cast<int32_t*>(tasks.data_ptr()), static_cast<int32_t*>(requests.data_ptr()));
  }
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess) << "flat-bin assign launch failed";
}

int64_t HPCMaxPlannerCTAs(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  int per_sm, sms;
  TVM_FFI_ICHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                     &per_sm, hpc_original::assign_attention_decode_task_kernel<2048, 128>, 128,
                     0) == cudaSuccess);
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) ==
                 cudaSuccess);
  return per_sm * sms;
}

void HPCFlatRun(TensorView q, TensorView k, TensorView v, TensorView indptr, TensorView indices,
                TensorView tail, TensorView state_ptr, TensorView cta_ptr, TensorView tasks,
                TensorView requests, TensorView dummy_tiles, TensorView meta, TensorView tmp_v,
                TensorView tmp_s, TensorView out, TensorView lse, double scale) {
  ffi::CUDADeviceGuard guard(q.device().device_id);
  auto stream = get_stream(q.device());
  using T = __nv_bfloat16;
  using Params = BatchDecodeParams<T, T, T, int32_t>;
  Params params;
  auto ks = k.strides(), vs = v.strides();
  params.paged_kv = paged_kv_t<T, int32_t>(
      k.size(2), 64, 128, q.size(0), QKVLayout::kNHD, static_cast<T*>(k.data_ptr()),
      static_cast<T*>(v.data_ptr()), ks.data(), vs.data(),
      static_cast<int32_t*>(indices.data_ptr()), static_cast<int32_t*>(indptr.data_ptr()),
      static_cast<int32_t*>(tail.data_ptr()));
  params.q = static_cast<T*>(q.data_ptr());
  params.o = static_cast<T*>(tmp_v.data_ptr());
  params.lse = static_cast<float*>(tmp_s.data_ptr());
  params.num_qo_heads = q.size(1);
  params.q_stride_n = q.stride(0);
  params.q_stride_h = q.stride(1);
  params.window_left = -1;
  params.sm_scale = scale;
  params.request_indices = static_cast<int32_t*>(requests.data_ptr());
  params.kv_tile_indices = static_cast<int32_t*>(dummy_tiles.data_ptr());
  params.kv_chunk_size_ptr = static_cast<int32_t*>(meta.data_ptr());
  params.partition_kv = true;
  auto capability = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(capability, STAGES, {
    constexpr int smem =
        std::max(2 * STAGES * 4 * 2 * 128 * sizeof(T), 2 * 4 * 128 * sizeof(float)) +
        std::max(128 * sizeof(T*), 2 * 4 * 2 * sizeof(float));
    auto kernel = HPCFlatBinDecode<STAGES, Params>;
    TVM_FFI_ICHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem) == cudaSuccess);
    dim3 grid(cta_ptr.size(0) - 1), threads(16, 4, 2);
    const int32_t* cp = static_cast<int32_t*>(cta_ptr.data_ptr());
    const int32_t* tp = static_cast<int32_t*>(tasks.data_ptr());
    void* all_args[] = {&params, &cp, &tp};
    TVM_FFI_ICHECK(cudaLaunchKernel((void*)kernel, grid, threads, all_args, smem, stream) ==
                   cudaSuccess)
        << "flat-bin decode launch failed";
  });
  auto status = VariableLengthMergeStates(
      static_cast<T*>(tmp_v.data_ptr()), static_cast<float*>(tmp_s.data_ptr()),
      static_cast<int32_t*>(state_ptr.data_ptr()), static_cast<T*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()), q.size(0) * k.size(2), nullptr, 4, 128, false, stream);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(assign, HPCFlatAssign);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_planner_ctas, HPCMaxPlannerCTAs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, HPCFlatRun);
