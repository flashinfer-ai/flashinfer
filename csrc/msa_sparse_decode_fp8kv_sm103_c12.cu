// TVM-FFI binding for the MSA block-sparse decode fused C12 core (bf16 Q, f16 P; split-K merge in
// kernel). The device code lives in
// include/flashinfer/attention/msa_sparse_decode_fp8kv_sm103/c12.cuh;
// this TU keeps the host-side launch glue and the exported entry point.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/attention/msa_sparse_decode_fp8kv_sm103/c12.cuh>

namespace ffi = tvm::ffi;

namespace msa_sm103 {
void EnsureAttributes(const void* kernel, int smem, bool nonportable, bool load_balance);
}

template <int NSTAGE, int CLU, bool HALFOUT = false, bool QUARTER = false, bool HALFPAGE = false,
          int CTA_THREADS = 256, bool BF16_Q_F16_P = false>
static cudaError_t launch(int B, const void* qp, const void* kp, const void* vp, const void* tkp,
                          const void* ptp, const void* slp, void* op, int pages, int bpc, int nblk,
                          cudaStream_t stream) {
  dim3 grid(CLU, QUARTER ? B * 4 : (HALFOUT ? B * 2 : B));
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(CTA_THREADS);
  config.dynamicSmemBytes = SMEM_VARIANT(NSTAGE, CLU, HALFPAGE, BF16_Q_F16_P);
  msa_sm103::EnsureAttributes(
      reinterpret_cast<const void*>(
          msa_kernel<NSTAGE, CLU, HALFOUT, QUARTER, HALFPAGE, CTA_THREADS, BF16_Q_F16_P>),
      config.dynamicSmemBytes, CLU > 8, false);
  config.stream = stream;
  cudaLaunchAttribute scheduling{};
  if constexpr (COMPACT_FALLBACK(NSTAGE, CLU, HALFPAGE, BF16_Q_F16_P)) {
    scheduling.id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    scheduling.val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = &scheduling;
    config.numAttrs = 1;
  }
  if constexpr (BF16_Q_F16_P) {
    cudaStreamCaptureStatus capture;
    cudaError_t status = cudaStreamIsCapturing(stream, &capture);
    if (status != cudaSuccess) return status;
    if (capture != cudaStreamCaptureStatusNone) {
      // Capture can defer launch validation until graph instantiation.
      // Check residency before recording a node that cannot be launched.
      int active = 0;
      status = cudaOccupancyMaxActiveClusters(
          &active, msa_kernel<NSTAGE, CLU, HALFOUT, QUARTER, HALFPAGE, CTA_THREADS, BF16_Q_F16_P>,
          &config);
      if (status != cudaSuccess) return status;
      if (active == 0) return cudaErrorInvalidClusterSize;
    }
  }
  return cudaLaunchKernelEx(
      &config, msa_kernel<NSTAGE, CLU, HALFOUT, QUARTER, HALFPAGE, CTA_THREADS, BF16_Q_F16_P>,
      (const __nv_bfloat16*)qp, (const uint8_t*)kp, (const uint8_t*)vp, (const int*)tkp,
      (const int*)ptp, (const int*)slp, (__nv_bfloat16*)op, pages, bpc, nblk);
}

void C12Run(ffi::TensorView q, ffi::TensorView kc, ffi::TensorView vc, ffi::TensorView topk,
            ffi::TensorView ptab, ffi::TensorView seqlens, ffi::TensorView out) {
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, q.device().device_id);
  const int B = (int)q.shape()[0];
  const int pages = (int)ptab.shape()[1];
  const int nblk = pages < TOPKC ? pages : TOPKC;

  int cap = 1;
  while (cap < nblk && cap < 8) cap *= 2;
  int splits = 1;
  while (splits < cap && B * splits * 2 <= 304) splits *= 2;
  int bpc = (nblk + splits - 1) / splits;

  const void* qp = q.data_ptr();
  const void* kp = kc.data_ptr();
  const void* vp = vc.data_ptr();
  const void* tkp = topk.data_ptr();
  const void* ptp = ptab.data_ptr();
  const void* slp = seqlens.data_ptr();
  void* op = out.data_ptr();
  if (nblk == 16 && B == 1)
    launch<1, 16, false, true>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (nblk == 3 && B <= 2) {
    const cudaError_t status = launch<1, 12, true, false, true, 128, true>(
        B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
    if (status == cudaErrorInvalidClusterSize) {
      // Some GB300 green contexts cannot place this non-portable cluster.
      // A rejected launch has not touched out. Use this core's existing
      // full-page/three-rank geometry, retaining bf16 Q on the same stream.
      (void)cudaGetLastError();
      const cudaError_t fallback_status = launch<1, 3, true, false, false, 256, true>(
          B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
      if (fallback_status != cudaSuccess)
        TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(fallback_status);
    } else if (status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << cudaGetErrorString(status);
    }
  } else if (nblk == 3 && B <= 16)
    launch<1, 4, true>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (nblk == 3 && B <= 32)
    launch<1, 4>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (nblk == 3)
    launch<1, 3>(B, qp, kp, vp, tkp, ptp, slp, op, pages, 1, nblk, stream);
  else if (splits == 1)
    launch<4, 1>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
  else if (splits == 2)
    launch<2, 2>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
  else if (splits == 4)
    launch<2, 4>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
  else
    launch<2, 8>(B, qp, kp, vp, tkp, ptp, slp, op, pages, bpc, nblk, stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(c12_split_run, C12Run);
