// TVM-FFI binding for the MSA block-sparse decode direct-K short-sequence core.
// The device code lives in
// include/flashinfer/attention/msa_sparse_decode_fp8kv_sm103/dkf16_short.cuh;
// this TU keeps the host-side launch glue and the exported entry point.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/attention/msa_sparse_decode_fp8kv_sm103/dkf16_short.cuh>

namespace ffi = tvm::ffi;

namespace msa_sm103 {
void EnsureAttributes(const void* kernel, int smem, bool nonportable, bool load_balance);
}

namespace dkf16short {

template <int NSTAGE, int CLU, int OCC, bool HALFOUT = false>
static void launchf(int B, const void* qp, const void* kp, const void* vp, const void* tkp,
                    const void* ptp, const void* slp, void* op, int pages, int bpc, int nblk,
                    cudaStream_t stream) {
  msa_sm103::EnsureAttributes(reinterpret_cast<const void*>(dkf_kernel<NSTAGE, CLU, OCC, HALFOUT>),
                              SMEM_F(NSTAGE, CLU), CLU > 8, CLU > 1);
  for (int base = 0; base < B;) {
    const int chunk = min(B - base, 65535);
    dim3 grid(CLU, HALFOUT ? chunk * 2 : chunk);
    dkf_kernel<NSTAGE, CLU, OCC, HALFOUT><<<grid, 256, SMEM_F(NSTAGE, CLU), stream>>>(
        (const __nv_bfloat16*)qp + (size_t)base * HQ * DH, (const uint8_t*)kp, (const uint8_t*)vp,
        (const int*)tkp + (size_t)base * TOPKC, (const int*)ptp + (size_t)base * pages,
        (const int*)slp + base, (__nv_bfloat16*)op + (size_t)base * HQ * DH, pages, bpc, nblk);
    // A rejected launch (e.g. a cluster geometry a reduced-resource context
    // cannot place) must never return success with an unwritten `out`.
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
      TVM_FFI_THROW(RuntimeError) << "specialized msa decode launch failed: "
                                  << cudaGetErrorString(launch_status);
    }
    base += chunk;
  }
}

}  // namespace dkf16short

void DirectKShortRun(ffi::TensorView q, ffi::TensorView kc, ffi::TensorView vc,
                     ffi::TensorView topk, ffi::TensorView ptab, ffi::TensorView seqlens,
                     ffi::TensorView out) {
  using namespace dkf16short;
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, q.device().device_id);
  const int B = (int)q.shape()[0];
  const int pages = (int)ptab.shape()[1];
  const int nblk = pages < TOPKC ? pages : TOPKC;
  const void* qp = q.data_ptr();
  const void* kp = kc.data_ptr();
  const void* vp = vc.data_ptr();
  const void* tkp = topk.data_ptr();
  const void* ptp = ptab.data_ptr();
  const void* slp = seqlens.data_ptr();
  void* op = out.data_ptr();
#define LF(NS, C, O, BPC)                                                          \
  do {                                                                             \
    launchf<NS, C, O>(B, qp, kp, vp, tkp, ptp, slp, op, pages, BPC, nblk, stream); \
    return;                                                                        \
  } while (0)
#define LFH(NS, C, O, BPC)                                                               \
  do {                                                                                   \
    launchf<NS, C, O, true>(B, qp, kp, vp, tkp, ptp, slp, op, pages, BPC, nblk, stream); \
    return;                                                                              \
  } while (0)
  if (nblk <= 3) {
    // Match the staged B16/B48 routes' CTA count while keeping K out of
    // shared memory: each CTA owns one 64-dimension output half.
    if (B == 16 || B == 48) LFH(1, 3, 2, 1);
    LF(1, 3, 2, 1);
  }
  if (nblk == TOPKC && B <= 8) LF(1, 16, 2, 1);
  if (nblk == TOPKC && B == 16) LFH(1, 8, 2, 2);
  if (nblk == TOPKC && B == 32) LF(1, 8, 2, 2);
  if (nblk == TOPKC && (B == 48 || B == 64)) LF(1, 4, 2, 4);
  int cap = 1;
  while (cap < nblk && cap < 8) cap *= 2;
  int splits = 1;
  while (splits < cap && B * splits * 2 <= 304) splits *= 2;
  const int bpc = (nblk + splits - 1) / splits;
  const int ns = bpc >= 2 ? 2 : 1;
  switch (splits) {
    case 8:
      if (ns == 1) LF(1, 8, 2, 1);
      LF(2, 8, 2, bpc);
    case 4:
      if (ns == 1) LF(1, 4, 2, 1);
      LF(2, 4, 2, bpc);
    case 2:
      if (ns == 1) LF(1, 2, 2, 1);
      LF(2, 2, 2, bpc);
    default:
      if (ns == 1) LF(1, 1, 2, 1);
      LF(2, 1, 2, bpc);
  }
#undef LF
#undef LFH
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(direct_k_short_run, DirectKShortRun);
