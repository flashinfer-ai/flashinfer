// TVM-FFI binding for the MSA block-sparse decode native-F16 tensor-core core (exact K/V dequant in
// registers). The device code lives in
// include/flashinfer/attention/msa_sparse_decode_fp8kv_sm103/f16core.cuh;
// this TU keeps the host-side launch glue and the exported entry point.
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <flashinfer/attention/msa_sparse_decode_fp8kv_sm103/f16core.cuh>

namespace ffi = tvm::ffi;

static bool g_f16_attr[64] = {};  // per-device one-time attribute opt-in

template <int NSTAGE, int CLU, int OCC, int OSPLIT = 1>
static void launch(int B, const void* qp, const void* kp, const void* vp, const void* tkp,
                   const void* ptp, const void* slp, void* op, int pages, int bpc, int nblk,
                   cudaStream_t stream) {
  const int chmax = 65535 / OSPLIT;
  for (int base = 0; base < B;) {
    const int chunk = min(B - base, chmax);
    dim3 grid(CLU, chunk * OSPLIT);
    msaf16_kernel<NSTAGE, CLU, OCC, OSPLIT><<<grid, 256, SMEM_O(NSTAGE, CLU, OSPLIT), stream>>>(
        (const __nv_bfloat16*)qp + (size_t)base * HQ * DH, (const uint8_t*)kp, (const uint8_t*)vp,
        (const int*)tkp + (size_t)base * TOPKC, (const int*)ptp + (size_t)base * pages,
        (const int*)slp + base, (__nv_bfloat16*)op + (size_t)base * HQ * DH, pages, bpc, nblk);
    base += chunk;
  }
}

template <int NSTAGE, int CLU, int OCC, int OSPLIT = 1, bool LOAD_BALANCE = false>
static void setattr() {
  cudaFuncSetAttribute(msaf16_kernel<NSTAGE, CLU, OCC, OSPLIT>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_O(NSTAGE, CLU, OSPLIT));
  if (CLU > 8)
    cudaFuncSetAttribute(msaf16_kernel<NSTAGE, CLU, OCC, OSPLIT>,
                         cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if constexpr (LOAD_BALANCE)
    cudaFuncSetAttribute(msaf16_kernel<NSTAGE, CLU, OCC, OSPLIT>,
                         cudaFuncAttributeClusterSchedulingPolicyPreference,
                         cudaClusterSchedulingPolicyLoadBalancing);
}

void F16Run(ffi::TensorView q, ffi::TensorView kc, ffi::TensorView vc, ffi::TensorView topk,
            ffi::TensorView ptab, ffi::TensorView seqlens, ffi::TensorView out) {
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, q.device().device_id);
  const int B = (int)q.shape()[0];
  const int pages = (int)ptab.shape()[1];
  const int nblk = pages < TOPKC ? pages : TOPKC;

  int attr_di = 0;
  cudaGetDevice(&attr_di);
  if (!g_f16_attr[attr_di & 63]) {
    setattr<1, 16, 1>();
    setattr<1, 16, 1, 4>();
    setattr<1, 16, 1, 2>();
    setattr<1, 3, 1, 4, true>();
    setattr<1, 3, 1, 8, true>();
    setattr<1, 3, 1, 2, true>();
    setattr<1, 3, 2, 2, true>();
    setattr<1, 3, 1>();
    setattr<1, 3, 2>();
    setattr<1, 1, 2>();
    setattr<2, 1, 2>();
    setattr<3, 1, 2>();
    setattr<1, 2, 2>();
    setattr<2, 2, 2>();
    setattr<3, 2, 2>();
    setattr<1, 4, 2>();
    setattr<2, 4, 2>();
    setattr<3, 4, 2>();
    setattr<1, 8, 2>();
    setattr<2, 8, 2>();
    // The former <1,4,4>/<1,8,4> occupancy-4 variants had no launch site and
    // are incompatible with the P-aliases-Q layout (Q must be hoisted).
    g_f16_attr[attr_di & 63] = true;
  }

  const void* qp = q.data_ptr();
  const void* kp = kc.data_ptr();
  const void* vp = vc.data_ptr();
  const void* tkp = topk.data_ptr();
  const void* ptp = ptab.data_ptr();
  const void* slp = seqlens.data_ptr();
  void* op = out.data_ptr();

#define LAUNCH(NS, C, O, BPC)                                                     \
  do {                                                                            \
    launch<NS, C, O>(B, qp, kp, vp, tkp, ptp, slp, op, pages, BPC, nblk, stream); \
    return;                                                                       \
  } while (0)
#define LAUNCHQ(NS, C, O, BPC)                                                       \
  do {                                                                               \
    launch<NS, C, O, 4>(B, qp, kp, vp, tkp, ptp, slp, op, pages, BPC, nblk, stream); \
    return;                                                                          \
  } while (0)
#define LAUNCHE(NS, C, O, BPC)                                                       \
  do {                                                                               \
    launch<NS, C, O, 8>(B, qp, kp, vp, tkp, ptp, slp, op, pages, BPC, nblk, stream); \
    return;                                                                          \
  } while (0)
#define LAUNCHH(NS, C, O, BPC)                                                       \
  do {                                                                               \
    launch<NS, C, O, 2>(B, qp, kp, vp, tkp, ptp, slp, op, pages, BPC, nblk, stream); \
    return;                                                                          \
  } while (0)

  if (nblk == 16 && B == 1) LAUNCHQ(1, 16, 1, 1);
  if (nblk == 16 && B == 2) LAUNCHH(1, 16, 1, 1);
  if (nblk == 3) {
    if (B <= 2) LAUNCHE(1, 3, 1, 1);
    if (B <= 8) LAUNCHQ(1, 3, 1, 1);
    if (B <= 16) LAUNCHH(1, 3, 2, 1);
    if (B <= 48) LAUNCHH(1, 3, 2, 1);
    LAUNCH(1, 3, 2, 1);
  }

  int cap = 1;
  while (cap < nblk && cap < 8) cap *= 2;
  int splits = 1;
  while (splits < cap && B * splits * 2 <= 304) splits *= 2;
  int bpc = (nblk + splits - 1) / splits;
  int ns = bpc >= 3 ? 3 : bpc;

  switch (splits) {
    case 1:
      if (ns == 1) LAUNCH(1, 1, 2, bpc);
      if (ns == 2) LAUNCH(2, 1, 2, bpc);
      LAUNCH(3, 1, 2, bpc);
    case 2:
      if (ns == 1) LAUNCH(1, 2, 2, bpc);
      if (ns == 2) LAUNCH(2, 2, 2, bpc);
      LAUNCH(3, 2, 2, bpc);
    case 4:
      if (ns == 1) LAUNCH(1, 4, 2, bpc);
      LAUNCH(2, 4, 2, bpc);
    default:
      if (ns == 1) LAUNCH(1, 8, 2, bpc);
      LAUNCH(2, 8, 2, bpc);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(f16_run, F16Run);
