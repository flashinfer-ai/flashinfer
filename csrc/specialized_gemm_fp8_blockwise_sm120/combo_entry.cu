#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/tvm_ffi.h>

namespace ffi = tvm::ffi;

void RunPeer(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
             ffi::TensorView Y);
void RunTable(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
              ffi::TensorView Y);
void RunA003(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
             ffi::TensorView Y);
void RunTuned(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
              ffi::TensorView Y);
void RunD26(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
            ffi::TensorView Y);
void RunR2Narrow(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                 ffi::TensorView Y);
void RunFX(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
           ffi::TensorView Y);
void RunFX103(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
              ffi::TensorView Y);
void RunAsyncDeep(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                  ffi::TensorView Y);
void RunXP(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
           ffi::TensorView Y);
void RunG5(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
           ffi::TensorView Y);
namespace c00_impl {
void Run(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
         ffi::TensorView Y);
}

void Run(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
         ffi::TensorView Y) {
  // Make A's device current for the whole dispatch: every Run* adapter below
  // reads its stream via TVMFFIEnvGetStream(kDLCUDA, A.device().device_id)
  // and launches on it, which requires the matching device context.
  ffi::CUDADeviceGuard device_guard(A.device().device_id);
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);

  if (K == 2048) {
    if (N == 64) {
      if (M >= 2 && M <= 8)
        RunG5(A, B, As, Bs, Y);
      else
        RunD26(A, B, As, Bs, Y);
      return;
    }
    if (N == 1024) {
      if (M <= 2)
        RunD26(A, B, As, Bs, Y);
      else if (M <= 16)
        c00_impl::Run(A, B, As, Bs, Y);
      else
        RunXP(A, B, As, Bs, Y);
      return;
    }
    if (N == 9216) {
      if (M <= 1)
        RunTable(A, B, As, Bs, Y);
      else if (M <= 4)
        RunD26(A, B, As, Bs, Y);
      else if (M <= 8)
        RunTable(A, B, As, Bs, Y);
      else
        RunA003(A, B, As, Bs, Y);
      return;
    }
    if (N == 12288) {
      if (M <= 2)
        RunD26(A, B, As, Bs, Y);
      else if (M <= 4)
        RunTuned(A, B, As, Bs, Y);
      else if (M <= 16)
        RunA003(A, B, As, Bs, Y);
      else
        RunFX(A, B, As, Bs, Y);
      return;
    }
  }
  if (N == 2048 && K == 512) {
    if (M <= 8)
      RunTable(A, B, As, Bs, Y);
    else if (M <= 16)
      c00_impl::Run(A, B, As, Bs, Y);
    else
      RunTable(A, B, As, Bs, Y);
    return;
  }
  if (N == 2048 && K == 4096) {
    if (M <= 8)
      RunR2Narrow(A, B, As, Bs, Y);
    else if (M == 16)
      RunFX103(A, B, As, Bs, Y);
    else
      RunPeer(A, B, As, Bs, Y);
    return;
  }
  RunTable(A, B, As, Bs, Y);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
