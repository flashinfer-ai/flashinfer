#include "tvm_ffi_utils.h"

namespace flashinfer::mamba {
void replayssm_materialize(TensorView, TensorView, TensorView, TensorView, TensorView, TensorView,
                           TensorView, TensorView, TensorView, TensorView, TensorView, TensorView,
                           TensorView, TensorView, TensorView, int64_t, int64_t, int64_t,
                           tvm::ffi::Optional<TensorView>);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(replayssm_materialize, flashinfer::mamba::replayssm_materialize);
