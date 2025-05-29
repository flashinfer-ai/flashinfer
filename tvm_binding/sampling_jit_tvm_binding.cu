#include "tvm_binding_utils.h"

void SamplingFromProbs(DLTensor* probs, DLTensor* output, DLTensor* maybe_indices,
                       bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
                       int64_t cuda_stream);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sampling_from_probs, SamplingFromProbs);
