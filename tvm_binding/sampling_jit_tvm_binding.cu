#include "tvm_binding_utils.h"
#include "batch_decode_config.inc"



void SamplingFromProbs(DLTensor* probs, DLTensor* output,
    std::optional<DLTensor*> maybe_indices, bool deterministic,
    DLTensor* philox_seeds, DLTensor* philox_offsets, int64_t cuda_stream);



TVM_DLL_EXPORT_TYPED_FUNC(sampling_from_probs, SamplingFromProbs);