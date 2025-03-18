#include <flashinfer/sampling.cuh>
#include "batch_decode_config.inc"
#include "tvm_binding_utils.h"



// TODO: change the philox seeds and offsets to DLTensor once the underlying API for sampling changes to support multiple seeds
void SamplingFromProbs(DLTensor* probs, DLTensor* output,
    std::optional<DLTensor*> maybe_indices, bool deterministic,
    uint64_t philox_seed, uint64_t philox_offset, int64_t cuda_stream){
    
        CHECK(probs->ndim == 2) << "Probs should have 2 dimensions";
        unsigned int batch_size = output->shape[0];
        unsigned int vocab_size = probs->shape[1];
        

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
        float* probs_cast = static_cast<float*>(probs->data) + probs->byte_offset;
        float* output_cast = static_cast<int*>(output->data) + output->byte_offset;
        int* maybe_indices_cast = maybe_indices.has_value() ? static_cast<int*>(maybe_indices->data) + maybe_indices->byte_offset : nullptr;
        
        cudaError_t status = sampling::SamplingFromProb(
            probs_cast, ouptut_cast, maybe_indices_cast,
            batch_size, vocab_size, deterministic, philox_seed, philox_offset, stream);
        CHECK(status == cudaSuccess)
            << "SamplingFromProbs failed with error " << cudaGetErrorString(status);

    }
