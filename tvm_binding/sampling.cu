#include <flashinfer/attention/hopper/attention_updater.cuh>
#include <flashinfer/attention/hopper/variant_helper.cuh>
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/sampling.cuh>

#include "tvm_binding_utils.h"

using namespace flashinfer;

// TODO: change the philox seeds and offsets to DLTensor once the underlying API for sampling
// changes to support multiple seeds
void SamplingFromProbs(DLTensor* probs, DLTensor* output, DLTensor* maybe_indices,
                       bool deterministic, uint64_t philox_seed, uint64_t philox_offset,
                       int64_t cuda_stream) {
  CHECK(probs->ndim == 2) << "Probs should have 2 dimensions";
  unsigned int batch_size = output->shape[0];
  unsigned int vocab_size = probs->shape[1];

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  float* probs_cast = static_cast<float*>(probs->data) + probs->byte_offset;
  int* output_cast = static_cast<int*>(output->data) + output->byte_offset;
  int* maybe_indices_cast =
      maybe_indices ? static_cast<int*>(maybe_indices->data) + maybe_indices->byte_offset : nullptr;

  cudaError_t status =
      sampling::SamplingFromProb(probs_cast, output_cast, maybe_indices_cast, batch_size,
                                 vocab_size, deterministic, philox_seed, philox_offset, stream);
  CHECK(status == cudaSuccess) << "SamplingFromProbs failed with error "
                               << cudaGetErrorString(status);
}
