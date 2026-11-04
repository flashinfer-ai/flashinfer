#include <flashinfer/attention/sparse_mla_sm120/kernels/cute_prefill/v16_score_swizzle.cuh>
using namespace flashinfer::sparse_mla_sm120::cute_prefill;
using namespace flashinfer::sparse_mla_sm120::cute_prefill::v16_score_swizzle;
extern "C" int launch(const Params* p, cudaStream_t stream) {
  auto rc = cudaFuncSetAttribute(splitk_prefill, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Shared));
  if (rc != cudaSuccess) return rc;
  splitk_prefill<<<p->tokens * ((p->heads + 31) / 32), 384, sizeof(Shared), stream>>>(*p);
  return cudaGetLastError();
}
