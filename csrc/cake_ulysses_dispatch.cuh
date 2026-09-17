// Native launch bridge for the generated Ulysses kernels.
#ifndef FLASHINFER_CSRC_CAKE_ULYSSES_DISPATCH_CUH_
#define FLASHINFER_CSRC_CAKE_ULYSSES_DISPATCH_CUH_
#include "flashinfer/comm/ulysses_all_to_all.cuh"
namespace flashinfer::comm::ulysses {
cudaError_t LaunchGeneratedUlysses(UlyssesA2A* fa, void* inp, int dtype, int B, int S_local,
                                   int H_local, int D, int mode, cudaStream_t stream);
}  // namespace flashinfer::comm::ulysses
#endif
