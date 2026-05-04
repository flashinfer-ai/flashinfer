// clang-format off
#include "ssu_incremental_config.inc"
#include <flashinfer/mamba/ssu_incremental.cuh>
#include <flashinfer/mamba/kernel_ssu_incremental.cuh>
// clang-format on

namespace flashinfer::mamba::incremental {

template void launchSsuIncremental<input_t, dt_t, weight_t, matrixA_t, state_t, stateIndex_t,
                                   state_scale_t>(SsuIncrementalParams&, cudaStream_t);

}  // namespace flashinfer::mamba::incremental
