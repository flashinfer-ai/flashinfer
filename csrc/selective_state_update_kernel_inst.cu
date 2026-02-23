// clang-format off
// config.inc MUST come before the header: it defines DIM, DSTATE, NTOKENS_MTP
// constexprs that the header's function templates rely on. Reordering breaks compilation.
#include "selective_state_update_config.inc"
#include <flashinfer/mamba/selective_state_update.cuh>
// clang-format on

namespace flashinfer::mamba {

template void invokeSelectiveStateUpdate<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(
    SelectiveStateUpdateParams&, SSUAlgorithm, cudaStream_t);

namespace mtp {
template void invokeSelectiveStateUpdateMTP<input_t, weight_t, matrixA_t, state_t, stateIndex_t>(
    SelectiveStateMTPParams&, SSUAlgorithm, cudaStream_t);
}  // namespace mtp

}  // namespace flashinfer::mamba
