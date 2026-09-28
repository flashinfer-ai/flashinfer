/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Native launch bridge for the generated Blackwell DCP all-to-all kernels.
#ifndef FLASHINFER_CSRC_CAKE_DCP_ALLTOALL_DISPATCH_CUH_
#define FLASHINFER_CSRC_CAKE_DCP_ALLTOALL_DISPATCH_CUH_
#include <cuda_runtime.h>

#include "tensorrt_llm/kernels/helixAllToAll.h"

namespace flashinfer::comm::dcp {
// Launches the generated kernel when ``params`` names one of its exact
// validated routes (CP2/CP4, 128 x 2-byte partial outputs, two FP32
// statistics, contiguous rows, automatic channel count) and returns true.
// Returns false without launching anything otherwise, so the caller runs the
// portable helix kernel. Both kernels share the workspace layout produced by
// ``computeHelixWorkspaceSizePerRank`` and ``initializeHelixWorkspace``.
bool LaunchGeneratedDcpAllToAll(tensorrt_llm::kernels::HelixAllToAllParams const& params,
                                bool allowVariableField1, bool enablePdl, cudaStream_t stream);
}  // namespace flashinfer::comm::dcp
#endif  // FLASHINFER_CSRC_CAKE_DCP_ALLTOALL_DISPATCH_CUH_
