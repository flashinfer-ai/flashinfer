/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tinygemm2_sm100_binding_common.cuh"

// The generated device TU emits its own fixed-width typedefs and an opaque
// CUtensorMap typedef so it carries no host-header dependency; isolate those
// names from the CUDA/TVM headers this binding already includes. The extern
// "C" kernel symbol is renamed per variant so the four variant TUs link into
// one module. (The heavy CUDA headers were all included by the common header
// above, so their include guards make the re-includes inside the generated
// TU no-ops under these macros.)
#define uint8_t tinygemm2_sm100_generated_uint8_t
#define uint16_t tinygemm2_sm100_generated_uint16_t
#define uint32_t tinygemm2_sm100_generated_uint32_t
#define uint64_t tinygemm2_sm100_generated_uint64_t
#define int32_t tinygemm2_sm100_generated_int32_t
#define int16_t tinygemm2_sm100_generated_int16_t
#define CUtensorMap tinygemm2_sm100_generated_CUtensorMap
#define kernel_flashinfer_tinygemm2 tinygemm2_sm100_stage8_kernel
#include "tinygemm2_sm100_stage8.cu"
#undef kernel_flashinfer_tinygemm2
#undef CUtensorMap
#undef int16_t
#undef int32_t
#undef uint64_t
#undef uint32_t
#undef uint16_t
#undef uint8_t

namespace flashinfer {
namespace tinygemm2_sm100 {

static_assert(THREADS == kThreads, "generated variant disagrees on block size");
static_assert(SMEM_TOTAL == 101504, "generated variant disagrees on dynamic smem footprint");
static_assert(USE_PDL == 0, "binding paired with the wrong PDL flavor of the generated TU");

// out = input @ weight.T + bias (bf16, fp32 accumulation), column-major
// epilogue identical to csrc/tinygemm2.cu.
void RunStage8(TensorView input, TensorView weight, TensorView bias, TensorView out) {
  const ProblemDims dims = CheckInputs(input, weight, bias, out);
  const CUtensorMap weight_map = EncodeWeightTma(weight);
  const CUtensorMap activation_map = EncodeActivationTma(input);
  const cudaStream_t stream = get_stream(input.device());
  LaunchVariant(&tinygemm2_sm100_stage8_kernel, SMEM_TOTAL, /*pdl=*/false, weight_map,
                activation_map, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()), dims, stream);
}

}  // namespace tinygemm2_sm100
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(stage8_op, flashinfer::tinygemm2_sm100::RunStage8);
