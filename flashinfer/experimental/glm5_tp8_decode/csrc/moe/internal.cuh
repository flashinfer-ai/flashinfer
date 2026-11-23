/* Copyright (c) 2026 by FlashInfer team. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "../moe_api.h"

namespace flashinfer::glm5_tp8::moe_detail {

const void* q1_kernel();
const void* q4_kernel();
cudaError_t launch_q1(const MoeArguments&, cudaStream_t, bool, bool);
cudaError_t launch_q4(const MoeArguments&, cudaStream_t, bool, bool);

inline cudaError_t launch(const void* kernel, void* arguments, int block, int dynamic_shared_bytes,
                          cudaStream_t stream, bool enable_pdl, bool cooperative) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(148);
  config.blockDim = dim3(block);
  config.dynamicSmemBytes = dynamic_shared_bytes;
  config.stream = stream;
  cudaLaunchAttribute attributes[2]{};
  unsigned count = 0;
  if (enable_pdl) {
    attributes[count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[count++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (cooperative) {
    attributes[count].id = cudaLaunchAttributeCooperative;
    attributes[count++].val.cooperative = 1;
  }
  config.attrs = attributes;
  config.numAttrs = count;
  void* params[] = {arguments};
  return cudaLaunchKernelExC(&config, kernel, params);
}

}  // namespace flashinfer::glm5_tp8::moe_detail
