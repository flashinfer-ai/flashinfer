/*
 * Copyright (c) 2025 by FlashInfer team.
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
#ifndef FLASHINFER_GEMM_SCHEDULER_CUH_
#define FLASHINFER_GEMM_SCHEDULER_CUH_

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

#include "../utils.cuh"

namespace flashinfer {

struct GemmPlanInfo {
  int64_t num_ctas;

  GemmPlanInfo() : num_ctas(0) {}

  // convert GemmPlanInfo to std::vector<int64_t>
  std::vector<int64_t> ToVector() const { return {num_ctas}; }

  // From std::vector<int64_t> to GemmPlanInfo
  void FromVector(const std::vector<int64_t>& vec) {
    if (vec.size() != 1) {
      std::ostringstream err_msg;
      err_msg << "GemmPlanInfo::FromVector: vec.size() should be 1, but got " << vec.size();
      FLASHINFER_ERROR(err_msg.str());
    }
    num_ctas = vec[0];
  }
};

inline cudaError_t GemmPlan(uint32_t num_ctas, GemmPlanInfo& plan_info) {
  int dev_id = 0;
  int num_sms = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  if (num_ctas > 0 && num_ctas < num_sms) {
    plan_info.num_ctas = num_ctas;
  } else {
    plan_info.num_ctas = num_sms;
  }
  return cudaSuccess;
}

}  // namespace flashinfer
#endif  // FLASHINFER_GEMM_SCHEDULER_CUH_
