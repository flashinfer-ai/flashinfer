/*
 * Copyright (c) 2026 by FlashInfer team.
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

// clang-format off
// Include order matters: headers → config (defines Ktraits) → kernel (uses Ktraits)
#include <algorithm>
#include <flashinfer/norm/ln_silu_headers.cuh>
#include "rmsnorm_silu_config.inc"
#include <flashinfer/norm/ln_fwd_silu_kernel.cuh>
// clang-format on

#include "tvm_ffi_utils.h"

void rmsnorm_silu(TensorView output, TensorView input, TensorView weight, double eps,
                  TensorView workspace, TensorView scale_row_out, int64_t sm_count) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight);
  CHECK_DEVICE(input, weight);
  CHECK_DIM(2, input);
  CHECK_DIM(2, output);
  CHECK_DIM(1, weight);

  int rows = input.size(0);
  int cols = input.size(1);
  TVM_FFI_ICHECK_EQ(cols, HIDDEN_SIZE) << "Input cols must match compiled HIDDEN_SIZE";
  TVM_FFI_ICHECK_EQ(output.size(0), rows);

  ffi::CUDADeviceGuard device_guard(input.device().device_id);
  const cudaStream_t stream = get_stream(input.device());

  // Grid dimensions (same logic as Sm100RmsNormSiluEngine::execute)
  int ctas_per_col_max = (rows + WARPS_M - 1) / WARPS_M;
  int ctas_per_col;
  if (KERNEL_CFG == 2) {
    ctas_per_col = ctas_per_col_max;
  } else {
    ctas_per_col =
        std::min(static_cast<int>(sm_count) * DESIRED_OCCUPANCY / CTAS_PER_ROW, ctas_per_col_max);
  }
  ctas_per_col = std::max(ctas_per_col, 1);

  dim3 grid(CTAS_PER_ROW * ctas_per_col);
  dim3 block(WARPS_M * WARPS_N * 32);

  // Pack kernel params
  PersistentLnFwdParams params{};
  params.rows = rows;
  params.cols = cols;
  params.ctas_per_col = ctas_per_col;
  params.isRMSNorm = true;
  params.noScale = false;
  params.noBias = true;
  params.isBatchFirst = true;
  params.batchSize = 1;
  params.seqLen = rows;
  params.epsilon = static_cast<float>(eps);
  params.x = input.data_ptr();
  params.z = output.data_ptr();
  params.gamma = weight.data_ptr();

  // Workspace layout (128-byte aligned segments)
  char* ws_ptr = static_cast<char*>(workspace.data_ptr());

  // [0] rs: rows * sizeof(float)
  params.rs = ws_ptr;
  int64_t off = static_cast<int64_t>(rows) * sizeof(float);
  off = ((off + 127) / 128) * 128;

  // [aligned] fp8_scale: sizeof(float)
  if (isFP8Out) {
    params.fp8_out = true;
    float* default_scale = reinterpret_cast<float*>(ws_ptr + off);
    // Set scale = 1.0f via cudaMemcpyAsync from host
    static const float one = 1.0f;
    cudaMemcpyAsync(default_scale, &one, sizeof(float), cudaMemcpyHostToDevice, stream);
    params.scale = default_scale;
  }
  off += sizeof(float);
  off = ((off + 127) / 128) * 128;

  // scale_row: passed as separate output tensor (NVFP4 only)
  if (isFP4Out) {
    params.scale_row = scale_row_out.data_ptr();
  }

  // [aligned] cooperative workspace + barriers (multi-CTA only)
  if (CTAS_PER_ROW > 1) {
    params.workspace = ws_ptr + off;
    int64_t coop_ws_size =
        static_cast<int64_t>(ctas_per_col) * WARPS_M * CTAS_PER_ROW * sizeof(float) * 2 * 2;
    off += coop_ws_size;
    off = ((off + 127) / 128) * 128;

    params.barrier = reinterpret_cast<int*>(ws_ptr + off);
    cudaMemsetAsync(params.barrier, 0, 2 * ctas_per_col * sizeof(int32_t), stream);
  }

  reduced_divisor divisor(rows);

  ln_fwd_kernel<<<grid, block, 0, stream>>>(params, divisor);
}
