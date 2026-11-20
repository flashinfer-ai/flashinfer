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

// Cake W8A8 MXFP8 MegaMoE EP16 (SM103), setup stage `quantize_weights`.
//
// Hand-written tvm-ffi launcher for the generated kernel translation unit of
// the same stage (cake_w8a8_megamoe_ep16_quantize_weights_kernel.cu). The
// program quantizes a bf16 [experts, rows, columns] weight to e4m3 with
// block-32 scales in the packed layout selected by `scale_tile_n256_k128`
// (1: 256x128 tiles for w13, 2: 128x128 tiles for the swapped w2), optionally
// interleaving gate and up rows. The trailing `ktile_major_rows` scalar selects
// the FP8 payload placement: 0 writes the row-major payload; R = 256 (w13) or
// R = 128 (w2) writes the k-tile-major payload [experts][rows / R][columns /
// 128][R][128 B] that the FC1 / FC2 A tensor maps address (manifest
// route_contract.weight_layout_rule). The scale carriers are unaffected. The
// binding checks the tensor metadata and the index bounds of that contract,
// resolves the environment stream and performs one launch with the generated
// program's block size.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

// Device entry point of the kernel translation unit (manifest declaration).
// clang-format off
extern "C" __global__ void kernel_cake_w8a8_megamoe_ep16_quantize_weights(__nv_bfloat16* __restrict__ source, uint8_t* __restrict__ quantized, uint8_t* __restrict__ scales_cp, int experts, int rows, int columns, int gate_width, int interleave_gate_up, int scale_tile_n256_k128, int ktile_major_rows);
// clang-format on

namespace cake_w8a8_megamoe_ep16_quantize_weights {

constexpr int64_t kScaleBlock = 32;     // elements per MXFP8 scale byte
constexpr int64_t kW13KtileRows = 256;  // FC1 A box height: the interleaved w13 payload
constexpr int64_t kW2KtileRows = 128;   // FC2 A box height: the swapped w2 payload
constexpr unsigned kBlockThreads = 256;

void Run(TensorView source, TensorView quantized, TensorView scales_cp, int64_t experts,
         int64_t rows, int64_t columns, int64_t gate_width, int64_t interleave_gate_up,
         int64_t scale_tile_n256_k128, int64_t ktile_major_rows, int64_t grid_x) {
  check_cuda_tensor(source, "source");
  const DLDevice dev = source.device();
  ffi::CUDADeviceGuard device_guard(dev.device_id);

  check_dtype(source, dl_bfloat16, "source");
  check_contiguous(source, "source");
  check_cuda_tensor(quantized, "quantized");
  const DLDataType quantized_dtype = quantized.dtype();
  TVM_FFI_CHECK(
      (quantized_dtype.code == dl_float8_e4m3fn.code || quantized_dtype.code == dl_uint8.code) &&
          quantized_dtype.bits == 8 && quantized_dtype.lanes == 1,
      TypeError)
      << "quantized must be float8_e4m3fn or uint8, got dtype code="
      << static_cast<int>(quantized_dtype.code)
      << " bits=" << static_cast<int>(quantized_dtype.bits)
      << " lanes=" << static_cast<int>(quantized_dtype.lanes);
  check_contiguous(quantized, "quantized");
  check_same_device(quantized, source, "quantized", "source");
  check_cuda_tensor(scales_cp, "scales_cp");
  check_dtype(scales_cp, dl_uint8, "scales_cp");
  check_contiguous(scales_cp, "scales_cp");
  check_same_device(scales_cp, source, "scales_cp", "source");

  TVM_FFI_CHECK(experts >= 1 && experts <= 2147483647LL, ValueError)
      << "experts must be in [1, 2147483647], got " << experts;
  TVM_FFI_CHECK(rows >= 1 && rows <= 2147483647LL, ValueError)
      << "rows must be in [1, 2147483647], got " << rows;
  TVM_FFI_CHECK(columns >= kScaleBlock && columns <= 2147483647LL && columns % kScaleBlock == 0,
                ValueError)
      << "columns must be a positive multiple of " << kScaleBlock << " within i32, got " << columns;
  TVM_FFI_CHECK(interleave_gate_up == 0 || interleave_gate_up == 1, ValueError)
      << "interleave_gate_up must be 0 or 1, got " << interleave_gate_up;
  TVM_FFI_CHECK(scale_tile_n256_k128 == 1 || scale_tile_n256_k128 == 2, ValueError)
      << "scale_tile_n256_k128 must be 1 (256x128 tiles) or 2 (128x128 tiles), got "
      << scale_tile_n256_k128;
  // The scale plane is addressed tile by tile; rows and columns must tile exactly.
  const int64_t tile_rows = scale_tile_n256_k128 == 1 ? 256 : 128;
  TVM_FFI_CHECK(rows % tile_rows == 0 && columns % 128 == 0, ValueError)
      << "rows must be a multiple of " << tile_rows << " and columns a multiple of 128 for "
      << "scale_tile_n256_k128=" << scale_tile_n256_k128 << ", got (" << rows << ", " << columns
      << ")";
  // Interleaving reads the second (up) half, rows [gate_width, gate_width + rows / 2),
  // into the first 128 rows of each 256-row output chunk (up128 | gate128).
  TVM_FFI_CHECK(gate_width >= 0 && gate_width <= rows, ValueError)
      << "gate_width must be in [0, rows], got " << gate_width;
  TVM_FFI_CHECK(interleave_gate_up == 0 || 2 * gate_width <= rows, ValueError)
      << "interleaving needs gate_width + rows / 2 <= rows, got gate_width=" << gate_width
      << " rows=" << rows;
  // FP8 payload placement. The exported FC1 reads the interleaved (w13) payload
  // in 256-row boxes and the exported FC2 reads the swapped (mode 2, w2) payload
  // in 128-row boxes; 0 is the row-major payload, which neither program reads.
  TVM_FFI_CHECK(ktile_major_rows == 0 || ktile_major_rows == kW2KtileRows ||
                    ktile_major_rows == kW13KtileRows,
                ValueError)
      << "ktile_major_rows must be 0 (row-major payload), " << kW2KtileRows << " or "
      << kW13KtileRows << " (k-tile-major payload, box height), got " << ktile_major_rows;
  TVM_FFI_CHECK(interleave_gate_up == 0 || ktile_major_rows != kW2KtileRows, ValueError)
      << "the interleaved gate|up payload is addressed in " << kW13KtileRows
      << "-row boxes: ktile_major_rows must be " << kW13KtileRows << " (or 0), got "
      << ktile_major_rows;
  TVM_FFI_CHECK(scale_tile_n256_k128 != 2 || ktile_major_rows != kW13KtileRows, ValueError)
      << "the swapped (scale_tile_n256_k128=2) payload is addressed in " << kW2KtileRows
      << "-row boxes: ktile_major_rows must be " << kW2KtileRows << " (or 0), got "
      << ktile_major_rows;
  TVM_FFI_CHECK(ktile_major_rows == 0 || rows % ktile_major_rows == 0, ValueError)
      << "rows must be a multiple of ktile_major_rows=" << ktile_major_rows << ", got " << rows;
  TVM_FFI_CHECK(source.ndim() == 3, ValueError) << "source must be 3-D, got ndim=" << source.ndim();
  TVM_FFI_CHECK(source.size(0) == experts && source.size(1) == rows && source.size(2) == columns,
                ValueError)
      << "source must be [" << experts << ", " << rows << ", " << columns << "], got ["
      << source.size(0) << ", " << source.size(1) << ", " << source.size(2) << "]";
  const int64_t elements = experts * rows * columns;
  TVM_FFI_CHECK(elements / kScaleBlock <= 2147483647LL, ValueError)
      << "experts * rows * columns / " << kScaleBlock << " must fit in i32, got "
      << elements / kScaleBlock;
  TVM_FFI_CHECK(quantized.numel() == elements, ValueError)
      << "quantized must hold " << elements << " elements, got " << quantized.numel();
  TVM_FFI_CHECK(scales_cp.numel() == elements / kScaleBlock, ValueError)
      << "scales_cp must hold " << elements / kScaleBlock << " bytes, got " << scales_cp.numel();
  TVM_FFI_CHECK(grid_x > 0 && grid_x <= 2147483647LL, ValueError)
      << "grid_x must be in [1, 2147483647], got " << grid_x;

  const cudaStream_t stream = get_stream(dev);

  __nv_bfloat16* p_source = static_cast<__nv_bfloat16*>(source.data_ptr());
  uint8_t* p_quantized = static_cast<uint8_t*>(quantized.data_ptr());
  uint8_t* p_scales_cp = static_cast<uint8_t*>(scales_cp.data_ptr());
  int32_t v_experts = static_cast<int32_t>(experts);
  int32_t v_rows = static_cast<int32_t>(rows);
  int32_t v_columns = static_cast<int32_t>(columns);
  int32_t v_gate_width = static_cast<int32_t>(gate_width);
  int32_t v_interleave_gate_up = static_cast<int32_t>(interleave_gate_up);
  int32_t v_scale_tile_n256_k128 = static_cast<int32_t>(scale_tile_n256_k128);
  int32_t v_ktile_major_rows = static_cast<int32_t>(ktile_major_rows);
  void* kargs[] = {&p_source,
                   &p_quantized,
                   &p_scales_cp,
                   &v_experts,
                   &v_rows,
                   &v_columns,
                   &v_gate_width,
                   &v_interleave_gate_up,
                   &v_scale_tile_n256_k128,
                   &v_ktile_major_rows};

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), 1, 1);
  config.blockDim = dim3(kBlockThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  const cudaError_t status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(kernel_cake_w8a8_megamoe_ep16_quantize_weights),
      kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC(kernel_cake_w8a8_megamoe_ep16_quantize_weights) failed: "
      << cudaGetErrorString(status);
}

}  // namespace cake_w8a8_megamoe_ep16_quantize_weights

TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_weights, cake_w8a8_megamoe_ep16_quantize_weights::Run);
