/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace cutlass::fmha::kernel {

struct HostPrecomputedTileScheduler {
  struct Arguments {
    int* work_indptr;
    int* qo_tile_indices;
    int* qo_head_indices;
    int* batch_indices;
  };

  struct Params {
    int* work_indptr;
    int* qo_tile_indices;
    int* qo_head_indices;
    int* batch_indices;
    int num_sm;
  };

  int work_ptr;
  int work_ptr_end;
  int qo_tile_idx;
  int batch_idx;
  int qo_head_idx;
  bool is_valid_tile;
  Params params;

  CUTLASS_DEVICE
  HostPrecomputedTileScheduler(Params const& params) {
    work_ptr = params.work_indptr[blockIdx.x];
    work_ptr_end = params.work_indptr[blockIdx.x + 1];
    if (work_ptr < work_ptr_end) {
      qo_tile_idx = params.qo_tile_indices[work_ptr];
      batch_idx = params.batch_indices[work_ptr];
      qo_head_idx = params.qo_head_indices[work_ptr];
      is_valid_tile = true;
    } else {
      is_valid_tile = false;
    }
    if (threadIdx.x % 32 == 0) {
      printf("blockIdx.x: %d, work_ptr: %d, work_ptr_end: %d, is_valid_tile: %d\n", blockIdx.x,
             work_ptr, work_ptr_end, is_valid_tile);
    }
  }

  static Params to_underlying_arguments(Arguments const& args, KernelHardwareInfo hw_info) {
    return {args.work_indptr, args.qo_tile_indices, args.qo_head_indices, args.batch_indices,
            hw_info.sm_count};
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(params.num_sm);
    return grid;
  }

  CUTLASS_DEVICE
  bool is_valid() { return is_valid_tile; }

  CUTLASS_DEVICE
  auto get_block_coord() {
    return make_coord(qo_tile_idx, _0{}, make_coord(qo_head_idx, batch_idx));
  }

  CUTLASS_DEVICE
  HostPrecomputedTileScheduler& operator++() {
    work_ptr++;
    if (work_ptr < work_ptr_end) {
      qo_tile_idx = params.qo_tile_indices[work_ptr];
      batch_idx = params.batch_indices[work_ptr];
      qo_head_idx = params.qo_head_indices[work_ptr];
      is_valid_tile = true;
    } else {
      is_valid_tile = false;
    }
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
