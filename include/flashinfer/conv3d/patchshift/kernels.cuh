/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#pragma once

// Stable single-translation-unit assembly point for the device implementation.
// The include order is intentional: later kernel families reuse primitives and
// storage types declared by earlier families. Do not compile the detail headers
// as separate translation units without revalidating PTX and resource usage.

#include <flashinfer/conv3d/patchshift/common.cuh>
#include <flashinfer/conv3d/patchshift/problem.cuh>
#include <flashinfer/conv3d/patchshift/weight_layout.cuh>

namespace flashinfer::conv3d::patchshift::detail {

// Compatibility alias kept local to the implementation namespace while the
// imported kernels are transitioned from their standalone namespace.
namespace patchshift = ::flashinfer::conv3d::patchshift;

#include <flashinfer/conv3d/patchshift/detail/kernels/mainloop.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/cluster_b_c32.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/c64_and_hybrid.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/cluster_a_hybrid_c96.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/cluster_b_c64.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/cluster_a.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/output_tail.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/small_grid.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/m32_c64_small_grid.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/m32_d1_shallow_c64.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/m64_c64_small_grid.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/m64_cluster_b.cuh>
#include <flashinfer/conv3d/patchshift/detail/kernels/m64n128_micro_d1.cuh>

}  // namespace flashinfer::conv3d::patchshift::detail
