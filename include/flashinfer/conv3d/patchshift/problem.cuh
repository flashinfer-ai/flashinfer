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

namespace flashinfer::conv3d::patchshift {

// The public problem description contains only runtime tensor extents. Kernel,
// padding, stride, dilation, layout, and dtype are fixed by this operator.
struct Conv3dProblem {
  int n;
  int d;
  int h;
  int w;
  int c;
  int k;
};

constexpr bool IsSupportedProblem(const Conv3dProblem& problem) {
  return problem.n > 0 && problem.d > 0 && problem.h > 0 && problem.w > 0 && problem.c > 0 &&
         problem.c % 8 == 0 && problem.k > 0;
}

}  // namespace flashinfer::conv3d::patchshift
