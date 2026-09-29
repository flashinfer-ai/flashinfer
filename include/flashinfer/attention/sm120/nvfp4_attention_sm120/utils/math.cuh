/*
 * Copyright (c) 2025 by SageAttention team.
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

#include <flashinfer/math.cuh>

namespace nvfp4_attention {

using flashinfer::math::add;
using flashinfer::math::exp2_fma_poly;
using flashinfer::math::fma;
using flashinfer::math::MaxOp;
using flashinfer::math::mul;
using flashinfer::math::ptx_exp2;
using flashinfer::math::SumOp;

}  // namespace nvfp4_attention
