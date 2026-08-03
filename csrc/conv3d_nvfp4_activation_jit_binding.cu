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

#include "tvm_ffi_utils.h"

void nvfp4_conv3d_quantize_activation(TensorView input, TensorView global_scale,
                                      TensorView packed_output, TensorView scale_output,
                                      int64_t pad_height, int64_t pad_width, int64_t tile_variant);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_conv3d_quantize_activation, nvfp4_conv3d_quantize_activation);
