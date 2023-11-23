/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <torch/extension.h>

torch::Tensor single_decode_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                          torch::Tensor tmp, unsigned int rotary_mode,
                                          unsigned int layout, float sm_scale, float rope_scale,
                                          float rope_theta);

torch::Tensor single_prefill_with_kv_cache(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                           torch::Tensor tmp, bool causal, unsigned int layout,
                                           unsigned int rotary_mode, bool allow_fp16_qk_reduction,
                                           float rope_scale, float rope_thet);
