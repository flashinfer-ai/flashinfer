/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "decoder_impl_common.h"
#include "decoder_params.h"
#include "pytorch_extension_utils.h"

void trtllm_paged_attention(at::Tensor& out, at::Tensor& query, at::Tensor& key_value_cache,
                            at::Tensor& workspace_buffer, int64_t num_q_heads, int64_t num_kv_heads,
                            double scale, at::Tensor& block_tables, at::Tensor& seq_lens,
                            int64_t block_size, int64_t max_seq_len,
                            const std::string kv_cache_dtype, double k_scale, double v_scale);
