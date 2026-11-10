#pragma once

/*
 * Public C++ interface for BGMV MoE kernels (TVM-FFI).
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <cstdint>

// Forward declarations for TVM-FFI dispatch functions.
// These are defined in moe_bgmv_ops.cu and exported via moe_bgmv_binding.cu.

void bgmv_moe_shrink(TensorView y, TensorView x, TensorView w_ptr, TensorView sorted_token_ids,
                     TensorView expert_ids, TensorView lora_indices, int64_t lora_stride,
                     bool per_pair_input);

void bgmv_moe_expand(TensorView y, TensorView x, TensorView w_ptr, TensorView sorted_token_ids,
                     TensorView expert_ids, TensorView topk_weights, TensorView lora_indices,
                     TensorView slice_start_loc, int64_t first_feat_out, int64_t lora_stride,
                     bool finalize);
