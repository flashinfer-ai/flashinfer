/*
 * TVM-FFI binding for BGMV MoE kernels.
 *
 * Exports two functions:
 *   - bgmv_moe_shrink
 *   - bgmv_moe_expand
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_bgmv_ops.cu"

// Forward declarations
void bgmv_moe_shrink(TensorView y, TensorView x, TensorView w_ptr, TensorView sorted_token_ids,
                     TensorView expert_ids, TensorView lora_indices, int64_t lora_stride,
                     bool per_pair_input);

void bgmv_moe_expand(TensorView y, TensorView x, TensorView w_ptr, TensorView sorted_token_ids,
                     TensorView expert_ids, TensorView topk_weights, TensorView lora_indices,
                     TensorView slice_start_loc, int64_t first_feat_out, int64_t lora_stride,
                     bool finalize);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bgmv_moe_shrink, bgmv_moe_shrink);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(bgmv_moe_expand, bgmv_moe_expand);
