/*
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_bgmv_config.h"
#include "moe_bgmv_impl.cuh"

// Expand only (in_T=nv_half, Y=float32, W_T=nv_half).
// Shrink is covered by moe_bgmv_fp16_fp16_fp16.cu.

#define INST_MOE_BGMV_EXPAND_ONLY(in_T, out_T, W_T, narrow, wide) \
  INST_MOE_BGMV_EXPAND_SLICED(narrow, wide, in_T, W_T)

FOR_MOE_ALL_WIDE_NARROW(INST_MOE_BGMV_EXPAND_ONLY, nv_half, float, nv_half)
