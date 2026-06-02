/*
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_bgmv_config.h"
#include "moe_bgmv_impl.cuh"

// Shrink only with mixed precision (in_T=float, out_T=nv_half, W_T=nv_half).

#define INST_MOE_BGMV_SHRINK_ONLY(in_T, out_T, W_T, narrow, wide) \
  INST_MOE_BGMV_SHRINK_SLICED(wide, narrow, in_T, out_T, W_T)

FOR_MOE_ALL_WIDE_NARROW(INST_MOE_BGMV_SHRINK_ONLY, float, nv_half, nv_half)
