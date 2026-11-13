/*
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include "moe_bgmv_config.h"
#include "moe_bgmv_impl.cuh"

// Shrink + expand (in_T=out_T=W_T=nv_bfloat16).
FOR_MOE_ALL_WIDE_NARROW(INST_MOE_BGMV_TWOSIDE, nv_bfloat16, nv_bfloat16, nv_bfloat16)
