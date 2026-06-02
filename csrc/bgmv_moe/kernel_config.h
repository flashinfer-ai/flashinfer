#pragma once

/*
 * BGMV MoE kernel tuning parameters.
 *
 * Target: H100/H200 (sm_90, 228 KB shared memory per SM)
 * Also supports sm_70+ (V100, A100) with reduced pipeline depth.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

struct MoeShrinkKernelConfig {
  static constexpr int tx = 32;        // threads per warp (x-dimension)
  static constexpr int ty = 4;         // number of warps (y-dimension)
  static constexpr int vec_size = 8;   // elements per vectorized load
  static constexpr int rank_tile = 8;  // rank elements per block (8x X reuse)

  // Multi-pair decode path: PPB=4 pairs per block for decode,
  // PPB=1 for prefill (grid already saturates GPU).
  static constexpr int pairs_per_block_prefill = 1;
  static constexpr int pairs_per_block_decode = 4;
  static constexpr int decode_threshold = 32;

  // Pipeline depth: 3 stages on sm_90 decode (216 KB / 228 KB = 95%),
  // 2 stages for prefill (36 KB, leaves room for occupancy).
  static constexpr int num_stages_default = 2;
  static constexpr int num_stages_extended = 3;

  // Shared memory budget (decode, PPB=4, 3 stages, RANK_TILE=8, fp16):
  //   X: 3 * 4 * 1024 * 2 =  24 KB
  //   W: 3 * 4 * 8 * 1024 * 2 = 192 KB
  //   y: 4 * 8 * 4 * 4 =  512 B
  //   Total: ~216 KB (fits 228 KB on H100/H200)
};

struct MoeExpandKernelConfig {
  static constexpr int tz = 4;
  static constexpr int vec_size = 8;
};
