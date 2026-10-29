# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Vocabulary shared by kernel templates and data-only configs."""

MASK_NONE = 0
MASK_PADDED = 1 << 0
MASK_CAUSAL = 1 << 1
MASK_SWA = 1 << 2

SCHED_NATURAL = 0
SCHED_LPT = 1
SCHED_LPT_L2 = 2

DTYPE_E4M3 = 0
DTYPE_E5M2 = 1
DTYPE_BF16 = 2
DTYPE_FP16 = 3
# Output-only block-scaled codes (per-tensor FP8 fprop epilogue): the data
# dtype plus a scale-factor tensor written alongside O.
DTYPE_O_NVFP4 = 4  # E2M1 data (2 per byte) + E4M3 scale per 16 d-elements
DTYPE_O_MXFP8 = 5  # E4M3 data + UE8M0 scale per 32 d-elements
# Output-only fp32 code (4 B/element): a backward kernel's per-Q-head PARTIAL that a
# later fold sums in fp32 and rounds once (never an io / payload dtype).
DTYPE_FP32 = 6

# Scale-factor block along d for each output code; 0 = no block scaling.
O_BLOCK_SCALE_BY_DTYPE = {DTYPE_E4M3: 0, DTYPE_E5M2: 0, DTYPE_BF16: 0, DTYPE_FP16: 0, DTYPE_O_NVFP4: 16, DTYPE_O_MXFP8: 32}

# --- THD metadata layout: the one word count the HOST sizes a buffer with ----
# The THD layout itself (offsets, unit decode, descriptor patchers) lives in
# ``tile_dsl.thd``, whose import binds the CuTe DSL (``cutlass.experimental``).
# ``sdpa/bwd/api_dsl.py`` sizes the SM100 THD workspace with this count at module
# scope, and that module is also the SM80 backward wrapper's entry, which must
# import WITHOUT the DSL so it can decline a too-old DSL by version before any
# kernel module loads (AGENTS.md Rule 7; ``test_sdpa_sm80_cutedsl_gate.py``).  So
# the pure-Python count is defined here and re-exported by ``thd`` next to the
# offsets it closes -- the backward metadata buffer, in int32 words:
#   [ seq_kv_lens(B) | cu_q(B+1) | cu_k(B+1) | remap(B) | live | ctr | row_off(B+1) | unit prefix(B+1) ]
#   = THD_BWD_WORK_PREFIX_OFF(b) + (b + 1).
THD_BWD_META_WORDS = lambda b: 6 * b + 6  # noqa: E731
