# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The attention kernel body as engine blocks : host operand
setup (``operands``), the TMA producer over paged KV (``producer``), the two MMA folds issued per
instruction (``mma``), the softmax warpgroup executing the spec's tile step on the engine's plans
(``softmax``), the correction warpgroup (rescale, chunk merge, epilogue) (``correction``), the
composition (``kernel``) and the compile wrapper (``compile``). Written against the CuTe DSL
library; the structure follows NVIDIA's CUTLASS 4.8 Blackwell FMHA example (BSD-3-Clause,
``examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/fmha.py``) and reuses the GEMM
blocks' conventions; the arithmetic is ``spec.py``'s."""
