# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The GEMM kernel body as engine blocks: operand setup, TMA producer, the tcgen05 mainloop (the engine's blocked fold),
the epilogue (the specification's chain, then the store) and the compile wrapper. The pipeline and TMEM scaffolding
follow the CUTLASS CuTe-DSL Blackwell GEMM examples (Apache-2.0); the composition and every numeric choice are this
package's."""
