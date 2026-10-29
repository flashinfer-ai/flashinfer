# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""cudnn.gemm.frost: JIT fused GEMM kernels from cuDNN graphs via the CuTe DSL.

User code uses the plain cuDNN frontend API. The analyzer reads the python IR
(``graph.nodes``) directly; :class:`cudnn.gemm.frost.engine.FrostGemmEngine`
is the engine the graph API dispatches to (listed in ``cudnn/engines/manifest.py``).
"""

from .graph_analyzer import build_gemm_plan, probe_gemm_plan

__all__ = ["build_gemm_plan", "probe_gemm_plan"]
