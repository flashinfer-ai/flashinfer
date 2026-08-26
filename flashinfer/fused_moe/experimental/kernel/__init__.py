# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA source behind the experimental fused MoE routing entry points.

``moe_routing_sm120.cu`` is one translation unit holding all three kernels and
all three tvm_ffi entry points.  It is compiled on demand by
:func:`flashinfer.jit.moe_routing.gen_moe_routing_sm120_module`, which reads it
from this directory by path -- nothing imports this package to reach it, and
JIT output goes to ``FLASHINFER_GEN_SRC_DIR`` / ``FLASHINFER_JIT_DIR``, never
back into the package.

This package init is therefore deliberately empty of imports: importing
``flashinfer`` must not import a kernel.  It exists so the ``.cu`` ships as
package data in a non-editable install (see ``pyproject.toml``).
"""
