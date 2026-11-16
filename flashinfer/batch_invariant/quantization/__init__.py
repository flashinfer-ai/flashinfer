# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant quantization classes: ``mxfp8_quantize`` / ``mxfp4_quantize`` (``mx_api``) and ``nvfp4_quantize``
(``nvfp4_api``) under ``batch_invariant=True``. Light imports only: the CuTe-DSL kernel modules are imported on first use.

The kernels' compiled-object cache modules are registered here (the kernel modules are their roots: an edit to a kernel
module, or to anything it imports within the mode's directories, invalidates its objects and nothing else's)."""

from .. import jit as _jit

_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_mx_quant", ("batch_invariant/quantization/mx_kernel.py",)
)
_jit.MODULE_ROOTS.setdefault(
    "batch_invariant_nvfp4_quant", ("batch_invariant/quantization/nvfp4_kernel.py",)
)
