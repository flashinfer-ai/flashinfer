"""Prepared serialized block-FP8 linear via native MXFP8 GEMM.

``plan(Caps)`` declares the exact capacity and precision contract without
allocating or compiling.  A ``PreparationSession`` produces the prepared plan
used by ``bind`` or ``run``.  Runtime calls only use launchers retained by
that plan; they never resolve a policy or compile a kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="block_fp8_linear",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Weight",
        "DenseGemmConfig",
        "BlockFp8LinearQuery",
        "plan",
        "bind",
        "run",
        "pack_weight",
        "quantize_input",
        "is_supported",
    ),
    dtypes=("bf16", "fp16"),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/block_fp8_linear.py",),
    ),
    test_path="tests/experimental/b12x/gemm/test_block_fp8_linear.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        DenseGemmConfig,
        BlockFp8LinearQuery,
        Caps,
        Plan,
        Weight,
        bind,
        is_supported,
        pack_weight,
        plan,
        quantize_input,
        run,
    )

install_lazy_api(globals(), META)
