"""Fused MLA WO-A / WO-B projections as prepared native MXFP8 GEMMs.

Declare ``plan(Caps(...))``, prepare it through ``PreparationSession``, then
bind caller-owned tensors/scratch and run with the returned prepared
``Plan``.  Quantizer entry points follow the same ready-plan boundary.
Packing remains a one-time caller-owned weight operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="wo_projection",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "InvRopeBinding",
        "Weights",
        "WoProjectionConfig",
        "WoProjectionQuery",
        "MXFP8Rows",
        "plan",
        "bind",
        "bind_inv_rope",
        "run",
        "run_inv_rope",
        "pack_weights",
        "quantize_input",
        "quantize_input_inv_rope",
        "quantize_input_b",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("mxfp8",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/wo_projection.py", "b12x/gemm/wo_quant_cute.py"),
    ),
    test_path="tests/experimental/b12x/gemm/test_wo_projection.py",
    since="0.7.0",
    notes=(
        "The planned run path is BF16-only; the standalone quantize_input* "
        "facet also compiles for fp16."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        InvRopeBinding,
        MXFP8Rows,
        Plan,
        Weights,
        WoProjectionConfig,
        WoProjectionQuery,
        bind,
        bind_inv_rope,
        is_supported,
        pack_weights,
        plan,
        quantize_input,
        quantize_input_b,
        quantize_input_inv_rope,
        run,
        run_inv_rope,
    )

install_lazy_api(globals(), META)
