# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""BF16 -> NVFP4 TMA tile quantizer for SM12x.

Quantizes a [M, K] BF16 tensor (M, K multiples of 128) into packed FP4
values plus e4m3 scales in the dense-GEMM MMA layout (sf vec 16), using a
128x128 TMA tile kernel. ``plan(m, k)`` compiles the shape (host-side,
cached); ``allocate_outputs`` sizes the output pair; ``run`` launches into
the caller's outputs (allocation-free, capture safe). There is no bind step:
the outputs container is the binding.

Example:
    from flashinfer.experimental.sm12x.quantization import nvfp4

    plan = nvfp4.plan(m=256, k=512)
    outs = nvfp4.allocate_outputs(plan)
    nvfp4.run(plan=plan, x=x_bf16, global_scale=gs, outputs=outs)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="nvfp4",
    group="quantization",
    api_style="planned",
    entry_points=(
        "Outputs",
        "Plan",
        "plan",
        "allocate_outputs",
        "run",
        "is_supported",
    ),
    dtypes=("bf16",),
    recipes=("nvfp4",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/quantization/",),
    ),
    test_path="tests/experimental/quantization/test_nvfp4.py",
    since="0.7.0",
    notes=(
        "No bind step: outputs are caller-allocated via allocate_outputs. "
        "Dead at the original port baseline; revived by the CUTLASS DSL 4.6 "
        "migration."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Outputs,
        Plan,
        allocate_outputs,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
