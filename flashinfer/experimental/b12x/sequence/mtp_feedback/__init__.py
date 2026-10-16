"""MTP token/multi-stream feedback fusion.

The op normalizes and projects a token embedding alongside the target model's
pre-final multi-stream residual state.  State normalization intentionally uses
one flattened ``S*H`` variance group; the hidden projection is shared across
streams.  The result is a caller-owned BF16 ``[T, S, H]`` draft-layer input.

``plan(Caps(...), invocation=invocation_from_tensors(...))`` declares capacity
and input alignment. ``PreparationSession`` compiles and primes the complete
projection/normalization route. ``bind`` consumes its prepared ``Plan``;
``run`` uses only the stored launchers, with live rows remaining dynamic.
The explicitly named ``reference`` module is a PyTorch correctness oracle and
is never a runtime fallback for the public GPU entry point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mtp_feedback",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "MtpFeedbackConfig",
        "MtpFeedbackQuery",
        "plan",
        "invocation_from_tensors",
        "bind",
        "run",
        "reference",
        "is_supported",
    ),
    dtypes=("bf16",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="fa097786643f49d9e9591fd8b2eb0cb3398d8f79",
        paths=("b12x/sequence/mtp_feedback/",),
    ),
    test_path="tests/experimental/b12x/sequence/test_mtp_feedback.py",
    since="1.3.0",
    notes=(
        "Qualified Qwen S=4,H=2560 uses mandatory capacity-specialized "
        "CuTeDSL projection GEMMs with runtime live-row grids. Triton is used "
        "only by the normalization and reduction auxiliaries feeding those "
        "projections. Other geometries or tensor contracts are unsupported."
    ),
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Binding,
        Caps,
        MtpFeedbackConfig,
        MtpFeedbackQuery,
        Plan,
        bind,
        is_supported,
        plan,
        invocation_from_tensors,
        reference,
        run,
    )

install_lazy_api(globals(), META)
