"""Prepared contiguous batched and packed-varlen attention for SM12x."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="varlen",
    group="attention",
    api_style="prepared",
    entry_points=(
        "BatchedBinding",
        "Plan",
        "VarlenAttentionConfig",
        "VarlenAttentionQuery",
        "VarlenBinding",
        "bind",
        "bind_batched",
        "plan",
        "plan_batched",
        "run",
        "run_batched",
        "is_supported",
    ),
    dtypes=("bf16", "fp16"),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/attention/contiguous/",),
    ),
    test_path="tests/experimental/b12x/attention/test_varlen.py",
    since="0.7.0",
    notes=("Reduced-assurance tier: correctness-tested against a torch reference.",),
)

if TYPE_CHECKING:
    from .api import *  # noqa: F401,F403

install_lazy_api(globals(), META)
