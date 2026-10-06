"""Prepared BF16 decode-time vocabulary projection.

``plan`` declares immutable projection geometry, ``PreparationSession`` selects and
primes one backend, and ``bind``/``run`` consume only its prepared ``Plan``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="bf16_vocab_projection",
    group="gemm",
    api_style="prepared",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "Bf16VocabProjectionConfig",
        "Bf16VocabProjectionQuery",
        "plan",
        "bind",
        "run",
        "is_supported",
    ),
    dtypes=("bf16",),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="11814a27",
        paths=("b12x/gemm/bf16_vocab_projection",),
    ),
    test_path="tests/experimental/b12x/gemm/test_bf16_vocab_projection.py",
    since="1.3.0",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Bf16VocabProjectionConfig,
        Bf16VocabProjectionQuery,
        Caps,
        Plan,
        bind,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
