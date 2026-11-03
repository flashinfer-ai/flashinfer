"""Stateful prime-learned-embedding residual support.

``plan`` / ``bind`` / ``run_decode``, ``run_prefill``, or ``run_mixed`` consume
already-projected PLE keys and values, compute the gated residual contribution,
and update the caller-owned dilated-convolution state. PLE token hashing is the
separate :mod:`b12x.sequence.ple_hash` op.

``plan`` declares capacity and invocation metadata without preparation.
``PreparationSession`` resolves and primes the native pipeline before ``bind``
may consume its prepared ``Plan``. Bindings retain that plan alongside
caller-owned tensors and scratch views. Exact PyTorch oracles are available from
:mod:`b12x.sequence.ple.reference` and are not used as runtime fallbacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="ple",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Plan",
        "Binding",
        "PleConfig",
        "PleQuery",
        "plan",
        "invocation_from_tensors",
        "bind",
        "run_decode",
        "run_mixed",
        "run_prefill",
        "is_supported",
    ),
    dtypes=("bf16", "int64"),
    recipes=("decode", "prefill", "mixed"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="3a437ab51680",
        paths=(
            "b12x/_lib/scratch.py",
            "b12x/_lib/scratch_layout.py",
        ),
    ),
    test_path="tests/experimental/b12x/sequence/test_ple.py",
    since="1.3.0",
    notes=(
        "Decode state keeps one guaranteed token in the base window and "
        "retains additional speculative candidates in the extended tail. "
        "The Triton implementation is a correctness reference and is not "
        "throughput-qualified."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        PleConfig,
        PleQuery,
        bind,
        is_supported,
        plan,
        invocation_from_tensors,
        run_decode,
        run_mixed,
        run_prefill,
    )

install_lazy_api(globals(), META)
