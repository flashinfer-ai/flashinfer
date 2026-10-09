"""Prepared causal depthwise convolution over raw projected BF16 inputs.

``plan(Caps(...))`` declares geometry; ``PreparationSession`` compiles and
primes the pipeline; ``bind`` maps caller-owned tensors and scratch; ``run``
updates live history slots and writes SiLU outputs without allocation.
Convolution accumulates in FP32 and rounds to BF16 before SiLU. No projection,
normalization, input gate, or residual addition is performed.

Packed device metadata supports decode, prefill, and mixed batches. History is
``[slot, channels, dilation * (kernel_size - 1) + max_speculative_tokens]``
in chronological order. Prefill commits all input rows. Decode commits its
first input and retains speculative candidates in the tail; the following
call selects history at ``num_accepted_tokens - 1``. Ordinary decode accepts
one token. Fresh slots ignore all prior history, including speculative tails.

Counts must be within the declared capacities, query boundaries must be
monotone and span the live tokens, and nonnegative state slots must be unique
and in range. Negative slots produce zero output and do not update history.
Decode query lengths and accepted counts are in ``[1, 1 + max_speculative_tokens]``;
empty requests are allowed and leave state unchanged. Inactive metadata is
not read. These device-value invariants are caller responsibilities.
"""

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="short_conv",
    group="sequence",
    api_style="planned",
    entry_points=(
        "Caps",
        "Binding",
        "Plan",
        "plan",
        "invocation_from_tensors",
        "bind",
        "run",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16", "int32", "int64"),
    recipes=("decode", "prefill", "mixed"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/flashinfer-ai/flashinfer",
        commit="98fb87a461183f98",
        paths=("flashinfer/experimental/b12x/sequence/ple/_kernels.py",),
    ),
    test_path="tests/experimental/b12x/sequence/test_short_conv.py",
    notes="Raw-input short convolution sharing the PLE history implementation.",
)

if TYPE_CHECKING:
    from .api import (
        Binding,
        Caps,
        Plan,
        bind,
        clear_caches,
        invocation_from_tensors,
        is_supported,
        plan,
        run,
    )

install_lazy_api(globals(), META)
