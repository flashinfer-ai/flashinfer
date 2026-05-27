# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatcher between the modular and monolithic CuTe DSL MLA decode kernels.

Both implementations share the public ``backend="cute-dsl"`` user surface in
``trtllm_batch_decode_with_kv_cache_mla``.  Implementation selection is
controlled by the ``cute_dsl_impl`` kwarg, with three valid values:

* ``"auto"`` (default) — library picks the right implementation.
  Monolithic by default, automatically promoted to modular when the call
  uses a feature monolithic doesn't support (currently: ``sinks``).
* ``"modular"`` — strict.  Always run the modular implementation.
* ``"monolithic"`` — strict.  Always run the monolithic implementation;
  raise :class:`ValueError` if the call uses any modular-only feature.
  No silent fallback — the contract is "you asked for monolithic, you
  get monolithic, or a clear error".

The strict modes exist so users can pin the implementation for
differential debugging or perf characterisation without worrying that
the dispatcher will silently substitute something else.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

VALID_IMPLS = ("auto", "modular", "monolithic")
_DEFAULT_IMPL = "monolithic"

# One-shot log so users can confirm which impl they actually got, without
# spamming on every kernel call.  Keyed by resolved impl so the very first
# call of each impl in a process logs once.
_logged_impls: set[str] = set()


# Modular-only features.  When any of these is requested:
#   * ``cute_dsl_impl="auto"``       → silently promote to modular
#   * ``cute_dsl_impl="modular"``    → already correct, run as-is
#   * ``cute_dsl_impl="monolithic"`` → raise ValueError
#
# Add new entries here as more variants are exposed through the standalone
# signature.
MODULAR_ONLY_KWARGS = ("sinks",)


def _has_modular_only_feature(kwargs: dict) -> Optional[str]:
    """Return the name of the first modular-only kwarg present (and non-None),
    or None if no modular-only feature was requested."""
    for name in MODULAR_ONLY_KWARGS:
        if kwargs.get(name) is not None:
            return name
    return None


def _resolve_impl(*, requested: str, kwargs: dict) -> str:
    """Map a user request and call kwargs to a concrete impl name.

    See module docstring for the contract.  ``requested`` must already be
    one of :data:`VALID_IMPLS`.
    """
    if requested not in VALID_IMPLS:
        raise ValueError(
            f"Invalid cute_dsl_impl={requested!r}; expected one of {VALID_IMPLS}"
        )

    needs_modular = _has_modular_only_feature(kwargs)

    if requested == "auto":
        return "modular" if needs_modular is not None else _DEFAULT_IMPL

    if requested == "monolithic" and needs_modular is not None:
        raise ValueError(
            f"cute_dsl_impl='monolithic' was requested but the call uses "
            f"{needs_modular!r}, which is only supported by the modular "
            f"implementation. Use cute_dsl_impl='auto' (default, picks the "
            f"right impl based on the call) or cute_dsl_impl='modular'."
        )

    return requested  # "modular" or "monolithic"


def cute_dsl_mla_decode(*args, cute_dsl_impl: str = "auto", **kwargs):
    """Run CuTe DSL MLA decode using the resolved implementation.

    Forwards all positional and keyword arguments verbatim to the underlying
    implementation. See
    :func:`flashinfer.cute_dsl.attention.wrappers.batch_mla.cute_dsl_mla_decode`
    (modular, supports ``sinks=``) and
    :func:`flashinfer.cute_dsl.attention.monolithic.mla_decode.cute_dsl_mla_decode`
    (monolithic, no variant support) — their signatures are otherwise
    identical.

    Parameters
    ----------
    cute_dsl_impl : str, default ``"auto"``
        ``"auto"`` (default) lets the dispatcher pick: monolithic by
        default, modular when the call uses a modular-only feature
        (currently ``sinks``).  ``"modular"`` and ``"monolithic"`` are
        strict — the dispatcher will not silently switch implementations,
        and ``"monolithic"`` raises :class:`ValueError` if the call uses
        a modular-only feature.
    """
    impl = _resolve_impl(requested=cute_dsl_impl, kwargs=kwargs)

    if impl not in _logged_impls:
        _logged_impls.add(impl)
        logger.info(
            "flashinfer.cute_dsl MLA decode using impl=%s",
            impl,
        )

    # Imports are deferred so that selecting one impl never imports/JITs the
    # other.  Each impl's import path triggers heavy CuTe-DSL machinery.
    if impl == "monolithic":
        # Strip modular-only kwargs.  The strict resolver above guarantees
        # they're all None when we land here, but the monolithic standalone
        # signature predates these kwargs and would TypeError on them.
        kwargs = {k: v for k, v in kwargs.items() if k not in MODULAR_ONLY_KWARGS}
        from .monolithic.mla_decode import cute_dsl_mla_decode as _impl
    else:
        from .wrappers.batch_mla import cute_dsl_mla_decode as _impl
    return _impl(*args, **kwargs)
