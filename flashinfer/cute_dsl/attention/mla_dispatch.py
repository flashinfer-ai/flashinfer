# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Dispatcher between the modular and monolithic CuTe DSL MLA decode kernels.

Both implementations share the public ``backend="cute-dsl"`` user surface in
``trtllm_batch_decode_with_kv_cache_mla``.  Implementation selection is
controlled by the ``cute_dsl_impl`` kwarg, with three valid values:

* ``"auto"`` (default) — library picks the right implementation.
  Monolithic by default, automatically promoted to modular when the call
  uses a feature monolithic doesn't support (currently: ``sinks``).
  Compact variable Q and DCP select the monolithic implementation.
* ``"modular"`` — strict.  Always run the modular implementation; reject
  monolithic-only features such as compact variable Q and DCP.
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
MONOLITHIC_ONLY_KWARGS = ("cum_seq_lens_q", "max_q_len")
DCP_KWARGS = (
    "enable_dcp",
    "cp_world",
    "cp_rank",
    "causal_seqlens_kv_global",
)


def _has_modular_only_feature(kwargs: dict) -> Optional[str]:
    """Return the name of the first modular-only kwarg present (and non-None),
    or None if no modular-only feature was requested."""
    for name in MODULAR_ONLY_KWARGS:
        if kwargs.get(name) is not None:
            return name
    return None


def _has_monolithic_only_feature(kwargs: dict) -> Optional[str]:
    """Return the first requested feature implemented only by monolithic."""
    for name in MONOLITHIC_ONLY_KWARGS:
        if kwargs.get(name) is not None:
            return name
    return None


def _validate_dcp_kwargs(kwargs: dict) -> bool:
    """Validate dispatcher-level DCP selection and return whether it is enabled.

    Tensor shape, dtype, and device validation belongs to the monolithic
    wrapper, which has access to the normalized query batch. The dispatcher
    only enforces the static feature contract and prevents non-default DCP
    arguments from being silently ignored.
    """
    enable_dcp = kwargs.get("enable_dcp", False)
    if not isinstance(enable_dcp, bool):
        raise TypeError(f"enable_dcp must be a bool, got {type(enable_dcp).__name__}")

    cp_world = kwargs.get("cp_world", 1)
    cp_rank = kwargs.get("cp_rank", 0)
    causal_seqlens_kv_global = kwargs.get("causal_seqlens_kv_global")
    if not isinstance(cp_world, int) or isinstance(cp_world, bool) or cp_world <= 0:
        raise ValueError(f"cp_world must be a positive integer, got {cp_world!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool):
        raise TypeError(f"cp_rank must be an integer, got {type(cp_rank).__name__}")
    if enable_dcp and not 0 <= cp_rank < cp_world:
        raise ValueError(
            f"cp_rank must satisfy 0 <= cp_rank < cp_world, got "
            f"cp_rank={cp_rank}, cp_world={cp_world}"
        )
    if not enable_dcp:
        nondefault = []
        if cp_world != 1:
            nondefault.append(f"cp_world={cp_world!r}")
        if cp_rank != 0:
            nondefault.append(f"cp_rank={cp_rank!r}")
        if causal_seqlens_kv_global is not None:
            nondefault.append("causal_seqlens_kv_global")
        if nondefault:
            raise ValueError(
                "DCP arguments require enable_dcp=True; got " + ", ".join(nondefault)
            )
    return enable_dcp


def _resolve_impl(*, requested: str, kwargs: dict) -> str:
    """Map a user request and call kwargs to a concrete impl name.

    See module docstring for the contract.  ``requested`` must already be
    one of :data:`VALID_IMPLS`.
    """
    if requested not in VALID_IMPLS:
        raise ValueError(
            f"Invalid cute_dsl_impl={requested!r}; expected one of {VALID_IMPLS}"
        )

    enable_dcp = _validate_dcp_kwargs(kwargs)
    needs_modular = _has_modular_only_feature(kwargs)
    needs_monolithic = _has_monolithic_only_feature(kwargs)

    if needs_modular is not None and needs_monolithic is not None:
        raise ValueError(
            "No CuTeDSL MLA implementation supports both "
            f"{needs_modular!r} (modular-only) and "
            f"{needs_monolithic!r} (monolithic-only)."
        )

    if enable_dcp and needs_modular is not None:
        raise ValueError(
            f"DCP cannot be combined with {needs_modular!r}: DCP requires the "
            "monolithic CuTeDSL MLA implementation, while the requested feature "
            "requires the modular implementation."
        )
    if enable_dcp and needs_monolithic is not None:
        raise ValueError("DCP does not support cum_seq_lens_q / max_q_len")

    if requested == "auto":
        if needs_modular is not None:
            return "modular"
        if enable_dcp or needs_monolithic is not None:
            return "monolithic"
        return _DEFAULT_IMPL

    if requested == "monolithic" and needs_modular is not None:
        raise ValueError(
            f"cute_dsl_impl='monolithic' was requested but the call uses "
            f"{needs_modular!r}, which is only supported by the modular "
            f"implementation. Use cute_dsl_impl='auto' (default, picks the "
            f"right impl based on the call) or cute_dsl_impl='modular'."
        )

    if requested == "modular" and enable_dcp:
        raise ValueError(
            "cute_dsl_impl='modular' was requested with enable_dcp=True, but "
            "DCP is only supported by the monolithic CuTeDSL MLA implementation. "
            "Use cute_dsl_impl='auto' or cute_dsl_impl='monolithic'."
        )

    if requested == "modular" and needs_monolithic is not None:
        raise ValueError(
            f"cute_dsl_impl='modular' was requested but the call uses "
            f"{needs_monolithic!r}, which is only supported by the "
            "monolithic implementation."
        )

    return requested  # "modular" or "monolithic"


def cute_dsl_mla_decode(*args, cute_dsl_impl: str = "auto", **kwargs):
    """Run CuTe DSL MLA decode using the resolved implementation.

    Forwards positional arguments and supported keyword arguments to the
    resolved implementation. See
    :func:`flashinfer.cute_dsl.attention.wrappers.batch_mla.cute_dsl_mla_decode`
    (modular, supports ``sinks=``) and
    :func:`flashinfer.cute_dsl.attention.monolithic.mla_decode.cute_dsl_mla_decode`
    (monolithic, supports compact variable Q and static DCP).
    Implementation-specific keyword
    arguments are validated before dispatch and stripped only when their
    default values make them irrelevant to the selected implementation.

    Parameters
    ----------
    cute_dsl_impl : str, default ``"auto"``
        ``"auto"`` (default) lets the dispatcher pick: monolithic by
        default, modular when the call uses a modular-only feature
        (currently ``sinks``); compact variable Q and DCP require monolithic.
        ``"modular"`` and
        ``"monolithic"`` are strict — the dispatcher will not silently switch
        implementations, and raises :class:`ValueError` for an incompatible
        feature request.
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
        from .monolithic.mla_decode import (
            cute_dsl_mla_decode as _monolithic_impl,
        )

        return _monolithic_impl(*args, **kwargs)
    else:
        # Symmetrically strip inactive monolithic-only and default-valued DCP
        # kwargs after the strict resolver has established that none requests
        # a feature.
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in MONOLITHIC_ONLY_KWARGS and k not in DCP_KWARGS
        }
        from .wrappers.batch_mla import cute_dsl_mla_decode as _modular_impl

        return _modular_impl(*args, **kwargs)
