# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer's switch for cudnn-frontend's opt-in FROST SDPA engines.

cudnn-frontend ships its CuTe-DSL ("FROST") SDPA engines behind an opt-in that
the frontend reads once, at ``import cudnn``: ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES``.
Those engines are what make cuDNN decode fast for the popular decode shapes on
Blackwell (the d128 / d256 decode tiles, multi-token rows, attention sinks at
``q_len_per_req == 1``); the classic backend engine serves the same graphs
correctly but slowly, and rejects a sink at ``q_len_per_req == 1``.

FlashInfer imports cudnn lazily from several modules (decode, prefill, gemm,
...), so a FlashInfer-level switch has to be applied before the first of them
runs. ``flashinfer/__init__.py`` calls :func:`configure_cudnn_frost_engines`
before any submodule import; everything else here is read at plan time.

Only stdlib imports: this module must stay importable before torch and cudnn.
"""

import functools
import os
import sys
import warnings
from typing import Optional, Tuple

FI_FROST_ENV = "FLASHINFER_CUDNN_FROST_ENGINES"
FE_FROST_ENV = "CUDNN_FRONTEND_ENABLE_FROST_ENGINES"
# The first cudnn-frontend release whose FROST rows serve paged decode (the
# d128 / d256 decode tiles, sink at s_q == 1, d192x128 paged KV).
FROST_DECODE_MIN_FRONTEND: Tuple[int, int, int] = (1, 30, 0)

_TRUE = ("1", "true", "yes", "on")


def _is_true(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in _TRUE


def frost_engines_requested() -> bool:
    """Whether the FROST engines are switched on for this process.

    ``FLASHINFER_CUDNN_FROST_ENGINES`` wins when set; otherwise the frontend's
    own ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` is honored, so a user who already
    sets the frontend variable sees no change.
    """
    fi = os.environ.get(FI_FROST_ENV)
    if fi is not None:
        return _is_true(fi)
    return _is_true(os.environ.get(FE_FROST_ENV))


def configure_cudnn_frost_engines() -> bool:
    """Apply ``FLASHINFER_CUDNN_FROST_ENGINES`` to the frontend's switch.

    Idempotent; returns whether the engines are requested. When the variable is
    unset, the frontend's own variable is left untouched. When cudnn was already
    imported before FlashInfer, the frontend has read its switch and this call
    cannot change it: a ``RuntimeWarning`` says so and the effective setting is
    returned.
    """
    fi = os.environ.get(FI_FROST_ENV)
    if fi is None:
        return frost_engines_requested()
    want = "1" if _is_true(fi) else "0"
    current = os.environ.get(FE_FROST_ENV)
    if "cudnn" in sys.modules and (_is_true(current) != _is_true(want)):
        warnings.warn(
            f"{FI_FROST_ENV}={fi!r} was applied after cudnn was already imported; "
            "cudnn-frontend read its FROST-engine switch at import, so this process "
            f"keeps {FE_FROST_ENV}={current!r}. Import flashinfer before cudnn, or set "
            f"{FE_FROST_ENV} in the environment.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _is_true(current)
    os.environ[FE_FROST_ENV] = want
    return want == "1"


def cudnn_frontend_version() -> Optional[Tuple[int, ...]]:
    """The installed cudnn-frontend python package's version, or ``None``.

    ``"1.31.0.dev123"`` parses to ``(1, 31, 0)``; a missing package or a
    version string without a numeric prefix gives ``None``.
    """
    try:
        import cudnn  # noqa: PLC0415 -- lazy on purpose (see module docstring)
    except Exception:  # noqa: BLE001 -- any import failure means "not available"
        return None
    raw = getattr(cudnn, "__version__", None)
    if not raw:
        return None
    parts = []
    for piece in str(raw).split("+")[0].split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else None


@functools.lru_cache(maxsize=None)
def _warn_frontend_too_old(version: Tuple[int, ...]) -> None:
    warnings.warn(
        f"{FI_FROST_ENV} is set but the installed cudnn-frontend is "
        f"{'.'.join(map(str, version))}; its FROST rows do not serve paged decode "
        f"(that needs {'.'.join(map(str, FROST_DECODE_MIN_FRONTEND))}+), so cuDNN "
        'decode runs on the backend engine and backend="auto" keeps fa2.',
        RuntimeWarning,
        stacklevel=3,
    )


def frost_decode_engines_available(compute_capability: Tuple[int, int]) -> bool:
    """Whether the FROST decode tiles can serve cuDNN decode in this process.

    True when the engines are requested, the installed frontend is at least
    :data:`FROST_DECODE_MIN_FRONTEND`, and the device is SM100 / SM103 (the
    Blackwell parts with a decode tile; Rubin has its own rows without one).
    """
    if not frost_engines_requested():
        return False
    version = cudnn_frontend_version()
    if version is None:
        return False
    if version < FROST_DECODE_MIN_FRONTEND:
        _warn_frontend_too_old(version)
        return False
    return tuple(compute_capability) in ((10, 0), (10, 3))
