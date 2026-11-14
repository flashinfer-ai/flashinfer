"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

The A/B knobs shared by the Cake LM-head training facades
(``flashinfer.cake_lm_head_det_train`` and the non-deterministic one): the
``FLASHINFER_CAKE_LM_HEAD_LOSS_*`` environment variables, their parsing and the
binding-cache configuration.  Private: no facade exports these names.

The package backends take every switch as a keyword argument and read no
environment variable.  A facade's switch keyword left at ``None`` falls back to
its variable here (:func:`resolve_knobs`), and the resolved value is passed
down.  Each facade reads the two binding-cache variables once, when it is
imported (:func:`binding_cache_settings`), and applies them to its backend's
cache on first use (:func:`configure_binding_cache`).
"""

import json
import os
from types import ModuleType
from typing import Any, Dict, Optional, Union

CHECK_WITHOUT_COUNT = (
    "check_num_valid_rows=True needs num_valid_rows (the count it checks)"
)

# "0" turns valid-row compaction off (``compact_rows``); anything else: on.
COMPACT_ROWS_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_COMPACT_ROWS"
# "0" turns the fused weight-gradient cast off (``fuse_dw_cast``).
FUSE_DW_CAST_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_FUSE_DW_CAST"
# "0" turns the fused dX finalize off (``dx_finalize``).
DX_FINALIZE_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_DX_FINALIZE"
# "0" / "false" / "off" / "no": never; "1" / "true" / "on" / "yes": every
# multi-chunk call; anything else (the default): "auto" (``dw_stream``).
DW_STREAM_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM"
# "0" turns the hidden valid-row count off (``hidden_count``).
HIDDEN_COUNT_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_HIDDEN_COUNT"
# A JSON object of explicit per-GEMM knobs (``gemm_tuning``); unset / empty:
# none.
GEMM_TUNING_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING"
# "0" disables the backend's binding cache (read at import, applied on first
# use).
BINDING_CACHE_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE"
# Positive integer: the binding cache's capacity (read at import, applied on
# first use; unset / blank: the backend's default of 64 bindings).
BINDING_CACHE_CAPACITY_ENV = "FLASHINFER_CAKE_LM_HEAD_LOSS_BINDING_CACHE_CAPACITY"

DW_STREAM_OFF = ("0", "false", "off", "no")
DW_STREAM_ON = ("1", "true", "on", "yes")


def env_switch(value: Optional[bool], name: str) -> bool:
    """``value`` when given, else the variable ``name``: on unless set to ``"0"``."""
    if value is not None:
        return bool(value)
    return os.environ.get(name, "1") != "0"


def env_dw_stream(value: Optional[Union[bool, str]]) -> Union[bool, str]:
    """``value`` when given, else ``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM`` as
    ``False`` (never), ``True`` (every multi-chunk call) or ``"auto"``."""
    if value is not None:
        return value
    raw = os.environ.get(DW_STREAM_ENV, "auto").strip().lower()
    if raw in DW_STREAM_OFF:
        return False
    if raw in DW_STREAM_ON:
        return True
    return "auto"


def env_gemm_tuning(
    value: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """``value`` when given, else ``FLASHINFER_CAKE_LM_HEAD_LOSS_GEMM_TUNING`` parsed
    as a JSON object (``None`` when unset / blank; invalid JSON or a non-object is a
    ``ValueError``); the backend validates the knobs themselves."""
    if value is not None:
        return value
    raw = os.environ.get(GEMM_TUNING_ENV, "").strip()
    if not raw:
        return None
    try:
        knobs = json.loads(raw)
    except ValueError as exc:
        raise ValueError(
            f"${GEMM_TUNING_ENV} must be a JSON object of GEMM knobs: {exc}"
        ) from exc
    if not isinstance(knobs, dict):
        raise ValueError(
            f"${GEMM_TUNING_ENV} must be a JSON object of GEMM knobs, got {raw!r}"
        )
    return knobs


def resolve_knobs(
    *,
    compact_rows: Optional[bool],
    fuse_dw_cast: Optional[bool],
    dx_finalize: Optional[bool],
    dw_stream: Optional[Union[bool, str]],
    hidden_count: Optional[bool],
    gemm_tuning: Optional[Dict[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    """The switch keywords one call passes to the backend: every explicit value
    as given, every ``None`` replaced by its environment variable."""
    return dict(
        compact_rows=env_switch(compact_rows, COMPACT_ROWS_ENV),
        fuse_dw_cast=env_switch(fuse_dw_cast, FUSE_DW_CAST_ENV),
        dx_finalize=env_switch(dx_finalize, DX_FINALIZE_ENV),
        dw_stream=env_dw_stream(dw_stream),
        hidden_count=env_switch(hidden_count, HIDDEN_COUNT_ENV),
        gemm_tuning=env_gemm_tuning(gemm_tuning),
    )


def binding_cache_settings() -> Dict[str, Any]:
    """``enabled`` / ``capacity`` of the backend's binding cache from the two
    binding-cache variables (``capacity`` ``None`` = the backend's default)."""
    enabled = os.environ.get(BINDING_CACHE_ENV, "1") != "0"
    raw = os.environ.get(BINDING_CACHE_CAPACITY_ENV)
    capacity: Optional[int] = None
    if raw is not None and raw.strip():
        try:
            capacity = int(raw)
        except ValueError:
            capacity = 0
        if capacity < 1:
            raise ValueError(
                f"{BINDING_CACHE_CAPACITY_ENV} must be a positive integer, got {raw!r}"
            )
    return dict(enabled=enabled, capacity=capacity)


def configure_binding_cache(cake_backend: ModuleType, settings: Dict[str, Any]) -> None:
    """Apply ``settings`` (``enabled`` / ``capacity`` as returned by
    :func:`binding_cache_settings`) to ``cake_backend.BINDING_CACHE`` in place;
    a ``capacity`` of ``None`` is the backend's default."""
    capacity = settings["capacity"]
    cake_backend.BINDING_CACHE.configure(
        enabled=settings["enabled"],
        capacity=cake_backend.BINDING_CACHE_DEFAULT_CAPACITY
        if capacity is None
        else capacity,
    )
