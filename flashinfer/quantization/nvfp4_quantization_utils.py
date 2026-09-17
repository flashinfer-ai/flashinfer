"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared NVFP4 quantization helpers.
"""

from dataclasses import dataclass, fields
from enum import IntEnum
import os
from typing import Any, Optional, Union
import warnings

import torch


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0


class NVFP44Over6ErrMode(IntEnum):
    """Error metric for selecting between NVFP4 4over6 scale candidates."""

    MAE = 0
    MSE = 1

    def __repr__(self) -> str:
        # ``IntEnum`` defaults to ``<NVFP44Over6ErrMode.MAE: 0>``, which is not
        # valid Python.  A qualified repr (matching ``QuantVariant`` in
        # flashinfer/fused_moe/api.py) keeps the ``eval(repr(cfg))`` round-trip
        # the unified MoE config tree guarantees.
        return f"{type(self).__name__}.{self.name}"


class _UnsetType:
    """Type of :data:`_UNSET`, the "argument omitted" default of ``nvfp4_4over6``.

    The public type of every ``nvfp4_4over6=`` parameter is
    ``Optional[NVFP44Over6Config]``: ``None`` means 4over6 off and a config
    means on with exactly that recipe.  Omitting the argument is the third
    case -- read the legacy ``FLASHINFER_NVFP4_4OVER6*`` environment
    variables -- and it is spelled by the *default value* rather than by a
    third public type, so that retiring the environment shim later only flips
    the default to ``None`` and changes no signature.

    ``__reduce__`` returning the module attribute name keeps ``pickle`` and
    ``copy`` returning this same object, so ``is _UNSET`` stays valid on a
    round-tripped config.
    """

    def __repr__(self) -> str:
        return "<nvfp4_4over6 unset: read FLASHINFER_NVFP4_4OVER6*>"

    def __reduce__(self) -> str:
        return "_UNSET"


#: Default of every public ``nvfp4_4over6=`` parameter: "not passed".  Typed
#: ``Any`` so the parameters keep the honest annotation
#: ``Optional[NVFP44Over6Config]``.  Never pass it explicitly; omit the
#: argument instead.
_UNSET: Any = _UnsetType()


@dataclass(frozen=True)
class NVFP44Over6Config:
    """NVFP4 4over6 configuration shared by Python drivers and kernels.

    Parameters
    ----------
    e4m3_max : int
        Upper bound of the E4M3 block scale, either ``448`` (full range) or
        ``256``.  Also pins the per-tensor global scale to
        ``1 / (e4m3_max * 6)``.
    err_mode : NVFP44Over6ErrMode or str
        Error metric used to choose between the two scale candidates.  A
        string (``"MAE"`` / ``"MSE"``) is normalized to the enum member.
    err_use_fast_math : bool
        Evaluate the candidate error in fp16 rather than exactly.
    """

    e4m3_max: int = 448
    err_mode: Union[NVFP44Over6ErrMode, str] = NVFP44Over6ErrMode.MAE
    err_use_fast_math: bool = False

    def __post_init__(self) -> None:
        if self.e4m3_max not in (256, 448):
            raise ValueError("NVFP4 4over6 E4M3 max must be either 256 or 448.")
        try:
            if isinstance(self.err_mode, str):
                err_mode = NVFP44Over6ErrMode[self.err_mode.upper()]
            else:
                err_mode = NVFP44Over6ErrMode(self.err_mode)
        except (KeyError, ValueError):
            raise ValueError("NVFP4 4over6 error mode must be MAE or MSE.") from None
        object.__setattr__(self, "err_mode", err_mode)

    @property
    def err_mode_name(self) -> str:
        if isinstance(self.err_mode, str):
            return self.err_mode.upper()
        return self.err_mode.name

    def __repr__(self) -> str:
        # Non-default fields only, matching MoEFinalizeConfig / ExecutionConfig
        # in flashinfer/fused_moe/api.py.  Read off ``self`` rather than the
        # class so the pytest-id subclass in tests/utils/test_fp4_quantize.py
        # does not misreport itself.
        defaults = {f.name: f.default for f in fields(NVFP44Over6Config)}
        parts = []
        if self.e4m3_max != defaults["e4m3_max"]:
            parts.append(f"e4m3_max={self.e4m3_max!r}")
        if self.err_mode != defaults["err_mode"]:
            parts.append(f"err_mode={self.err_mode!r}")
        if self.err_use_fast_math != defaults["err_use_fast_math"]:
            parts.append(f"err_use_fast_math={self.err_use_fast_math!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


#: Wire codes for the packed recipe crossing the torch-custom-op / TVM-FFI
#: boundary.  Mirrored in csrc/nv_internal/tensorrt_llm/kernels/nvfp4Recipe.h.
NVFP4_4OVER6_CODE_FROM_ENV = -1
NVFP4_4OVER6_CODE_STANDARD = 0

#: Number of bits :func:`nvfp4_4over6_code` defines, mirroring
#: ``kNVFP44Over6KnownBits`` in
#: csrc/nv_internal/tensorrt_llm/kernels/nvfp4Recipe.h.  Widening the wire
#: format means bumping both together: both decoders reject codes with bits
#: above this set, so a newer producer fails loudly against an older consumer
#: instead of having its extra bits truncated.
NVFP4_4OVER6_KNOWN_BITS = 5


def env_flag_enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


def current_nvfp4_4over6_config() -> NVFP44Over6Config | None:
    """Build the 4over6 recipe from the ``FLASHINFER_NVFP4_4OVER6*`` env vars.

    This is what an omitted ``nvfp4_4over6=`` argument resolves to.  Prefer
    :func:`resolve_nvfp4_4over6`, which routes here only when the caller did
    not pass a recipe.
    """
    if not env_flag_enabled("FLASHINFER_NVFP4_4OVER6"):
        return None

    return NVFP44Over6Config(
        e4m3_max=256
        if env_flag_enabled("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256")
        else 448,
        err_mode=os.environ.get("FLASHINFER_NVFP4_4OVER6_ERR_MODE", "MAE"),
        err_use_fast_math=env_flag_enabled("FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH"),
    )


def nvfp4_4over6_fp8_input_error(input_dtype: torch.dtype) -> ValueError:
    """Build the error for a recipe an FP8-input quantizer cannot honor.

    Every FP8->FP4 path hardcodes the standard recipe -- the ``FP8_TO_FP4``
    branch of ``invokeFP4Quantization``
    (csrc/nv_internal/cpp/kernels/quantization.cu) and the CuTe-DSL
    ``process_nvfp4_block_fp8`` -- so the only alternative to refusing is
    pairing the caller's global scale with a 448 clamp it was not built for.
    Shared, so the CUDA and CuTe-DSL backends of one public API refuse in the
    same words.
    """
    return ValueError(
        "NVFP4 4over6 (from the nvfp4_4over6= argument or the "
        "FLASHINFER_NVFP4_4OVER6* environment) requires fp16 or bf16 input, "
        f"got {input_dtype}."
    )


def resolve_nvfp4_4over6(
    setting: Optional[NVFP44Over6Config] = _UNSET,
) -> NVFP44Over6Config | None:
    """Resolve a public ``nvfp4_4over6=`` argument to the recipe the kernels use.

    This is the *only* place the ``FLASHINFER_NVFP4_4OVER6*`` environment
    variables are consulted on the Python side, so precedence is decided
    exactly once.

    Parameters
    ----------
    setting : NVFP44Over6Config or None
        Omitted (the default): read the environment; when it enables 4over6
        a ``FutureWarning`` points at the argument, because the environment
        variables are a compatibility shim.  ``None``: 4over6
        off, environment ignored.  An :class:`NVFP44Over6Config`: on with
        exactly that recipe, environment ignored (no per-field merge).

    Returns
    -------
    NVFP44Over6Config or None
        The resolved recipe, where ``None`` means 4over6 is off.  This is the
        value every kernel driver below the public boundary expects.
    """
    if setting is _UNSET:
        config = current_nvfp4_4over6_config()
        if config is not None:
            # FutureWarning, not DeprecationWarning: the default filters hide
            # DeprecationWarning unless it is attributed to __main__, and this
            # one has to reach the framework maintainers who set the variable.
            warnings.warn(
                "The FLASHINFER_NVFP4_4OVER6* environment variables are "
                f"deprecated; pass nvfp4_4over6={config!r} explicitly instead.",
                FutureWarning,
                stacklevel=3,
            )
        return config
    if setting is None:
        return None
    if isinstance(setting, NVFP44Over6Config):
        if type(setting) is not NVFP44Over6Config:
            # Canonicalize subclasses (e.g. the pytest-id subclass in
            # tests/utils/test_fp4_quantize.py): a subclass compares and hashes
            # differently, which would split the @functools.cache keyed kernel
            # compilations and the on-disk CuTe-DSL artifacts.
            return NVFP44Over6Config(
                e4m3_max=setting.e4m3_max,
                err_mode=setting.err_mode,
                err_use_fast_math=setting.err_use_fast_math,
            )
        return setting
    raise TypeError(
        "nvfp4_4over6 must be an NVFP44Over6Config or None (or omitted); got "
        f"{type(setting).__name__}."
    )


def nvfp4_4over6_cache_key(config: NVFP44Over6Config | None) -> str:
    """Canonical short token for a **resolved** recipe.

    Used both by the CuTe-DSL kernel-specialization name and by the MoE
    autotuner's cache-key extras, so one recipe cannot be served a kernel or a
    tactic tuned for another.
    """
    if config is None:
        return "off"
    return (
        f"4over6_{config.e4m3_max}_{config.err_mode_name}"
        f"_{int(config.err_use_fast_math)}"
    )


def nvfp4_4over6_code(config: NVFP44Over6Config | None) -> int:
    """Pack a **resolved** recipe into the int64 the custom ops / TVM-FFI take.

    Deliberately typed on the resolved two-state value, so FlashInfer's own
    Python cannot emit :data:`NVFP4_4OVER6_CODE_FROM_ENV` — a type-level proof
    that the environment is consulted only in :func:`resolve_nvfp4_4over6`.

    Layout: bit0 enabled, bit1 ``e4m3_max == 256``, bits2-3 ``err_mode``,
    bit4 ``err_use_fast_math``.
    """
    if config is None:
        return NVFP4_4OVER6_CODE_STANDARD
    return (
        1
        | (int(config.e4m3_max == 256) << 1)
        | (int(config.err_mode) << 2)
        | (int(config.err_use_fast_math) << 4)
    )


def nvfp4_4over6_from_code(code: int) -> Optional[NVFP44Over6Config]:
    """Inverse of :func:`nvfp4_4over6_code`, for logs, repro tooling and tests.

    ``-1`` (:data:`NVFP4_4OVER6_CODE_FROM_ENV`) decodes to the "argument
    omitted" sentinel, whose ``repr`` names the environment variables.

    Mirrors ``resolveNVFP4Recipe`` in
    csrc/nv_internal/tensorrt_llm/kernels/nvfp4Recipe.h *including its
    rejections*, and in the same order.  Both decoders read the same wire
    format off publicly callable ops, so a code one accepts and the other
    rejects -- or worse, silently reinterprets -- would make a Python-side
    log or repro disagree with the kernel that actually ran.  Bump the two
    together.

    Raises
    ------
    ValueError
        If ``code`` is not one :func:`nvfp4_4over6_code` can produce, nor
        :data:`NVFP4_4OVER6_CODE_FROM_ENV`.
    """
    if code < 0:
        if code != NVFP4_4OVER6_CODE_FROM_ENV:
            raise ValueError(
                f"Unsupported NVFP4 4over6 code {code}: the only negative code "
                f"is {NVFP4_4OVER6_CODE_FROM_ENV} (read the environment)."
            )
        return _UNSET
    if code >> NVFP4_4OVER6_KNOWN_BITS:
        raise ValueError(f"Unsupported NVFP4 4over6 code {code}: unknown bits set.")
    if not code & 1:
        # 4over6 off.  Covers NVFP4_4OVER6_CODE_STANDARD.
        return None
    # Validated rather than handed to NVFP44Over6ErrMode(), so the two spare
    # encodings report the same "expected 0=MAE or 1=MSE" as the C++ decoder
    # instead of a bare enum-lookup failure.
    err_mode_bits = (code >> 2) & 0x3
    if err_mode_bits not in (0, 1):
        raise ValueError(
            f"Unsupported NVFP4 4over6 error mode {err_mode_bits} "
            "(expected 0=MAE or 1=MSE)."
        )
    return NVFP44Over6Config(
        e4m3_max=256 if code & 0x2 else 448,
        err_mode=NVFP44Over6ErrMode(err_mode_bits),
        err_use_fast_math=bool((code >> 4) & 0x1),
    )


#: Relative tolerance for checking that a per-token global scale was built from
#: the same recipe as the quantizer: loose enough for a float32 round trip of
#: ``1 / (448 * 6)``, far tighter than the 448 / 256 ratio it exists to catch.
NVFP4_PER_TOKEN_SCALE_RTOL = 1e-3


def nvfp4_per_token_scale_inv(nvfp4_4over6_config: NVFP44Over6Config | None) -> float:
    """The per-token inverse global scale a recipe implies: ``1 / (e4m3_max * 6)``."""
    return 1.0 / (nvfp4_e4m3_max(nvfp4_4over6_config) * FLOAT4_E2M1_MAX)


def nvfp4_e4m3_max(nvfp4_4over6_config: NVFP44Over6Config | None = None) -> float:
    """E4M3 block-scale clamp implied by a **resolved** recipe.

    ``None`` (the default) is standard NVFP4.  Unlike the quantizers, omitting
    the recipe here does *not* read the environment: the pre-existing contract
    of the scale helpers is that no recipe means the 448 clamp, and every
    caller passes the recipe it resolved for the matching quantize call.
    """
    if nvfp4_4over6_config is not None:
        return float(nvfp4_4over6_config.e4m3_max)
    return FLOAT8_E4M3_MAX


def make_nvfp4_global_scale(
    input_tensor: torch.Tensor,
    per_token_activation: bool,
    global_scale: float | None = None,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
) -> torch.Tensor:
    """Build the NVFP4 global scale implied by a **resolved** 4over6 recipe.

    The recipe used here **must** match the one the quantizer runs with: the
    E4M3 clamp appears both in this scale and in the kernel's candidate
    search, and a mismatch silently rescales the whole tensor.  Pass the same
    ``NVFP44Over6Config`` (or ``None``) here as ``nvfp4_4over6=`` on the
    quantize call; when the quantize call leaves ``nvfp4_4over6`` unset, pass
    ``resolve_nvfp4_4over6()`` here.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Tensor whose amax seeds the scale (per-tensor mode only).
    per_token_activation : bool
        Per-token activation mode, where the scale is the pure function
        ``1 / (e4m3_max * 6)``.
    global_scale : float, optional
        Explicit per-tensor scale, bypassing the amax reduction.
    nvfp4_4over6_config : NVFP44Over6Config or None
        Resolved recipe; ``None`` (the default) means 4over6 is off.  Like
        :func:`nvfp4_e4m3_max`, omitting it does not read the environment.
    """
    e4m3_max = nvfp4_e4m3_max(nvfp4_4over6_config)
    if per_token_activation:
        scale = nvfp4_per_token_scale_inv(nvfp4_4over6_config)
    elif global_scale is not None:
        scale = global_scale
    else:
        amax = input_tensor.abs().max().to(torch.float32)
        if amax == 0:
            return torch.full(
                (1,),
                torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=input_tensor.device,
            )
        return (e4m3_max * FLOAT4_E2M1_MAX / amax).reshape(1).to(input_tensor.device)

    return torch.tensor(
        [scale],
        dtype=torch.float32,
        device=input_tensor.device,
    )


def nvfp4_4over6_mode_label(
    per_token_activation: bool, nvfp4_4over6_config: NVFP44Over6Config | None
) -> str:
    parts = []
    if per_token_activation:
        parts.append("per-token")
    if nvfp4_4over6_config is not None:
        parts.append(f"4over6-{nvfp4_4over6_config.err_mode_name.lower()}")
        parts.append(f"e4m3-{nvfp4_4over6_config.e4m3_max}")
        parts.append(
            "err-fastmath" if nvfp4_4over6_config.err_use_fast_math else "err-exactmath"
        )
    return ", ".join(parts) if parts else "standard"


__all__ = [
    "FLOAT4_E2M1_MAX",
    "FLOAT8_E4M3_MAX",
    "NVFP4_4OVER6_CODE_FROM_ENV",
    "NVFP4_4OVER6_CODE_STANDARD",
    "NVFP4_4OVER6_KNOWN_BITS",
    "NVFP4_PER_TOKEN_SCALE_RTOL",
    "NVFP44Over6Config",
    "NVFP44Over6ErrMode",
    "current_nvfp4_4over6_config",
    "env_flag_enabled",
    "make_nvfp4_global_scale",
    "nvfp4_4over6_cache_key",
    "nvfp4_4over6_code",
    "nvfp4_4over6_fp8_input_error",
    "nvfp4_4over6_from_code",
    "nvfp4_4over6_mode_label",
    "nvfp4_e4m3_max",
    "nvfp4_per_token_scale_inv",
    "resolve_nvfp4_4over6",
]
