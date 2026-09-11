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
from enum import Enum, IntEnum
import os
from typing import Union

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


class NVFP4Recipe(Enum):
    """The two NVFP4 scale-selection states that are not a 4over6 recipe.

    A 4over6 setting has *three* states and ``None`` cannot spell all of
    them: internally ``None`` has always meant "4over6 disabled", while a
    public parameter's default must mean "derive from the environment" to
    stay backwards compatible.  These two members name the states that are
    not an :class:`NVFP44Over6Config`:

    ``FROM_ENV``
        Read ``FLASHINFER_NVFP4_4OVER6`` and friends at call time.  This is
        the default everywhere and is byte-for-byte the behaviour that
        predates the ``nvfp4_4over6=`` parameter.
    ``STANDARD``
        Plain NVFP4.  The environment is not consulted at all.

    An ``Enum`` rather than a bespoke sentinel: singleton identity,
    ``pickle``, ``copy.deepcopy``, hashing and static narrowing come for
    free, and ``.value`` is a JSON token so a framework-level quantization
    config round-trips through ``json.dumps`` without a custom encoder.
    """

    FROM_ENV = "from_env"
    STANDARD = "standard"

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


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


#: Public three-state 4over6 setting.  ``NVFP4Recipe.FROM_ENV`` (``None`` is an
#: alias) derives the recipe from the environment, ``NVFP4Recipe.STANDARD``
#: turns 4over6 off, and an :class:`NVFP44Over6Config` turns it on with exactly
#: that recipe.  Below :func:`resolve_nvfp4_4over6` everything keeps the
#: two-state ``NVFP44Over6Config | None`` convention where ``None`` means off.
NVFP44Over6Setting = Union[NVFP4Recipe, NVFP44Over6Config, None]


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

    This is the implementation of :attr:`NVFP4Recipe.FROM_ENV`.  Prefer
    :func:`resolve_nvfp4_4over6`, which routes here only when the caller did
    not pin a recipe.
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


def nvfp4_4over6_is_from_env(setting: NVFP44Over6Setting) -> bool:
    """Whether ``setting`` defers to the environment (``FROM_ENV`` or ``None``)."""
    return setting is None or setting is NVFP4Recipe.FROM_ENV


def nvfp4_4over6_fp8_input_error(
    setting: NVFP44Over6Setting, input_dtype: torch.dtype
) -> ValueError:
    """Build the error for a recipe an FP8-input quantizer cannot honor.

    Every FP8->FP4 path hardcodes the standard recipe -- the ``FP8_TO_FP4``
    branch of ``invokeFP4Quantization``
    (csrc/nv_internal/cpp/kernels/quantization.cu) and the CuTe-DSL
    ``process_nvfp4_block_fp8`` -- so the only alternative to refusing is
    pairing the caller's global scale with a 448 clamp it was not built for.
    Shared, so the CUDA and CuTe-DSL backends of one public API refuse in the
    same words.

    Takes the **unresolved** ``setting``: below :func:`resolve_nvfp4_4over6` an
    environment-derived recipe is indistinguishable from an explicitly passed
    one, and sending a caller to look at an argument they never passed is a
    dead end.
    """
    source = (
        "FLASHINFER_NVFP4_4OVER6"
        if nvfp4_4over6_is_from_env(setting)
        else "the nvfp4_4over6= argument"
    )
    return ValueError(
        f"NVFP4 4over6 (requested via {source}) requires fp16 or bf16 input, "
        f"got {input_dtype}."
    )


def resolve_nvfp4_4over6(
    setting: NVFP44Over6Setting = NVFP4Recipe.FROM_ENV,
) -> NVFP44Over6Config | None:
    """Collapse the public three-state setting to the internal two-state one.

    This is the *only* place the ``FLASHINFER_NVFP4_4OVER6*`` environment
    variables are consulted on the Python side, so precedence is decided
    exactly once.

    Parameters
    ----------
    setting : NVFP4Recipe, NVFP44Over6Config or None
        ``NVFP4Recipe.FROM_ENV`` (or ``None``) reads the environment;
        ``NVFP4Recipe.STANDARD`` returns ``None`` without reading it; an
        :class:`NVFP44Over6Config` is returned as-is.

    Returns
    -------
    NVFP44Over6Config or None
        The resolved recipe, where ``None`` means 4over6 is off.  This is the
        value every kernel driver below the public boundary expects.
    """
    if nvfp4_4over6_is_from_env(setting):
        return current_nvfp4_4over6_config()
    if setting is NVFP4Recipe.STANDARD:
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
        "nvfp4_4over6 must be NVFP4Recipe.FROM_ENV, NVFP4Recipe.STANDARD, an "
        f"NVFP44Over6Config, or None; got {type(setting).__name__}."
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


def nvfp4_4over6_from_code(code: int) -> NVFP44Over6Setting:
    """Inverse of :func:`nvfp4_4over6_code`, for logs, repro tooling and tests.

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
                f"is {NVFP4_4OVER6_CODE_FROM_ENV} (FROM_ENV)."
            )
        return NVFP4Recipe.FROM_ENV
    if code >> NVFP4_4OVER6_KNOWN_BITS:
        raise ValueError(f"Unsupported NVFP4 4over6 code {code}: unknown bits set.")
    if not code & 1:
        # STANDARD: 4over6 off.  Covers NVFP4_4OVER6_CODE_STANDARD.
        return NVFP4Recipe.STANDARD
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


def _resolve_for_scale(
    nvfp4_4over6_config: NVFP44Over6Config | None,
    nvfp4_4over6: NVFP44Over6Setting,
) -> NVFP44Over6Config | None:
    """Reconcile the legacy resolved parameter with the three-state one.

    ``nvfp4_4over6`` defaults to ``NVFP4Recipe.STANDARD`` rather than
    ``FROM_ENV`` on the scale helpers: their pre-existing contract is that
    ``nvfp4_4over6_config=None`` means "no 4over6", and silently promoting
    that to an environment read would change the scale of every existing
    caller.
    """
    if nvfp4_4over6 is NVFP4Recipe.STANDARD:
        return nvfp4_4over6_config
    if nvfp4_4over6_config is not None:
        raise ValueError(
            "pass either nvfp4_4over6_config (resolved) or nvfp4_4over6 "
            "(three-state), not both."
        )
    return resolve_nvfp4_4over6(nvfp4_4over6)


def nvfp4_e4m3_max(
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
    *,
    nvfp4_4over6: NVFP44Over6Setting = NVFP4Recipe.STANDARD,
) -> float:
    """E4M3 block-scale clamp implied by a recipe.

    Parameters
    ----------
    nvfp4_4over6_config : NVFP44Over6Config or None
        Legacy **resolved** recipe; ``None`` means 4over6 is off.
    nvfp4_4over6 : NVFP4Recipe, NVFP44Over6Config or None
        Three-state public spelling.  Pass exactly one of the two.
    """
    config = _resolve_for_scale(nvfp4_4over6_config, nvfp4_4over6)
    if config is not None:
        return float(config.e4m3_max)
    return FLOAT8_E4M3_MAX


def make_nvfp4_global_scale(
    input_tensor: torch.Tensor,
    per_token_activation: bool,
    global_scale: float | None = None,
    nvfp4_4over6_config: NVFP44Over6Config | None = None,
    *,
    nvfp4_4over6: NVFP44Over6Setting = NVFP4Recipe.STANDARD,
) -> torch.Tensor:
    """Build the NVFP4 global scale implied by a 4over6 recipe.

    The recipe used here **must** match the one passed to the quantizer: the
    E4M3 clamp appears both in this scale and in the kernel's candidate
    search, and a mismatch silently rescales the whole tensor.  Pass the same
    ``nvfp4_4over6=`` value to both.

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
        Legacy **resolved** recipe; ``None`` means 4over6 is off.
    nvfp4_4over6 : NVFP4Recipe, NVFP44Over6Config or None
        Three-state public spelling.  Pass exactly one of the two.
    """
    config = _resolve_for_scale(nvfp4_4over6_config, nvfp4_4over6)
    e4m3_max = nvfp4_e4m3_max(config)
    if per_token_activation:
        scale = 1.0 / (e4m3_max * FLOAT4_E2M1_MAX)
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
    "NVFP44Over6Config",
    "NVFP44Over6ErrMode",
    "NVFP44Over6Setting",
    "NVFP4Recipe",
    "current_nvfp4_4over6_config",
    "env_flag_enabled",
    "make_nvfp4_global_scale",
    "nvfp4_4over6_cache_key",
    "nvfp4_4over6_code",
    "nvfp4_4over6_fp8_input_error",
    "nvfp4_4over6_from_code",
    "nvfp4_4over6_is_from_env",
    "nvfp4_4over6_mode_label",
    "nvfp4_e4m3_max",
    "resolve_nvfp4_4over6",
]
