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
"""

"""Availability probes for the optional CuTe DSL stack.

These probes answer "is the CuTe DSL installed, and can it target this
device?" so callers can dispatch to a CuTe-DSL backend or fall back to a
plain-CUDA one. They must therefore be importable *without* the DSL: this
module imports nothing from ``cutlass`` at module scope, and the few
functions that need the DSL's own metadata import it lazily inside a
``try``. Keep it that way -- a top-level ``import cutlass`` here would make
the guard depend on the thing it guards, and ``import flashinfer`` would
hard-require ``nvidia-cutlass-dsl`` again.

``flashinfer.cute_dsl.utils`` re-exports everything defined here for
backwards compatibility, but that module *does* require the DSL. Import
from this module on any path that runs before the DSL is known present.
"""

import functools
import importlib.util
import os
from typing import Optional

__all__ = [
    "is_cute_dsl_available",
    "is_rubin_cute_dsl_available",
    "is_cute_dsl_experimental_available",
    "is_cute_dsl_arch_supported",
    "cute_dsl_compile_arch",
    "require_cute_dsl_arch",
]


def is_cute_dsl_available() -> bool:
    r"""Return ``True`` when the optional CuTe DSL stack is importable.

    Probes for ``cutlass`` and ``cutlass.cute`` via :func:`importlib.util.find_spec`.
    Used by higher-level wrappers to decide whether to dispatch to a CuTe-DSL
    backend (e.g. :func:`flashinfer.quantization.mxfp4_quantize`,
    :class:`flashinfer.cute_dsl.attention.wrappers.BatchDecodeCuteDSLWrapper`)
    or fall back to a plain-CUDA implementation.

    Returns
    -------
    bool
        ``True`` if both ``cutlass`` and ``cutlass.cute`` are importable in the
        current Python environment.
    """
    return (
        importlib.util.find_spec("cutlass") is not None
        and importlib.util.find_spec("cutlass.cute") is not None
    )


@functools.cache
def is_rubin_cute_dsl_available() -> bool:
    r"""Return ``True`` when the installed CuTe DSL exposes the Rubin (SM107) API.

    The SM107 kernels are built on ``cutlass.utils.rubin_helpers``, which is only
    present from CuTe DSL 4.8 onwards. FlashInfer continues to support older DSL
    releases, so those kernels are imported lazily and this probe decides whether
    they are offered at all: on an older DSL the rest of the package still works
    and only the SM107 CuTe DSL paths are unavailable.

    Returns
    -------
    bool
        ``True`` if ``cutlass.utils.rubin_helpers`` is importable.
    """
    return (
        is_cute_dsl_available()
        and importlib.util.find_spec("cutlass.utils.rubin_helpers") is not None
    )


@functools.cache
def is_cute_dsl_experimental_available() -> bool:
    r"""Return ``True`` when the installed CuTe DSL exposes ``cutlass.experimental``.

    The namespace landed in CuTe DSL 4.7, while FlashInfer's dependency floor is
    4.6.2, so kernels built on it probe before importing and an older DSL loses
    only those kernels.

    Returns
    -------
    bool
        ``True`` if ``cutlass.experimental`` is importable.
    """
    return (
        is_cute_dsl_available()
        and importlib.util.find_spec("cutlass.experimental") is not None
    )


@functools.cache
def is_cute_dsl_arch_supported(
    major: int, minor: int, native_only: bool = False
) -> bool:
    r"""Return ``True`` when the installed CuTe DSL can target compute
    capability ``(major, minor)``.

    :func:`is_cute_dsl_available` only checks that the package is importable;
    the installed DSL may still lack the *device's* architecture (its
    ``cutlass.base_dsl.arch.Arch`` enum resolves names like ``sm_107a`` and
    raises ``KeyError`` for unknown members — e.g. a DSL release that
    predates the device). Dispatchers must consult this before selecting a
    CuTe-DSL backend so unsupported devices get a clean fallback/skip instead
    of a ``KeyError`` from deep inside kernel compilation.

    When the device's own architecture is missing but the DSL has the
    family-conditional target for its major line (e.g. ``sm_100f`` for an
    sm_107 device), kernels restricted to family-portable features still
    compile and run correctly; this probe then pins the DSL's default target
    via the ``CUTE_DSL_ARCH`` environment variable and reports the arch
    supported. Pass ``native_only=True`` for kernels that require
    architecture-specific instructions (e.g. block-scaled ``tcgen05.mma``
    kinds, which the DSL only accepts for ``sm_100a``/``sm_103a`` targets).
    """
    if not is_cute_dsl_available():
        return False
    try:
        from cutlass.base_dsl.arch import Arch

        for name in (f"sm_{major}{minor}a", f"sm_{major}{minor}"):
            try:
                Arch[name]
                return True
            except KeyError:
                continue
        if native_only:
            return False
        family = _family_fallback_arch(major, minor)
        if family is not None:
            # The device's native arch is absent, but the DSL has the
            # family-conditional target (e.g. ``sm_100f`` for an sm_107
            # device); family-portable kernels compile and run correctly
            # when the DSL targets it. Ground truth is the target the DSL
            # captured at first ``cutlass`` import (from ``CUTE_DSL_ARCH``),
            # not ``os.environ`` now: an env var set after that import does
            # not retarget the DSL, so checking the environment here would
            # report supported while ``cute.compile`` still fails.
            target = _dsl_captured_arch()
            if target is None:
                # Internal API unavailable: fall back to the env-var proxy.
                target = os.environ.get("CUTE_DSL_ARCH", "")
            if target.replace("_", "").lower() == family.replace("_", "").lower():
                return True
        return False
    except Exception:
        # Arch module layout changed or import failed: fall back to
        # "package available" semantics rather than disabling the backend.
        return True


def _family_fallback_arch(major: int, minor: int) -> Optional[str]:
    r"""Return the DSL's family-conditional target covering ``(major,
    minor)`` (e.g. ``"sm_100f"`` for an sm_107 device), or ``None``."""
    try:
        from cutlass.base_dsl.arch import Arch

        name = f"sm_{major}0f"
        Arch[name]
        return name
    except Exception:
        return None


def _dsl_captured_arch() -> Optional[str]:
    r"""Return the compile target the DSL captured when ``cutlass`` was first
    imported (e.g. ``"sm_100f"`` when ``CUTE_DSL_ARCH`` was exported before
    the process started), or ``None`` if the internal API is unavailable."""
    try:
        from cutlass.cutlass_dsl import CuTeDSL

        return str(CuTeDSL._get_dsl().envar.arch)
    except Exception:
        return None


def cute_dsl_compile_arch(major: int, minor: int) -> str:
    r"""Return the arch name to pass to ``cute.GPUArch`` for a device.

    Prefers the device's own arch (e.g. ``sm_107a``). When the installed DSL
    predates the device but is targeting the family-conditional arch because
    the user exported ``CUTE_DSL_ARCH=sm_100f``, returns that instead --
    handing it ``sm_107a`` would raise ``KeyError`` from inside the DSL's
    ``Arch`` enum, which is the failure this exists to prevent.

    Raises :class:`NotImplementedError` when neither is available, so callers
    fail with an actionable message instead of a bare ``KeyError``.
    """
    from cutlass.base_dsl.arch import Arch

    for name in (f"sm_{major}{minor}a", f"sm_{major}{minor}"):
        try:
            Arch[name]
            return name
        except KeyError:
            continue
    # Native arch absent. is_cute_dsl_arch_supported returns True here only
    # when the DSL is already targeting the family arch, which is exactly when
    # compiling for it is valid.
    if is_cute_dsl_arch_supported(major, minor):
        family = _family_fallback_arch(major, minor)
        if family is not None:
            return family
    raise NotImplementedError(
        f"the installed CuTe DSL cannot target sm_{major}{minor}; export "
        f"CUTE_DSL_ARCH=sm_{major}0f before starting the process to build "
        f"family-portable kernels on this device"
    )


def require_cute_dsl_arch(device, native_only: bool = False) -> None:
    r"""Raise :class:`NotImplementedError` when the installed CuTe DSL cannot
    target ``device``'s architecture (see :func:`is_cute_dsl_arch_supported`)."""
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    if not is_cute_dsl_arch_supported(major, minor, native_only=native_only):
        hint = ""
        if not native_only:
            family = _family_fallback_arch(major, minor)
            if family is not None:
                hint = (
                    f"; family-portable CuTe-DSL kernels can run on this device "
                    f"when CUTE_DSL_ARCH={family} is exported in the environment "
                    f"before the process starts (see the release notes)"
                )
        raise NotImplementedError(
            f"the installed CuTe DSL does not support sm_{major}{minor} on this device{hint}"
        )
