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

"""Kernel registry of the ``specialized`` MoE backend.

Kernels are found, not listed: every subpackage of ``kernels/`` whose name does
not start with ``_`` must export ``KERNEL``, a :class:`SpecializedMoEKernel`
subclass whose ``name`` is the subpackage name. Adding a kernel therefore adds
one subpackage and edits no shared file. Kernel modules import no device code at
module scope, so discovery is cheap. A subpackage that fails to import or to
declare its kernel is skipped with a warning and recorded in
:func:`discovery_errors`; the other kernels keep working.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Type

from ....jit.core import logger
from . import kernels as _kernels_package
from .kernel import SpecializedMoEKernel

_discovery_errors: Dict[str, str] = {}


@functools.cache
def registered_kernels() -> Mapping[str, Type[SpecializedMoEKernel]]:
    """The registered kernel classes by name, in name order (read-only)."""
    found: Dict[str, Type[SpecializedMoEKernel]] = {}
    modules = sorted(
        pkgutil.iter_modules(_kernels_package.__path__), key=lambda info: info.name
    )
    for info in modules:
        if not info.ispkg or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{_kernels_package.__name__}.{info.name}")
            kernel = getattr(module, "KERNEL", None)
            if not (
                isinstance(kernel, type) and issubclass(kernel, SpecializedMoEKernel)
            ):
                raise TypeError("KERNEL is not a SpecializedMoEKernel subclass")
            if kernel.name != info.name:
                raise ValueError(
                    f"KERNEL.name {kernel.name!r} differs from its subpackage name"
                )
            if not (
                kernel.weight_view
                and kernel.quant_variants
                and kernel.routing_modes
                and kernel.activation_classes
                and kernel.compute_capabilities
            ):
                raise ValueError(
                    "KERNEL must declare weight_view, quant_variants, "
                    "routing_modes, activation_classes and compute_capabilities"
                )
        except Exception as exc:  # one broken kernel must not disable the others
            _discovery_errors[info.name] = f"{type(exc).__name__}: {exc}"[:2000]
            logger.warning(
                f"MoELayer specialized backend: skipping kernel {info.name!r} "
                f"({_discovery_errors[info.name]})"
            )
            continue
        found[info.name] = kernel
    return MappingProxyType(found)


def discovery_errors() -> Dict[str, str]:
    """Subpackages of ``kernels/`` that could not be registered, with the reason."""
    registered_kernels()
    return dict(_discovery_errors)


def supports_arch(arch: int) -> bool:
    """Whether any registered kernel runs on compute capability ``arch``."""
    return any(arch in k.compute_capabilities for k in registered_kernels().values())


def stats() -> Dict[str, Dict[str, Any]]:
    """Each registered kernel's dispatch and build diagnostics, by kernel name.

    Side-effect free like each kernel's own ``stats()``: nothing is built, no
    CUDA call is made and no file is read."""
    return {name: kernel.stats() for name, kernel in registered_kernels().items()}


def gen_specialized_moe_aot_modules(sm_capabilities: Mapping[str, bool]) -> List[Any]:
    """The JIT specs the registered kernels precompile for ``sm_capabilities``
    (called by ``flashinfer.aot``)."""
    specs: List[Any] = []
    for kernel in registered_kernels().values():
        specs.extend(kernel.aot_jit_specs(sm_capabilities))
    return specs
