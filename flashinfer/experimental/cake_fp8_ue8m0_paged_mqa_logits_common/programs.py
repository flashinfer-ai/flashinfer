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

# Loading the Cake-generated CuTe DSL programs of the FP8 / UE8M0 paged
# MQA-logits backends: one program per architecture (and, for decode, per
# next_n) under each backend's csrc/, imported from its file and compiled for
# the device's exact architecture through FlashInfer's CuTe DSL JIT cache.

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch

from ...utils import get_compute_capability

ARCHES = {(10, 3): "sm_103a", (10, 7): "sm_107a"}
SUPPORTED_COMPUTE_CAPABILITIES = tuple(ARCHES)


def program_arch(device_index: int) -> str:
    """The architecture of the device's programs."""
    arch = ARCHES[get_compute_capability(torch.device("cuda", device_index))]
    # The programs name their architecture; the JIT cache labels its module
    # directory from CUTE_DSL_ARCH when set.
    requested = os.environ.get("CUTE_DSL_ARCH")
    if requested and requested.replace("_", "") != arch.replace("_", ""):
        raise ValueError(
            f"CUTE_DSL_ARCH={requested} does not match the device architecture {arch}"
        )
    return arch


def program_files(csrc: Path, pattern: str) -> tuple[str, ...]:
    """Every generated program of a backend: its programs share one JIT module
    directory, whose cache key must cover all of them."""
    return tuple(str(path) for path in sorted(csrc.glob(pattern)))


def import_program(path: Path, owner: str) -> ModuleType:
    """The generated program at ``path``, named after its backend module
    ``owner``."""
    spec = importlib.util.spec_from_file_location(
        f"{owner}._generated_{path.stem}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_launcher(path: Path, launcher: Callable, parameters: tuple[str, ...]) -> None:
    """Raise unless the program's launcher takes exactly ``parameters``, the
    arguments its backend binds, in that order."""
    found = tuple(inspect.signature(launcher).parameters)
    if found != parameters:
        raise RuntimeError(
            f"{path.name} launches with {found}; this adapter binds {parameters}"
        )


def on_device(device_index: int) -> contextlib.AbstractContextManager:
    """The TVM-FFI environment stream is the current device's: make the
    tensors' device current for the launch, unless it already is."""
    if device_index == torch.cuda.current_device():
        return contextlib.nullcontext()
    return torch.cuda.device(device_index)
