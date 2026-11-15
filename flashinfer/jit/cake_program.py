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

Runtime of the generated Cake in-tree loaders.

A generated loader (``flashinfer/jit/cake_<op>.py``) publishes one
:class:`CakeProgram` constant per exported program: the two CUDA translation
units under ``csrc/<operator_dir>/`` (the device source and its tvm-ffi
binding), the exact targets the program ships for, the compile options of
each target, the binding's argument order and the launch it performs.  The
constants are data; this module turns one of them into a FlashInfer JIT
specification (:meth:`CakeProgram.spec`), builds and loads it
(:meth:`CakeProgram.load`) and lists the specifications an ahead-of-time
build registers (:func:`aot_specs`).

A program is compiled for exactly one target per library.  The module name
is ``<program>_<target>_<digest>``, the digest being the first twenty hex
digits of the SHA-256 over the bytes of the two sources in their listed
order followed by the compact JSON of the target and its compile options,
so a changed source or option never reuses a stale library and two targets
never share one.  The build opts out of FlashInfer's default fast math and
passes ``--fmad=false`` followed by the target's recorded compile options,
which reproduces the numerics the sources were validated with.

The Cake exporter writes this file byte-identically into FlashInfer; do not
edit it in place.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
    sm110a_nvcc_flags,
    sm120a_nvcc_flags,
    sm121a_nvcc_flags,
)

__all__ = [
    "CAPABILITY_TARGET",
    "DIGEST_DIGITS",
    "GRID_ARGS",
    "TARGET_FLAGS",
    "CakeProgram",
    "aot_specs",
    "content_digest",
    "operator_csrc_dir",
]

#: The nvcc flag set of each exact target a program may ship for.
TARGET_FLAGS: dict[str, Sequence[str]] = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
    "sm_110a": sm110a_nvcc_flags,
    "sm_120a": sm120a_nvcc_flags,
    "sm_121a": sm121a_nvcc_flags,
}
#: The exact target of a device's compute capability.
CAPABILITY_TARGET: dict[tuple[int, int], str] = {
    (9, 0): "sm_90a",
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
    (11, 0): "sm_110a",
    (12, 0): "sm_120a",
    (12, 1): "sm_121a",
}
#: The three grid arguments every binding takes after its host parameters.
GRID_ARGS: tuple[str, str, str] = ("grid_x", "grid_y", "grid_z")
#: Hex digits of the content digest in a module name.
DIGEST_DIGITS = 20
_FMAD_OFF = "--fmad=false"


def operator_csrc_dir(operator_dir: str) -> Path:
    """The directory holding the sources of ``operator_dir``: the installed
    ``csrc`` data of the wheel, else the checkout's ``csrc/``."""
    installed = jit_env.FLASHINFER_CSRC_DIR / operator_dir
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / operator_dir
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        f"the Cake sources of {operator_dir!r} were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def content_digest(
    kernel: bytes, binding: bytes, target: str, options: Sequence[str]
) -> str:
    """The first :data:`DIGEST_DIGITS` hex digits of the SHA-256 over the kernel bytes, the binding bytes and
    ``{"options": [...], "target": "..."}`` as compact sorted JSON, in that order and without separators."""
    digest = hashlib.sha256()
    digest.update(kernel)
    digest.update(binding)
    identity = {"options": list(options), "target": target}
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()[:DIGEST_DIGITS]


@dataclass(frozen=True)
class CakeProgram:
    """One exported program: its sources, targets, options, argument order and launch.

    Every field is a string, number, boolean or tuple of those, so a constant
    is hashable (the caches below key on it) and a generated loader spells it
    as literals.  ``compile_options`` pairs each target with its options;
    ``args`` are the binding's host parameters in call order followed by the
    three grid arguments, and ``tensors`` the subset passed as tensors.  The
    launch fields (``block``, ``cluster``, ``dynamic_smem_bytes``, ``use_pdl``,
    ``cooperative``) state the launch the binding performs; the grid is the
    caller's, through the grid arguments.  The host module that owns the
    operation assembles the positional call from ``args`` once, at
    preparation, and calls ``entry(target)`` with it; nothing here interprets
    an argument plan per call.
    """

    name: str
    operator_dir: str
    kernel: str
    sources: tuple[str, str]
    targets: tuple[str, ...]
    compile_options: tuple[tuple[str, tuple[str, ...]], ...]
    args: tuple[str, ...]
    tensors: tuple[str, ...]
    block: tuple[int, int, int]
    cluster: tuple[int, int, int]
    dynamic_smem_bytes: int
    use_pdl: bool
    cooperative: bool
    needs_device_linking: bool

    def options_for(self, target: str) -> tuple[str, ...]:
        """The recorded compile options of ``target``; a target the program does not ship raises ``KeyError``."""
        if target not in self.targets:
            raise KeyError(
                f"{self.name} ships no {target!r} build; its targets are {list(self.targets)}"
            )
        return dict(self.compile_options)[target]

    def target_for(self, capability: tuple[int, int]) -> str | None:
        """The exact target this program ships for a device of ``capability``, or ``None``."""
        target = CAPABILITY_TARGET.get((int(capability[0]), int(capability[1])))
        return target if target in self.targets else None

    def source_paths(self) -> tuple[Path, ...]:
        """The two translation units, under :func:`operator_csrc_dir`."""
        root = operator_csrc_dir(self.operator_dir)
        return tuple(root / source for source in self.sources)

    def digest(self, target: str) -> str:
        """The content digest of the two sources and the target's options (the sources are read once per
        process)."""
        kernel, binding = _source_bytes(self)
        return content_digest(kernel, binding, target, self.options_for(target))

    def module_name(self, target: str) -> str:
        """``<name>_<target>_<digest>``: the JIT module of this program on ``target``."""
        return f"{self.name}_{target}_{self.digest(target)}"

    def spec(self, target: str) -> JitSpec:
        """The JIT specification of this program on ``target`` (one per process)."""
        return _spec(self, target)

    def load(self, target: str) -> Any:
        """The built and loaded tvm-ffi module of this program on ``target`` (one per process)."""
        return _load(self, target)

    def entry(self, target: str) -> Any:
        """The binding's ``run`` function on ``target``."""
        return self.load(target).run

    def launch_record(self) -> dict[str, Any]:
        """The launch the binding performs, as the exporter records it."""
        return {
            "block": self.block,
            "cluster": self.cluster,
            "cooperative": self.cooperative,
            "dynamic_smem_bytes": self.dynamic_smem_bytes,
            "use_pdl": self.use_pdl,
        }


@functools.cache
def _source_bytes(program: CakeProgram) -> tuple[bytes, bytes]:
    kernel, binding = program.source_paths()
    return kernel.read_bytes(), binding.read_bytes()


@functools.cache
def _spec(program: CakeProgram, target: str) -> JitSpec:
    options = program.options_for(target)
    return gen_jit_spec(
        name=program.module_name(target),
        sources=list(program.source_paths()),
        extra_cuda_cflags=[*TARGET_FLAGS[target], _FMAD_OFF, *options],
        extra_include_paths=[operator_csrc_dir(program.operator_dir)],
        needs_device_linking=program.needs_device_linking,
        use_fast_math=False,
    )


@functools.cache
def _load(program: CakeProgram, target: str) -> Any:
    return program.spec(target).build_and_load()


def aot_specs(programs: Sequence[CakeProgram], target: str) -> tuple[JitSpec, ...]:
    """The JIT specifications of every program of ``programs`` shipping ``target``, in order."""
    if target not in TARGET_FLAGS:
        raise ValueError(
            f"unknown target {target!r}; the targets are {sorted(TARGET_FLAGS)}"
        )
    return tuple(
        program.spec(target) for program in programs if target in program.targets
    )
