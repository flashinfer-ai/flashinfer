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

# Program source of the Cake NVFP4 MegaMoE route: the generated loader
# ``flashinfer/jit/cake_mega_moe_nvfp4.py`` publishes one ``CakeProgram``
# constant per program of :data:`PROGRAM_IDENTIFIERS`, each for sm_107a. A
# program table resolves an identifier to a :class:`ProgramHandle`: the
# binding's argument names in call order (ending in the grid triple), its
# recorded launch and the launch callable. Another table (a test's) is
# installed with :func:`set_programs`. The loader is imported on first use.

from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import torch

from .cake_plan import GRID_ARGS, PROGRAM_IDENTIFIERS, STAGE

LOADER_MODULE = "flashinfer.jit.cake_mega_moe_nvfp4"
TARGET = "sm_107a"
CAPABILITY = (10, 7)


@dataclass(frozen=True)
class ProgramHandle:
    """One loaded program: ``args`` the binding's argument names in call
    order (ending in :data:`GRID_ARGS`); ``launch(*values)`` launches with the
    values in that order on the tvm-ffi environment stream."""

    identifier: str
    args: Tuple[str, ...]
    launch: Callable[..., None]
    cluster: Tuple[int, int, int] = (1, 1, 1)
    cooperative: bool = False
    use_pdl: bool = False

    def __post_init__(self) -> None:
        if tuple(self.args[-3:]) != GRID_ARGS:
            raise RuntimeError(
                f"{self.identifier}: the binding's arguments end with "
                f"{list(self.args[-3:])}, not {list(GRID_ARGS)}"
            )
        if len(set(self.args)) != len(self.args):
            raise RuntimeError(
                f"{self.identifier}: duplicate argument names in {self.args}"
            )

    @property
    def kernel_args(self) -> Tuple[str, ...]:
        return tuple(self.args[: -len(GRID_ARGS)])


class ProgramTable(Protocol):
    def handle(self, identifier: str, device: torch.device) -> ProgramHandle: ...

    def available(self) -> bool: ...


def _expected_launch(identifier: str) -> Dict[str, Any]:
    """The launch each program's binding must perform (the host's protocol)."""
    mega = identifier.startswith("MEGA_")
    return {
        "use_pdl": mega,
        "cooperative": identifier in ("ROUTER_T4096", "ROUTER_T8192"),
        "cluster": (2, 1, 1) if mega else (1, 1, 1),
    }


class ShippedPrograms:
    """The generated loader's ``CakeProgram`` constants, loaded once per
    identifier."""

    def __init__(self) -> None:
        self._handles: Dict[str, ProgramHandle] = {}

    @staticmethod
    @functools.cache
    def loader() -> Any:
        return importlib.import_module(LOADER_MODULE)

    def available(self) -> bool:
        try:
            loader = self.loader()
        except ImportError:
            return False
        return all(
            TARGET in getattr(getattr(loader, identifier, None), "targets", ())
            for identifier in PROGRAM_IDENTIFIERS
        )

    def handle(self, identifier: str, device: torch.device) -> ProgramHandle:
        from flashinfer.utils import get_compute_capability

        handle = self._handles.get(identifier)
        if handle is not None:
            return handle
        program = getattr(self.loader(), identifier, None)
        if program is None:
            raise RuntimeError(
                f"{LOADER_MODULE} publishes no program {identifier!r}; the route needs "
                f"every program of {list(PROGRAM_IDENTIFIERS)}"
            )
        capability = tuple(get_compute_capability(device))
        if capability != CAPABILITY or program.target_for(capability) != TARGET:
            raise RuntimeError(
                f"the Cake NVFP4 MegaMoE route runs on sm_107a; {identifier} ships "
                f"{list(program.targets)} and the device is {capability[0]}.{capability[1]}"
            )
        record = program.launch_record()
        expected = _expected_launch(identifier)
        if tuple(program.cluster) != expected["cluster"] or any(
            bool(record[key]) != expected[key] for key in ("use_pdl", "cooperative")
        ):
            raise RuntimeError(
                f"{identifier}: the host launches it with {expected}; the loader "
                f"records {record} and cluster {tuple(program.cluster)}"
            )
        module = program.load(TARGET)
        handle = ProgramHandle(
            identifier=identifier,
            args=tuple(program.args),
            launch=module.run,
            cluster=tuple(program.cluster),
            cooperative=bool(record["cooperative"]),
            use_pdl=bool(record["use_pdl"]),
        )
        self._handles[identifier] = handle
        return handle


_programs: Optional[ProgramTable] = None


def programs() -> ProgramTable:
    global _programs
    if _programs is None:
        _programs = ShippedPrograms()
    return _programs


def set_programs(table: Optional[ProgramTable]) -> None:
    """Install another program table (``None``: the shipped loader again)."""
    global _programs
    _programs = table


def programs_available() -> bool:
    return programs().available()


__all__ = [
    "CAPABILITY",
    "LOADER_MODULE",
    "ProgramHandle",
    "ProgramTable",
    "STAGE",
    "ShippedPrograms",
    "TARGET",
    "programs",
    "programs_available",
    "set_programs",
]
