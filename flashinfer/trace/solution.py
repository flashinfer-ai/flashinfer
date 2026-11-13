# Copyright (c) 2025 by FlashInfer team.
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

"""First-class, strongly-typed schema for trace Solutions.

A ``Solution`` is a concrete implementation for a given fi_trace ``Definition``,
identified by the definition name. The schema mirrors flashinfer-bench Solution
objects but is implemented with standard-library dataclasses so it adds no
runtime dependency. It is the single source of truth shared by trace collection
and ``flashinfer.trace_apply``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class SupportedLanguages(str, Enum):
    """Languages a solution implementation may be written in."""

    PYTHON = "python"
    TRITON = "triton"
    CUTEDSL = "cutedsl"
    TILELANG = "tilelang"
    CPP = "cpp"
    CUDA = "cuda"
    CUTLASS = "cutlass"
    CUTILE = "cutile"


class SupportedBindings(str, Enum):
    """Binding ABI for C++/CUDA-family solutions."""

    TVM_FFI = "tvm-ffi"
    TORCH = "torch"


# Languages loaded by preparing a Python module and resolving an entry point.
# Everything else is built/linked as a C++/CUDA extension.
_PYTHON_FAMILY = frozenset(
    {
        SupportedLanguages.PYTHON,
        SupportedLanguages.TRITON,
        SupportedLanguages.CUTEDSL,
        SupportedLanguages.CUTILE,
        SupportedLanguages.TILELANG,
    }
)


@dataclass(frozen=True)
class SourceFile:
    """A single source file in a solution implementation (content inlined)."""

    path: str
    """Relative path of the file, including name and extension."""
    content: str
    """Complete text content of the source file."""

    def __post_init__(self) -> None:
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(f"Invalid source path (absolute not allowed): {self.path}")
        if ".." in src_path.parts:
            raise ValueError(
                f"Invalid source path (parent traversal not allowed): {self.path}"
            )


@dataclass(frozen=True)
class BuildSpec:
    """How to build and call a solution implementation."""

    language: SupportedLanguages
    target_hardware: Sequence[str]
    """Compute capabilities / SM targets this solution supports, e.g. ["sm100"]."""
    entry_point: str
    """``"<file_path>::<function_name>"`` — the callable/exported symbol."""
    dependencies: Sequence[str] = field(default_factory=tuple)
    destination_passing_style: bool = False
    """If True, the solution writes into output buffers passed as trailing args
    and returns None; otherwise it returns the output value(s)."""
    binding: Optional[SupportedBindings] = None
    setup_entry_point: Optional[str] = None
    """``"<file_path>::<function_name>"`` of an optional untimed setup step,
    ``setup(context) -> state``, run once per weight set before the entry point
    is called (see :mod:`flashinfer.trace_apply.setup`). Python-family
    solutions only."""
    teardown_entry_point: Optional[str] = None
    """``"<file_path>::<function_name>"`` of ``teardown(state)``; requires
    ``setup_entry_point``."""
    setup_inputs: Sequence[str] = field(default_factory=tuple)
    """Definition inputs whose live tensors setup receives and whose identity
    keys the retained state. Empty with a setup means one state per process."""
    setup_state_param: str = "state"
    """The keyword parameter the entry point takes the setup state under. A
    field rather than a fixed name because trace_apply calls Python-family
    solutions by keyword, so the wrapper must know the name; an existing kernel
    keeps whatever parameter name it already uses and records it here."""

    def __post_init__(self) -> None:
        """Validate language, entry point, hardware, binding and the setup fields."""
        try:
            language = SupportedLanguages(self.language)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported solution language: {self.language!r}"
            ) from exc

        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"Invalid entry point: {self.entry_point!r}. "
                'Expected "<file_path>::<function_name>".'
            )
        if not self.target_hardware:
            raise ValueError("BuildSpec.target_hardware must be non-empty")

        binding = None
        if self.binding is not None:
            try:
                binding = SupportedBindings(self.binding)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported solution binding: {self.binding!r}"
                ) from exc

        for name, entry in (
            ("setup_entry_point", self.setup_entry_point),
            ("teardown_entry_point", self.teardown_entry_point),
        ):
            if entry is not None and (
                entry.count("::") != 1 or not all(entry.split("::"))
            ):
                raise ValueError(
                    f"Invalid {name}: {entry!r}. "
                    'Expected "<file_path>::<function_name>".'
                )
        if (
            isinstance(self.setup_inputs, str)
            or not isinstance(self.setup_inputs, Sequence)
            or not all(
                isinstance(item, str) and item.isidentifier()
                for item in self.setup_inputs
            )
            or len(set(self.setup_inputs)) != len(self.setup_inputs)
        ):
            raise ValueError(
                f"setup_inputs must be distinct identifiers: {self.setup_inputs!r}"
            )
        setup_inputs = tuple(self.setup_inputs)
        if self.setup_entry_point is None:
            if self.teardown_entry_point is not None:
                raise ValueError("teardown_entry_point requires setup_entry_point")
            if setup_inputs:
                raise ValueError("setup_inputs requires setup_entry_point")
        elif language not in _PYTHON_FAMILY:
            raise ValueError(
                f"setup_entry_point requires a Python-family language, "
                f"not {language.value!r}"
            )
        if (
            not isinstance(self.setup_state_param, str)
            or not self.setup_state_param.isidentifier()
        ):
            raise ValueError(f"Invalid setup_state_param: {self.setup_state_param!r}")
        if self.setup_state_param in setup_inputs:
            raise ValueError(
                f"setup_state_param {self.setup_state_param!r} is also a setup input"
            )

        object.__setattr__(self, "language", language)
        object.__setattr__(self, "target_hardware", tuple(self.target_hardware))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(
            self, "destination_passing_style", bool(self.destination_passing_style)
        )
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "setup_inputs", setup_inputs)

    @property
    def is_python_family(self) -> bool:
        """True for languages loaded as a Python module (python/triton/cutedsl/
        tilelang); False for C++/CUDA-family languages that must be compiled.
        This single distinction also fixes the calling convention: C++/CUDA
        (TVM-FFI) solutions take positional args, Python-family take keyword.
        """
        return self.language in _PYTHON_FAMILY

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BuildSpec":
        """Build a BuildSpec from JSON, refusing an implicit DPS with a setup."""
        if data.get("setup_entry_point") is not None and not isinstance(
            data.get("destination_passing_style"), bool
        ):
            # The state-bearing run ABI never inherits a default
            # (FlashInfer defaults to False, flashinfer-bench to True), and a
            # null would silently read as False.
            raise ValueError(
                "destination_passing_style must be declared explicitly, as a "
                "boolean, when setup_entry_point is present"
            )
        return cls(
            language=data["language"],
            target_hardware=data.get("target_hardware", ()),
            entry_point=data["entry_point"],
            dependencies=data.get("dependencies", ()),
            destination_passing_style=data.get("destination_passing_style", False),
            binding=data.get("binding"),
            setup_entry_point=data.get("setup_entry_point"),
            teardown_entry_point=data.get("teardown_entry_point"),
            setup_inputs=data.get("setup_inputs", ()),
            setup_state_param=data.get("setup_state_param", "state"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON, emitting setup fields only when a setup is declared."""
        out: dict[str, Any] = {
            "language": self.language.value,
            "target_hardware": list(self.target_hardware),
            "entry_point": self.entry_point,
            "dependencies": list(self.dependencies),
            "destination_passing_style": self.destination_passing_style,
        }
        if self.binding is not None:
            out["binding"] = self.binding.value
        if self.setup_entry_point is not None:
            out["setup_entry_point"] = self.setup_entry_point
            if self.teardown_entry_point is not None:
                out["teardown_entry_point"] = self.teardown_entry_point
            out["setup_inputs"] = list(self.setup_inputs)
            out["setup_state_param"] = self.setup_state_param
        return out


def _entry_part(entry: Optional[str], index: int) -> Optional[str]:
    """File (0) or symbol (1) half of an optional ``"<file_path>::<symbol>"``."""
    return None if entry is None else entry.split("::", 1)[index]


@dataclass(frozen=True, eq=False)
class Solution:
    """A concrete implementation for a given fi_trace Definition."""

    name: str
    definition: str
    """Name of the Definition this solution implements."""
    author: str
    spec: BuildSpec
    sources: Sequence[SourceFile]
    description: Optional[str] = None
    _hash_cache: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Check the spec type, source uniqueness and that entry files are sources."""
        if not isinstance(self.spec, BuildSpec):
            raise TypeError("Solution.spec must be a BuildSpec")
        if not self.sources:
            raise ValueError("Solution.sources must be non-empty")

        sources = tuple(self.sources)
        seen: set[str] = set()
        for src in sources:
            if not isinstance(src, SourceFile):
                raise TypeError("Solution.sources entries must be SourceFile")
            if src.path in seen:
                raise ValueError(f"Duplicate source path: {src.path!r}")
            seen.add(src.path)

        entry_file = self.spec.entry_point.split("::", 1)[0]
        if entry_file not in seen:
            raise ValueError(f"Entry source file {entry_file!r} not found in sources")
        for role, path in (
            ("Setup", self.setup_file()),
            ("Teardown", self.teardown_file()),
        ):
            if path is not None and path not in seen:
                raise ValueError(f"{role} source file {path!r} not found in sources")

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "_hash_cache", self._compute_hash())

    def entry_symbol(self) -> str:
        return self.spec.entry_point.split("::", 1)[1]

    def entry_file(self) -> str:
        return self.spec.entry_point.split("::", 1)[0]

    def setup_symbol(self) -> Optional[str]:
        return _entry_part(self.spec.setup_entry_point, 1)

    def setup_file(self) -> Optional[str]:
        return _entry_part(self.spec.setup_entry_point, 0)

    def teardown_symbol(self) -> Optional[str]:
        return _entry_part(self.spec.teardown_entry_point, 1)

    def teardown_file(self) -> Optional[str]:
        return _entry_part(self.spec.teardown_entry_point, 0)

    def entry_source(self) -> Optional[SourceFile]:
        entry = self.entry_file()
        for src in self.sources:
            if src.path == entry:
                return src
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Solution":
        definition = data.get("definition")
        if not definition:
            raise ValueError("Solution requires a non-empty 'definition'")
        spec = data.get("spec")
        if not isinstance(spec, Mapping):
            raise TypeError(f"Solution {definition!r} requires a 'spec' object")
        sources = data.get("sources")
        if not isinstance(sources, list):
            raise TypeError(f"Solution {definition!r} requires a 'sources' list")
        return cls(
            name=data["name"],
            definition=definition,
            author=data["author"],
            spec=BuildSpec.from_dict(spec),
            sources=[SourceFile(path=s["path"], content=s["content"]) for s in sources],
            description=data.get("description"),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "definition": self.definition,
            "author": self.author,
            "spec": self.spec.to_dict(),
            "sources": [{"path": s.path, "content": s.content} for s in self.sources],
        }
        if self.description is not None:
            out["description"] = self.description
        return out

    def _compute_hash(self) -> str:
        """Hash the definition, build-affecting spec fields and sorted sources."""
        h = hashlib.sha256()
        # Sort sources by path so the hash is independent of source ordering.
        sorted_sources = sorted(self.sources, key=lambda s: s.path)
        parts = (
            self.definition,
            self.spec.language.value,
            self.spec.entry_point,
            self.spec.binding.value if self.spec.binding else "",
            # Build-affecting spec fields: a different SM target or call ABI must
            # not share a build-cache key with otherwise-identical sources.
            *self.spec.target_hardware,
            str(int(self.spec.destination_passing_style)),
            *self.spec.dependencies,
            *(item for s in sorted_sources for item in (s.path, s.content)),
        )
        if self.spec.setup_entry_point is not None:
            # Appended only when declared, so every Solution without a setup
            # keeps the hash (and build-cache key) it had before the fields
            # existed; framed as one JSON document so the fields cannot run
            # into each other (("a", ["bc"]) and ("ab", ["c"]) differ).
            parts += (
                json.dumps(
                    [
                        "setup",
                        self.spec.setup_entry_point,
                        self.spec.teardown_entry_point,
                        self.spec.setup_state_param,
                        list(self.spec.setup_inputs),
                    ]
                ),
            )
        for part in parts:
            h.update(part.encode())
        return h.hexdigest()

    def hash(self) -> str:
        """Deterministic content hash (used for build-cache directory naming)."""
        return self._hash_cache

    def __hash__(self) -> int:
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return self._hash_cache == other._hash_cache


__all__ = [
    "BuildSpec",
    "Solution",
    "SourceFile",
    "SupportedBindings",
    "SupportedLanguages",
]
