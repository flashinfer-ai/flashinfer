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

"""Strong-typed data definitions for solution implementations.

The schema mirrors flashinfer-bench Solution objects, but is implemented with
standard-library dataclasses to avoid adding a runtime dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class SupportedLanguages(str, Enum):
    """Supported programming languages for solution implementations."""

    PYTHON = "python"
    TRITON = "triton"
    CPP = "cpp"
    CUDA = "cuda"
    TILELANG = "tilelang"


class SupportedBindings(str, Enum):
    """Supported bindings for C++/CUDA solution implementations."""

    TVM_FFI = "tvm-ffi"
    TORCH = "torch"


@dataclass(frozen=True)
class SourceFile:
    """A single source code file in a solution implementation."""

    path: str
    """The relative path of the file, including its name and extension."""
    content: str
    """The complete text content of the source file."""

    def __post_init__(self) -> None:
        src_path = Path(self.path)
        if src_path.is_absolute():
            raise ValueError(f"Invalid source path (absolute path not allowed): {self.path}")
        if ".." in src_path.parts:
            raise ValueError(
                f"Invalid source path (parent directory traversal not allowed): {self.path}"
            )

@dataclass(frozen=True)
class BuildSpec:
    """Build specification for a solution implementation."""

    language: SupportedLanguages
    """The primary programming language."""
    target_hardware: Sequence[str]
    """List of hardware this solution is compatible with."""
    entry_point: str
    """The exact path to the function to be called: ``<file_path>::<function>``."""
    dependencies: Sequence[str] = field(default_factory=list)
    """Optional list of required libraries or packages."""
    destination_passing_style: bool = True
    """Whether the solution accepts output tensors as the last arguments."""
    binding: Optional[SupportedBindings] = None
    """The binding type to use for C++/CUDA solutions."""

    def __post_init__(self) -> None:
        try:
            language = SupportedLanguages(self.language)
        except ValueError as exc:
            raise ValueError(f"Unsupported solution language: {self.language!r}") from exc

        if self.entry_point.count("::") != 1:
            raise ValueError(
                f"Invalid entry point format: {self.entry_point}. Expected "
                '"<file_path>::<function_name>".'
            )

        if not self.target_hardware:
            raise ValueError("BuildSpec.target_hardware must be non-empty")
        target_hardware = tuple(self.target_hardware)
        dependencies = tuple(self.dependencies)

        binding = None
        if self.binding is not None:
            try:
                binding = SupportedBindings(self.binding)
            except ValueError as exc:
                raise ValueError(f"Unsupported solution binding: {self.binding!r}") from exc

        object.__setattr__(self, "language", language)
        object.__setattr__(self, "target_hardware", target_hardware)
        object.__setattr__(self, "entry_point", self.entry_point)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "destination_passing_style", bool(self.destination_passing_style))
        object.__setattr__(self, "binding", binding)

@dataclass(frozen=True, eq=False)
class Solution:
    """A concrete implementation for a given fi_trace Definition."""

    name: str
    """A unique, human-readable name for this specific solution."""
    definition: str
    """The name of the Definition this implementation solves."""
    author: str
    """The name of the author or agent system that created this solution."""
    spec: BuildSpec
    """Technical specifications for building and executing this solution."""
    sources: Sequence[SourceFile]
    """Array of source code files representing the complete implementation."""
    description: Optional[str] = None
    """Optional human-readable description of the solution's technique."""
    _hash_cache: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, BuildSpec):
            raise TypeError("Solution.spec must be a BuildSpec")
        if not self.sources:
            raise ValueError("Solution.sources must be non-empty")

        sources = tuple(self.sources)
        seen_paths: set[str] = set()
        for source in sources:
            if not isinstance(source, SourceFile):
                raise TypeError("Solution.sources entries must be SourceFile")
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.spec.entry_point.split("::")[0]
        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")

        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "_hash_cache", self._compute_hash())

    def get_entry_symbol(self) -> str:
        """Extract the function/symbol name from the entry point specification."""

        return self.spec.entry_point.split("::")[-1]

    def get_entry_source(self) -> Optional[SourceFile]:
        """Get the entry source file specified in the build spec."""

        entry_path = self.spec.entry_point.split("::")[0]
        for source in self.sources:
            if source.path == entry_path:
                return source
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Solution":
        """Build a Solution from a JSON-compatible dictionary."""

        solution_definition = data.get("definition")
        if solution_definition is None:
            raise ValueError("Solution dictionary requires a definition")

        spec_data = data.get("spec")
        if not isinstance(spec_data, Mapping):
            raise TypeError(f"Solution '{solution_definition}' requires a spec object")

        sources_data = data.get("sources")
        if not isinstance(sources_data, list):
            raise TypeError(f"Solution '{solution_definition}' requires a sources list")

        return cls(
            name=data["name"],
            definition=solution_definition,
            author=data["author"],
            spec=BuildSpec(
                language=spec_data["language"],
                target_hardware=spec_data["target_hardware"],
                entry_point=spec_data["entry_point"],
                dependencies=spec_data.get("dependencies", ()),
                destination_passing_style=spec_data.get(
                    "destination_passing_style", True
                ),
                binding=spec_data.get("binding"),
            ),
            sources=[
                SourceFile(path=source["path"], content=source["content"])
                for source in sources_data
            ],
            description=data.get("description"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""

        spec = self.spec
        out = {
            "name": self.name,
            "definition": self.definition,
            "author": self.author,
            "spec": {
                "language": spec.language.value,
                "target_hardware": list(spec.target_hardware),
                "entry_point": spec.entry_point,
                "dependencies": list(spec.dependencies),
                "destination_passing_style": spec.destination_passing_style,
            },
            "sources": [
                {"path": source.path, "content": source.content}
                for source in self.sources
            ],
        }
        if spec.binding is not None:
            out["spec"]["binding"] = spec.binding.value
        if self.description is not None:
            out["description"] = self.description
        return out

    def _compute_hash(self) -> str:
        """Compute a deterministic hash of the solution content."""

        h = hashlib.sha1()
        for part in (
            self.definition,
            self.spec.language.value,
            self.spec.entry_point,
            self.spec.binding.value if self.spec.binding else "",
            *self.spec.dependencies,
            *(item for source in self.sources for item in (source.path, source.content)),
        ):
            h.update(part.encode())
        return h.hexdigest()

    def hash(self) -> str:
        """Return the memoized deterministic hash of the solution content."""

        return self._hash_cache

    def __hash__(self) -> int:
        return hash(self._hash_cache)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Solution):
            return NotImplemented
        return self._hash_cache == other._hash_cache
