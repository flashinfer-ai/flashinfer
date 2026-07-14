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

"""Loading the apply config: a folder of solution (and definition) JSON files.

Trace Apply is fed by pointing ``FLASHINFER_TRACE_APPLY_PATH`` at a directory in the
flashinfer-bench layout::

    <root>/
      definitions/**/*.json   # the problem specs (axes, inputs incl. dtype, fi_api)
      solutions/**/*.json      # the implementations (one per file)
      workloads/ traces/ ...   # ignored by apply

``solutions/`` provides the kernels to substitute; ``definitions/`` provides the
routing identity (const axes + input dtypes). Everything else is ignored.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flashinfer.trace.solution import Solution

ENABLE_ENV = "FLASHINFER_TRACE_APPLY"
PATH_ENV = "FLASHINFER_TRACE_APPLY_PATH"


# ---------------------------------------------------------------------------
# Definition (typed parse of the dict fi_trace emits)
# ---------------------------------------------------------------------------


def _pick(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


@dataclass(slots=True)
class Axis:
    type: str  # "var" or "const"
    value: int | None = None  # set when type == "const"
    description: str = ""


@dataclass(slots=True)
class IOSpec:
    shape: list[str] | None  # axis names; None for scalars
    dtype: str
    optional: bool = False
    # For outputs: the API parameter this output is written into (``out=`` buffer
    # or, for in-place APIs, an input buffer).
    param: str | None = None
    description: str = ""


@dataclass(slots=True)
class Definition:
    name: str
    op_type: str
    axes: dict[str, Axis]
    inputs: dict[str, IOSpec]
    outputs: dict[str, IOSpec]
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Definition":
        return cls(
            name=d["name"],
            op_type=d.get("op_type", ""),
            axes={
                k: Axis(**_pick(v, "type", "value", "description"))
                for k, v in d.get("axes", {}).items()
            },
            inputs={
                k: IOSpec(
                    **_pick(v, "shape", "dtype", "optional", "param", "description")
                )
                for k, v in d.get("inputs", {}).items()
            },
            outputs={
                k: IOSpec(
                    **_pick(v, "shape", "dtype", "optional", "param", "description")
                )
                for k, v in d.get("outputs", {}).items()
            },
            tags=list(d.get("tags", [])),
        )

    def fi_api(self) -> str | None:
        """The importable target attribute, encoded in tags as ``fi_api:<path>``."""
        for tag in self.tags:
            if tag.startswith("fi_api:"):
                return tag[len("fi_api:") :]
        return None

    def const_axes(self) -> dict[str, int]:
        return {
            name: a.value
            for name, a in self.axes.items()
            if a.type == "const" and a.value is not None
        }

    def input_dtypes(self) -> frozenset[tuple[str, str]]:
        """``{(input_name, dtype)}`` over **required tensor** inputs.

        Part of the routing identity so dtype-specialized solutions (fp16 vs
        bf16) for the same shape route distinctly. Scalars (shape is None) are
        excluded (dtype is not a meaningful kernel specialization there), and so
        are *optional* inputs — an optional input present at trace time but
        absent at a call would otherwise make the runtime key mismatch.
        """
        return frozenset(
            (name, io.dtype)
            for name, io in self.inputs.items()
            if io.shape is not None
            and not io.optional
            and io.dtype
            and io.dtype != "unknown"
        )


# ---------------------------------------------------------------------------
# Folder loading
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TraceConfig:
    """Parsed contents of a trace-apply folder."""

    definitions: dict[str, Definition]  # by definition name
    solutions: dict[tuple[str, str], Solution]  # by (definition, solution name)


def resolve_trace_path(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the folder, preferring the explicit arg over ``FLASHINFER_TRACE_APPLY_PATH``."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(PATH_ENV)
    if not env:
        raise RuntimeError(
            f"{PATH_ENV} is not set. Trace Apply requires an explicit local trace "
            "folder; auto-download is not supported."
        )
    return Path(env).expanduser().resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON at {path}: {e}") from e


def load_config(path: str | os.PathLike | None = None) -> TraceConfig:
    """Load definitions + solutions from the trace folder.

    Strict: a malformed definition/solution file raises (we never silently skip
    something we were asked to apply). Only ``definitions/`` and ``solutions/``
    are read; other subdirs (``workloads/``, ``traces/``) are ignored.
    """
    root = resolve_trace_path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Trace path is not a directory: {root}")

    defs_dir = root / "definitions"
    sols_dir = root / "solutions"
    if not sols_dir.is_dir():
        raise FileNotFoundError(f"Missing expected subdirectory: {sols_dir}")

    definitions: dict[str, Definition] = {}
    if defs_dir.is_dir():
        for p in sorted(defs_dir.rglob("*.json")):
            defn = Definition.from_dict(_load_json(p))
            if defn.name in definitions:
                raise ValueError(f"Duplicate definition name {defn.name!r} (in {p})")
            definitions[defn.name] = defn

    solutions: dict[tuple[str, str], Solution] = {}
    for p in sorted(sols_dir.rglob("*.json")):
        sol = Solution.from_dict(_load_json(p))
        if sol.definition not in definitions:
            raise ValueError(
                f"Solution {(sol.definition, sol.name)!r} references unknown "
                f"definition {sol.definition!r} (in {p})"
            )
        key = (sol.definition, sol.name)
        if key in solutions:
            raise ValueError(f"Duplicate solution {key} (in {p})")
        solutions[key] = sol

    return TraceConfig(definitions=definitions, solutions=solutions)


__all__ = [
    "ENABLE_ENV",
    "PATH_ENV",
    "Axis",
    "Definition",
    "IOSpec",
    "TraceConfig",
    "load_config",
    "resolve_trace_path",
]
