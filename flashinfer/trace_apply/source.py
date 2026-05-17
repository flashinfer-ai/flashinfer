from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from flashinfer.trace_apply.schema import Definition, Solution, TraceRecord

_TRACE_ENV = "FLASHINFER_TRACE_PATH"


@dataclass(slots=True)
class TracePaths:
    """Resolved roots within a FlashInfer Trace directory."""

    root: Path
    definitions: Path
    solutions: Path
    traces: Path
    blob: Path

    @classmethod
    def from_root(cls, root: str | os.PathLike) -> TracePaths:
        r = Path(root).expanduser().resolve()
        if not r.is_dir():
            raise FileNotFoundError(f"FlashInfer Trace path is not a directory: {r}")
        paths = cls(
            root=r,
            definitions=r / "definitions",
            solutions=r / "solutions",
            traces=r / "traces",
            blob=r / "blob",
        )
        # Only definitions and solutions are strictly required; traces and blob
        # may be absent in a hand-assembled minimal trace.
        for required in (paths.definitions, paths.solutions):
            if not required.is_dir():
                raise FileNotFoundError(f"Missing expected subdirectory: {required}")
        return paths


def resolve_trace_path(explicit: str | os.PathLike | None = None) -> Path:
    """Resolve the trace root, preferring the explicit argument over the env var."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get(_TRACE_ENV)
    if not env:
        raise RuntimeError(
            f"{_TRACE_ENV} is not set. Trace Apply requires an explicit local trace path; "
            "auto-download is not supported."
        )
    return Path(env).expanduser().resolve()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_definitions(paths: TracePaths) -> dict[str, Definition]:
    """Load every Definition under <root>/definitions/<op_type>/<name>.json."""
    out: dict[str, Definition] = {}
    for path in sorted(paths.definitions.rglob("*.json")):
        with path.open() as f:
            data = json.load(f)
        defn = Definition.from_dict(data)
        if defn.name in out:
            raise ValueError(f"Duplicate definition name: {defn.name} (in {path})")
        out[defn.name] = defn
    return out


def load_solutions(paths: TracePaths) -> dict[tuple[str, str], Solution]:
    """Load every Solution. Keyed by (definition_name, solution_name)."""
    out: dict[tuple[str, str], Solution] = {}
    for path in sorted(paths.solutions.rglob("*.json")):
        with path.open() as f:
            data = json.load(f)
        sol = Solution.from_dict(data)
        key = (sol.definition, sol.name)
        if key in out:
            raise ValueError(f"Duplicate solution: {key} (in {path})")
        out[key] = sol
    return out


def iter_traces(paths: TracePaths) -> Iterator[TraceRecord]:
    """Stream TraceRecords from every JSONL under <root>/traces/."""
    if not paths.traces.is_dir():
        return
    for path in sorted(paths.traces.rglob("*.jsonl")):
        with path.open() as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSONL at {path}:{line_no}: {e}") from e
                yield TraceRecord.from_dict(data)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Trace:
    """A loaded FlashInfer Trace: definitions, solutions, and trace records."""

    paths: TracePaths
    definitions: dict[str, Definition]
    solutions: dict[tuple[str, str], Solution]
    records: list[TraceRecord]


def load_trace(path: str | os.PathLike | None = None) -> Trace:
    """Load definitions, solutions, and trace records from the given path or
    `FLASHINFER_TRACE_PATH`.
    """
    paths = TracePaths.from_root(resolve_trace_path(path))
    return Trace(
        paths=paths,
        definitions=load_definitions(paths),
        solutions=load_solutions(paths),
        records=list(iter_traces(paths)),
    )
