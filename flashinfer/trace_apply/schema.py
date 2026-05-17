from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _pick(d: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Axis:
    type: str  # "var" or "const"
    value: int | None = None  # set when type == "const"
    description: str = ""


@dataclass(slots=True)
class IOSpec:
    shape: list[str] | None  # list of axis names; None for scalars
    dtype: str
    optional: bool = False
    description: str = ""


@dataclass(slots=True)
class Definition:
    name: str
    op_type: str
    axes: dict[str, Axis]
    inputs: dict[str, IOSpec]
    outputs: dict[str, IOSpec]
    tags: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Definition:
        return cls(
            name=d["name"],
            op_type=d["op_type"],
            axes={k: Axis(**_pick(v, "type", "value", "description")) for k, v in d.get("axes", {}).items()},
            inputs={k: IOSpec(**_pick(v, "shape", "dtype", "optional", "description")) for k, v in d.get("inputs", {}).items()},
            outputs={k: IOSpec(**_pick(v, "shape", "dtype", "optional", "description")) for k, v in d.get("outputs", {}).items()},
            tags=list(d.get("tags", [])),
            constraints=list(d.get("constraints", [])),
            description=d.get("description", ""),
        )

    def fi_api(self) -> str | None:
        """Importable target attribute the engine calls (e.g. flashinfer.gqa_paged_decode).

        Encoded in tags as `fi_api:<dotted.path>`. Returns None if absent.
        """
        for tag in self.tags:
            if tag.startswith("fi_api:"):
                return tag[len("fi_api:") :]
        return None

    def const_axes(self) -> dict[str, int]:
        return {name: a.value for name, a in self.axes.items() if a.type == "const" and a.value is not None}

    def var_axis_names(self) -> list[str]:
        return [name for name, a in self.axes.items() if a.type == "var"]


# ---------------------------------------------------------------------------
# Solution
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SourceFile:
    path: str
    content: str


@dataclass(slots=True)
class BuildSpec:
    language: str
    target_hardware: list[str]
    entry_point: str
    dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BuildSpec:
        return cls(
            language=d["language"],
            target_hardware=list(d.get("target_hardware", [])),
            entry_point=d["entry_point"],
            dependencies=list(d.get("dependencies", [])),
        )


@dataclass(slots=True)
class Solution:
    name: str
    definition: str
    author: str
    spec: BuildSpec
    sources: list[SourceFile]
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Solution:
        return cls(
            name=d["name"],
            definition=d["definition"],
            author=d["author"],
            spec=BuildSpec.from_dict(d["spec"]),
            sources=[SourceFile(path=s["path"], content=s["content"]) for s in d.get("sources", [])],
            description=d.get("description", ""),
        )


# ---------------------------------------------------------------------------
# TraceRecord
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Correctness:
    max_absolute_error: float | None = None
    max_relative_error: float | None = None


@dataclass(slots=True)
class Performance:
    latency_ms: float
    reference_latency_ms: float | None = None
    speedup_factor: float | None = None


@dataclass(slots=True)
class Environment:
    hardware: str
    libs: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Workload:
    axes: dict[str, int]
    uuid: str = ""


@dataclass(slots=True)
class TraceRecord:
    definition: str
    solution: str
    workload: Workload
    status: str
    environment: Environment
    performance: Performance | None = None
    correctness: Correctness | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TraceRecord:
        wl = d["workload"]
        ev = d.get("evaluation", {})
        env = ev.get("environment", {})
        perf = ev.get("performance")
        corr = ev.get("correctness")
        return cls(
            definition=d["definition"],
            solution=d["solution"],
            workload=Workload(axes=dict(wl.get("axes", {})), uuid=wl.get("uuid", "")),
            status=ev.get("status", "UNKNOWN"),
            environment=Environment(
                hardware=env.get("hardware", ""),
                libs=dict(env.get("libs", {})),
            ),
            performance=(
                Performance(**_pick(perf, "latency_ms", "reference_latency_ms", "speedup_factor"))
                if perf
                else None
            ),
            correctness=(
                Correctness(**_pick(corr, "max_absolute_error", "max_relative_error"))
                if corr
                else None
            ),
        )
