"""Per-query defaults and exhaustive legal kernel parameter spaces."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Generic, TypeVar

from .types import DeviceIdentity, FrozenMapping

QueryT = TypeVar("QueryT")
ConfigT = TypeVar("ConfigT")

class ParameterBinding(str, Enum):
    COMPILE = "compile"
    RUNTIME = "runtime"


def _identity(value: object) -> str:
    if isinstance(value, FrozenMapping):
        value = value.to_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


@dataclass(frozen=True, kw_only=True)
class Knob:
    """One parameter dimension, with its binding time and conditional values.

    A component may supply shape-dependent values through ``parameters`` on its
    tuning contract. Runtime binding is a contract: varying that parameter must
    not change the actual compiled-program keys for a fixed compile assignment.
    """

    name: str
    values: tuple[object, ...] | None
    binding: ParameterBinding = ParameterBinding.COMPILE
    when: FrozenMapping = FrozenMapping()
    otherwise: object = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("kernel parameters require an identifier name")
        object.__setattr__(self, "binding", ParameterBinding(self.binding))
        if not isinstance(self.when, FrozenMapping):
            object.__setattr__(self, "when", FrozenMapping(self.when))
        if self.values is not None:
            object.__setattr__(self, "values", tuple(self.values))
            identities = tuple(_identity(value) for value in self.values)
            if not identities or len(identities) != len(set(identities)):
                raise ValueError("parameter values must be nonempty and unique")
        _identity(self.otherwise)

    def active(self, assignment: Mapping[str, object]) -> bool:
        return all(
            name in assignment and _same_value(assignment[name], value)
            for name, value in self.when.items()
        )


@dataclass(frozen=True, kw_only=True)
class ParameterSpace:
    """Finite axes with correctness and optional efficiency predicates.

    ``predicates`` always apply. ``efficiency_predicates`` only prune the search
    when ``exhaustive`` is false; components capture that policy in their query.
    """

    knobs: tuple[Knob, ...]
    predicates: tuple[Callable[[Mapping[str, object]], bool], ...] = ()
    efficiency_predicates: tuple[Callable[[Mapping[str, object]], bool], ...] = ()
    exhaustive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "knobs", tuple(self.knobs))
        object.__setattr__(self, "predicates", tuple(self.predicates))
        object.__setattr__(self, "efficiency_predicates", tuple(self.efficiency_predicates))
        if type(self.exhaustive) is not bool:
            raise TypeError("exhaustive must be a boolean")
        names = set()
        for knob in self.knobs:
            if knob.name in names or set(knob.when) - names:
                raise ValueError(
                    "conditional parameters must uniquely follow their dependencies"
                )
            if knob.values is None:
                raise ValueError(
                    f"plan did not resolve values for parameter {knob.name!r}"
                )
            names.add(knob.name)
        if not names:
            raise ValueError(
                "parameter spaces must identify at least their implementation"
            )

    @classmethod
    def create(
        cls,
        knobs: Iterable[Knob],
        *,
        values: Mapping[str, Iterable[object]] = FrozenMapping(),
        predicates: Iterable[Callable[[Mapping[str, object]], bool]] = (),
        efficiency_predicates: Iterable[Callable[[Mapping[str, object]], bool]] = (),
        exhaustive: bool = False,
    ) -> ParameterSpace:
        """Resolve planner ranges while preserving the declared binding times."""
        knobs = tuple(knobs)
        if set(values) - {knob.name for knob in knobs}:
            raise ValueError("parameter-space provider returned undeclared dimensions")
        return cls(
            knobs=tuple(
                replace(knob, values=tuple(values[knob.name]))
                if knob.name in values
                else knob
                for knob in knobs
            ),
            predicates=tuple(predicates),
            efficiency_predicates=tuple(efficiency_predicates),
            exhaustive=exhaustive,
        )

    def _passes_predicates(self, assignment: Mapping[str, object]) -> bool:
        # Predicates must not mutate or retain the enumeration's working mapping.
        # An exception is a planner defect, not a reason to silently drop a choice.
        return all(predicate(assignment) for predicate in self.predicates) and (
            self.exhaustive
            or all(predicate(assignment) for predicate in self.efficiency_predicates)
        )

    def validate(self, assignment: Mapping[str, object]) -> None:
        if set(assignment) != {knob.name for knob in self.knobs}:
            raise ValueError(
                "assignment fields differ from the plan's parameter dimensions"
            )
        for knob in self.knobs:
            values = knob.values if knob.active(assignment) else (knob.otherwise,)
            if not any(_same_value(assignment[knob.name], value) for value in values):
                raise ValueError(
                    f"parameter {knob.name!r} is outside its eligible values"
                )
        if not all(predicate(assignment) for predicate in self.predicates):
            raise ValueError("assignment fails the plan's correctness predicates")
        if not self.exhaustive and not all(
            predicate(assignment) for predicate in self.efficiency_predicates
        ):
            raise ValueError("assignment fails the plan's efficiency predicates")

    def _product(
        self, index: int, assignment: dict[str, object]
    ) -> Iterator[dict[str, object]]:
        if index == len(self.knobs):
            yield assignment
            return
        knob = self.knobs[index]
        values = knob.values if knob.active(assignment) else (knob.otherwise,)
        for value in values:
            assignment[knob.name] = value
            yield from self._product(index + 1, assignment)
        del assignment[knob.name]

    def configurations(self) -> Iterator[FrozenMapping]:
        for assignment in self._product(0, {}):
            if self._passes_predicates(assignment):
                yield FrozenMapping(assignment)

    def compile_assignment(self, assignment: Mapping[str, object]) -> FrozenMapping:
        return FrozenMapping(
            {
                knob.name: assignment[knob.name]
                for knob in self.knobs
                if knob.binding is ParameterBinding.COMPILE
            }
        )

    def runtime_assignment(self, assignment: Mapping[str, object]) -> FrozenMapping:
        return FrozenMapping(
            {
                knob.name: assignment[knob.name]
                for knob in self.knobs
                if knob.binding is ParameterBinding.RUNTIME
            }
        )


@dataclass(frozen=True, kw_only=True)
class EligiblePlan(Generic[ConfigT]):
    space: ParameterSpace
    candidates: tuple[tuple[FrozenMapping, ConfigT], ...]
    cartesian_count: int
    legal_count: int

    @property
    def equivalent_count(self) -> int:
        return self.legal_count - len(self.candidates)


@dataclass(frozen=True, kw_only=True)
class TuningConfiguration(Generic[QueryT, ConfigT]):
    query: QueryT
    encoded_query: FrozenMapping
    device: DeviceIdentity | None
    space: ParameterSpace | None
    default: ConfigT
    pinned: ConfigT | None
    contract: TuningContract[QueryT, ConfigT]


@dataclass(frozen=True, kw_only=True)
class TuningContract(Generic[QueryT, ConfigT]):
    component_id: str
    query_schema_version: int
    config_schema_version: int
    query_fields: frozenset[str]
    config_fields: frozenset[str]
    encode_query: Callable[[QueryT], Mapping[str, object]]
    encode_config: Callable[[ConfigT], Mapping[str, object]]
    decode_config: Callable[[FrozenMapping], ConfigT]
    validate_query: Callable[[QueryT, DeviceIdentity | None], None]
    validate_config: Callable[[QueryT, ConfigT, DeviceIdentity | None], None]
    default_config: Callable[[QueryT, DeviceIdentity | None], ConfigT]
    knobs: tuple[Knob, ...]
    semantic_version: int = 1
    candidate_contract_version: int = 1
    parameters: Callable[
        [QueryT, DeviceIdentity | None],
        Mapping[str, Iterable[object]] | ParameterSpace,
    ] | None = None
    materialize: Callable[[QueryT, DeviceIdentity | None, FrozenMapping], ConfigT] | None = None
    equivalence_key: Callable[[QueryT, DeviceIdentity | None, ConfigT], object] | None = None

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", self.component_id):
            raise ValueError(f"invalid component ID {self.component_id!r}")
        for version in (
            self.query_schema_version, self.config_schema_version,
            self.semantic_version, self.candidate_contract_version,
        ):
            if type(version) is not int or version <= 0:
                raise ValueError("contract versions must be positive integers")
        for name in ("query_fields", "config_fields"):
            value = frozenset(getattr(self, name))
            if not value or any(not isinstance(item, str) or not item for item in value):
                raise ValueError("contract fields must be nonempty strings")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "knobs", tuple(self.knobs))
        if not self.knobs:
            raise ValueError("contracts require at least an implementation knob")
        if self.parameters is None and any(knob.values is None for knob in self.knobs):
            raise ValueError("dynamic axes require a parameter-space provider")

    def parameter_space(self, query, device) -> ParameterSpace:
        values = {} if self.parameters is None else self.parameters(query, device)
        if isinstance(values, ParameterSpace):
            return values
        return ParameterSpace.create(self.knobs, values=values)

    def configure(self, query, *, device, override=None, search=True) -> TuningConfiguration:
        self.validate_query(query, device)
        encoded = FrozenMapping(self.encode_query(query))
        if set(encoded) != self.query_fields:
            raise ValueError("query codec fields differ from the contract")
        # A complete pin is authoritative; an unused default may not reject it.
        default = override if override is not None else self.default_config(query, device)
        self.validate_config(query, default, device)
        self.config_payload(default)
        return TuningConfiguration(
            query=query, encoded_query=encoded, device=device,
            space=self.parameter_space(query, device) if search else None, default=default,
            pinned=override, contract=self,
        )

    def config_payload(self, config) -> FrozenMapping:
        payload = FrozenMapping(self.encode_config(config))
        if set(payload) != self.config_fields:
            raise ValueError("config codec fields differ from the contract")
        return payload

    def _lower(self, query, device, assignment):
        payload = FrozenMapping(assignment)
        config = (
            self.decode_config(payload) if self.materialize is None
            else self.materialize(query, device, payload)
        )
        self.validate_config(query, config, device)
        self.config_payload(config)
        return config

    def lower(self, query, device, assignment):
        self.validate_query(query, device)
        self.parameter_space(query, device).validate(assignment)
        return self._lower(query, device, assignment)

    def iterate(self, configuration: TuningConfiguration) -> CandidateIterator:
        if configuration.contract is not self:
            raise ValueError("configuration belongs to another contract")
        if configuration.space is None:
            raise ValueError("candidate enumeration requires a search configuration")
        return CandidateIterator(configuration)

    def eligible_plan(self, query, device, *, eligible=None) -> EligiblePlan:
        configuration = self.configure(query, device=device)
        iterator = self.iterate(configuration)
        candidates = tuple(
            candidate for candidate in iterator
            if eligible is None or eligible(*candidate)
        )
        if not candidates:
            raise ValueError(f"no eligible configurations for {self.component_id}")
        return EligiblePlan(
            space=configuration.space, candidates=candidates,
            cartesian_count=iterator.cartesian_count,
            legal_count=iterator.legal_count,
        )

    def choices(self, query, device):
        yield from self.eligible_plan(query, device).candidates


class CandidateIterator:
    """A resumable product; each step performs at most one metadata action."""

    def __init__(self, configuration: TuningConfiguration):
        self.configuration = configuration
        self._product = configuration.space._product(0, {})
        self._seen = set()
        self.cartesian_count = 0
        self.legal_count = 0
        self.effective_count = 0
        self.done = False

    def step(self):
        if self.done:
            raise StopIteration
        try:
            raw = next(self._product)
        except StopIteration:
            self.done = True
            raise
        self.cartesian_count += 1
        cfg = self.configuration
        if not cfg.space._passes_predicates(raw):
            return None
        assignment = FrozenMapping(raw)
        try:
            config = cfg.contract._lower(cfg.query, cfg.device, assignment)
        except ValueError:
            return None
        self.legal_count += 1
        effective = (
            cfg.contract.config_payload(config)
            if cfg.contract.equivalence_key is None
            else cfg.contract.equivalence_key(cfg.query, cfg.device, config)
        )
        key = _identity(effective)
        if key in self._seen:
            return None
        self._seen.add(key)
        self.effective_count += 1
        return assignment, config

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            value = self.step()
            if value is not None:
                return value


@dataclass(frozen=True, kw_only=True)
class BackendConfig:
    backend: str

    @classmethod
    def from_config(cls, payload: FrozenMapping):
        if set(payload) != {"backend"} or not isinstance(payload["backend"], str):
            raise ValueError("fixed configs require exactly a string backend")
        return cls(backend=payload["backend"])

    def to_dict(self):
        return {"backend": self.backend}


def make_fixed_contract(*, component_id, query_type, backend) -> TuningContract:
    query_fields = frozenset(item.name for item in fields(query_type))

    def validate_query(query, device):
        if not isinstance(query, query_type):
            raise TypeError(f"query must be {query_type.__name__}")

    def validate_config(query, config, device):
        if not isinstance(config, BackendConfig):
            raise TypeError("config must be BackendConfig")
        if config.backend != backend:
            raise ValueError(f"unsupported {component_id} backend {config.backend!r}")

    return TuningContract(
        component_id=component_id, query_schema_version=1, config_schema_version=1,
        query_fields=query_fields, config_fields=frozenset({"backend"}),
        encode_query=lambda query: {name: getattr(query, name) for name in query_fields},
        encode_config=BackendConfig.to_dict, decode_config=BackendConfig.from_config,
        validate_query=validate_query, validate_config=validate_config,
        default_config=lambda query, device: BackendConfig(backend=backend),
        knobs=(Knob(name="backend", values=(backend,)),),
    )
