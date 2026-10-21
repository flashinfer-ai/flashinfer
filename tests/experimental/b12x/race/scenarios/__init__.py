"""Race-test scenarios: one factory per b12x kernel family.

A factory takes ``(device, case)`` and returns a ``Scenario`` (see
``_base.py``) whose ``forward`` launches the kernel under test from fixed,
seeded inputs into ``outputs``. ``reset`` restores anything ``forward``
mutates and is recorded ahead of it in the same graph. Factories prepare and
freeze their plans before returning, so capture records only launches. A
frozen preparation session blocks compilation process-wide, so callers must
``close()`` one scenario before building the next.

Each family module defines ``FAMILIES`` (name -> factory), ``CASES`` (name ->
curated cases) and ``FUZZ`` (name -> function drawing a random case from a
``random.Random``), and may define ``MUTATIONS`` (fixed races that
``mutations.py`` puts back to prove a scenario catches them; see there).
Family modules must not import b12x at module level, because mutations are
applied before b12x is first imported. Outputs must be bit-exact functions of the inputs: sort
or otherwise canonicalize results whose order depends on atomics, and pick
deterministic options where a kernel offers them; ``ReplayCheck`` rejects a
scenario whose quiet replays disagree. Size grids from the SM count (see
``_rows_for_waves``), since races between CTAs need several waves.
"""

from __future__ import annotations

import importlib
import pkgutil
import random

from ._base import Scenario  # noqa: F401

# Every module in this package not starting with "_" is a family module.
_MODULES = sorted(
    m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_")
)

FAMILIES: dict = {}
CASES: dict = {}
FUZZ: dict = {}
MUTATIONS: dict = {}
for _name in _MODULES:
    _module = importlib.import_module(f"{__name__}.{_name}")
    for _registry, _source in (
        (FAMILIES, _module.FAMILIES),
        (CASES, _module.CASES),
        (FUZZ, _module.FUZZ),
        (MUTATIONS, getattr(_module, "MUTATIONS", {})),
    ):
        overlap = set(_registry) & set(_source)
        if overlap:
            raise RuntimeError(f"duplicate race scenario families: {sorted(overlap)}")
        _registry.update(_source)


def fuzz_cases(family: str, count: int, seed: int) -> list[dict]:
    rng = random.Random(f"{family}:{seed}")
    return [FUZZ[family](rng) for _ in range(count)] if family in FUZZ else []
