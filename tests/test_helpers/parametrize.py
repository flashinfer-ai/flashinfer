"""Pytest support for regular and full Cartesian parameter matrices."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class _ProductSpec:
    argnames: tuple[str, ...]
    axes: tuple[tuple[Any, ...], ...]
    regular: tuple[tuple[Any, ...], ...]
    ids: Callable[[tuple[Any, ...]], str] | None

    def case_index(self, case: tuple[Any, ...]) -> int:
        index = 0
        for axis, value in zip(self.axes, case, strict=True):
            index = index * len(axis) + axis.index(value)
        return index

    def case_id(self, case: tuple[Any, ...]) -> str:
        index = self.case_index(case)
        return (
            f"{self.ids(case)}-case-{index}"
            if self.ids is not None
            else f"case-{index}"
        )


def parametrize_product(
    argnames: str | Sequence[str],
    axes: Sequence[Iterable[Any]],
    *,
    regular: Sequence[Sequence[Any]],
    ids: Callable[[tuple[Any, ...]], str] | None = None,
):
    """Parametrize a test with selected cases, or the full product under ``--full``."""
    names = (
        tuple(name.strip() for name in argnames.split(","))
        if isinstance(argnames, str)
        else tuple(argnames)
    )
    axis_values = tuple(tuple(axis) for axis in axes)
    regular_cases = tuple(tuple(case) for case in regular)

    if not names or any(not name for name in names):
        raise ValueError("parametrize_product requires at least one parameter name")
    if len(names) != len(axis_values):
        raise ValueError(
            f"parametrize_product received {len(names)} names but "
            f"{len(axis_values)} value axes"
        )
    if any(not axis for axis in axis_values):
        raise ValueError("parametrize_product value axes cannot be empty")

    for case in regular_cases:
        if len(case) != len(names):
            raise ValueError(
                f"regular case {case!r} has {len(case)} values; expected {len(names)}"
            )
        for name, value, axis in zip(names, case, axis_values, strict=True):
            if value not in axis:
                raise ValueError(
                    f"regular case {case!r} uses unknown {name} value {value!r}"
                )

    spec = _ProductSpec(names, axis_values, regular_cases, ids)
    seen_cases: set[int] = set()
    for case in regular_cases:
        index = spec.case_index(case)
        if index in seen_cases:
            raise ValueError(f"duplicate regular case: {case!r}")
        seen_cases.add(index)

    def decorate(function):
        if hasattr(function, "_flashinfer_product_spec"):
            raise ValueError("parametrize_product cannot decorate the same test twice")
        function._flashinfer_product_spec = spec
        return function

    return decorate


def pytest_addoption(parser):
    parser.getgroup("flashinfer").addoption(
        "--full",
        action="store_true",
        default=False,
        help="run full parameter matrices instead of regular subsets",
    )


def pytest_generate_tests(metafunc):
    spec = getattr(metafunc.function, "_flashinfer_product_spec", None)
    if spec is None:
        return

    cases = (
        list(product(*spec.axes))
        if metafunc.config.getoption("--full")
        else list(spec.regular)
    )
    metafunc.parametrize(
        spec.argnames,
        cases,
        ids=[spec.case_id(case) for case in cases],
    )
