"""Pytest support for regular and full Cartesian parameter matrices."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class _ProductSpec:
    argnames: tuple[str, ...]
    argname_groups: tuple[tuple[str, ...], ...]
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

    def expand_case(self, case: tuple[Any, ...]) -> tuple[Any, ...]:
        expanded = []
        for names, value in zip(self.argname_groups, case, strict=True):
            if len(names) == 1:
                expanded.append(value)
            else:
                expanded.extend(value)
        return tuple(expanded)


def pairwise_product_cases(
    axes: Sequence[Iterable[Any]],
) -> tuple[tuple[Any, ...], ...]:
    """Select a deterministic subset covering every pair of axis values."""
    axis_values = tuple(tuple(axis) for axis in axes)
    if not axis_values or any(not axis for axis in axis_values):
        raise ValueError("pairwise_product_cases requires non-empty value axes")
    if len(axis_values) == 1:
        return tuple((value,) for value in axis_values[0])

    axis_sizes = tuple(map(len, axis_values))
    required_pairs = {
        (left_axis, left_value, right_axis, right_value)
        for left_axis in range(len(axis_values))
        for right_axis in range(left_axis + 1, len(axis_values))
        for left_value in range(axis_sizes[left_axis])
        for right_value in range(axis_sizes[right_axis])
    }
    candidate_indices = set()
    # Seed one full row for every required pair. The two fixed positions make
    # that pair coverable; modular values fill the remaining axes without
    # materializing the full Cartesian product.
    for left_axis in range(len(axis_values)):
        for right_axis in range(left_axis + 1, len(axis_values)):
            for left_value in range(axis_sizes[left_axis]):
                for right_value in range(axis_sizes[right_axis]):
                    candidate = tuple(
                        (left_value + right_value + axis_index + left_axis + right_axis)
                        % axis_size
                        for axis_index, axis_size in enumerate(axis_sizes)
                    )
                    candidate = (
                        candidate[:left_axis]
                        + (left_value,)
                        + candidate[left_axis + 1 : right_axis]
                        + (right_value,)
                        + candidate[right_axis + 1 :]
                    )
                    candidate_indices.add(candidate)

    def covered_pairs(candidate):
        return {
            (left_axis, candidate[left_axis], right_axis, candidate[right_axis])
            for left_axis in range(len(axis_values))
            for right_axis in range(left_axis + 1, len(axis_values))
        }

    candidates = sorted(candidate_indices)
    selected = []
    while required_pairs:
        best = max(
            candidates,
            key=lambda candidate: len(covered_pairs(candidate) & required_pairs),
        )
        selected.append(best)
        required_pairs.difference_update(covered_pairs(best))
        candidates.remove(best)

    return tuple(
        tuple(axis_values[axis][value] for axis, value in enumerate(candidate))
        for candidate in selected
    )


def parametrize_product(
    argnames: str | Sequence[str],
    axes: Sequence[Iterable[Any]],
    *,
    regular: Sequence[Sequence[Any]]
    | Callable[[Sequence[Iterable[Any]]], Sequence[Sequence[Any]]],
    ids: Callable[[tuple[Any, ...]], str] | None = None,
):
    """Parametrize a test with selected cases, or the full product under ``--full``."""
    argname_groups = (
        tuple((name.strip(),) for name in argnames.split(","))
        if isinstance(argnames, str)
        else tuple(
            tuple(name.strip() for name in group.split(",")) for group in argnames
        )
    )
    names = tuple(name for group in argname_groups for name in group)
    axis_values = tuple(tuple(axis) for axis in axes)
    selected_regular = regular(axis_values) if callable(regular) else regular
    regular_cases = tuple(tuple(case) for case in selected_regular)

    if not names or any(not name for name in names):
        raise ValueError("parametrize_product requires at least one parameter name")
    if len(argname_groups) != len(axis_values):
        raise ValueError(
            f"parametrize_product received {len(argname_groups)} name groups but "
            f"{len(axis_values)} value axes"
        )
    if any(not axis for axis in axis_values):
        raise ValueError("parametrize_product value axes cannot be empty")

    for group, axis in zip(argname_groups, axis_values, strict=True):
        if len(group) == 1:
            continue
        for value in axis:
            if not isinstance(value, (tuple, list)) or len(value) != len(group):
                raise ValueError(
                    f"parameter group {','.join(group)!r} requires "
                    f"{len(group)} values per axis entry; got {value!r}"
                )

    for case in regular_cases:
        if len(case) != len(argname_groups):
            raise ValueError(
                f"regular case {case!r} has {len(case)} axis values; "
                f"expected {len(argname_groups)}"
            )
        for group, value, axis in zip(argname_groups, case, axis_values, strict=True):
            if value not in axis:
                raise ValueError(
                    f"regular case {case!r} uses unknown {','.join(group)} "
                    f"value {value!r}"
                )

    spec = _ProductSpec(names, argname_groups, axis_values, regular_cases, ids)
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
        [spec.expand_case(case) for case in cases],
        ids=[spec.case_id(case) for case in cases],
    )
