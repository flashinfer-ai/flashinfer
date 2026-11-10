# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lazy resources shared by nested autotune contexts and closed on exit."""

from collections.abc import Callable, Hashable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Any, ContextManager, Optional, TypeVar

T = TypeVar("T")
_resources: ContextVar[Optional[tuple[ExitStack, dict[Hashable, Any]]]] = ContextVar(
    "autotune_resources", default=None
)


@contextmanager
def tuning_resources_scope() -> Iterator[None]:
    """Keep resources until the outermost context exits, including on error.

    Separate threads have independent scopes. Nested contexts inherit the
    enclosing scope so bucket overrides do not repeatedly initialize workers.
    Resources are created only on demand by ``get_tuning_resource``.
    """
    if _resources.get() is not None:
        yield
        return
    with ExitStack() as stack:
        token = _resources.set((stack, {}))
        try:
            yield
        finally:
            _resources.reset(token)


def get_tuning_resource(
    key: Hashable, factory: Callable[[], ContextManager[T]]
) -> Optional[T]:
    """Return a scoped resource, or None outside an autotune context.

    Factories must return context managers whose entry value is not None.
    The autotuner serializes preparation/profiling, so resource creation does
    not need another lock.
    """
    scope = _resources.get()
    if scope is None:
        return None
    stack, values = scope
    if key not in values:
        values[key] = stack.enter_context(factory())
    return values[key]
