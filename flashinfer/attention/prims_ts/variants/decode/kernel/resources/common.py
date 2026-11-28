# Copyright (c) 2026 by FlashInfer team.
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

"""Base class of the decode kernel resources."""

from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    TaskLocalVariable,
)
from ..device.common import Constexpr
from typing import ClassVar


class DecodeGenResourceBase(MemoryResource):
    """Base for decode-gen resources.

    Captured schedules let the framework manage variable lifecycle per work
    tile.

    consumer_vars / producer_vars are marked Constexpr so that the @cute.jit
    tracer's tree_flatten does NOT traverse them during dynamic-if
    serialization.  The framework accesses these dicts via
    object.__getattribute__ which bypasses both the Constexpr filter and
    the __getattribute__ guard.
    """

    consumer_vars: Constexpr[dict] = None
    producer_vars: Constexpr[dict] = None
    task_local_specs: ClassVar[tuple[tuple, ...]] = ()

    def __post_init__(self) -> None:
        """Materialize task-local variables and placeholder resource state."""
        for name, dtype, default, docs in self.task_local_specs:
            object.__setattr__(
                self,
                name,
                TaskLocalVariable(dtype=dtype, default=default, docs=docs),
            )
        self.init_placeholder_state()

    def init_placeholder_state(self) -> None:
        """Hook for subclasses to install placeholder arrays before tracing."""
        return
