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

"""Task-local variables declared by MLA resource classes (1CTA and 2CTA)."""

from cutlass.experimental.task_scheduling.resources import TaskLocalVariable


def install_task_local_specs(resource: object, specs: tuple[tuple, ...]) -> None:
    """Install TaskLocalVariable fields declared by resource classes."""
    for spec in specs:
        field_name, dtype, default, docs = spec[:4]
        runtime_slot_name = spec[4] if len(spec) > 4 else None
        object.__setattr__(
            resource,
            field_name,
            TaskLocalVariable(
                dtype=dtype,
                default=default,
                docs=docs,
                runtime_slot_name=runtime_slot_name,
            ),
        )
