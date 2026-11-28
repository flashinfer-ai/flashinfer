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

"""Ordering-only resources of the decode kernel.

They carry no buffer: one keeps a sparse membership row live until every
softmax stream releases it, the other orders schedule-token reuse.
"""

import cutlass.cute as cute
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass


@dataclass(kw_only=True)
class ScheduleTokenThrottleResource(MemoryResource):
    """Order persistent load consumption before the scheduler reuses a slot."""

    @consumer_work
    @cute.jit
    def consume_schedule_token(self, stage_info: StageInfo) -> None:
        """Wait until the load task no longer needs the current schedule token."""
        del stage_info

    @producer_work
    @cute.jit
    def publish_schedule_token(self, stage_info: StageInfo) -> None:
        """Publish that the load task has consumed the current schedule token."""
        del stage_info


@dataclass(kw_only=True)
class SparseMembershipLifetimeResource(MemoryResource):
    """Keep the membership row live until every softmax stream releases it."""
