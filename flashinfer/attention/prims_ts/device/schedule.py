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

"""Task-schedule steps shared by the decode and MLA kernels.

Each helper is one step that a task schedule issues on an optional resource:
page-offset handoffs and the token-throttle head and tail.
"""

from cutlass.experimental.task_scheduling.resources import MemoryResource
from .stage import WorkStage


def page_offsets_produce(
    smem_page_offsets: MemoryResource, label: str, section: WorkStage
) -> None:
    """Produce page IDs for one K/V load label in a schedule section."""
    # ProdAcquire: reserve a page-offset SMEM pipeline stage.
    smem_page_offsets.acquire()
    # ProdWork: load the page-table entries for K0/K1/V0/V1.
    getattr(smem_page_offsets, label)(section=section)
    # ProdCommit: publish the staged offsets to LoadTask.
    smem_page_offsets.commit()


def page_offsets_release(smem_page_offsets: MemoryResource | None) -> None:
    """Release the page-offset stage after the matching K/V load is issued."""
    if smem_page_offsets is None:
        return
    # ConsRelease: the load resource no longer needs these page IDs.
    smem_page_offsets.release()


def schedule_token_throttle_head(
    schedule_token_throttle: MemoryResource | None,
) -> None:
    """Let the scheduler recycle a schedule token slot once Load owns its tile state."""
    if schedule_token_throttle is None:
        return
    schedule_token_throttle.acquire()
    schedule_token_throttle.publish_schedule_token()
    schedule_token_throttle.commit()


def schedule_token_throttle_tail(
    schedule_token_throttle: MemoryResource | None,
) -> None:
    """Pace scheduler schedule token reuse against the persistent Load task."""
    if schedule_token_throttle is None:
        return
    schedule_token_throttle.wait()
    schedule_token_throttle.consume_schedule_token()
    schedule_token_throttle.release()
