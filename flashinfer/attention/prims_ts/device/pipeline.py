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

"""Pipeline variants shared by the context and MLA kernels."""

import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, dsl_user_op, if_generate
from cutlass.pipeline import PipelineAsync, PipelineState
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class StructuredWaitPipelineAsync(PipelineAsync):
    """PipelineAsync with an explicit public-primitive retry loop."""

    @dsl_user_op
    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self.retry_wait(self.sync_object_full, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.retry_wait(self.sync_object_empty, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def retry_wait(
        self,
        sync_object: object,
        state: PipelineState,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        while not sync_object.try_wait(
            state.index,
            state.phase,
            loc=loc,
            ip=ip,
        ):
            pass
