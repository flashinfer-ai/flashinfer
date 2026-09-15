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

"""JIT module for the host-side balanced PrimsTS MLA planner."""

import functools

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


@functools.cache
def gen_prims_balanced_mla_plan_module() -> JitSpec:
    """Build the native PrimsTS host scheduler used between graph replays."""

    return gen_jit_spec(
        "prims_balanced_mla_plan",
        [
            jit_env.FLASHINFER_CSRC_DIR / "prims_balanced_mla_plan.cu",
            jit_env.FLASHINFER_CSRC_DIR / "prims_balanced_mla_scheduler.cu",
        ],
    )


__all__ = ["gen_prims_balanced_mla_plan_module"]
