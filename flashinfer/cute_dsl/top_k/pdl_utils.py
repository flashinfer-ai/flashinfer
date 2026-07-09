# Copyright (c) 2025 by FlashInfer team.
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
"""PDL (Program-Dependent Launch) grid-dependency control PTX helpers.

These inline PTX wrappers are used by the GVR Top-K kernel to express
grid-level dependency scheduling on Blackwell (sm_100+).
"""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def griddepcontrol_wait(*, loc=None, ip=None) -> None:
    """Wait for the previous kernel's grid to finish before proceeding.

    Ensures the previous grid's blocks have completed and memory-flushed
    before any instruction after this point is issued.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.wait;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def griddepcontrol_launch_dependents(*, loc=None, ip=None) -> None:
    """Hint the hardware to launch dependent kernels earlier.

    Does not affect correctness; only performance. Avoids the latency of
    waiting for the current grid to fully complete before the dependent
    kernel starts executing.
    """
    llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="griddepcontrol.launch_dependents;",
        constraints="",
        has_side_effects=True,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
