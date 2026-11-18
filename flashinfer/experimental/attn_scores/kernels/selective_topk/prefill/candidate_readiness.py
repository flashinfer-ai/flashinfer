# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Release/acquire publication for four candidate banks per row."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


# At most 1M keys per row keeps each bank count below the 24-bit payload limit.
COUNT_BITS = 24
OVERFLOW_BITS = 28
COUNT_MASK = (1 << COUNT_BITS) - 1
BANK_ARRIVAL = 1 << COUNT_BITS
READY_MASK = 0xF << COUNT_BITS
READY_VALUE = 4 << COUNT_BITS
OVERFLOW_UNIT = 1 << OVERFLOW_BITS


@dsl_user_op
def acquire_count(ptr, *, loc=None, ip=None):
    value = llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        "ld.acquire.gpu.global.u32 $0, [$1];",
        "=r,l",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(value)


@dsl_user_op
def exchange_count(ptr, value, *, loc=None, ip=None):
    old = llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [
            ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            cutlass.Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "atom.relaxed.gpu.global.exch.b32 $0, [$1], $2;",
        "=r,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(old)


@cute.jit
def decode_count(value):
    decoded = value
    if (value & READY_MASK) == READY_VALUE:
        decoded = (value & COUNT_MASK) + ((value >> OVERFLOW_BITS) << COUNT_BITS)
    return decoded


@cute.jit
def load_count(ptr):
    return decode_count(acquire_count(ptr))
