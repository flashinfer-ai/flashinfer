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

"""Dependency-neutral semantic contract for PrimTS block-sparse attention."""

_SIGNED_INT32_MAX = (1 << 31) - 1


def _validate_sparse_block_size(value: object, name: str) -> int:
    """Return a supported semantic sparse block size.

    Query and KV block sizes are independent, but each must be represented by
    an exact number of 64-token execution fragments. Rejecting ``bool``
    explicitly is necessary because it is a subclass of ``int`` in Python.
    """

    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a positive multiple of 64 expressed as a Python "
            "integer, got bool"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be a positive multiple of 64 expressed as a Python "
            f"integer, got {type(value).__name__}"
        )
    if value <= 0 or value % 64 != 0:
        raise ValueError(f"{name} must be a positive multiple of 64")
    if value > _SIGNED_INT32_MAX:
        raise OverflowError(
            f"{name} must be a positive multiple of 64 fitting in signed int32"
        )
    return value
