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

_FINE_BLOCK_SIZES = (8, 16, 32)
_PREPARED_KV_ROUTE_SIZE = 128
_MAX_KV_ATOM_SIZE = 64
_SIGNED_INT32_MAX = (1 << 31) - 1


def _validate_sparse_block_size(value: object, name: str) -> int:
    """Return a supported semantic sparse block size.

    Query and KV block sizes are independent. Fine blocks map directly to
    Q/KV atoms; coarse blocks must contain an exact number of 64-token atoms.
    Rejecting ``bool`` explicitly is necessary because it is a subclass of
    ``int`` in Python.
    """

    requirement = f"{name} must be 8, 16, 32, or a positive multiple of 64"
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{requirement} expressed as a Python integer, got {type(value).__name__}"
        )
    if value not in _FINE_BLOCK_SIZES and (value <= 0 or value % 64 != 0):
        raise ValueError(requirement)
    if value > _SIGNED_INT32_MAX:
        raise OverflowError(f"{name} must fit in signed int32")
    return value


def _canonical_block_sparse_q_tile_size(q_block_size: int) -> int:
    """Select the canonical PrimTS Q tile for a semantic Q block."""

    q_block_size = _validate_sparse_block_size(q_block_size, "q_block_size")
    if q_block_size in _FINE_BLOCK_SIZES:
        return q_block_size
    return 128 if q_block_size % 128 == 0 else 64


def _block_sparse_kv_atom_size(kv_block_size: int) -> int:
    """Return the independently addressable fragment used in a prepared route.

    This is route-metadata granularity, not the TMA load tile. Coarse BSR
    blocks use KV64 fragments so either half can be addressed independently.
    Their primary TensorMap box is still KV128: B128/B256 (and every multiple
    of B128) naturally produce KV128 loads, while B64 and odd coarse block
    sizes use KV128 whenever the two route fragments are physically adjacent.
    """

    return min(
        _validate_sparse_block_size(kv_block_size, "kv_block_size"),
        _MAX_KV_ATOM_SIZE,
    )


def _prepared_kv_routes_are_block_aligned(
    kv_block_size: int,
    kv_route_size: int,
) -> bool:
    """Return whether each prepared route stays within one semantic BSR block."""

    return (
        _validate_sparse_block_size(kv_block_size, "kv_block_size")
        % kv_route_size
        == 0
    )
