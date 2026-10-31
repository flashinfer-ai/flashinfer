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
"""Immutable tile and pipeline configuration for the full FP8 indexer.

The installed FP8 MMA supports M64/M128 for
CtaGroup.ONE and M128/M256 for CtaGroup.TWO. This scorer passes M directly to
one MMA atom, so the M256 definition cannot launch without a different supported
decomposition/CTA design and corresponding TMEM and consumer ownership review.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _FP8MQAKernelConfig:
    """Explicit configuration for strict-FP32 paged MQA scoring.

    Parameters
    ----------
    compute_block_kv : int
        KV tokens per MMA tile (the M extent). Executable dispatch currently
        supports only 128.
    num_epi_subtiles : int
        Number of head subdivisions used to reduce and store each score tile.
    w_cache_regs : int
        Register-budget setting for the cached-weight epilogue.
    query_tile : int, optional
        Queries grouped within each request, default 1. The logical N extent is
        ``heads * query_tile``; queries from different requests are not grouped.
    num_q_stages : int or None, optional
        Q/weight pipeline depth, either 2 or 3. None uses the scorer default.
    num_kv_stages : int or None, optional
        KV/scale pipeline depth, either 2 or 3. None uses the scorer default.

    Notes
    -----
    Instances are immutable. Construct a new instance and rebuild the plan and
    captured graph to change a configuration. Construction checks field types
    and positive ranges; :meth:`validate_geometry` checks a supplied shape,
    while :meth:`require_dispatchable` checks implemented tile support.
    The scorer additionally validates FP32 accumulation/output, strict math,
    device resources and launch bindings.
    """

    compute_block_kv: int
    num_epi_subtiles: int
    w_cache_regs: int
    query_tile: int = 1
    num_q_stages: int | None = None
    num_kv_stages: int | None = None

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (
                self.compute_block_kv,
                self.num_epi_subtiles,
                self.w_cache_regs,
                self.query_tile,
            )
        ):
            raise TypeError("FP8 configuration fields must be exact integers")
        if (
            min(
                self.compute_block_kv,
                self.num_epi_subtiles,
                self.w_cache_regs,
                self.query_tile,
            )
            <= 0
        ):
            raise ValueError("FP8 configuration fields must be positive")
        if self.compute_block_kv != 128:
            raise ValueError("compute_block_kv must be 128 for the single-CTA scorer")
        for stages in (self.num_q_stages, self.num_kv_stages):
            if stages is not None and (type(stages) is not int or stages not in (2, 3)):
                raise ValueError("explicit Q/KV stages must be 2 or 3")

    @property
    def identity(self):
        """Return the tile/epilogue/cache triple, excluding grouping and depths."""
        return (self.compute_block_kv, self.num_epi_subtiles, self.w_cache_regs)

    @property
    def launch_identity(self):
        """Return the config cache key, including grouping and explicit depths.

        With both depths omitted, return the four-field default-policy key.
        Otherwise append both depth settings, preserving any omitted default.
        """
        legacy = (*self.identity, self.query_tile)
        if self.num_q_stages is None and self.num_kv_stages is None:
            return legacy
        return (*legacy, self.num_q_stages, self.num_kv_stages)

    @property
    def schedule_span(self):
        """KV tokens covered by the tile's two math groups in one schedule step."""
        return 2 * self.compute_block_kv  # Existing two math groups, no CTA change.

    def padded_columns(self, columns):
        """Round a nonnegative integer score width up to the schedule span.

        Returns zero for zero columns. Padding changes storage extent only;
        callers must preserve each row's eligible physical-key bound.
        """
        if type(columns) is not int or columns < 0:
            raise ValueError("column count must be a nonnegative exact integer")
        return (
            (columns + self.schedule_span - 1) // self.schedule_span
        ) * self.schedule_span

    def smem_bytes(self, heads, *, dimension=128, page=128, next_n=1):
        """Estimate staged shared-memory bytes after geometric validation.

        ``heads`` and ``dimension`` count head elements, ``page`` counts KV
        tokens per page, and ``next_n`` counts queries per group. The estimate
        uses three stages for omitted Q/KV depths and the ordinary scorer's
        single-UMMA-stage layout. Compiled resources and device fit require
        separate validation; this estimate is not a measured allocation.
        """
        # Conservative staged allocation estimate, not generated resource data.
        self.validate_geometry(
            heads=heads, dimension=dimension, page=page, next_n=next_n
        )
        n = heads * next_n
        weight_stride = (n * 4 + 127) // 128 * 128
        q_stages = 3 if self.num_q_stages is None else self.num_q_stages
        kv_stages = 3 if self.num_kv_stages is None else self.num_kv_stages
        # For <=3 Q/KV stages and U1 the actual barrier struct fits in 256 B.
        # Each separate data allocation begins on a 128 B boundary.
        align = lambda size: (size + 127) // 128 * 128
        return (
            256
            + 2 * align(self.compute_block_kv * dimension * kv_stages)
            + 2 * align(self.compute_block_kv * 4 * kv_stages)
            + align(n * dimension * q_stages)
            + align(weight_stride * q_stages)
        )

    def validate_geometry(self, *, heads, dimension, page, next_n):
        """Check shape constraints without compiling or allocating resources.

        ``heads`` and ``dimension`` are head count and head dimension;
        ``page`` is KV tokens per page and ``next_n`` is queries per group.
        Raise ValueError for invalid dimensions, page/tile relationships or
        epilogue subdivisions, and TypeError for noninteger geometry. A valid
        geometry still requires :meth:`require_dispatchable` and the scorer's
        dtype, math-policy and device-resource checks before execution.
        """
        if any(type(value) is not int for value in (heads, dimension, page, next_n)):
            raise TypeError("geometry fields must be exact integers")
        if min(heads, dimension, page, next_n) <= 0:
            raise ValueError("geometry fields must be positive")
        if dimension % 32:
            raise ValueError("head dimension must be a multiple of the MMA K32")
        if self.compute_block_kv != 128:
            raise ValueError("unsupported FP8 MMA M extent")
        n = heads * next_n
        if self.query_tile > 1 and next_n != self.query_tile:
            raise ValueError("query grouping must match the configured tile")
        if not 8 <= n <= 256 or n % 8:
            raise ValueError("MMA N must be a multiple of eight in [8, 256]")
        if self.compute_block_kv % page or self.compute_block_kv // page > 4:
            raise ValueError("invalid page-to-compute-tile relationship")
        if heads % self.num_epi_subtiles or (heads // self.num_epi_subtiles) % 4:
            raise ValueError("invalid epilogue head subdivision")

    def require_dispatchable(self):
        """Reject M64/M256; the implemented single-CTA scorer supports M128.

        Raises ValueError for a tile lacking supported schedule/store and TMEM
        ownership. This check does not compile a kernel or query a device.
        """
        if self.compute_block_kv != 128:
            raise ValueError("compute_block_kv must be 128 for the single-CTA scorer")
