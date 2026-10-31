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
"""Immutable request-local metadata for the full FP8 scorer.

No Torch/CUDA dependency: the same rows are validated by the public launch ABI.
Rows are (real output base, valid queries, physical end for each query slot).
"""

from dataclasses import dataclass


def validate_query_group_rows(rows, group_ends, query_tile, output_rows):
    """Validate the host row ABI consumed by the grouped full-score launcher.

    Each row is ``(output_base, valid_queries, *physical_ends)`` with exactly
    ``query_tile`` exclusive KV ends. Real rows must partition
    ``[0, output_rows)`` in order; padded ends must be zero and each group end
    must equal its maximum valid end. Output extent and group ends must fit int32.
    Return None on success and raise ValueError on a violated row contract.
    Inputs are neither mutated nor transferred to a device. This validates
    the row encoding only; the caller validates its device tensors separately.
    """
    if type(query_tile) is not int or query_tile <= 0:
        raise ValueError("positive integer query tile required")
    if (
        type(output_rows) is not int
        or not 0 <= output_rows < 2**31
        or len(rows) != len(group_ends)
    ):
        raise ValueError("invalid grouped output extent")
    cursor = 0
    for row, maximum in zip(rows, group_ends, strict=False):
        if (
            len(row) != 2 + query_tile
            or any(type(x) is not int for x in row)
            or type(maximum) is not int
            or not 0 <= maximum < 2**31
        ):
            raise ValueError("integer grouped row ABI required")
        base, valid, *ends = row
        if base != cursor or not 1 <= valid <= query_tile:
            raise ValueError("group rows must partition real output rows")
        if any(e < 0 or e > maximum for e in ends[:valid]):
            raise ValueError("query end outside group bound")
        if any(ends[valid:]) or max(ends[:valid]) != maximum:
            raise ValueError("invalid padding or group maximum")
        cursor += valid
    if cursor != output_rows:
        raise ValueError("group mapping must cover every real output row")


@dataclass(frozen=True)
class _FP8QueryGroups:
    """Immutable host metadata for request-local full-score query grouping.

    Use :meth:`build` to validate prefixes and construct the fields. Query
    prefixes count real rows; token prefixes count original, uncompressed
    tokens. All stored KV ends and ``physical_prefix`` use physical KV slots.
    No device tensors, query payloads or page tables are allocated here.

    ``rows`` contains ``(output_base, valid_queries, *physical_ends)`` for each
    group, with zero ends in padded slots. ``group_ends`` and ``request_ids``
    give each group's maximum physical end and originating request.
    ``gather_rows`` maps flattened group slots to real input query rows;
    padded slots repeat a valid row in that group and are identified by
    ``invalid_queries`` so callers can zero their gathered Q/weights.
    ``real_ends`` contains one physical end per real output row.
    ``physical_prefix`` accumulates each request's floored physical length.
    Empty-query requests produce no groups. Grouping never crosses a request.
    """

    query_tile: int
    q_prefix: tuple
    token_prefix: tuple
    ratio: int
    rows: tuple
    group_ends: tuple
    request_ids: tuple
    gather_rows: tuple
    invalid_queries: tuple
    real_ends: tuple
    physical_prefix: tuple

    @classmethod
    def build(cls, q_prefix, token_prefix, ratio, query_tile):
        """Build grouping metadata from cumulative host integer prefixes.

        ``q_prefix`` and ``token_prefix`` must have the same batch-plus-one
        length, start at zero and be nondecreasing. A request with Q queries
        and L original KV tokens requires ``L >= Q``. For ratio 1 or 4, query
        i's exclusive physical end is ``(L - Q + i + 1) // ratio``; physical
        request lengths are floored before forming their cumulative prefix.

        ``query_tile`` is a positive integer number of queries per group.
        The last group of a request may contain padded slots, but output
        bases and valid counts cover only real rows. Return an _FP8QueryGroups
        containing tuples independent of the input sequences; raise ValueError
        for invalid prefix, ratio, grouping or row-encoding contracts.
        """
        if type(ratio) is not int or ratio not in (1, 4):
            raise ValueError("ratio must be 1 or 4")
        if type(query_tile) is not int or query_tile <= 0:
            raise ValueError("positive integer query tile required")
        q_prefix, token_prefix = tuple(q_prefix), tuple(token_prefix)
        if len(q_prefix) < 2 or len(q_prefix) != len(token_prefix):
            raise ValueError("matching request prefixes required")
        for prefix in (q_prefix, token_prefix):
            if (
                any(type(x) is not int for x in prefix)
                or prefix[0] != 0
                or any(a > b for a, b in zip(prefix, prefix[1:], strict=False))
            ):
                raise ValueError(
                    "nondecreasing integer prefixes starting at zero required"
                )
        rows, maxima, requests, gathers, invalid, real_ends = [], [], [], [], [], []
        physical = [0]
        for req, (qb, qe, kb, ke) in enumerate(
            zip(q_prefix, q_prefix[1:], token_prefix, token_prefix[1:], strict=False)
        ):
            count, tokens = qe - qb, ke - kb
            if tokens < count:
                raise ValueError("original KV length must be at least Q length")
            physical.append(physical[-1] + tokens // ratio)
            ends = tuple((tokens - count + i + 1) // ratio for i in range(count))
            real_ends.extend(ends)
            for first in range(0, count, query_tile):
                valid = min(query_tile, count - first)
                slot_ends = ends[first : first + valid] + (0,) * (query_tile - valid)
                rows.append((qb + first, valid, *slot_ends))
                maxima.append(max(slot_ends))
                requests.append(req)
                # Dummy slots stay within this request/group, then are zeroed.
                gathers.extend(
                    qb + first + t if t < valid else qb + first
                    for t in range(query_tile)
                )
                invalid.extend(t >= valid for t in range(query_tile))
        validate_query_group_rows(rows, maxima, query_tile, q_prefix[-1])
        return cls(
            query_tile,
            q_prefix,
            token_prefix,
            ratio,
            tuple(rows),
            tuple(maxima),
            tuple(requests),
            tuple(gathers),
            tuple(invalid),
            tuple(real_ends),
            tuple(physical),
        )
