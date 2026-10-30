"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Tests for the experimental FP8 paged MQA prefill logits over UE8M0-scaled
# keys (flashinfer.attn_scores.fp8_ue8m0_paged_mqa_logits_prefill): an FP32
# reference over the cache bytes for causal staircases (cold prompts, appends
# after a cached prefix, several requests, rows at tile edges), untouched
# positions, a row-strided block table, a wider caller-owned output, agreement
# bit for bit with the decode entry point on 4-row groups, the plan's
# coverage, and the input checks.

import dataclasses

import numpy as np
import pytest
import torch

from flashinfer.attn_scores import (
    fp8_ue8m0_paged_mqa_logits_decode,
    fp8_ue8m0_paged_mqa_logits_prefill,
    plan_fp8_ue8m0_paged_mqa_logits_prefill,
)
from flashinfer.experimental.cake_fp8_ue8m0_paged_mqa_logits_prefill import (
    ITEM_FIELDS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    plan_work,
)

HEADS, HEAD_DIM, PAGE = 32, 128, 64
TOKEN_BYTES = HEAD_DIM + 4
SCALE_OFFSET = PAGE * HEAD_DIM
TILE = 128  # keys per tile
ROWS = 8  # rows per work item

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in SUPPORTED_COMPUTE_CAPABILITIES,
        reason=f"needs compute capability in {SUPPORTED_COMPUTE_CAPABILITIES}",
    ),
    pytest.mark.filterwarnings("ignore::flashinfer.api_logging.ExperimentalWarning"),
]


@dataclasses.dataclass(frozen=True)
class Case:
    q: torch.Tensor  # [rows + ROWS, HEADS, HEAD_DIM] float8_e4m3fn
    kv_fused: torch.Tensor  # [num_blocks, PAGE, 1, TOKEN_BYTES] uint8
    weights: torch.Tensor  # [rows + ROWS, HEADS] float32
    block_tables: torch.Tensor  # [requests, width] int32, -1 past each request's pages
    row_offsets: tuple[int, ...]
    row_counts: tuple[int, ...]
    prefixes: tuple[int, ...]

    def lengths(self) -> list[tuple[int, int]]:
        """(row, its visible key count) for every scored row."""
        return [
            (offset + j, prefix + j + 1)
            for offset, count, prefix in zip(
                self.row_offsets, self.row_counts, self.prefixes, strict=True
            )
            for j in range(count)
        ]


def make_case(
    shapes: list[tuple[int, int]], seed: int = 0, gap: int = 0, table_stride: int = 1
) -> Case:
    """Requests of (rows, prefix), ``gap`` unscored rows between requests;
    keys of per-token magnitudes 2^-8 .. 2^7 quantized like vLLM's indexer
    cache: scale 2^ceil(log2(amax / 448)), stored as four equal UE8M0 bytes
    after the block's FP8 values; shuffled blocks. ``table_stride`` spaces the
    block-table rows (a row-strided view)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-(rows + prefix) // PAGE) for rows, prefix in shapes]
    num_blocks = sum(pages) + 3
    order = torch.randperm(num_blocks, generator=generator, device="cuda")
    table = torch.full(
        (len(shapes) * table_stride, max(pages) + 2),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    start = 0
    for b, count in enumerate(pages):
        table[b * table_stride, :count] = order[start : start + count].to(torch.int32)
        start += count
    magnitude = torch.randint(
        -8, 8, (num_blocks, PAGE, 1), generator=generator, device="cuda"
    )
    keys = torch.randn(num_blocks, PAGE, HEAD_DIM, generator=generator, device="cuda")
    keys = keys * torch.exp2(magnitude.float())
    exponent = torch.ceil(torch.log2(keys.abs().amax(-1).clamp(min=1e-4) / 448.0))
    values = (keys / torch.exp2(exponent)[..., None]).to(torch.float8_e4m3fn)
    scale_bytes = (exponent + 127).to(torch.uint8)[..., None].expand(-1, -1, 4)
    kv_fused = torch.cat(
        (values.view(torch.uint8).flatten(1), scale_bytes.flatten(1)), dim=1
    ).view(num_blocks, PAGE, 1, TOKEN_BYTES)
    offsets, offset = [], 0
    for rows, _ in shapes:
        offsets.append(offset)
        offset += rows + gap
    total = offset + ROWS
    q = torch.randn(total, HEADS, HEAD_DIM, generator=generator, device="cuda")
    weights = torch.randn(total, HEADS, generator=generator, device="cuda")
    return Case(
        q=(q * 4).to(torch.float8_e4m3fn),
        kv_fused=kv_fused,
        weights=weights * HEADS**-0.5,
        block_tables=table[::table_stride],
        row_offsets=tuple(offsets),
        row_counts=tuple(rows for rows, _ in shapes),
        prefixes=tuple(prefix for _, prefix in shapes),
    )


def reference(case: Case) -> dict[int, torch.Tensor]:
    """Each scored row's FP32 scores over the keys it sees, from the cache bytes."""
    blocks = case.kv_fused.view(case.kv_fused.shape[0], -1)
    result = {}
    for b, (offset, count, prefix) in enumerate(
        zip(case.row_offsets, case.row_counts, case.prefixes, strict=True)
    ):
        length = prefix + count
        pages = -(-length // PAGE)
        own = blocks[case.block_tables[b, :pages].long()]
        values = own[:, :SCALE_OFFSET].contiguous().view(torch.float8_e4m3fn)
        values = values.float().view(-1, HEAD_DIM)[:length]
        exponents = own[:, SCALE_OFFSET:].reshape(pages, PAGE, 4)[..., 0]
        scales = torch.exp2(exponents.float() - 127).view(-1)[:length]
        for j in range(count):
            visible = prefix + j + 1
            per_head = (case.q[offset + j].float() @ values[:visible].T).relu()
            weighted = per_head * case.weights[offset + j, :, None]
            result[offset + j] = weighted.sum(0) * scales[:visible]
    return result


def run(case: Case, pitch: int | None = None) -> torch.Tensor:
    work = plan_fp8_ue8m0_paged_mqa_logits_prefill(
        case.row_offsets, case.row_counts, case.prefixes, case.q.device
    )
    longest = max(
        prefix + count
        for count, prefix in zip(case.row_counts, case.prefixes, strict=True)
    )
    out = torch.full(
        (case.q.shape[0], pitch or longest), float("nan"), device=case.q.device
    )
    result = fp8_ue8m0_paged_mqa_logits_prefill(
        case.q, case.kv_fused, case.weights, case.block_tables, work, out=out
    )
    assert result is out
    torch.cuda.synchronize()
    return out


def check_rows(out: torch.Tensor, case: Case) -> None:
    """Scores where a row sees, untouched (the NaN fill) everywhere else."""
    want = reference(case)
    scored = set()
    for row, length in case.lengths():
        scored.add(row)
        got = out[row]
        torch.testing.assert_close(
            got[:length], want[row], rtol=1e-3, atol=1e-4 * float(want[row].abs().max())
        )
        assert got[length:].isnan().all(), f"row {row} written past its keys"
    for row in range(out.shape[0]):
        if row not in scored:
            assert out[row].isnan().all(), f"unscored row {row} written"


SHAPES = {
    "cold": [(300, 0)],
    "append": [(77, 5000)],
    "one_row": [(1, 129)],
    "tile_edges": [(8, TILE - 8), (9, TILE - 1), (16, 2 * TILE)],
    "mixed": [(13, 0), (64, 1000), (9, 70000), (40, 3)],
    "long_prefix": [(24, 200000)],
}


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference(shape: str) -> None:
    case = make_case(SHAPES[shape])
    check_rows(run(case), case)


def test_gaps_strided_table_and_wide_out() -> None:
    """Rows between requests are not scored; a row-strided block table and a
    wider output pitch."""
    case = make_case([(20, 300), (33, 0), (5, 9000)], gap=11, table_stride=3)
    longest = max(c + p for c, p in zip(case.row_counts, case.prefixes, strict=True))
    check_rows(run(case, pitch=longest + 37), case)


def test_default_out_is_padded_for_topk() -> None:
    case = make_case([(10, 1001)])
    work = plan_fp8_ue8m0_paged_mqa_logits_prefill(
        case.row_offsets, case.row_counts, case.prefixes, case.q.device
    )
    out = fp8_ue8m0_paged_mqa_logits_prefill(
        case.q, case.kv_fused, case.weights, case.block_tables, work
    )
    assert out.shape == (10, 1012) and out.stride(0) % 4 == 0
    want = reference(case)
    for row, length in case.lengths():
        torch.testing.assert_close(
            out[row, :length],
            want[row],
            rtol=1e-3,
            atol=1e-4 * float(want[row].abs().max()),
        )


def test_matches_decode_entry_point_bitwise() -> None:
    """Rows in 4-row groups are fp8_ue8m0_paged_mqa_logits_decode's next_n = 4
    staircase: the same MMA and summation order, so the same bits."""
    rows, prefix = 64, 3000
    case = make_case([(rows, prefix)])
    out = run(case)
    groups = rows // 4
    decode = fp8_ue8m0_paged_mqa_logits_decode(
        case.q[:rows].view(groups, 4, HEADS, HEAD_DIM),
        case.kv_fused,
        case.weights[:rows],
        case.block_tables[:1].expand(groups, -1),
        prefix + 4 * torch.arange(1, groups + 1, dtype=torch.int32, device="cuda"),
        prefix + rows,
    )
    for row, length in case.lengths():
        assert torch.equal(
            out[row, :length].view(torch.int32), decode[row, :length].view(torch.int32)
        ), row


@pytest.mark.parametrize("sms", [1, 7, 148])
def test_plan_covers_every_row_and_tile_once(sms: int) -> None:
    counts, prefixes, offsets = [300, 9, 64, 0, 3], [0, 70000, 1000, 5, 127], []
    offset = 0
    for count in counts:
        offsets.append(offset)
        offset += count + 2
    work = plan_work(offsets, counts, prefixes, sms)
    assert work.dtype == np.int32 and work.shape[1] == ITEM_FIELDS
    seen: dict[tuple[int, int], int] = {}
    for first_row, request, first_len, rows, t_begin, t_end, *_ in work.tolist():
        assert 1 <= rows <= ROWS and t_begin < t_end
        j0 = first_row - offsets[request]
        assert first_len == prefixes[request] + j0 + 1
        for tile in range(t_begin, t_end):
            seen[(first_row, tile)] = seen.get((first_row, tile), 0) + 1
    expected = {
        (offsets[i] + j0, tile)
        for i, (count, prefix) in enumerate(zip(counts, prefixes, strict=True))
        for j0 in range(0, count, ROWS)
        for tile in range(-(-(prefix + min(j0 + ROWS, count)) // TILE))
    }
    assert set(seen) == expected and set(seen.values()) == {1}


def test_input_checks() -> None:
    case = make_case([(16, 100)])
    work = plan_fp8_ue8m0_paged_mqa_logits_prefill(
        case.row_offsets, case.row_counts, case.prefixes, case.q.device
    )
    out = torch.empty(case.q.shape[0], 128, device="cuda")
    call = fp8_ue8m0_paged_mqa_logits_prefill
    with pytest.raises(ValueError, match="at least 8 rows"):
        call(
            case.q[:4],
            case.kv_fused,
            case.weights[:4],
            case.block_tables,
            work,
            out=out,
        )
    with pytest.raises(ValueError, match="weights"):
        call(
            case.q,
            case.kv_fused,
            case.weights.double(),
            case.block_tables,
            work,
            out=out,
        )
    with pytest.raises(ValueError, match="out must hold"):
        call(
            case.q,
            case.kv_fused,
            case.weights,
            case.block_tables,
            work,
            out=torch.empty(case.q.shape[0], 64, device="cuda"),
        )
    with pytest.raises(ValueError, match="block_tables"):
        call(
            case.q, case.kv_fused, case.weights, case.block_tables.long(), work, out=out
        )
    # 116 keys need 2 blocks; a 1-block table would read past its rows.
    with pytest.raises(ValueError, match="2 blocks per request"):
        call(
            case.q, case.kv_fused, case.weights, case.block_tables[:, :1], work, out=out
        )
    with pytest.raises(ValueError, match="host"):
        plan_fp8_ue8m0_paged_mqa_logits_prefill(
            torch.tensor([0], device="cuda"), [16], [100], case.q.device
        )


def test_public_apis_are_marked_experimental() -> None:
    assert plan_fp8_ue8m0_paged_mqa_logits_prefill.is_experimental
    assert fp8_ue8m0_paged_mqa_logits_prefill.is_experimental
