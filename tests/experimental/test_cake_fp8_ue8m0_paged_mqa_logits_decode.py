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

# Tests for the experimental FP8 paged MQA decode logits over UE8M0-scaled
# keys (flashinfer.attn_scores.fp8_ue8m0_paged_mqa_logits_decode): an FP32
# reference over the cache bytes for next_n 1, 2 and 4, the -inf and untouched
# positions, a row-strided per-token block table, a wider caller-owned output,
# CUDA-graph replay with changed lengths, agreement with fp8_paged_mqa_logits
# on the same keys with FP32 scales, and the input checks.

import dataclasses

import pytest
import torch

from flashinfer.attn_scores import (
    fp8_paged_mqa_logits,
    fp8_ue8m0_paged_mqa_logits_decode,
)
from flashinfer.experimental.cake_fp8_ue8m0_paged_mqa_logits_decode import (
    NEXT_NS,
    SUPPORTED_COMPUTE_CAPABILITIES,
)

HEADS, HEAD_DIM, PAGE = 32, 128, 64
TOKEN_BYTES = HEAD_DIM + 4
SCALE_OFFSET = PAGE * HEAD_DIM
TILE = 256  # positions per tile: two math warpgroups of 128 keys

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
    q: torch.Tensor  # [B, next_n, HEADS, HEAD_DIM] float8_e4m3fn
    kv_fused: torch.Tensor  # [num_blocks, PAGE, 1, TOKEN_BYTES] uint8
    weights: torch.Tensor  # [B * next_n, HEADS] float32
    block_tables: torch.Tensor  # [B, width] int32, -1 past each request's pages
    seq_lens: torch.Tensor  # [B] int32
    max_seq_len: int

    @property
    def next_n(self) -> int:
        return self.q.shape[1]

    @property
    def rows(self) -> int:
        return self.q.shape[0] * self.next_n


def make_case(seq_lens: list[int], next_n: int, seed: int = 0) -> Case:
    """Keys of per-token magnitudes 2^-8 .. 2^7 quantized like vLLM's indexer
    cache: scale 2^ceil(log2(amax / 448)), stored as four equal UE8M0 bytes
    after the block's FP8 values; shuffled blocks."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pages = [-(-length // PAGE) for length in seq_lens]
    num_blocks = sum(pages) + 3
    order = torch.randperm(num_blocks, generator=generator, device="cuda")
    block_tables = torch.full(
        (len(seq_lens), max(pages, default=0) + 2), -1, dtype=torch.int32, device="cuda"
    )
    start = 0
    for b, count in enumerate(pages):
        block_tables[b, :count] = order[start : start + count].to(torch.int32)
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
    q = torch.randn(
        len(seq_lens), next_n, HEADS, HEAD_DIM, generator=generator, device="cuda"
    )
    weights = torch.randn(
        len(seq_lens) * next_n, HEADS, generator=generator, device="cuda"
    )
    return Case(
        q=(q * 4).to(torch.float8_e4m3fn),
        kv_fused=kv_fused,
        weights=weights * HEADS**-0.5,
        block_tables=block_tables,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        max_seq_len=max(seq_lens, default=0),
    )


def reference(case: Case) -> list[torch.Tensor]:
    """Each row's FP32 scores over the positions it sees, from the cache bytes."""
    blocks = case.kv_fused.view(case.kv_fused.shape[0], -1)
    rows = []
    for b, length in enumerate(case.seq_lens.tolist()):
        pages = -(-length // PAGE)
        own = blocks[case.block_tables[b, :pages].long()]
        values = own[:, :SCALE_OFFSET].contiguous().view(torch.float8_e4m3fn)
        values = values.float().view(-1, HEAD_DIM)[:length]
        exponents = own[:, SCALE_OFFSET:].reshape(pages, PAGE, 4)[..., 0]
        scales = torch.exp2(exponents.float() - 127).view(-1)[:length]
        for t in range(case.next_n):
            visible = max(length - case.next_n + t + 1, 0)
            per_head = (case.q[b, t].float() @ values[:visible].T).relu()
            weighted = per_head * case.weights[b * case.next_n + t, :, None]
            rows.append(weighted.sum(0) * scales[:visible])
    return rows


def check_rows(out: torch.Tensor, case: Case) -> None:
    """Scores where a row sees, -inf through the end of its request's last
    tile, untouched (the NaN fill) after it."""
    lengths = case.seq_lens.tolist()
    for row, want in enumerate(reference(case)):
        got = out[row]
        visible = want.numel()
        tile_end = min(-(-lengths[row // case.next_n] // TILE) * TILE, got.numel())
        if visible:
            torch.testing.assert_close(
                got[:visible], want, rtol=1e-3, atol=1e-4 * float(want.abs().max())
            )
        assert torch.isneginf(got[visible:tile_end]).all(), f"row {row}: masked"
        assert got[tile_end:].isnan().all(), f"row {row}: past the last tile"


def run(case: Case, output: torch.Tensor | None, **overrides) -> torch.Tensor:
    args = dict(
        q=case.q,
        kv_fused=case.kv_fused,
        weights=case.weights,
        block_tables=case.block_tables,
        seq_lens=case.seq_lens,
        max_seq_len=case.max_seq_len,
        out=output,
    )
    return fp8_ue8m0_paged_mqa_logits_decode(**{**args, **overrides})


def nan_output(case: Case, rows: int = 0, cols: int = 0) -> torch.Tensor:
    shape = (rows or case.rows, cols or case.max_seq_len)
    return torch.full(shape, float("nan"), device="cuda")


def misaligned_output(case: Case) -> torch.Tensor:
    """A contiguous [rows, max_seq_len] buffer 4 bytes past an aligned one."""
    flat = torch.full((case.rows * case.max_seq_len + 1,), float("nan"), device="cuda")
    return flat[1:].view(case.rows, case.max_seq_len)


CASES = [
    [1],
    [1000],
    [64, 256, 257, 511],
    [0, 700, 3],
    [5000, 1500, 70000],
    list(range(1, 3001, 10)),  # more requests than SMs: one CTA each
]


@pytest.mark.parametrize("next_n", NEXT_NS)
@pytest.mark.parametrize("seq_lens", CASES, ids=lambda lens: f"{len(lens)}x{max(lens)}")
def test_matches_reference(seq_lens: list[int], next_n: int) -> None:
    case = make_case(seq_lens, next_n)
    out = nan_output(case)
    logits = run(case, out)
    assert logits.shape == (case.rows, case.max_seq_len)
    assert logits.data_ptr() == out.data_ptr()
    check_rows(out, case)


@pytest.mark.parametrize("next_n", NEXT_NS)
def test_allocates_the_output(next_n: int) -> None:
    case = make_case([300, 4000], next_n)
    logits = run(case, None)
    assert logits.shape == (case.rows, case.max_seq_len)
    assert logits.dtype == torch.float32
    lengths = case.seq_lens.tolist()
    for row, want in enumerate(reference(case)):
        visible = want.numel()
        torch.testing.assert_close(
            logits[row, :visible], want, rtol=1e-3, atol=1e-4 * float(want.abs().max())
        )
        tile_end = min(-(-lengths[row // next_n] // TILE) * TILE, case.max_seq_len)
        assert torch.isneginf(logits[row, visible:tile_end]).all()


@pytest.mark.parametrize("next_n", NEXT_NS)
def test_row_strided_block_table(next_n: int) -> None:
    """A per-token table (a request's next_n rows equal) read in place
    through table[::next_n]."""
    case = make_case([5000, 1500, 70000], next_n)
    per_token = case.block_tables.repeat_interleave(next_n, dim=0)
    compact = run(case, nan_output(case))
    strided = run(case, nan_output(case), block_tables=per_token[::next_n])
    torch.testing.assert_close(strided, compact, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("next_n", NEXT_NS)
def test_wider_output(next_n: int) -> None:
    """Extra rows stay untouched; the column count is the row pitch."""
    case = make_case([5000, 1500, 70000], next_n)
    tight = run(case, nan_output(case))
    out = nan_output(case, rows=case.rows + 3, cols=case.max_seq_len + 100)
    wide = run(case, out)
    assert wide.shape == tight.shape and wide.data_ptr() == out.data_ptr()
    torch.testing.assert_close(wide, tight, rtol=0, atol=0, equal_nan=True)
    assert out[case.rows :].isnan().all()


@pytest.mark.parametrize("next_n", NEXT_NS)
def test_cuda_graph_replay_with_changed_lengths(next_n: int) -> None:
    case = make_case([5000, 1500, 70000], next_n)
    out = nan_output(case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(case, out)  # compile and warm up outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(case, out)
    # Within each request's pages: shorter, and up to its last page's end.
    for lengths in ([4000, 1536, 65000], [5000, 1, 70000]):
        case.seq_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check_rows(out, case)


@pytest.mark.parametrize("next_n", NEXT_NS)
def test_agrees_with_fp32_scales(next_n: int) -> None:
    """fp8_paged_mqa_logits on the same keys with each exponent stored as an
    FP32 scale computes the same scores up to the FP32 summation order."""
    case = make_case([5000, 1500, 70000], next_n)
    blocks = case.kv_fused.view(case.kv_fused.shape[0], -1)
    exponents = blocks[:, SCALE_OFFSET:].reshape(-1, PAGE, 4)[..., 0]
    fp32_scales = torch.exp2(exponents.float() - 127)
    kv_fp32 = torch.cat(
        (blocks[:, :SCALE_OFFSET], fp32_scales.view(torch.uint8)), dim=1
    ).view(case.kv_fused.shape)
    ours = run(case, nan_output(case))
    theirs = fp8_paged_mqa_logits(
        case.q,
        kv_fp32,
        case.weights,
        case.block_tables,
        case.seq_lens,
        case.max_seq_len,
    )
    lengths = case.seq_lens.tolist()
    for row in range(case.rows):
        visible = max(lengths[row // next_n] - next_n + row % next_n + 1, 0)
        want = theirs[row, :visible]
        torch.testing.assert_close(
            ours[row, :visible], want, rtol=1e-4, atol=1e-5 * float(want.abs().max())
        )


def test_empty_batch() -> None:
    case = make_case([], 1)
    logits = run(case, None, max_seq_len=100)
    assert logits.shape == (0, 100)


@pytest.mark.parametrize(
    "overrides, match",
    [
        (lambda c: {"q": c.q[:, :1].expand(-1, 3, -1, -1).contiguous()}, "next_n"),
        (lambda c: {"q": c.q.repeat(1, 1, 2, 1)}, r"q must be"),
        (lambda c: {"kv_fused": c.kv_fused.view(-1, PAGE, TOKEN_BYTES)}, "kv_fused"),
        (lambda c: {"weights": c.weights[:, :16]}, "weights"),
        (
            lambda c: {"block_tables": c.block_tables.t().contiguous().t()},
            "block_tables",
        ),
        (lambda c: {"seq_lens": c.seq_lens.long()}, "seq_lens"),
        (lambda c: {"max_seq_len": 1.5}, "max_seq_len"),
        (lambda c: {"out": nan_output(c, cols=c.max_seq_len - 1)}, "out must be"),
        (lambda c: {"out": misaligned_output(c)}, "16-byte aligned"),
    ],
)
def test_rejects_unsupported_inputs(overrides, match: str) -> None:
    case = make_case([1000, 300], 1)
    with pytest.raises(ValueError, match=match):
        run(case, nan_output(case), **overrides(case))
