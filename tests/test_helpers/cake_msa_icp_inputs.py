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

Synthetic MiniMax-M3 ICP indexer problems and their CPU oracle.

Resolves one problem description (world size, page size, query length,
batch, context lengths and layout options) into per-request integers,
materialises every rank's tensors in the chain's layouts (FP8 queries, the
row-sharded FP8 key fragments inside compound pages with a configurable
pitch, the block table with a poisoned tail, the per-token geometry) and
provides the exact reference: FP32 full-page scores, the canonical top-16 of
every (token, head) row and the merged owner output.  Shared by the
experimental Cake ICP indexer tests and their benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

HEAD_DIM = 128
NUM_HEADS = 4
TOPK = 16
PAD_PAGE = -2147480000  # poisoned page-table tail: a read is a bug, never a page
E4M3_NAN = 0x7F  # byte pattern of an FP8 E4M3 NaN (poison outside the key fragments)
# Compound page layouts: the index-key fragment (rows_per_rank x 128 bytes) sits
# behind the other page components; the pitch is the whole page.
COMPOUND_LAYOUTS = {
    "nvfp4_compound": (
        9,
        2,
    ),  # offset = 4.5 x fragment, pitch = 5.5 x fragment (45056 / 36864 for 64 rows)
    "fp8_compound": (
        16,
        2,
    ),  # offset = 8 x fragment, pitch = 9 x fragment (73728 / 65536 for 64 rows)
}


def compound_layout(
    rows_per_rank: int, index_pitch: str = "nvfp4_compound"
) -> tuple[int, int]:
    """``(page_pitch_bytes, index_offset_bytes)`` of the production compound page for ``rows_per_rank`` rows."""
    if index_pitch not in COMPOUND_LAYOUTS:
        raise ValueError(f"unknown index_pitch {index_pitch!r}")
    numerator, denominator = COMPOUND_LAYOUTS[index_pitch]
    region = int(rows_per_rank) * HEAD_DIM
    offset = region * numerator // denominator
    return offset + region, offset


@dataclass
class IcpCase:
    """One problem resolved into per-request integers (host side)."""

    label: str
    world_size: int
    page_size: int
    rows_per_rank: int
    query_len: int
    batch: int
    score_columns: int
    page_pitch_bytes: int
    index_offset_bytes: int
    page_table_order: str
    active_rows_pattern: str
    seed: int
    context: list[int]  # per request: visible tokens INCLUDING the decode rows
    positions: list[
        int
    ]  # per token (T = query_len * batch): causal position of the row
    active: list[int]  # per token, 1 = scored / selected
    query_start_loc: list[int]  # [batch + 1]
    pages_per_request: list[int]  # ceil(context / page_size)
    block_table: list[
        list[int]
    ]  # [batch][score_columns], PAD_PAGE beyond the live prefix
    num_physical_pages: int

    @property
    def tokens(self) -> int:
        return self.query_len * self.batch

    @property
    def heads_local(self) -> int:
        return NUM_HEADS // self.world_size


def _resolve_context(params: dict[str, Any], rng: torch.Generator) -> list[int]:
    batch = int(params["batch"])
    ctx = params["context_tokens"]
    pattern = params.get("context_pattern", "uniform")
    if pattern == "uniform":
        return [int(ctx)] * batch
    if pattern == "half_short_half_long":
        short, long_ = int(ctx[0]), int(ctx[1])
        return [short if i < batch // 2 else long_ for i in range(batch)]
    if pattern == "random_uniform":
        lo, hi = int(ctx[0]), int(ctx[1])
        draws = torch.randint(lo, hi + 1, (batch,), generator=rng)
        return [int(v) for v in draws]
    if pattern == "zero_length_every_third_request":
        return [0 if i % 3 == 2 else int(ctx) for i in range(batch)]
    if pattern == "list":
        if len(ctx) != batch:
            raise ValueError(
                f"context_pattern 'list' needs {batch} contexts, got {len(ctx)}"
            )
        return [int(v) for v in ctx]
    raise ValueError(f"unknown context_pattern {pattern!r}")


def _place_newest(
    ctx: int, page_size: int, position_in_block: str, rank_rows: int
) -> int:
    """Position of the newest token of a request with ``ctx`` visible tokens.

    ``ctx`` stays the number of visible tokens; only the alignment of the
    newest token inside its block is adjusted, so the three placements put the
    forced block's newest row at the block start (boundary), in the middle
    (mid) or at the end (end) of the rank fragments.
    """
    if ctx <= 0:
        return -1
    p = ctx - 1
    base = (p // page_size) * page_size
    if position_in_block == "boundary":
        target = base
    elif position_in_block == "mid":
        target = base + min(page_size - 1, (page_size // 2) + 3)
    elif position_in_block == "end":
        target = base + page_size - 1
    else:
        raise ValueError(f"unknown position_in_block {position_in_block!r}")
    return max(target, 0)


def build_case(label: str, params: dict[str, Any]) -> IcpCase:
    """Resolve ``params`` (``world_size``, ``page_size``, ``query_len``, ``batch``,
    ``score_columns``, ``context_tokens``, ``seed`` and the optional layout
    options) into an :class:`IcpCase`."""
    rng = torch.Generator(device="cpu").manual_seed(int(params["seed"]))
    world = int(params["world_size"])
    page = int(params["page_size"])
    rows_per_rank = int(params.get("rows_per_rank", page // world))
    q = int(params["query_len"])
    batch = int(params["batch"])
    columns = int(params["score_columns"])
    contexts = _resolve_context(params, rng)
    pos_mode = params.get("position_in_block", "mid")
    newest = [_place_newest(c, page, pos_mode, rows_per_rank) for c in contexts]
    # history length as seen by the scorer: the newest position + 1 (decode rows already written)
    contexts = [n + 1 if n >= 0 else 0 for n in newest]
    positions: list[int] = []
    active: list[int] = []
    qsl = [0]
    pattern = params.get("active_rows_pattern", "all")
    for b in range(batch):
        for i in range(q):
            positions.append(newest[b] - (q - 1) + i if newest[b] >= 0 else 0)
            is_active = contexts[b] > 0
            if pattern == "padded" and (b * q + i) % 3 == 2:
                is_active = False
            active.append(1 if is_active else 0)
        qsl.append(qsl[-1] + q)
    pages_per_request = [(c + page - 1) // page for c in contexts]
    if max(pages_per_request, default=0) > columns:
        raise ValueError(
            f"{label}: context needs more blocks than score_columns={columns}"
        )
    order = params.get("page_table_order", "identity")
    table: list[list[int]] = []
    if order == "shared_prefix":
        shared = max(pages_per_request) // 4
        private_total = sum(max(0, n - shared) for n in pages_per_request)
        nphys = shared + private_total + 8
        cursor = shared
        for n in pages_per_request:
            own = list(range(shared))[: min(shared, n)]
            rest = list(range(cursor, cursor + max(0, n - shared)))
            cursor += max(0, n - shared)
            table.append(own + rest)
    else:
        nphys = sum(pages_per_request) + 8
        physical = list(range(nphys))
        if order == "shuffled":
            physical = [int(v) for v in torch.randperm(nphys, generator=rng)]
        elif order != "identity":
            raise ValueError(f"unknown page_table_order {order!r}")
        cursor = 0
        for n in pages_per_request:
            table.append(physical[cursor : cursor + n])
            cursor += n
    full_table = [row + [PAD_PAGE] * (columns - len(row)) for row in table]
    default_pitch, default_offset = compound_layout(
        rows_per_rank, params.get("index_pitch", "nvfp4_compound")
    )
    return IcpCase(
        label=label,
        world_size=world,
        page_size=page,
        rows_per_rank=rows_per_rank,
        query_len=q,
        batch=batch,
        score_columns=columns,
        page_pitch_bytes=int(params.get("page_pitch_bytes", default_pitch)),
        index_offset_bytes=int(params.get("index_offset_bytes", default_offset)),
        page_table_order=order,
        active_rows_pattern=pattern,
        seed=int(params["seed"]),
        context=contexts,
        positions=positions,
        active=active,
        query_start_loc=qsl,
        pages_per_request=pages_per_request,
        block_table=full_table,
        num_physical_pages=nphys,
    )


@dataclass
class RankInputs:
    """Device tensors of one rank in the chain's layouts."""

    rank: int
    index_q: torch.Tensor  # fp8 [T, 4, 128]
    index_k: torch.Tensor  # fp8 view [pages, R, 128], stride(0) = page pitch
    compound_storage: torch.Tensor  # uint8 backing of the compound pages (kept alive)
    block_table: torch.Tensor  # int32 [B, columns]
    query_start_loc: torch.Tensor  # int32 [B + 1]
    seq_lens: torch.Tensor  # int32 [B]
    positions: torch.Tensor  # int64 [T]
    active_rows_u8: torch.Tensor  # uint8 [T]
    active_rows_bool: torch.Tensor  # bool [T]
    local_valid_blocks: torch.Tensor  # int32 [T]: readable columns on this rank
    forced_column: (
        torch.Tensor
    )  # int32 [T]: column of the forced block in the readable window, else -1
    forced_block: torch.Tensor  # int32 [T]: global forced block id (position // 128)
    n_ordinary: torch.Tensor  # int32 [T]: min(15, position // 128)
    scan_block_begin: int
    global_block_stride: int
    full_index_keys_host: (
        torch.Tensor
    )  # fp8 [pages, page_size, 128] on the host (oracle input)


def _random_fp8(shape, rng: torch.Generator, *, integer: bool) -> torch.Tensor:
    if integer:
        vals = torch.randint(-4, 5, shape, generator=rng).to(torch.float32)
    else:
        vals = torch.randn(shape, generator=rng) * 0.5
    return vals.to(torch.float8_e4m3fn)


def build_rank_inputs(
    case: IcpCase,
    rank: int,
    device,
    *,
    integer_data: bool = False,
    host_keys: torch.Tensor | None = None,
    host_q: torch.Tensor | None = None,
) -> RankInputs:
    """Materialise one rank's tensors.  Every rank derives the same logical
    pages from the case seed; ``host_keys`` / ``host_q`` reuse an earlier
    rank's host tensors in the same process."""
    rng = torch.Generator(device="cpu").manual_seed(case.seed * 7919 + 17)
    P, R, W = case.page_size, case.rows_per_rank, case.world_size
    T = case.tokens
    if host_keys is None:
        host_keys = _random_fp8(
            (case.num_physical_pages, P, HEAD_DIM), rng, integer=integer_data
        )
    else:
        _ = _random_fp8(
            (case.num_physical_pages, P, HEAD_DIM), rng, integer=integer_data
        )  # keep rng in step
    if host_q is None:
        host_q = _random_fp8((T, NUM_HEADS, HEAD_DIM), rng, integer=integer_data)
    dev = torch.device(device)
    pitch, offset = case.page_pitch_bytes, case.index_offset_bytes
    if offset + R * HEAD_DIM > pitch:
        raise ValueError("index component does not fit the compound page")
    backing = torch.full(
        (case.num_physical_pages * pitch,), E4M3_NAN, dtype=torch.uint8, device=dev
    )
    view = backing.view(torch.float8_e4m3fn).as_strided(
        (case.num_physical_pages, R, HEAD_DIM), (pitch, HEAD_DIM, 1), offset
    )
    # rows [rank * R, (rank + 1) * R) of every logical page belong to this rank
    view.copy_(
        host_keys.reshape(case.num_physical_pages, W, R, HEAD_DIM)[:, rank].to(dev)
    )
    positions = torch.tensor(case.positions, dtype=torch.int64)
    seq_lens = torch.tensor(case.context, dtype=torch.int32)
    # readable columns per token on this rank: n = v // P + int(v % P > rank * R), v = min(seq_len, p + 1)
    req_of_token = torch.repeat_interleave(torch.arange(case.batch), case.query_len)
    visible = torch.minimum(
        seq_lens[req_of_token].to(torch.int64), positions + 1
    ).clamp(min=0)
    n_local = (visible // P + ((visible % P) > rank * R).to(torch.int64)).to(
        torch.int32
    )
    forced_block = (positions // 128).to(torch.int32)
    if P == 128:
        scan_begin, stride = 0, 1
        forced_col = forced_block.clone()
    else:
        # two ranks on 256-token pages: this rank holds 128-row half ``rank`` of every page -> global id 2 * col + rank
        scan_begin, stride = rank, 2
        forced_col = torch.where(
            forced_block % 2 == rank,
            forced_block // 2,
            torch.full_like(forced_block, -1),
        )
    active_u8 = torch.tensor(case.active, dtype=torch.uint8)
    # inactive rows read nothing; keep their geometry harmless
    n_local = torch.where(active_u8.bool(), n_local, torch.zeros_like(n_local))
    # the forced block is excluded locally only where this rank's fragment of it is readable
    # (column < n_local); otherwise the column is -1 (absent) and the merge still injects the global id
    forced_col = torch.where(
        active_u8.bool() & (forced_col < n_local),
        forced_col,
        torch.full_like(forced_col, -1),
    )
    n_ordinary = torch.clamp(forced_block, max=TOPK - 1)
    return RankInputs(
        rank=rank,
        index_q=host_q.to(dev),
        index_k=view,
        compound_storage=backing,
        block_table=torch.tensor(case.block_table, dtype=torch.int32).to(dev),
        query_start_loc=torch.tensor(case.query_start_loc, dtype=torch.int32).to(dev),
        seq_lens=seq_lens.to(dev),
        positions=positions.to(dev),
        active_rows_u8=active_u8.to(dev),
        active_rows_bool=active_u8.bool().to(dev),
        local_valid_blocks=n_local.to(dev),
        forced_column=forced_col.to(dev),
        forced_block=forced_block.to(dev),
        n_ordinary=n_ordinary.to(dev),
        scan_block_begin=scan_begin,
        global_block_stride=stride,
        full_index_keys_host=host_keys,
    )


# --------------------------------------------------------------------------- oracle


def reference_scores(
    case: IcpCase, inputs: RankInputs, tokens: list[int] | None = None
) -> dict[int, torch.Tensor]:
    """CPU FP32 full-page oracle: per active token ``[4, columns]`` scores of
    this rank's fragment (``-inf`` where not visible)."""
    P, R, rank = case.page_size, case.rows_per_rank, inputs.rank
    keys = inputs.full_index_keys_host
    q_all = inputs.index_q.detach().cpu().float()
    owned = (torch.arange(P) // R) == rank
    out: dict[int, torch.Tensor] = {}
    req_of_token = torch.repeat_interleave(torch.arange(case.batch), case.query_len)
    rows = (
        tokens
        if tokens is not None
        else [t for t in range(case.tokens) if case.active[t]]
    )
    for t in rows:
        if not case.active[t]:
            continue
        b = int(req_of_token[t])
        length = case.context[b]
        count = min((length + P - 1) // P, case.score_columns)
        scores = torch.full((NUM_HEADS, case.score_columns), float("-inf"))
        if count > 0:
            pages = torch.tensor(case.block_table[b][:count], dtype=torch.int64)
            if bool(((pages < 0) | (pages >= keys.shape[0])).any()):
                raise AssertionError("oracle would read a poisoned page-table tail")
            k = keys[pages].float()  # [count, P, 128]
            key_pos = torch.arange(count * P).reshape(count, P)
            visible = (
                (key_pos < length) & (key_pos <= case.positions[t]) & owned[None, :]
            )
            dots = (q_all[t] @ k.flatten(0, 1).T).reshape(NUM_HEADS, count, P)
            masked = torch.where(visible[None], dots, torch.tensor(float("-inf")))
            scores[:, :count] = masked.amax(dim=-1)
        out[t] = scores
    return out


def sortable_key(score: torch.Tensor, gid: torch.Tensor) -> torch.Tensor:
    """Canonical 64-bit selection key: score descending, global id ascending,
    ``-0.0 == +0.0``; the larger key wins."""
    s = score.to(torch.float32).clone()
    s[s == 0] = 0.0  # flush -0.0
    bits = s.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    neg = (bits & 0x80000000) != 0
    sortable = torch.where(neg, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    inv_gid = (~gid.to(torch.int64)) & 0xFFFFFFFF
    return (sortable << 32) | inv_gid


def reference_local_topk(
    scores: torch.Tensor, n_valid: int, forced_col: int, scan_begin: int, stride: int
) -> list[tuple[float, int]]:
    """Exact top-16 of one (token, head) row under the canonical key: columns
    ``[0, n_valid)`` minus ``forced_col`` are ordinary; ``(score, gid)`` pairs
    best first, padded with ``(-inf, -1)`` to 16 entries."""
    cols = [c for c in range(n_valid) if c != forced_col]
    vals = scores[cols]
    if bool(torch.isnan(vals).any()):
        raise AssertionError("NaN in an ordinary selector cell")
    gids = torch.tensor([scan_begin + stride * c for c in cols], dtype=torch.int64)
    if len(cols) == 0:
        return [(float("-inf"), -1)] * TOPK
    keys = sortable_key(vals, gids)
    order = torch.argsort(keys, descending=True)[:TOPK]
    out = []
    for i in order.tolist():
        v = float(vals[i])
        out.append((0.0 if v == 0 else v, int(gids[i])))
    while len(out) < TOPK:
        out.append((float("-inf"), -1))
    return out


def reference_merge(
    per_rank: list[list[tuple[float, int]]], forced_block: int, n_ordinary: int
) -> list[int]:
    """Merge the ranks' lists of one row: max per id, canonical key order,
    ``n_ordinary`` ordinary slots, the forced block injected once, ascending
    ids with a ``-1`` tail."""
    best: dict[int, float] = {}
    for lst in per_rank:
        for s, g in lst:
            if g < 0 or g == forced_block:
                continue
            best[g] = max(best.get(g, float("-inf")), s)
    if best:
        ids = torch.tensor(list(best.keys()), dtype=torch.int64)
        vals = torch.tensor(list(best.values()), dtype=torch.float32)
        order = torch.argsort(sortable_key(vals, ids), descending=True)[
            : max(0, n_ordinary)
        ]
        chosen = [int(ids[i]) for i in order]
    else:
        chosen = []
    if forced_block >= 0:
        chosen.append(forced_block)
    chosen = sorted(set(chosen))[:TOPK]
    return chosen + [-1] * (TOPK - len(chosen))


def local_topk_lists(
    case: IcpCase, inputs: RankInputs
) -> dict[int, list[list[tuple[float, int]]]]:
    """Exact local top-16 lists of one rank: ``{token: [list per head]}`` for every token (pads for inactive rows)."""
    scores = reference_scores(case, inputs)
    n_local = inputs.local_valid_blocks.cpu()
    forced_col = inputs.forced_column.cpu()
    empty = torch.full((case.score_columns,), float("-inf"))
    out: dict[int, list] = {}
    for t in range(case.tokens):
        row = scores.get(t)
        out[t] = [
            reference_local_topk(
                row[h] if row is not None else empty,
                int(n_local[t]),
                int(forced_col[t]),
                inputs.scan_block_begin,
                inputs.global_block_stride,
            )
            for h in range(NUM_HEADS)
        ]
    return out


def expected_out(
    case: IcpCase, all_lists: list[dict[int, list]], inputs: RankInputs, rank: int
) -> torch.Tensor:
    """The owner output of ``rank``: int32 ``[T, 4 // world_size, 16]`` =
    ``reference_merge`` over the ranks' lists for the heads ``rank`` owns."""
    hl = case.heads_local
    fb = inputs.forced_block.cpu()
    n_ord = inputs.n_ordinary.cpu()
    out = torch.empty((case.tokens, hl, TOPK), dtype=torch.int32)
    for t in range(case.tokens):
        for hloc in range(hl):
            h = rank * hl + hloc
            lists = [all_lists[r][t][h] for r in range(case.world_size)]
            out[t, hloc] = torch.tensor(
                reference_merge(lists, int(fb[t]), int(n_ord[t])), dtype=torch.int32
            )
    return out


def candidates_match(
    case: IcpCase, inputs: RankInputs, lists: dict[int, list], candidates: torch.Tensor
) -> dict:
    """Compare a rank's ``candidates`` (fp32 ``[T, 4, 16, 2]``) with its exact lists bit for bit."""
    import struct

    def fbits(v: float) -> int:
        return struct.unpack("<I", struct.pack("<f", v))[0]

    c = candidates.detach().cpu()
    sc = c[..., 0].contiguous()
    gi = c[..., 1].contiguous().view(torch.int32)
    mismatches = 0
    example = None
    for t in range(case.tokens):
        for h in range(NUM_HEADS):
            exp = [(fbits(s), g) for s, g in lists[t][h]]
            got = [(fbits(float(sc[t, h, j])), int(gi[t, h, j])) for j in range(TOPK)]
            if got != exp:
                mismatches += 1
                if example is None:
                    example = {"token": t, "head": h, "got": got[:6], "expect": exp[:6]}
    result = {
        "rows_checked": case.tokens * NUM_HEADS,
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }
    if example is not None:
        result["example"] = example
    return result


def case_params(
    *,
    world_size: int,
    query_len: int,
    batch: int,
    context_tokens,
    score_columns: int,
    page_size: int = 128,
    seed: int = 1132,
    **options,
) -> dict[str, Any]:
    """Problem parameters in the form :func:`build_case` reads."""
    params = dict(
        world_size=int(world_size),
        page_size=int(page_size),
        rows_per_rank=int(page_size) // int(world_size),
        query_len=int(query_len),
        batch=int(batch),
        score_columns=int(score_columns),
        context_tokens=context_tokens,
        seed=int(seed),
    )
    params.update(options)
    return params


# Small problems the single-GPU emulation test and the smoke benchmark use.
SMALL_CASES = {
    "tp2_q4_b8_ctx1500": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132001,
    ),
    "tp2_q1_b8_ctx1500": case_params(
        world_size=2,
        query_len=1,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132002,
    ),
    "tp4_q4_b8_ctx1500": case_params(
        world_size=4,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132003,
    ),
    "tp2_p256_q4_b8_ctx3000": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=3000,
        score_columns=24,
        page_size=256,
        seed=1132004,
    ),
    "tp2_q4_b8_padded": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132005,
        active_rows_pattern="padded",
    ),
    "tp2_q4_b8_zero_length": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132006,
        context_pattern="zero_length_every_third_request",
    ),
    "tp2_q4_b8_shuffled": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132007,
        page_table_order="shuffled",
    ),
    "tp2_q4_b8_shared_prefix": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132009,
        page_table_order="shared_prefix",
    ),
    "tp2_q4_b8_fp8pitch": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132010,
        index_pitch="fp8_compound",
    ),
    "tp2_q4_b8_position_end": case_params(
        world_size=2,
        query_len=4,
        batch=8,
        context_tokens=1500,
        score_columns=12,
        seed=1132008,
        position_in_block="end",
    ),
}
