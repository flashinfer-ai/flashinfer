"""Minor GPU metadata and local FP8 gather; no full-table dequantization."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..ple_hash._kernels import _request_ids_kernel, _source_token


@triton.jit
def _compress(ids, token_mask, token_map, num_tokens, compressed, V: tl.constexpr):
    i = tl.program_id(0)
    live = i < tl.load(num_tokens)
    raw = tl.load(ids + i, live, 0).to(tl.int64)
    # Token IDs outside the vocabulary compress to the excluded marker instead
    # of indexing past the token map.
    in_vocab = (raw >= 0) & (raw < V)
    included = tl.load(token_mask + i, live, 0)
    value = tl.load(token_map + raw, live & in_vocab & included, -1)
    tl.store(compressed + i, value)


@triton.jit
def _hash(compressed, starts, slots, history, num_tokens, request_ids, multipliers,
          primes, offsets, hashes, PAD: tl.constexpr):
    t, head = tl.program_id(0), tl.program_id(1)
    req = tl.load(request_ids + t)
    live = (t < tl.load(num_tokens)) & (req >= 0)
    start = tl.load(starts + req, live, 0)
    slot = tl.load(slots + req, live, 0)
    blocked = tl.full((), False, tl.int1)
    mixed = tl.full((), 0, tl.int64)
    order = head // 8 + 2
    for lag in tl.static_range(4):
        source = _source_token(compressed, history, start, slot, t - start, -lag, -1,
                               live, MAX_ORDER=4)
        blocked |= source == -1
        value = tl.where(blocked, PAD, source).to(tl.int64)
        multiplier = tl.load(multipliers + lag)
        mixed ^= tl.where(lag < order, value * multiplier, 0)
    prime = tl.load(primes + head)
    remainder = mixed % prime
    remainder = tl.where(remainder < 0, remainder + prime, remainder)
    result = remainder + tl.load(offsets + head)
    tl.store(hashes + t.to(tl.int64) * 24 + head, tl.where(live, result, -1))


@triton.jit(do_not_specialize=["prepared_tokens"])
def _lookup(
    weight,
    scales,
    hashes,
    num_tokens,
    out,
    prepared_tokens,
    T: tl.constexpr,
    ROWS: tl.constexpr,
    START: tl.constexpr,
    END: tl.constexpr,
    COMPACT: tl.constexpr,
    RESIDENT_SCALES: tl.constexpr,
):
    t, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 256)
    nt = tl.load(num_tokens)
    live = (t < nt) & (t < prepared_tokens) & (nt >= 0) & (nt <= T)
    row = tl.load(hashes + t.to(tl.int64) * 24 + head, live, -1).to(tl.int64)
    local = live & (row >= START) & (row < END) & (row < ROWS)
    if COMPACT:
        local_row = t.to(tl.int64) * 24 + head
    else:
        local_row = tl.where(local, row - tl.full((), START, tl.int64), 0)
    quant = tl.load(weight + local_row * 256 + col.to(tl.int64), local, 0.0).to(
        tl.float32
    )
    scale_row = local_row
    if RESIDENT_SCALES:
        scale_row = tl.where(local, row - tl.full((), START, tl.int64), 0)
    exponent = tl.load(
        scales + scale_row * 8 + (col // 32).to(tl.int64), local, 127
    ).to(tl.uint32)
    # E8M0 byte zero is 2^-127, not floating point zero; 255 is NaN.
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exponent == 0, 2.0**-127, scale)
    scale = tl.where(exponent == 255, float("nan"), scale)
    result = (quant * scale).to(tl.bfloat16)
    tl.store(out + t.to(tl.int64) * 6144 + head * 256 + col, tl.where(local, result, 0))


@torch.library.custom_op("b12x::engram_hash", mutates_args=("hashes", "compressed", "request_ids"))
def hash_op(plan_handle: int, ids: torch.Tensor, token_mask: torch.Tensor, starts: torch.Tensor,
            slots: torch.Tensor, history: torch.Tensor, num_seqs: torch.Tensor,
            num_tokens: torch.Tensor, hashes: torch.Tensor, compressed: torch.Tensor,
            request_ids: torch.Tensor, prepared_tokens: int) -> None:
    from b12x.preparation.types import plan_from_handle, require_prepared
    state = require_prepared(plan_from_handle(plan_handle), "sequence.engram", ids.device)
    state.run(type("_Binding", (), {
        "token_ids": ids, "token_mask": token_mask, "query_start_loc": starts,
        "request_slots": slots, "committed_history": history, "num_seqs": num_seqs,
        "num_tokens": num_tokens, "hash_ids": hashes, "compressed": compressed,
        "request_ids": request_ids,
    })(), prepared_tokens)


@hash_op.register_fake
def _hash_fake(*args, **kwargs):
    return None


@torch.library.custom_op("b12x::engram_lookup", mutates_args=("out",))
def lookup_op(plan_handle: int, weight: torch.Tensor, scale_bytes: torch.Tensor,
              hashes: torch.Tensor, num_tokens: torch.Tensor, out: torch.Tensor,
              prepared_tokens: int, clear_tail: bool) -> None:
    from b12x.preparation.types import plan_from_handle, require_prepared
    state = require_prepared(plan_from_handle(plan_handle), "sequence.engram", weight.device)
    state.run_lookup(type("_Binding", (), {
        "weight": weight, "scale_bytes": scale_bytes, "hash_ids": hashes,
        "num_tokens": num_tokens, "out": out,
    })(), prepared_tokens, clear_tail=clear_tail)


@lookup_op.register_fake
def _lookup_fake(*args, **kwargs):
    return None
