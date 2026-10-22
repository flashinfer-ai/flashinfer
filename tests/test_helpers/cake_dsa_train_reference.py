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

Seeded inputs and the chunked FP64 reference of the DSA sparse-attention
training problem (64 query heads, 512 latent + 64 rope, top-k selection),
shared by ``tests/experimental/test_cake_dsa_train.py``,
``tests/experimental/test_cake_dsa_train_api.py`` and
``benchmarks/bench_cake_dsa_train.py``.  The protocol follows the reference
tests of the FA sparse-MLA backward: per-document causal random top-k with
``-1`` padding, BF16 inputs, FP64 reference from the same BF16 values.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional

import torch

NUM_HEADS = 64
D_LATENT = 512
D_ROPE = 64
D_QK = D_LATENT + D_ROPE
DEFAULT_TOPK = 2048
DEFAULT_SCALE = D_QK**-0.5


def causal_topk_local(
    seq_q: int,
    seq_k: int,
    topk: int,
    gen: torch.Generator,
    device,
    *,
    self_including: bool = False,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Document-relative causal top-k for one document.

    Query ``i`` sits at key position ``seq_k - seq_q + i`` and may attend keys
    ``[0, pos]``: ``min(pos + 1, topk)`` distinct random valid slots first, then
    ``-1`` padding.  ``self_including`` forces the query's own key into the
    selection (the peaked-attention configuration).
    """
    out = torch.empty(seq_q, topk, dtype=torch.int32, device=device)
    n_keys = max(seq_k, topk)
    key_idx = torch.arange(n_keys, device=device)
    for r0 in range(0, seq_q, chunk_rows):
        r1 = min(seq_q, r0 + chunk_rows)
        pos = torch.arange(r0, r1, device=device) + (seq_k - seq_q)
        scores = torch.rand(r1 - r0, n_keys, device=device, generator=gen)
        if self_including:
            scores[torch.arange(r1 - r0, device=device), pos] = 2.0
        invalid = (key_idx[None, :] > pos[:, None]) | (key_idx >= seq_k)[None, :]
        scores.masked_fill_(invalid, float("-inf"))
        # Descending argsort rather than topk: identical selection for distinct random scores, and
        # independent of any vendor-specific topk override (masked slots sort last and become -1).
        idx = scores.argsort(dim=-1, descending=True)[:, :topk]
        val = scores.gather(-1, idx)
        out[r0:r1] = idx.masked_fill(torch.isinf(val), -1).to(torch.int32)
    return out


def _randn_bf16(shape, gen, device, chunk_rows: int = 4096) -> torch.Tensor:
    out = torch.empty(*shape, dtype=torch.bfloat16, device=device)
    for r0 in range(0, shape[0], chunk_rows):
        r1 = min(shape[0], r0 + chunk_rows)
        out[r0:r1].copy_(torch.randn(r1 - r0, *shape[1:], device=device, generator=gen))
    return out


@dataclass
class Inputs:
    """One problem in the split contract, with both index representations."""

    seq_q: list
    seq_k: list
    topk: int
    q_latent: torch.Tensor
    q_rope: torch.Tensor
    kv_latent: torch.Tensor
    k_rope: torch.Tensor
    dout: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    idx_local: torch.Tensor  # document-relative, -1 padded (valid first)
    idx_global: torch.Tensor  # global key rows, -1 padded
    topk_length: torch.Tensor
    own_key: torch.Tensor  # global key row of each query's own token

    @property
    def total_q(self) -> int:
        return int(self.q_latent.shape[0])

    @property
    def total_k(self) -> int:
        return int(self.kv_latent.shape[0])

    @property
    def max_seqlen_q(self) -> int:
        return max(self.seq_q)

    @property
    def max_seqlen_k(self) -> int:
        return max(self.seq_k)

    def bytes_inputs(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.q_latent,
                self.q_rope,
                self.kv_latent,
                self.k_rope,
                self.dout,
                self.idx_local,
                self.idx_global,
            )
        )


def globalize_gather_indices_loop(
    gather_kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Plain-Python reference of the varlen index rule (host tensors; loop-sized problems).

    Row ``t`` of document ``d`` (``cu_seqlens_q[d] <= t < cu_seqlens_q[d + 1]``) is the query at key
    position ``(seqlen_k[d] - seqlen_q[d]) + (t - cu_seqlens_q[d])`` of its document: the query segment
    is the tail of its key prefix.  Slot ``idx`` is valid when ``0 <= idx < seqlen_k[d]`` and, with
    ``causal``, ``idx <= that position``; valid slots become ``idx + cu_seqlens_k[d]``, the others ``-1``.
    A zero-length query segment contributes no rows.
    """
    local = gather_kv_indices.cpu().tolist()
    cu_q = [int(v) for v in cu_seqlens_q.cpu().tolist()]
    cu_k = [int(v) for v in cu_seqlens_k.cpu().tolist()]
    rows = []
    for d in range(len(cu_q) - 1):
        seqlen_q, seqlen_k = cu_q[d + 1] - cu_q[d], cu_k[d + 1] - cu_k[d]
        for i in range(seqlen_q):
            position = (seqlen_k - seqlen_q) + i
            rows.append(
                [
                    idx + cu_k[d]
                    if 0 <= idx < seqlen_k and (not causal or idx <= position)
                    else -1
                    for idx in local[cu_q[d] + i]
                ]
            )
    return torch.tensor(rows, dtype=torch.int32).reshape(
        len(rows), int(gather_kv_indices.shape[1])
    )


def make_inputs(
    seq_q,
    seq_k,
    *,
    seed: int,
    topk: int = DEFAULT_TOPK,
    self_including: bool = False,
    beta: float = 0.0,
    device="cuda",
) -> Inputs:
    """Seeded BF16 inputs for documents of lengths ``seq_q`` / ``seq_k``.

    ``beta > 0`` adds ``beta * (own key)`` to every query so its own key
    dominates the softmax (peaked attention); the query's own key must be in
    the selection (``self_including``) for that to be meaningful.
    """
    seq_q, seq_k = [int(x) for x in seq_q], [int(x) for x in seq_k]
    if len(seq_q) != len(seq_k) or any(
        lq > lk for lq, lk in zip(seq_q, seq_k, strict=True)
    ):
        raise ValueError(
            "documents need seq_q <= seq_k, one key length per query length"
        )
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total_q, total_k = sum(seq_q), sum(seq_k)
    q_latent = _randn_bf16((total_q, NUM_HEADS, D_LATENT), gen, device)
    q_rope = _randn_bf16((total_q, NUM_HEADS, D_ROPE), gen, device)
    kv_latent = _randn_bf16((total_k, D_LATENT), gen, device)
    k_rope = _randn_bf16((total_k, D_ROPE), gen, device)
    dout = _randn_bf16((total_q, NUM_HEADS, D_LATENT), gen, device)
    cu_q = [0] + torch.tensor(seq_q).cumsum(0).tolist()
    cu_k = [0] + torch.tensor(seq_k).cumsum(0).tolist()
    locals_, globals_, own = [], [], []
    for d, (lq, lk) in enumerate(zip(seq_q, seq_k, strict=True)):
        loc = causal_topk_local(
            lq, lk, topk, gen, device, self_including=self_including
        )
        locals_.append(loc)
        globals_.append(torch.where(loc >= 0, loc + cu_k[d], loc))
        own.append(torch.arange(lq, device=device) + (cu_k[d] + lk - lq))
    idx_local = torch.cat(locals_, 0).contiguous()
    idx_global = torch.cat(globals_, 0).contiguous()
    own_key = torch.cat(own)
    if beta:
        q_rope.add_(k_rope[own_key][:, None, :], alpha=beta)
        q_latent.add_(kv_latent[own_key][:, None, :], alpha=beta)
    return Inputs(
        seq_q=seq_q,
        seq_k=seq_k,
        topk=topk,
        q_latent=q_latent,
        q_rope=q_rope,
        kv_latent=kv_latent,
        k_rope=k_rope,
        dout=dout,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        idx_local=idx_local,
        idx_global=idx_global,
        topk_length=(idx_local >= 0).sum(-1).to(torch.int32).contiguous(),
        own_key=own_key,
    )


@torch.no_grad()
def reference_fp64(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rope: torch.Tensor,
    indices: torch.Tensor,
    *,
    dout: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: float = DEFAULT_SCALE,
    own_key: Optional[torch.Tensor] = None,
    chunk_rows: int = 256,
    consume: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """Chunked FP64 reference from the BF16 inputs.

    Returns ``out [T, 64, 512]``, natural-log ``lse [T, 64]`` (``-inf`` for
    fully masked rows), ``out_emu`` (the same forward with the numerics of a
    BF16-P kernel emulated in fp64: probabilities rounded to BF16 relative to
    the exact row maximum, row sum over the unrounded values, output rounded
    to BF16 -- its error against ``out`` is the floor such a kernel can reach)
    and, with ``dout``, ``dq_latent``, ``dq_rope``, ``dkv_latent [S, 512]``,
    ``dk_rope [S, 64]`` plus their ``*_emu`` BF16-P/dS numerics floors (P and dS
    rounded to BF16 at the MMA inputs, BF16 outputs); with ``own_key`` also the mean softmax mass on the
    query's own key (``self_weight``).  Slots that are ``-1``, ``>= S`` or
    ``>= topk_length[t]`` are invalid.

    With ``consume`` the per-query results are streamed instead of kept: after
    every ``chunk_rows`` query rows ``consume(r0, r1, chunk)`` receives the
    slice ``chunk`` (``out``, ``lse``, ``out_emu`` and, with ``dout``,
    ``dq_latent``, ``dq_rope``, ``dq_latent_emu``, ``dq_rope_emu``; FP64, rows
    ``r0:r1``) and the returned dict holds only the key-side results.  The
    numbers are those of the unstreamed call; only the retained memory differs
    (a 131072-query problem keeps 136 GiB of FP64 per-query results otherwise).
    """
    device = q_latent.device
    total_q, total_k = int(q_latent.shape[0]), int(kv_latent.shape[0])
    kf, vf = k_rope.double(), kv_latent.double()
    want_grad = dout is not None
    stream = consume is not None
    out = out_emu = lse = dql = dqr = dql_emu = dqr_emu = None
    if not stream:
        out = torch.empty(
            total_q, NUM_HEADS, D_LATENT, dtype=torch.float64, device=device
        )
        out_emu = torch.empty_like(out)
        lse = torch.empty(total_q, NUM_HEADS, dtype=torch.float64, device=device)
        dql = torch.empty_like(out) if want_grad else None
        dqr = (
            torch.empty(total_q, NUM_HEADS, D_ROPE, dtype=torch.float64, device=device)
            if want_grad
            else None
        )
    dkvl = (
        torch.zeros(total_k, D_LATENT, dtype=torch.float64, device=device)
        if want_grad
        else None
    )
    dkr = (
        torch.zeros(total_k, D_ROPE, dtype=torch.float64, device=device)
        if want_grad
        else None
    )
    # BF16-P/dS numerics floor of the backward: P and dS rounded to BF16 where a kernel feeds them to
    # its MMAs, FP32-exact accumulation, BF16 outputs (dQ directly, dKV/dKr after the FP32 sum).
    if not stream:
        dql_emu = torch.empty_like(dql) if want_grad else None
        dqr_emu = torch.empty_like(dqr) if want_grad else None
    dkvl_emu = torch.zeros_like(dkvl) if want_grad else None
    dkr_emu = torch.zeros_like(dkr) if want_grad else None
    self_w = torch.zeros((), dtype=torch.float64, device=device)
    slot = torch.arange(indices.shape[1], device=device)
    for r0 in range(0, total_q, chunk_rows):
        r1 = min(total_q, r0 + chunk_rows)
        ix = indices[r0:r1].long()
        valid = (ix >= 0) & (ix < total_k)
        if topk_length is not None:
            valid &= slot[None, :] < topk_length[r0:r1, None].long()
        ixs = ix.clamp(0, total_k - 1)
        kg, vg = kf[ixs], vf[ixs]
        qr, ql = q_rope[r0:r1].double(), q_latent[r0:r1].double()
        s = torch.einsum("thd,twd->thw", qr, kg) + torch.einsum("thd,twd->thw", ql, vg)
        s = (s * softmax_scale).masked_fill(~valid[:, None, :], float("-inf"))
        l = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - l[..., None]).nan_to_num(0.0)
        o = torch.einsum("thw,twd->thd", p, vg)
        if not stream:
            out[r0:r1], lse[r0:r1] = o, l
        m = s.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p_rel = torch.exp(s - m).nan_to_num(0.0)
        num = torch.einsum("thw,twd->thd", p_rel.to(torch.bfloat16).double(), vg)
        o_emu = (
            (num / p_rel.sum(-1, keepdim=True))
            .nan_to_num(0.0)
            .to(torch.bfloat16)
            .double()
        )
        if stream:
            chunk = dict(out=o, lse=l, out_emu=o_emu)
        else:
            out_emu[r0:r1] = o_emu
        del p_rel, num, o_emu
        if own_key is not None:
            is_own = (ix == own_key[r0:r1, None]) & valid
            self_w += (p * is_own[:, None, :]).sum()
        if want_grad:
            g = dout[r0:r1].double()
            dp = torch.einsum("thd,twd->thw", g, vg)
            ds = p * (dp - (g * o).sum(-1, keepdim=True)) * softmax_scale
            dqr_c = torch.einsum("thw,twd->thd", ds, kg)
            dql_c = torch.einsum("thw,twd->thd", ds, vg)
            if stream:
                chunk.update(dq_rope=dqr_c, dq_latent=dql_c)
            else:
                dqr[r0:r1], dql[r0:r1] = dqr_c, dql_c
            del dqr_c, dql_c
            dkr.index_add_(0, ix[valid], torch.einsum("thw,thd->twd", ds, qr)[valid])
            dvl = torch.einsum("thw,thd->twd", ds, ql) + torch.einsum(
                "thw,thd->twd", p, g
            )
            dkvl.index_add_(0, ix[valid], dvl[valid])
            p_b = p.to(torch.bfloat16).double()
            ds_b = ds.to(torch.bfloat16).double()
            dqr_emu_c = (
                torch.einsum("thw,twd->thd", ds_b, kg).to(torch.bfloat16).double()
            )
            dql_emu_c = (
                torch.einsum("thw,twd->thd", ds_b, vg).to(torch.bfloat16).double()
            )
            if stream:
                chunk.update(dq_rope_emu=dqr_emu_c, dq_latent_emu=dql_emu_c)
            else:
                dqr_emu[r0:r1], dql_emu[r0:r1] = dqr_emu_c, dql_emu_c
            del dqr_emu_c, dql_emu_c
            dkr_emu.index_add_(
                0, ix[valid], torch.einsum("thw,thd->twd", ds_b, qr)[valid]
            )
            dkvl_emu.index_add_(
                0,
                ix[valid],
                (
                    torch.einsum("thw,thd->twd", ds_b, ql)
                    + torch.einsum("thw,thd->twd", p_b, g)
                )[valid],
            )
            del p_b, ds_b
        if stream:
            consume(r0, r1, chunk)
            del chunk
    result = {} if stream else dict(out=out, lse=lse, out_emu=out_emu)
    if want_grad:
        if not stream:
            result.update(
                dq_latent=dql,
                dq_rope=dqr,
                dq_latent_emu=dql_emu,
                dq_rope_emu=dqr_emu,
            )
        result.update(
            dkv_latent=dkvl,
            dk_rope=dkr,
            dkv_latent_emu=dkvl_emu.to(torch.bfloat16).double(),
            dk_rope_emu=dkr_emu.to(torch.bfloat16).double(),
        )
    if own_key is not None:
        result["self_weight"] = (self_w / (total_q * NUM_HEADS)).item()
    return result


def rel_l2(a: torch.Tensor, ref: torch.Tensor) -> float:
    a = a.double().reshape(-1)
    ref = ref.double().reshape(-1)
    return (a - ref).norm().item() / max(ref.norm().item(), 1e-30)


def rel_l2_rows(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    a = a.double().reshape(a.shape[0], -1)
    ref = ref.double().reshape(ref.shape[0], -1)
    return (a - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)


def calibrate_beta(
    target_self_weight: float,
    *,
    seed: int,
    probe_len: int = 4096,
    topk: int = DEFAULT_TOPK,
    device="cuda",
) -> float:
    """``beta`` of :func:`make_inputs` giving the mean own-key softmax mass ``target``
    on a ``probe_len x probe_len`` peaked problem (bisection on the FP64 reference)."""
    lo, hi = 0.0, 8.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        inp = make_inputs(
            [probe_len],
            [probe_len],
            seed=seed,
            topk=topk,
            self_including=True,
            beta=mid,
            device=device,
        )
        w = reference_fp64(
            inp.q_latent,
            inp.q_rope,
            inp.kv_latent,
            inp.k_rope,
            inp.idx_global,
            own_key=inp.own_key,
        )["self_weight"]
        if w < target_self_weight:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# NoPE (zero-RoPE) problem: shared latent K/V, explicit scale 1/16, combined dkv
# ---------------------------------------------------------------------------

# GLM-5.3-Flash: softmax scale 1/16 (not 1/sqrt(512) nor 1/sqrt(576)), index pools of
# four tokens, ``K_slots = round_up(4 * P + 3, 128)`` with the pool capacity ``P``
# capped at 512 (2176 slots).
DEFAULT_NOPE_SCALE = 1.0 / 16
# Unit roundoffs of round-to-nearest fp32 and bf16 and the forward error constant gamma_n = n u / (1 - n u) (Higham,
# Accuracy and Stability of Numerical Algorithms, 2nd ed., section 3.1): a sum of n exact fp32 terms carries at most
# gamma_{n - 1} x the sum of their magnitudes of rounding error in any summation order.
FP32_UNIT_ROUNDOFF = 2.0**-24
BF16_UNIT_ROUNDOFF = 2.0**-8


def gamma_fp32(n: int) -> float:
    return n * FP32_UNIT_ROUNDOFF / (1.0 - n * FP32_UNIT_ROUNDOFF)


POOL_TOKENS = 4
TAIL_SLOTS = 3
SLOT_ALIGN = 128
DEFAULT_NOPE_TOPK = 2176


def nope_k_slots(pools: int, align: int = SLOT_ALIGN) -> int:
    """``round_up(4 * pools + 3, align)``: the physical slot count of a pool capacity."""
    return -(-(POOL_TOKENS * int(pools) + TAIL_SLOTS) // align) * align


# Recorded pool capacity per slot width: the model's capacities for its own widths (512 -> 2176,
# the profiled case; 256 -> 1152), the width-filling capacity for the sweep widths.
NOPE_POOLS_BY_K_SLOTS = {2176: 512, 1152: 256, 128: 31, 256: 63, 512: 127, 1024: 255}


def nope_pools(k_slots: int) -> int:
    """The recorded pool capacity of a slot width (``NOPE_POOLS_BY_K_SLOTS``), else the largest
    capacity the width holds, ``(k_slots - 3) // 4``."""
    return NOPE_POOLS_BY_K_SLOTS.get(
        int(k_slots), (int(k_slots) - TAIL_SLOTS) // POOL_TOKENS
    )


def pooled_topk_local(
    seq_q: int,
    seq_k: int,
    topk: int,
    gen: torch.Generator,
    device,
    *,
    pools: int,
    self_including: bool = False,
    chunk_rows: int = 1024,
) -> torch.Tensor:
    """Document-relative pooled selection for one document (the GLM-5.3 index structure).

    Query ``i`` sits at key position ``pos = seq_k - seq_q + i``.  Its row holds up to
    ``pools`` complete four-token pools drawn uniformly without replacement from the
    ``(pos + 1) // 4`` complete pools inside ``[0, pos]``, written in ascending pool order
    (keys ``4 p .. 4 p + 3``), then the causal tail of the open pool (keys
    ``4 * (pos // 4) .. pos``, 1-3 keys; none when ``pos % 4 == 3``, that pool being a
    complete candidate), then ``-1`` padding.  Valid slots come first and every valid key
    satisfies ``key <= pos``.  ``self_including`` forces the query's own pool when the own
    key is not already in the tail.  Needs ``4 * pools + 3 <= topk``.
    """
    pools = int(pools)
    if POOL_TOKENS * pools + TAIL_SLOTS > topk:
        raise ValueError(
            f"pooled selection needs 4 * pools + 3 <= topk, got pools={pools}, topk={topk}"
        )
    out = torch.full((seq_q, topk), -1, dtype=torch.int32, device=device)
    big = torch.iinfo(torch.int32).max
    quad = torch.arange(POOL_TOKENS, device=device)
    for r0 in range(0, seq_q, chunk_rows):
        r1 = min(seq_q, r0 + chunk_rows)
        n = r1 - r0
        pos = torch.arange(r0, r1, device=device) + (seq_k - seq_q)
        complete = (pos + 1) // POOL_TOKENS  # complete pools inside [0, pos]
        cand = max(int(complete.max()), pools, 1)
        pool_id = torch.arange(cand, device=device)
        scores = torch.rand(n, cand, device=device, generator=gen)
        if self_including:
            own_pool = pos // POOL_TOKENS
            in_tail = pos % POOL_TOKENS != POOL_TOKENS - 1
            rows = torch.arange(n, device=device)[~in_tail]
            scores[rows, own_pool[~in_tail]] = 2.0
        scores.masked_fill_(pool_id[None, :] >= complete[:, None], float("-inf"))
        # descending argsort rather than topk: identical selection for distinct random scores
        order = scores.argsort(dim=-1, descending=True)[:, :pools]
        chosen = order.masked_fill(torch.isinf(scores.gather(-1, order)), big)
        chosen = chosen.sort(dim=-1).values  # ascending pool order, invalid (big) last
        keys = chosen[:, :, None] * POOL_TOKENS + quad[None, None, :]
        keys = keys.masked_fill((chosen == big)[:, :, None], -1).reshape(
            n, POOL_TOKENS * pools
        )
        block = out[r0:r1]
        block[:, : POOL_TOKENS * pools] = keys.to(torch.int32)
        n_sel = torch.minimum(complete, torch.full_like(complete, pools))
        tail_n = torch.where(
            pos % POOL_TOKENS == POOL_TOKENS - 1, 0, pos % POOL_TOKENS + 1
        )
        base = POOL_TOKENS * n_sel  # first free slot after the pool keys
        first_tail = POOL_TOKENS * (pos // POOL_TOKENS)
        for j in range(TAIL_SLOTS):
            rows = torch.nonzero(tail_n > j, as_tuple=True)[0]
            block[rows, base[rows] + j] = (first_tail[rows] + j).to(torch.int32)
        valid = block >= 0
        assert bool((block[valid] <= pos[:, None].expand_as(block)[valid]).all()), (
            "pooled key after its query"
        )
        assert bool(((block >= 0).int().diff(dim=1) <= 0).all()), (
            "pooled row is not valid-first"
        )
    return out


@dataclass
class NopeInputs:
    """One NoPE problem: absorbed latent queries, the shared latent, both index representations."""

    seq_q: list
    seq_k: list
    topk: int
    q: torch.Tensor  # [T, 64, 512] BF16
    kv: torch.Tensor  # [S, 512] BF16 (K and V)
    dout: (
        torch.Tensor
    )  # [T, 64, 512] BF16, contiguous or the head-major view [512, T * 512, 1]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    idx_local: torch.Tensor  # document-relative, -1 padded (valid first)
    idx_global: torch.Tensor  # global key rows, -1 padded
    topk_length: torch.Tensor
    own_key: torch.Tensor  # global key row of each query's own token
    index: str  # "pooled" | "iid"
    pools: Optional[int]
    dout_layout: str  # "contiguous" | "head_major"

    @property
    def total_q(self) -> int:
        return int(self.q.shape[0])

    @property
    def total_k(self) -> int:
        return int(self.kv.shape[0])

    @property
    def max_seqlen_q(self) -> int:
        return max(self.seq_q)

    @property
    def max_seqlen_k(self) -> int:
        return max(self.seq_k)

    @property
    def num_segments(self) -> int:
        return len(self.seq_q)

    @property
    def valid_keys(self) -> int:
        return int((self.idx_local >= 0).sum().item())

    def bytes_inputs(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.q, self.kv, self.dout, self.idx_local, self.idx_global)
        )


def make_nope_inputs(
    seq_q,
    seq_k,
    *,
    seed: int,
    topk: int = DEFAULT_NOPE_TOPK,
    index: str = "pooled",
    pools: Optional[int] = None,
    self_including: bool = False,
    beta: float = 0.0,
    dout_layout: str = "contiguous",
    device="cuda",
) -> NopeInputs:
    """Seeded BF16 NoPE inputs for documents of lengths ``seq_q`` / ``seq_k``.

    ``index="pooled"`` draws the GLM-5.3 pool structure with capacity ``pools`` (default
    ``nope_pools(topk)``, the largest the width holds); ``"iid"`` draws
    :func:`causal_topk_local` (``min(pos + 1, topk)`` distinct keys).  ``beta > 0`` adds
    ``beta * (own key)`` to every query (peaked attention; pair with ``self_including``).
    ``dout_layout="head_major"`` stores ``dout`` as ``[64, T, 512]`` and returns the
    ``[T, 64, 512]`` view with strides ``[512, T * 512, 1]`` (the recorded trainer layout).
    """
    seq_q, seq_k = [int(x) for x in seq_q], [int(x) for x in seq_k]
    if len(seq_q) != len(seq_k) or any(
        lq > lk for lq, lk in zip(seq_q, seq_k, strict=True)
    ):
        raise ValueError(
            "documents need seq_q <= seq_k, one key length per query length"
        )
    if index not in ("pooled", "iid"):
        raise ValueError(f"index must be 'pooled' or 'iid', got {index!r}")
    if dout_layout not in ("contiguous", "head_major"):
        raise ValueError(
            f"dout_layout must be 'contiguous' or 'head_major', got {dout_layout!r}"
        )
    if pools is None:
        pools = nope_pools(topk) if index == "pooled" else None
    device = torch.device(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    total_q, total_k = sum(seq_q), sum(seq_k)
    q = _randn_bf16((total_q, NUM_HEADS, D_LATENT), gen, device)
    kv = _randn_bf16((total_k, D_LATENT), gen, device)
    if dout_layout == "head_major":
        dout = _randn_bf16((NUM_HEADS, total_q, D_LATENT), gen, device).permute(1, 0, 2)
    else:
        dout = _randn_bf16((total_q, NUM_HEADS, D_LATENT), gen, device)
    cu_q = [0] + torch.tensor(seq_q).cumsum(0).tolist()
    cu_k = [0] + torch.tensor(seq_k).cumsum(0).tolist()
    locals_, globals_, own = [], [], []
    for d, (lq, lk) in enumerate(zip(seq_q, seq_k, strict=True)):
        if index == "pooled":
            loc = pooled_topk_local(
                lq, lk, topk, gen, device, pools=pools, self_including=self_including
            )
        else:
            loc = causal_topk_local(
                lq, lk, topk, gen, device, self_including=self_including
            )
        locals_.append(loc)
        globals_.append(torch.where(loc >= 0, loc + cu_k[d], loc))
        own.append(torch.arange(lq, device=device) + (cu_k[d] + lk - lq))
    idx_local = torch.cat(locals_, 0).contiguous()
    idx_global = torch.cat(globals_, 0).contiguous()
    own_key = torch.cat(own)
    if beta:
        q.add_(kv[own_key][:, None, :], alpha=beta)
    return NopeInputs(
        seq_q=seq_q,
        seq_k=seq_k,
        topk=topk,
        q=q,
        kv=kv,
        dout=dout,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        idx_local=idx_local,
        idx_global=idx_global,
        topk_length=(idx_local >= 0).sum(-1).to(torch.int32).contiguous(),
        own_key=own_key,
        index=index,
        pools=pools,
        dout_layout=dout_layout,
    )


_LOG2E = 1.4426950408889634


def _out_emu_lazy_rescale(
    s: torch.Tensor,
    vg: torch.Tensor,
    *,
    tile: int = 64,
    threshold: float = 6.0,
    vote_group: int = 32,
    mma_k: int = 16,
) -> torch.Tensor:
    """Emulated forward output at the kernel's own rounding points (an fp32 value returned
    in fp64, before the final bf16 rounding; the caller rounds it to the BF16-P numerics
    floor of ``out`` and splits it into the bf16 high + low parts the backward reads).

    ``s`` ``[rows, 64, K]`` fp64 scaled scores (natural log units, ``-inf`` at invalid
    slots), ``vg`` ``[rows, K, 512]`` fp64 gathered values.  The forward kernel walks the
    index row in tiles of 64 consecutive slots, keeps a running row maximum ``mi`` (log2
    units) that it raises only when a tile's maximum exceeds it by more than ``threshold``
    (the lazy rescale; the vote is taken over ``vote_group`` consecutive heads of the
    token, so a head whose own tile maximum is above ``mi`` by less than the threshold
    still follows its group), rounds ``P = exp2(s * log2(e) - mi)`` to bf16 for the PV
    MMA, accumulates the denominator from the unrounded P, rescales numerator and
    denominator by ``exp2(mi_old - mi_new)`` at the rescale points and rounds the
    normalised output to bf16 once.

    The accumulation of ``O`` (bf16 P x bf16 V products, exact in fp32) and of the
    denominator ``l`` is fp32 in slot order: fp32 partial sums per ``mma_k``-key MMA step
    added to the fp32 accumulator, the rescale factors applied in fp32 per tile, and
    ``out = O * (1 / l)`` in fp32 (a plain fp32 reciprocal stands for the kernel's
    ``rcp.approx``, 2^-23 and below the modelled effect).  The fp32 accumulation is
    modelled statistically -- the same magnitude class, not the tensor core's exact
    summation order.  It matters only through ``delta`` on rows whose exact gradient is
    ~0 (one dominant key): an fp32-level realisation error eps ~ 1e-6 on the forward
    output flips ~eps * 2^16 ~ 6 % of the 16-bit hi + lo roundings, which widens the
    delta-path residue RMS by ~1.3x (sqrt(0.083 + 0.065) / 0.29); an fp64 accumulation
    under-estimates the per-row floor there by that factor.  ``ex2.approx`` (P, rescale
    factors) is not modelled: two orders below bf16 rounding.
    """
    assert torch.get_float32_matmul_precision() == "highest", (
        "the fp32 accumulation model needs exact fp32 matmul (no TF32)"
    )
    rows, heads, slots = s.shape
    s2 = s * _LOG2E
    pad = (-slots) % tile
    if pad:
        s2 = torch.nn.functional.pad(s2, (0, pad), value=float("-inf"))
        vg = torch.nn.functional.pad(vg, (0, 0, 0, pad))
    v32 = vg.float()  # bf16 values: exact in fp32
    mi = torch.full((rows, heads), float("-inf"), dtype=s2.dtype, device=s2.device)
    num = torch.zeros(rows, heads, vg.shape[-1], dtype=torch.float32, device=s2.device)
    den = torch.zeros(rows, heads, dtype=torch.float32, device=s2.device)
    groups = heads // vote_group
    for t in range(0, s2.shape[-1], tile):
        st = s2[..., t : t + tile]
        cur = st.amax(-1)
        # -inf - -inf is NaN: no vote from a still-empty row
        pred = (cur - mi) > threshold
        vote = (
            pred.view(rows, groups, vote_group)
            .any(-1, keepdim=True)
            .expand(rows, groups, vote_group)
            .reshape(rows, heads)
        )
        nxt = torch.where(vote, torch.maximum(cur, mi), mi)
        factor = torch.where(vote, torch.exp2(mi - nxt), torch.ones_like(mi))
        factor = torch.where(torch.isnan(factor), torch.ones_like(factor), factor)
        f32 = factor.float()
        num = num * f32[..., None]
        den = den * f32
        mi = nxt
        p = torch.exp2(st - mi[..., None]).nan_to_num(0.0)
        p_b = p.to(torch.bfloat16).float()
        for k in range(0, tile, mma_k):
            num = num + torch.einsum(
                "thw,twd->thd",
                p_b[..., k : k + mma_k],
                v32[:, t + k : t + k + mma_k],
            )
        den = den + p.float().sum(-1)
    out = (num * torch.reciprocal(den)[..., None]).nan_to_num(0.0)
    return out.double()


@torch.no_grad()
def nope_zero_row_dq_bound(
    dout: torch.Tensor, v: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Element-wise bound on the backward kernel's ``|dq|`` for query tokens whose exact
    ``dq`` is identically zero: ``dout [Z, 64, 512]`` are their output gradients and
    ``v [Z, 512]`` the one valid key of each (zero for a token without a valid key); returns
    the fp64 bound ``[Z, 64, 512]``.

    With one valid key the forward is exact -- ``P = 1``, ``l = 1``, ``out = v`` (a bf16
    value, so its low part is 0) -- and ``dS = P (dP - delta)`` cancels: ``dP = dout . v``
    and ``delta = dout . (hi + lo) = dout . v`` are the same exact value.  The kernel takes
    them from two fp32 reductions of the same 512 exact products (bf16 x bf16 fits fp32):
    the MMA accumulation of ``dP`` and the row-sum stage's dot product ``delta``.  Under
    round-to-nearest fp32 additions (``u = 2^-24``) each carries at most
    ``gamma_512 x sum_d |dout_d v_d|`` of error in any summation order, so
    ``|dP - delta| <= B = 2 gamma_512 sum_d |dout_d v_d|`` and, with ``dq = dS k``
    (``k = v``, the shared latent) and the bf16 roundings of ``dS`` and ``dq``,
    ``|dq_el| <= (1 + 2^-8)^2 softmax_scale B |v_el|``.  A token without a valid key has
    ``B = 0``: its ``dq`` must be exactly zero.
    """
    v = v.double()
    b = (dout.double().abs() * v.abs()[:, None, :]).sum(-1)
    b *= 2.0 * gamma_fp32(D_LATENT)
    return (
        (1.0 + BF16_UNIT_ROUNDOFF) ** 2
        * softmax_scale
        * b[:, :, None]
        * v.abs()[:, None, :]
    )


def reference_fp64_nope(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    dout: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    softmax_scale: float = DEFAULT_NOPE_SCALE,
    own_key: Optional[torch.Tensor] = None,
    chunk_rows: int = 256,
    consume: Optional[Callable[[int, int, dict], None]] = None,
    out_emu_model: str = "lazy_rescale",
) -> dict:
    """Chunked FP64 reference of the NoPE problem from the BF16 inputs (the oracle of
    GOAL §5 for the zero-RoPE family; ``indices`` are global key rows).

    ``out_emu_model``: ``"lazy_rescale"`` (default) places the bf16 rounding of P where
    the forward kernel places it (running row maximum with the threshold-6 lazy rescale
    over 64-slot tiles, see :func:`_out_emu_lazy_rescale`); ``"exact_max"`` rounds P
    relative to the exact row maximum (the earlier, optimistic model).

    The emulated gradients ``dq_emu`` / ``dkv_emu`` take the softmax row sum ``delta`` the
    way the backward kernel reads it: ``dout . (hi + lo)`` of the emulated forward output
    split into its bf16 high and bf16 low parts (the kernel's ``out`` and ``o_lo``), not
    the exact ``dout . out``.  ``dS = P (dP - delta)`` cancels on rows where one key
    carries most of the mass (two to four valid keys, a self key near 1), so the error of
    that ``delta`` -- the forward's bf16-P rounding carried by ``hi + lo`` and the 2^-18
    split residue -- is amplified by about ``1 / (1 - max P)`` there and dominates the
    floor of ``dq`` on those rows; the exact-``delta`` gradients ``dq`` / ``dkv`` stay the
    reference distance.  On that same amplified path the emulated gradients accumulate
    ``dP_j = dout . v_j`` (512 dims, the kernel's fp32 MMA) and ``delta = dout . (hi + lo)``
    (the fp32 dot product of the row-sum stage) in fp32, modelled statistically (torch's
    fp32 reduction order, not the kernel's; ~2^-24 x sqrt(512) relative each); everything
    else stays fp64.

    ``p = softmax_j(q . kv_j * softmax_scale)`` over the valid slots, ``out = p @ kv``.
    Returns ``out [T, 64, 512]``, natural-log ``lse [T, 64]`` (``-inf`` for fully masked
    rows), ``out_emu`` (BF16-P numerics floor, as :func:`reference_fp64`) and, with
    ``dout``, ``dq [T, 64, 512]``, the combined ``dkv [S, 512] = dk + dv`` (the gradient of
    the shared latent: ``dS^T q`` plus ``P^T dout``), ``dq_emu`` / ``dkv_emu`` (BF16 P/dS at
    the MMA inputs, BF16 outputs), ``dq_zero_rows [T]`` (the tokens with at most one valid
    key, whose exact ``dq`` is identically zero), ``dq_zero_bound [Z, 64, 512]`` (the
    element-wise bound on the kernel's ``dq`` on those tokens, in their order:
    :func:`nope_zero_row_dq_bound`) and, with ``own_key``, ``self_weight``.  Slots that are
    ``-1``, ``>= S`` or ``>= topk_length[t]`` are invalid.  ``consume`` streams the
    per-query results as in :func:`reference_fp64` (``out``, ``lse``, ``out_emu``, ``dq``,
    ``dq_emu``, ``dq_zero_rows``, ``dq_zero_bound``).
    """
    assert torch.get_float32_matmul_precision() == "highest", (
        "the fp32 accumulation model needs exact fp32 matmul (no TF32)"
    )
    device = q.device
    total_q, total_k = int(q.shape[0]), int(kv.shape[0])
    vf = kv.double()
    want_grad = dout is not None
    stream = consume is not None
    out = out_emu = lse = dq = dq_emu = None
    if not stream:
        out = torch.empty(
            total_q, NUM_HEADS, D_LATENT, dtype=torch.float64, device=device
        )
        out_emu = torch.empty_like(out)
        lse = torch.empty(total_q, NUM_HEADS, dtype=torch.float64, device=device)
        dq = torch.empty_like(out) if want_grad else None
        dq_emu = torch.empty_like(out) if want_grad else None
    dkv = (
        torch.zeros(total_k, D_LATENT, dtype=torch.float64, device=device)
        if want_grad
        else None
    )
    dkv_emu = torch.zeros_like(dkv) if want_grad else None
    zero_rows = torch.zeros(total_q, dtype=torch.bool, device=device)
    zero_bound = []
    self_w = torch.zeros((), dtype=torch.float64, device=device)
    slot = torch.arange(indices.shape[1], device=device)
    for r0 in range(0, total_q, chunk_rows):
        r1 = min(total_q, r0 + chunk_rows)
        ix = indices[r0:r1].long()
        valid = (ix >= 0) & (ix < total_k)
        if topk_length is not None:
            valid &= slot[None, :] < topk_length[r0:r1, None].long()
        ixs = ix.clamp(0, total_k - 1)
        vg = vf[ixs]
        ql = q[r0:r1].double()
        s = torch.einsum("thd,twd->thw", ql, vg)
        s = (s * softmax_scale).masked_fill(~valid[:, None, :], float("-inf"))
        l = torch.logsumexp(s, dim=-1)
        p = torch.exp(s - l[..., None]).nan_to_num(0.0)
        o = torch.einsum("thw,twd->thd", p, vg)
        if not stream:
            out[r0:r1], lse[r0:r1] = o, l
        if out_emu_model == "lazy_rescale":
            o_emu_full = _out_emu_lazy_rescale(s, vg)
        elif out_emu_model == "exact_max":
            m = s.amax(dim=-1, keepdim=True)
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            p_rel = torch.exp(s - m).nan_to_num(0.0)
            num = torch.einsum("thw,twd->thd", p_rel.to(torch.bfloat16).double(), vg)
            o_emu_full = (num / p_rel.sum(-1, keepdim=True)).nan_to_num(0.0)
            del p_rel, num
        else:
            raise ValueError(f"unknown out_emu_model {out_emu_model!r}")
        o_emu = o_emu_full.to(torch.bfloat16).double()
        # the kernel's forward output as its backward reads it: bf16 high part plus bf16 low part
        o_emu_hl = o_emu + (o_emu_full - o_emu).to(torch.bfloat16).double()
        del o_emu_full
        if stream:
            chunk = dict(out=o, lse=l, out_emu=o_emu)
        else:
            out_emu[r0:r1] = o_emu
        del o_emu
        if own_key is not None:
            is_own = (ix == own_key[r0:r1, None]) & valid
            self_w += (p * is_own[:, None, :]).sum()
        if want_grad:
            # tokens whose exact dq is identically zero (at most one valid key) and the element-wise bound on the
            # kernel's fp32 rounding residue there, from the inputs
            zero_c = valid.sum(-1) <= 1
            v_one = (vg[zero_c] * valid[zero_c][:, :, None]).sum(1)
            bound_c = nope_zero_row_dq_bound(dout[r0:r1][zero_c], v_one, softmax_scale)
            del v_one
            if stream:
                chunk.update(dq_zero_rows=zero_c, dq_zero_bound=bound_c)
            else:
                zero_rows[r0:r1] = zero_c
                zero_bound.append(bound_c)
            del zero_c, bound_c
            g = dout[r0:r1].double()
            dp = torch.einsum("thd,twd->thw", g, vg)
            ds = p * (dp - (g * o).sum(-1, keepdim=True)) * softmax_scale
            dq_c = torch.einsum("thw,twd->thd", ds, vg)
            if stream:
                chunk.update(dq=dq_c)
            else:
                dq[r0:r1] = dq_c
            del dq_c
            # combined latent gradient: the key part dS^T q and the value part P^T dout
            dkv_c = torch.einsum("thw,thd->twd", ds, ql) + torch.einsum(
                "thw,thd->twd", p, g
            )
            dkv.index_add_(0, ix[valid], dkv_c[valid])
            del dkv_c
            p_b = p.to(torch.bfloat16).double()
            # emulated dS: fp32 dP (dout . v over the 512 dims, the kernel's fp32 MMA) and fp32 delta (dout . (hi + lo)
            # of the emulated output, the row-sum stage), both statistically modelled in torch's reduction order;
            # P x (dP - delta) at bf16 for the MMA input
            g32 = g.float()
            dp32 = torch.einsum("thd,twd->thw", g32, vg.float())
            delta32 = (g32 * o_emu_hl.float()).sum(-1, keepdim=True)
            ds_b = (
                (p * (dp32.double() - delta32.double()) * softmax_scale)
                .to(torch.bfloat16)
                .double()
            )
            del g32, dp32, delta32
            dq_emu_c = (
                torch.einsum("thw,twd->thd", ds_b, vg).to(torch.bfloat16).double()
            )
            if stream:
                chunk.update(dq_emu=dq_emu_c)
            else:
                dq_emu[r0:r1] = dq_emu_c
            del dq_emu_c
            dkv_emu.index_add_(
                0,
                ix[valid],
                (
                    torch.einsum("thw,thd->twd", ds_b, ql)
                    + torch.einsum("thw,thd->twd", p_b, g)
                )[valid],
            )
            del p_b, ds_b
        del o_emu_hl
        if stream:
            consume(r0, r1, chunk)
            del chunk
    result = {} if stream else dict(out=out, lse=lse, out_emu=out_emu)
    if want_grad:
        if not stream:
            result.update(
                dq=dq,
                dq_emu=dq_emu,
                dq_zero_rows=zero_rows,
                dq_zero_bound=torch.cat(zero_bound, 0),
            )
        result.update(dkv=dkv, dkv_emu=dkv_emu.to(torch.bfloat16).double())
    if own_key is not None:
        result["self_weight"] = (self_w / (total_q * NUM_HEADS)).item()
    return result


def calibrate_beta_nope(
    target_self_weight: float,
    *,
    seed: int,
    probe_len: int = 4096,
    topk: int = DEFAULT_NOPE_TOPK,
    index: str = "iid",
    pools: Optional[int] = None,
    softmax_scale: float = DEFAULT_NOPE_SCALE,
    device="cuda",
) -> float:
    """``beta`` of :func:`make_nope_inputs` giving the mean own-key softmax mass ``target``
    on a ``probe_len x probe_len`` peaked NoPE problem at ``softmax_scale`` (bisection on
    :func:`reference_fp64_nope`; the RoPE calibration does not apply: it adds ``beta`` to
    both query parts at the 576-scale)."""
    lo, hi = 0.0, 8.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        inp = make_nope_inputs(
            [probe_len],
            [probe_len],
            seed=seed,
            topk=topk,
            index=index,
            pools=pools,
            self_including=True,
            beta=mid,
            device=device,
        )
        w = reference_fp64_nope(
            inp.q,
            inp.kv,
            inp.idx_global,
            softmax_scale=softmax_scale,
            own_key=inp.own_key,
        )["self_weight"]
        if w < target_self_weight:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
