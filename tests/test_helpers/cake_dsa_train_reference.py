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
shared by ``tests/experimental/test_cake_dsa_train.py`` and
``benchmarks/bench_cake_dsa_train.py``.  The protocol follows the reference
tests of the FA sparse-MLA backward: per-document causal random top-k with
``-1`` padding, BF16 inputs, FP64 reference from the same BF16 values.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """
    device = q_latent.device
    total_q, total_k = int(q_latent.shape[0]), int(kv_latent.shape[0])
    kf, vf = k_rope.double(), kv_latent.double()
    out = torch.empty(total_q, NUM_HEADS, D_LATENT, dtype=torch.float64, device=device)
    out_emu = torch.empty_like(out)
    lse = torch.empty(total_q, NUM_HEADS, dtype=torch.float64, device=device)
    want_grad = dout is not None
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
        out[r0:r1], lse[r0:r1] = o, l
        m = s.amax(dim=-1, keepdim=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p_rel = torch.exp(s - m).nan_to_num(0.0)
        num = torch.einsum("thw,twd->thd", p_rel.to(torch.bfloat16).double(), vg)
        out_emu[r0:r1] = (
            (num / p_rel.sum(-1, keepdim=True))
            .nan_to_num(0.0)
            .to(torch.bfloat16)
            .double()
        )
        del p_rel, num
        if own_key is not None:
            is_own = (ix == own_key[r0:r1, None]) & valid
            self_w += (p * is_own[:, None, :]).sum()
        if want_grad:
            g = dout[r0:r1].double()
            dp = torch.einsum("thd,twd->thw", g, vg)
            ds = p * (dp - (g * o).sum(-1, keepdim=True)) * softmax_scale
            dqr[r0:r1] = torch.einsum("thw,twd->thd", ds, kg)
            dql[r0:r1] = torch.einsum("thw,twd->thd", ds, vg)
            dkr.index_add_(0, ix[valid], torch.einsum("thw,thd->twd", ds, qr)[valid])
            dvl = torch.einsum("thw,thd->twd", ds, ql) + torch.einsum(
                "thw,thd->twd", p, g
            )
            dkvl.index_add_(0, ix[valid], dvl[valid])
            p_b = p.to(torch.bfloat16).double()
            ds_b = ds.to(torch.bfloat16).double()
            dqr_emu[r0:r1] = (
                torch.einsum("thw,twd->thd", ds_b, kg).to(torch.bfloat16).double()
            )
            dql_emu[r0:r1] = (
                torch.einsum("thw,twd->thd", ds_b, vg).to(torch.bfloat16).double()
            )
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
    result = dict(out=out, lse=lse, out_emu=out_emu)
    if want_grad:
        result.update(
            dq_latent=dql,
            dq_rope=dqr,
            dkv_latent=dkvl,
            dk_rope=dkr,
            dq_latent_emu=dql_emu,
            dq_rope_emu=dqr_emu,
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
