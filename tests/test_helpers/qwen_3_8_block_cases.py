# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Step builders for the Qwen 3.8 Flash-Next gated attention block tests: the Qwen 3.5 ``PagedCase`` (vLLM-shaped pools, a dense block
table, a flat slot mapping, logical KV lengths, the fp32 oracle of the whole sequence) extended with the in-block indexer's state --
the indexer band's raw keys of every position from the oracle's own projection, the PREFIX blocks compressed through the vendored
compress kernel into the paged compressed-key cache (page 0 the null page, NaN everywhere unwritten), the raw-key ring primed with the
last ``cap`` raw keys below the step, the position tables over the table's whole reach -- and the TWO-STAGE oracle: (1) the selection
(the counts rule, the canonical list form, the SET == the fp64 top-k of the kernel's own operands up to ties at the k-th boundary), (2)
the attention of every row over the KERNEL's list (the visible mask), then the gate and the out projection.  Opt-in on top of the Qwen
3.5 case: the 3.5 tests are untouched."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch

from tests.test_helpers.qwen_3_5_block_cases import NULL_PAGE, OUT_COS, PagedCase
from tests.test_helpers.qwen_3_5_block_reference import build_rope_tables, cosine

ATOL_O = 2e-2  # the block suite's bar on the gated O against the fp32 oracle fed the kernel's list (never widened here)
TIE_TOL = 1e-4  # the scorer suite's tie window at the k-th boundary (fp32 logits of O(1) vs the fp64 reference)
SENT_I32 = -7
__all__ = [
    "ATOL_O",
    "NULL_PAGE",
    "OUT_COS",
    "SENT_I32",
    "TIE_TOL",
    "Qsa38PagedCase",
    "attend",
    "bits",
    "random_weights_38",
    "ring_cap_for",
    "set_vs_topk",
    "unified_kv_cache",
    "visible_mask",
]


def bits(x: torch.Tensor) -> torch.Tensor:
    """The bit pattern of a bf16 tensor (an int16 view): a bitwise comparison that treats NaN as a value."""
    return x.contiguous().view(torch.int16) if x.dtype == torch.bfloat16 else x


def ring_cap_for(q_len: int, block_size: int = 4) -> int:
    """A serving stack's ring capacity rule: whole ``block_size``-token groups covering ``q_len`` rows (``4`` at q_len 1, ``8`` at 2..4,
    ``12`` at 5..8); meets the block's ``cap >= q_len + block_size - 1`` for every q_len."""
    return block_size * (-(-(block_size - 1 + int(q_len)) // block_size))


def random_weights_38(geometry, *, device, dtype=torch.bfloat16, seed=0, std=0.02):
    """``(Qwen38BlockWeights, w_o)`` drawn in fp32 and rounded once into ``dtype``: the four dense bands and the indexer band in one
    ``w_qkvg``, the QK norms 1.0, the two indexer norms ``1 + small`` (pre-folded, as the kernels multiply)."""
    from flashinfer.qwen_3_8_block import Qwen38BlockWeights

    g = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, scale=std):
        return (
            torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * scale
        ).to(dtype)

    w_qkvg = randn(geometry.n_qkvg, geometry.d_model)
    w_o = randn(geometry.d_model, geometry.num_qo_heads * geometry.head_dim)
    ones = torch.ones(geometry.head_dim, dtype=dtype, device=device)
    w_iq = (1.0 + 0.03 * randn(geometry.index_head_dim, scale=1.0).float()).to(dtype)
    w_ik = (1.0 + 0.02 * randn(geometry.index_head_dim, scale=1.0).float()).to(dtype)
    return Qwen38BlockWeights(w_qkvg, ones, ones.clone(), w_iq, w_ik), w_o


def unified_kv_cache(kv_cache_nhd: torch.Tensor) -> torch.Tensor:
    """A serving stack's UNIFIED row ``[num_pages, page_size, H_kv, 2 x D]`` (K at ``[..., :D]``, V at ``[..., D:]``) holding the same
    values as an NHD 5-D cache ``[num_pages, 2, page_size, H_kv, D]`` -- a new buffer; the block binds it by strides."""
    k, v = kv_cache_nhd[:, 0], kv_cache_nhd[:, 1]
    return torch.cat((k, v), dim=-1).contiguous()


def visible_mask(
    block_ids: torch.Tensor,
    positions: torch.Tensor,
    kv_len: int,
    block_size: int,
    top_k: int,
) -> torch.Tensor:
    """``[rows, kv_len]`` bool, True where a key is VISIBLE to the row, for ``rows`` query rows of ONE sequence at ``positions`` (0-based):
    the listed complete blocks (the entries below ``min(top_k, floor((p + 1) / block_size))``, ``-1`` contributing nothing) inside the
    causal range plus the open tail ``[block_size x floor((p + 1) / block_size), p]`` -- the per-row reading of the decode indexer's list
    (the row's own tail; a block listed at or past the row's visible count contributes nothing); a row at a negative position is dead."""
    rows, width = block_ids.shape
    dev = block_ids.device
    p = positions.to(dev, torch.long)
    n_vis = torch.minimum(p + 1, torch.full_like(p, int(kv_len))).clamp_min(0)
    count = torch.minimum(
        torch.div(p + 1, block_size, rounding_mode="floor"),
        torch.full_like(p, int(top_k)),
    ).clamp_min(0)
    count = torch.minimum(count, torch.full_like(count, width))
    ids = block_ids.to(torch.long)
    n_blocks_total = -(-int(kv_len) // block_size)
    read = torch.arange(width, device=dev)[None, :] < count[:, None]
    live = read & (ids >= 0) & (ids < n_blocks_total)
    target = torch.where(live, ids, torch.full_like(ids, n_blocks_total))
    member = torch.zeros(rows, n_blocks_total + 1, dtype=torch.bool, device=dev)
    if width:
        member.scatter_(1, target, True)
    member = member[:, :n_blocks_total]
    t = torch.arange(int(kv_len), device=dev)
    listed = member[:, t // block_size]
    tail_start = block_size * torch.div(n_vis, block_size, rounding_mode="floor")
    tail = (t[None, :] >= tail_start[:, None]) & (t[None, :] < n_vis[:, None])
    causal = t[None, :] < n_vis[:, None]
    return (listed & causal) | tail


def attend(ref, allowed, p_live, scale, dtype, h_q, group):
    """The fp32 oracle's attention of the LIVE rows of one sequence (positions ``p_live``) over its keys through the visible mask
    ``allowed`` ``[n_live, L]`` (the dense reference's post-norm / post-RoPE Q, K, V and pre-sigmoid gate), then the gate: ``(o_gated
    [n_live, h_q, D]`` in ``dtype`` -- one rounding of O, the gate in fp32, one rounding --, ``lse [h_q, n_live]`` fp32)``."""
    dev = allowed.device
    q_rows = ref.q[0][p_live].float()
    gate_rows = ref.gate[0][p_live].float()
    n_live, d = int(q_rows.shape[0]), int(q_rows.shape[-1])
    o = torch.zeros(n_live, h_q, d, dtype=torch.float32, device=dev)
    lse = torch.full((h_q, n_live), float("-inf"), dtype=torch.float32, device=dev)
    for hh in range(h_q):
        kk, vv = ref.k[0][:, hh // group].float(), ref.v[0][:, hh // group].float()
        sc = (q_rows[:, hh] @ kk.t()) * scale
        sc = sc.masked_fill(~allowed, float("-inf"))
        rmax = sc.amax(dim=-1)
        dead = torch.isinf(rmax) & (rmax < 0)
        safe = torch.where(dead, torch.zeros_like(rmax), rmax)
        pr = torch.where(allowed, torch.exp(sc - safe[:, None]), torch.zeros_like(sc))
        den = pr.sum(dim=-1)
        den_safe = torch.where(dead, torch.ones_like(den), den)
        oc = (pr @ vv) / den_safe[:, None]
        o[:, hh] = torch.where(dead[:, None], torch.zeros_like(oc), oc)
        lse[hh] = torch.where(
            dead, torch.full_like(den, float("-inf")), safe + torch.log(den)
        )
    o16 = o.to(dtype)
    og = (o16.float() * torch.sigmoid(gate_rows)).to(dtype)
    return og, lse


def set_vs_topk(
    in_set: torch.Tensor, scores: torch.Tensor, top_k: int, tol: float
) -> Tuple[int, int, float]:
    """Per row: the SET ``in_set`` ``[R, N]`` against the top-k of ``scores`` ``[R, N]`` (fp64; ``-inf`` = not visible); every member of the
    symmetric difference accepted iff its score lies within ``tol`` of the row's k-th score (the margin rule).  Returns ``(flips, rows,
    max_gap)``; asserts the rule."""
    rows, n = scores.shape
    k_eff = min(top_k, n)
    tk = torch.topk(scores, k_eff, dim=-1)
    exp_valid = torch.isfinite(tk.values)
    in_exp = torch.zeros(rows, n + 1, dtype=torch.bool, device=scores.device)
    in_exp.scatter_(
        1,
        torch.where(exp_valid, tk.indices, torch.full_like(tk.indices, n)),
        torch.ones_like(exp_valid),
    )
    in_exp = in_exp[:, :n]
    kth = (
        torch.where(exp_valid, tk.values, torch.full_like(tk.values, float("inf")))
        .min(dim=-1)
        .values
    )
    differs = in_set ^ in_exp
    flips = int(differs.sum())
    if flips == 0:
        return 0, 0, 0.0
    gap = (scores - kth[:, None]).abs()[differs]
    assert torch.isfinite(gap).all(), (
        f"{int((~torch.isfinite(gap)).sum())} flipped ids lie outside the row's visible range"
    )
    assert bool((gap <= tol).all()), (
        f"{flips} ids differ from the top-k beyond the margin: max gap {float(gap.max()):.3e} > tol {tol:.3e}"
    )
    return flips, int(differs.any(dim=-1).sum()), float(gap.max())


class Qsa38PagedCase(PagedCase):
    """The Qwen 3.5 paged case plus the in-block indexer's state for ONE decode step over ``kv_lens`` sequences (0 = a padded entry) with
    ``q_len`` rows each (every live sequence the same ``q_len``: the uniform batch).  Beyond the 3.5 case: ``k_raw_full[b]`` the band's
    raw key of every position (the oracle's own projection, one rounding), ``index_kv_cache`` with the prefix blocks compressed through
    the vendored kernel (``prime()``), ``index_state_cache`` with the last ``cap`` raw keys below the step, ``index_state_block_table``
    (block ``b + 1``; 0 for a padded entry), ``ring_slots`` (the stack's flat ring WRITE rows), ``position_tables`` over
    ``max_pages x page_size`` positions, ``block_ids_out`` / ``block_lens_out`` sentinel buffers; ``check_selection`` and
    ``check_out_from_list`` the two oracle stages, ``check_state`` the ring / cache rows."""

    def __init__(
        self,
        geometry,
        kv_lens: Sequence[int],
        q_len: int,
        *,
        page_size: int = 16,
        layout: str = "NHD",
        dtype=torch.bfloat16,
        seed: int = 0,
        device="cuda",
        table_width: Optional[int] = None,
        extra_pages: int = 2,
        cap: Optional[int] = None,
        weights=None,
    ):
        if weights is None:
            weights = random_weights_38(geometry, device=device, dtype=dtype, seed=seed)
        q_lens = [int(q_len) if L else 0 for L in kv_lens]
        # the 3.5 case's oracle reads the DENSE bands only: hand it a view of the four dense bands as "the 3.5 weights"
        from flashinfer.qwen_3_5_block import Qwen35BlockGeometry, Qwen35BlockWeights

        self.geometry38 = geometry
        g35 = Qwen35BlockGeometry(
            d_model=geometry.d_model,
            num_qo_heads=geometry.num_qo_heads,
            num_kv_heads=geometry.num_kv_heads,
            head_dim=geometry.head_dim,
            rope_dim=geometry.rope_dim,
            qk_norm=geometry.qk_norm,
            qk_norm_eps=geometry.qk_norm_eps,
            sm_scale=geometry.sm_scale,
        )
        w38, w_o = weights
        dense = Qwen35BlockWeights(
            w38.w_qkvg[: geometry.n_qkvg_dense].contiguous(), w38.w_q_norm, w38.w_k_norm
        )
        super().__init__(
            g35,
            kv_lens,
            q_lens,
            page_size=page_size,
            layout=layout,
            dtype=dtype,
            seed=seed,
            device=device,
            table_width=table_width,
            extra_pages=extra_pages,
            weights=(dense, w_o),
        )
        self.weights38 = w38
        self.q_len = int(q_len)
        # the UNIFORM batch (a serving stack's padded decode graph): every entry carries q_len rows, a padded entry's rows finite (zeros),
        # its slots -1, its reference rows 0 -- the 3.5 case laid out the LIVE rows only
        dm, rope_dim, dtype_ = geometry.d_model, geometry.rope_dim, dtype
        B, q = self.b, self.q_len
        h_u = torch.zeros(B * q, dm, dtype=dtype_, device=self.device)
        cos_u = torch.ones(B * q, rope_dim, dtype=dtype_, device=self.device)
        sin_u = torch.zeros(B * q, rope_dim, dtype=dtype_, device=self.device)
        slots_u = torch.full((B * q,), -1, dtype=torch.int32, device=self.device)
        ref_u = torch.zeros(B * q, dm, dtype=dtype_, device=self.device)
        live_u = torch.zeros(B * q, dtype=torch.bool, device=self.device)
        row = 0
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            h_u[b * q : (b + 1) * q] = self.hidden_states[row : row + q]
            cos_u[b * q : (b + 1) * q] = self.rope_cos[row : row + q]
            sin_u[b * q : (b + 1) * q] = self.rope_sin[row : row + q]
            slots_u[b * q : (b + 1) * q] = self.slot_mapping[row : row + q]
            ref_u[b * q : (b + 1) * q] = self.ref_out[row : row + q]
            live_u[b * q : (b + 1) * q] = True
            row += q
        (
            self.hidden_states,
            self.rope_cos,
            self.rope_sin,
            self.slot_mapping,
            self.ref_out,
            self.live_rows,
        ) = h_u, cos_u, sin_u, slots_u, ref_u, live_u
        self.t = B * q
        g = geometry
        d_i, h_i, bs, top_k = g.index_head_dim, g.index_heads, g.block_size, g.top_k
        self.block_size, self.top_k = bs, top_k
        dev = self.device
        self.cap = int(cap) if cap is not None else ring_cap_for(self.q_len, bs)
        self.rows_per_page = page_size // bs
        # the band's raw key head of every position: the oracle's own projection, one rounding
        self.w_band_k = w38.w_qkvg[g.n_qkvg_dense + h_i * d_i :]  # [d_i, d_model]
        self.k_raw_full: Dict[int, torch.Tensor] = {}
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            self.k_raw_full[b] = (
                self.h_full[b][0].float() @ self.w_band_k.float().t()
            ).to(dtype)
        # the position tables over the table's whole reach (the ceiling met with equality)
        width = int(self.block_table.shape[1])
        self.n_pos = width * page_size
        cos_t, sin_t = build_rope_tables(
            self.n_pos,
            g.rope_dim,
            base=self.ref_geom.rope_base,
            batch=1,
            device=dev,
            dtype=dtype,
        )
        self.position_tables = (cos_t.contiguous(), sin_t.contiguous())
        # the side caches: NaN everywhere unwritten (a gathered unwritten row would poison a score / an output), block 0 / page 0 the null ones
        self.index_kv_cache = torch.full(
            (self.num_pages, self.rows_per_page, 1, d_i),
            float("nan"),
            dtype=dtype,
            device=dev,
        )
        self.index_state_cache = torch.full(
            (self.b + 1, self.cap, 1, d_i), float("nan"), dtype=dtype, device=dev
        )
        self.index_state_block_table = torch.zeros(
            self.b, 1, dtype=torch.int32, device=dev
        )
        ring_slots = torch.full(
            (self.b * self.q_len,), -1, dtype=torch.int64, device=dev
        )
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            self.index_state_block_table[b, 0] = b + 1
            pos0 = L - self.q_len
            for p in range(max(0, pos0 - self.cap), pos0):
                self.index_state_cache[b + 1, p % self.cap, 0] = self.k_raw_full[b][p]
            for j in range(self.q_len):
                ring_slots[b * self.q_len + j] = (b + 1) * self.cap + (
                    pos0 + j
                ) % self.cap
        self.ring_slots = ring_slots
        self.block_ids_out = torch.full(
            (self.b * self.q_len, top_k), SENT_I32, dtype=torch.int32, device=dev
        )
        self.block_lens_out = torch.full(
            (self.b * self.q_len,), SENT_I32, dtype=torch.int32, device=dev
        )
        self._primed = False
        self._compress = {}
        self.prefix_bits = (bits(self.kv_cache).clone(),)

    # -- the vendored compress kernel (the one-rounding chain the block's own arms compute) ----------------------------------------
    def compress(
        self,
        k_raw: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pool: Optional[int] = None,
    ) -> torch.Tensor:
        """The vendored prefill compress over ``k_raw [B', S, d_i]`` at ``pool`` (the block size by default) with the position tables
        ``cos / sin [1, S_pos, rope_dim]`` -> ``[B', floor(S / pool), d_i]``; ``pool=1`` with the query norm = the indexer queries."""
        from flashinfer.experimental.qwen_3_5_block import api as _api
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.gated_attention_block.kernels.qsa_compress import (
            compile_qsa_compress,
            run_qsa_compress,
        )

        _api.configure_vendored_tree()
        g = self.geometry38
        pool = g.block_size if pool is None else int(pool)
        if pool not in self._compress:
            self._compress[pool] = compile_qsa_compress(
                dtype=self.dtype,
                d=g.index_head_dim,
                rope_dim=g.rope_dim,
                pool=pool,
                has_seq_lens=False,
                table_dtype=self.dtype,
            )
        k_raw = k_raw.contiguous()
        out = torch.full(
            (int(k_raw.shape[0]), int(k_raw.shape[1]) // pool, g.index_head_dim),
            float("nan"),
            dtype=self.dtype,
            device=self.device,
        )
        w = (
            self.weights38.w_ik_norm
            if pool == g.block_size
            else self.weights38.w_iq_norm
        )
        run_qsa_compress(
            self._compress[pool],
            k_raw,
            out,
            w,
            cos,
            sin,
            None,
            eps=float(g.index_norm_eps),
            norm_weight_offset=0.0,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        return out

    def prime(self) -> None:
        """The PREFIX blocks of every live sequence (``j < floor(pos0 / block_size)``) into the paged compressed-key cache through the
        vendored compress kernel (what a prefill wrote), once per case."""
        if self._primed:
            return
        bs = self.block_size
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            nb = max(L - self.q_len, 0) // bs
            if nb == 0:
                continue
            kb = self.compress(
                self.k_raw_full[b][None, : nb * bs], *self.position_tables
            )
            for j in range(nb):
                self.index_kv_cache[
                    int(self.block_table[b, j // self.rows_per_page]),
                    j % self.rows_per_page,
                    0,
                ] = kb[0, j]
        torch.cuda.synchronize()
        self._primed = True

    # -- the run's arguments -------------------------------------------------------------------------------------------------------
    def decode_args(self, *, ring_slots: bool = False) -> Dict[str, torch.Tensor]:
        """The ``run`` keywords: the 3.5 ones plus the side caches, the ring table and (``ring_slots``) the stack's ring slots."""
        args = dict(
            kv_cache=self.kv_cache,
            slot_mapping=self.slot_mapping,
            block_table=self.block_table,
            seq_lens=self.seq_lens,
            index_kv_cache=self.index_kv_cache,
            index_state_cache=self.index_state_cache,
            index_state_block_table=self.index_state_block_table,
        )
        if ring_slots:
            args["index_state_slot_mapping"] = self.ring_slots.to(
                self.slot_mapping.dtype
            )
        return args

    def plan_args(self) -> Dict[str, object]:
        """The ``prepare`` keywords beyond the 3.5 ones."""
        return dict(
            index_kv_cache=self.index_kv_cache,
            index_state_cache=self.index_state_cache,
            index_state_block_table=self.index_state_block_table,
            position_tables=self.position_tables,
        )

    def reset_outputs(self) -> None:
        self.block_ids_out.fill_(SENT_I32)
        self.block_lens_out.fill_(SENT_I32)

    # -- the oracles -----------------------------------------------------------------------------------------------------------------
    def positions(self, b: int) -> torch.Tensor:
        return torch.arange(self.q_len, device=self.device) + (
            self.kv_lens[b] - self.q_len
        )

    def n_vis(self, n_blocks_max: int) -> torch.Tensor:
        """The visible complete-block count per row ``[B, q_len]``: ``clamp(floor((pos + 1) / bs), 0, n_blocks_max)``."""
        return torch.tensor(
            [
                [
                    max(
                        0,
                        min((L - self.q_len + j + 1) // self.block_size, n_blocks_max),
                    )
                    for j in range(self.q_len)
                ]
                for L in self.kv_lens
            ],
            device=self.device,
        )

    def check_selection(
        self,
        ids: torch.Tensor,
        lens,
        qi: torch.Tensor,
        *,
        n_blocks_max: int,
        label: str = "selection",
    ) -> dict:
        """The two-stage oracle's FIRST stage on the kernel's own outputs: the counts rule (``min(top_k, n_vis)`` in ``lens``), the list a
        strictly descending valid prefix then ``-1``, every id below the row's visible count, dead rows all ``-1``, the SET == the fp64 top-k
        of the kernel's OWN ``(qi, kbar)`` (``kbar`` the cache rows through the table) up to ties."""
        B, s_q, top_k = self.b, self.q_len, self.top_k
        ids = ids.view(B, s_q, top_k)
        n_vis = self.n_vis(n_blocks_max)
        n_sel = torch.minimum(n_vis, torch.full_like(n_vis, top_k))
        if lens is not None:
            assert torch.equal(lens.view(B, s_q).cpu(), n_sel.cpu().to(torch.int32)), (
                f"{label}: block_lens_out != min(top_k, n_vis)"
            )
        valid = ids >= 0
        assert torch.equal(
            valid,
            torch.arange(top_k, device=self.device)[None, None, :]
            < n_sel.to(self.device)[..., None],
        ), f"{label}: the valid ids are not the prefix min(top_k, n_vis)"
        assert not (ids == SENT_I32).any(), f"{label}: a sentinel survived in the list"
        diffs = (ids[..., :-1] - ids[..., 1:]).masked_fill(~valid[..., 1:], 1)
        assert bool((diffs > 0).all()), (
            f"{label}: a row's list is not strictly descending"
        )
        assert bool(
            (
                torch.where(valid, ids, torch.full_like(ids, -1)).amax(-1)
                < n_vis.to(self.device)
            ).all()
        ), f"{label}: an id at or past the row's visible count"
        d_i, h_i = self.geometry38.index_head_dim, self.geometry38.index_heads
        qi = qi.view(B, s_q, h_i, d_i)
        flips = rows = 0
        gap = 0.0
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                assert bool((ids[b] == -1).all()), (
                    f"{label}: a padded entry selected something"
                )
                continue
            n = int(n_vis[b].max())
            if n == 0:
                continue
            pages = self.block_table[b, : -(-n // self.rows_per_page)].long()
            kbar = self.index_kv_cache[pages][:, :, 0].reshape(-1, d_i)[:n].double()
            sc = torch.relu(torch.einsum("shd,jd->shj", qi[b].double(), kbar)).sum(
                1
            ) / math.sqrt(d_i)
            limit = (
                torch.arange(n, device=self.device)[None, :]
                < n_vis[b].to(self.device)[:, None]
            )
            sc = sc.masked_fill(~limit, float("-inf"))
            m = torch.zeros(s_q, n + 1, dtype=torch.bool, device=self.device)
            m.scatter_(
                1,
                torch.where(
                    ids[b] >= 0, ids[b].long(), torch.full_like(ids[b], n).long()
                ),
                torch.ones_like(ids[b], dtype=torch.bool),
            )
            f, r, gp = set_vs_topk(m[:, :n], sc, top_k, TIE_TOL)
            flips, rows, gap = flips + f, rows + r, max(gap, gp)
        print(
            f"[{label}] selection: {flips} tie flips vs the fp64 top-k of the kernel's operands ({rows} rows, gap {gap:.2e})"
        )
        return dict(flips=flips, rows=rows, gap=gap)

    def oracle_from_list(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two-stage oracle's SECOND stage: the attention of every live row over the KERNEL's list, the gate, the out projection ->
        ``(out [t, d_model], o_gated [t, h_q, D])`` in the case's dtype (padded entries 0)."""
        g = self.geometry38
        B, s_q = self.b, self.q_len
        ids = ids.view(B, s_q, self.top_k)
        out = torch.zeros(B, s_q, g.d_model, dtype=self.dtype, device=self.device)
        og_all = torch.zeros(
            B, s_q, g.num_qo_heads, g.head_dim, dtype=self.dtype, device=self.device
        )
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            pos = self.positions(b)
            live = pos >= 0
            allowed = visible_mask(ids[b], pos, L, self.block_size, self.top_k)[live]
            og, _ = attend(
                self.refs[b],
                allowed,
                pos[live],
                g.scale,
                self.dtype,
                g.num_qo_heads,
                g.group,
            )
            og_all[b, live] = og
            out[b, live] = (
                og.reshape(-1, g.num_qo_heads * g.head_dim).float()
                @ self.w_o.float().t()
            ).to(self.dtype)
        return out.view(B * s_q, g.d_model), og_all.view(
            B * s_q, g.num_qo_heads, g.head_dim
        )

    def check_out_from_list(
        self,
        out: torch.Tensor,
        ids: torch.Tensor,
        *,
        label: str = "out",
        min_cos: float = OUT_COS,
    ) -> float:
        """``out`` ``[t, d_model]`` against the fp32 oracle fed the KERNEL's list: finite, padded entries exactly 0, cosine over the live rows
        above the budget."""
        out = out.reshape(self.b * self.q_len, self.geometry38.d_model)
        ref, _ = self.oracle_from_list(ids)
        live = (
            self.live_rows
            if self.t
            else torch.zeros(0, dtype=torch.bool, device=self.device)
        )
        assert torch.isfinite(out.float()).all(), f"{label}: non-finite rows"
        dead = ~live
        if dead.any():
            assert (out[dead] == 0).all(), (
                f"{label}: a padded entry's rows must be exactly 0"
            )
        c = cosine(out[live], ref[live])
        md = float((out[live].float() - ref[live].float()).abs().max())
        print(
            f"[{label}] vs the oracle fed the kernel's list: cos {c:.6f} max|diff| {md:.3e}"
        )
        assert c > min_cos, f"{label}: cos {c} <= {min_cos}"
        return c

    def check_state(self, raw_keys: torch.Tensor, *, label: str = "state") -> int:
        """After one step: the ring rows of the step BITWISE the given raw keys ``[t, d_i]`` (the block's own band projection), the block a
        row completes BITWISE the compress kernel over its four raw keys, the null ring block / page 0 untouched, the K / V pools' step
        rows within the 3.5 budget (``check_written_rows``).  Returns the number of completed blocks."""
        bs = self.block_size
        raw = raw_keys.view(self.b, self.q_len, -1)
        completed = 0
        cos_t, sin_t = self.position_tables
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            pos0 = L - self.q_len
            for j in range(self.q_len):
                p = pos0 + j
                if p < 0:
                    continue
                assert torch.equal(
                    bits(self.index_state_cache[b + 1, p % self.cap, 0]),
                    bits(raw[b, j]),
                ), f"{label}: ring row of position {p} is not the step's raw key"
                if (p + 1) % bs == 0:
                    blk = p // bs
                    k4 = torch.stack(
                        [
                            (
                                self.index_state_cache[b + 1, q % self.cap, 0]
                                if q < pos0
                                else raw[b, q - pos0]
                            )
                            for q in range(bs * blk, bs * blk + bs)
                        ]
                    )
                    kb = self.compress(
                        k4[None],
                        cos_t[:, bs * blk : bs * blk + 1],
                        sin_t[:, bs * blk : bs * blk + 1],
                    )[0, 0]
                    row = self.index_kv_cache[
                        int(self.block_table[b, blk // self.rows_per_page]),
                        blk % self.rows_per_page,
                        0,
                    ]
                    assert torch.equal(bits(row), bits(kb)), (
                        f"{label}: sequence {b} row {j}: block {blk} is not the compress of its four raw keys"
                    )
                    completed += 1
        assert (
            torch.isnan(self.index_state_cache[0].float()).all()
            and torch.isnan(self.index_kv_cache[NULL_PAGE].float()).all()
        ), f"{label}: the null ring block / page 0 moved"
        self.check_written_rows(label=label)
        return completed

    def check_written_rows(
        self, *, label: str = "pools", rtol: float = 2.0**-7, atol: float = 1e-2
    ) -> None:
        """The step's live rows' K / V in the pools against the oracle's post-RoPE K / V (the 3.5 budget), over the uniform slot mapping."""
        from tests.test_helpers.qwen_3_5_block_cases import cache_rows

        live = self.slot_mapping >= 0
        k_rows, v_rows = cache_rows(
            self.kv_cache,
            self.layout,
            self.slot_mapping[live].to(torch.long),
            self.page_size,
        )
        want_k, want_v = [], []
        for b, L in enumerate(self.kv_lens):
            if L == 0:
                continue
            want_k.append(self.refs[b].k[0, L - self.q_len :])
            want_v.append(self.refs[b].v[0, L - self.q_len :])
        want_k, want_v = torch.cat(want_k), torch.cat(want_v)
        print(
            f"[{label}] written K rows vs oracle cos {cosine(k_rows, want_k):.6f}; V rows cos {cosine(v_rows, want_v):.6f}"
        )
        torch.testing.assert_close(v_rows.float(), want_v.float(), rtol=rtol, atol=atol)
        torch.testing.assert_close(k_rows.float(), want_k.float(), rtol=rtol, atol=atol)

    def raw_keys_of_step(self) -> torch.Tensor:
        """The oracle's raw indexer keys of the step's rows ``[t, d_i]`` (the band projection of the step's hidden states, one rounding) --
        what the block's own GEMM emits up to its rounding; the ring rows are compared with the BLOCK's band through the workspace, this is
        the reference a from-scratch chain uses."""
        return (self.hidden_states.float() @ self.w_band_k.float().t()).to(self.dtype)
