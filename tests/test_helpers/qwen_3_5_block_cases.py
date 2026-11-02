# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Step builders for the Qwen 3.5 gated attention block tests: vLLM-shaped tensors (a 5-D FlashInfer paged cache, a dense block
table, a flat slot mapping, logical KV lengths) around the fp32 oracle of ``qwen_3_5_block_reference``."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tests.test_helpers.qwen_3_5_block_reference import (
    RefGeometry,
    build_rope_tables,
    cosine,
    gated_attention_block_reference,
)

OUT_COS = 0.999  # the dense suite's budget on ``out`` against the fp32 oracle
NULL_PAGE = 0  # the serving stacks' NULL block: a padded entry's table row points here


def ref_geometry(geometry) -> RefGeometry:
    return RefGeometry(
        d_model=geometry.d_model,
        h_q=geometry.num_qo_heads,
        h_kv=geometry.num_kv_heads,
        d_head=geometry.head_dim,
        rope_dim=geometry.rope_dim,
        qk_norm_eps=geometry.qk_norm_eps,
        attn_scale=geometry.sm_scale,
        qk_norm=geometry.qk_norm,
        causal_bottom_right=True,
    )


def random_weights(geometry, *, device, dtype=torch.bfloat16, seed=0, std=0.02):
    """``(Qwen35BlockWeights, w_o)`` drawn in fp32 and rounded once into ``dtype``, the norm weights 1.0."""
    from flashinfer.qwen_3_5_block import Qwen35BlockWeights

    g = torch.Generator(device=device).manual_seed(seed)
    w_qkvg = (
        torch.randn(
            geometry.n_qkvg,
            geometry.d_model,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * std
    ).to(dtype)
    w_o = (
        torch.randn(
            geometry.d_model,
            geometry.num_qo_heads * geometry.head_dim,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * std
    ).to(dtype)
    ones = torch.ones(geometry.head_dim, dtype=dtype, device=device)
    return Qwen35BlockWeights(w_qkvg, ones, ones.clone()), w_o


def new_kv_cache(
    num_pages: int,
    num_kv_heads: int,
    page_size: int,
    head_dim: int,
    layout: str,
    *,
    device,
    dtype=torch.bfloat16,
    fill: float = 0.0,
):
    """A FlashInfer 5-D paged cache, zero (finite) filled: NHD ``[N, 2, P, H, D]`` or HND ``[N, 2, H, P, D]``."""
    shape = (
        (num_pages, 2, page_size, num_kv_heads, head_dim)
        if layout.upper() == "NHD"
        else (num_pages, 2, num_kv_heads, page_size, head_dim)
    )
    return torch.full(shape, fill, device=device, dtype=dtype)


def kv_views_of(kv_cache, layout: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The block's ``[N, H, P, D]`` K / V views of a 5-D cache (what the API binds; strides only)."""
    k, v = kv_cache[:, 0], kv_cache[:, 1]
    if layout.upper() == "NHD":
        return k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    return k, v


def cache_rows(
    kv_cache, layout: str, slots: torch.Tensor, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The K and V rows ``[n, H, D]`` at flat ``slots`` of a 5-D cache, read through the layout as a torch caller would."""
    k, v = kv_views_of(kv_cache, layout)
    page, off = slots // page_size, slots % page_size
    return k[page, :, off], v[page, :, off]


def write_cache_rows(
    kv_cache,
    layout: str,
    slots: torch.Tensor,
    page_size: int,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
) -> None:
    """Write K / V rows ``[n, H, D]`` at flat ``slots`` through the layout (the prefix a torch caller filled)."""
    k, v = kv_views_of(kv_cache, layout)
    page, off = slots // page_size, slots % page_size
    k[page, :, off] = k_rows
    v[page, :, off] = v_rows


class PagedCase:
    """A batch of sequences with full histories, their fp32 oracles, and the serving stack's tensors for ONE step: ``kv_lens``
    logical lengths (0 = a padded entry) of which the last ``q_lens[b]`` tokens are the step's new tokens (prefix = the rest,
    written into the pools by the ORACLE's K / V rows); the step's ``hidden_states`` / ``rope`` rows, ``slot_mapping``,
    ``block_table`` (page 0 = NULL), ``seq_lens`` and ``query_start_loc``."""

    def __init__(
        self,
        geometry,
        kv_lens: Sequence[int],
        q_lens: Sequence[int],
        *,
        page_size: int = 16,
        layout: str = "NHD",
        dtype=torch.bfloat16,
        seed: int = 0,
        device="cuda",
        table_width: Optional[int] = None,
        extra_pages: int = 2,
        weights=None,
    ):
        self.geometry, self.page_size, self.layout, self.dtype, self.device = (
            geometry,
            page_size,
            layout.upper(),
            dtype,
            torch.device(device),
        )
        self.kv_lens, self.q_lens = [int(x) for x in kv_lens], [int(x) for x in q_lens]
        assert len(self.kv_lens) == len(self.q_lens)
        for L, q in zip(self.kv_lens, self.q_lens, strict=False):
            assert (L == 0 and q == 0) or 0 < q <= L, (L, q)
        self.ref_geom = ref_geometry(geometry)
        g = torch.Generator(device=self.device).manual_seed(seed)
        d, dm, rope = geometry.head_dim, geometry.d_model, geometry.rope_dim
        if weights is None:
            weights = random_weights(
                geometry, device=self.device, dtype=dtype, seed=seed
            )
        self.weights, self.w_o = weights
        self.b = len(self.kv_lens)
        self.t = sum(self.q_lens)
        L_max = max(max(self.kv_lens), 1)
        cos_t, sin_t = build_rope_tables(
            L_max,
            rope,
            base=self.ref_geom.rope_base,
            batch=1,
            device=self.device,
            dtype=dtype,
        )
        self.h_full: List[Optional[torch.Tensor]] = []
        self.refs = []
        for L in self.kv_lens:
            if L == 0:
                self.h_full.append(None)
                self.refs.append(None)
                continue
            h = torch.randn(
                1, L, dm, generator=g, device=self.device, dtype=torch.float32
            ).to(dtype)
            self.h_full.append(h)
            self.refs.append(
                gated_attention_block_reference(
                    h,
                    self.weights.w_qkvg,
                    self.weights.w_q_norm,
                    self.weights.w_k_norm,
                    cos_t[:, :L],
                    sin_t[:, :L],
                    self.w_o,
                    self.ref_geom,
                )
            )
        # the step's rows: the LAST q tokens of every live sequence, in sequence order
        rows_h, rows_cos, rows_sin, rows_ref = [], [], [], []
        for b, (L, q) in enumerate(zip(self.kv_lens, self.q_lens, strict=False)):
            if q == 0:
                continue
            rows_h.append(self.h_full[b][0, L - q :])
            rows_cos.append(cos_t[0, L - q : L])
            rows_sin.append(sin_t[0, L - q : L])
            rows_ref.append(self.refs[b].out[0, L - q :])
        self.hidden_states = (
            torch.cat(rows_h)
            if rows_h
            else torch.empty(0, dm, device=self.device, dtype=dtype)
        )
        self.rope_cos = (
            torch.cat(rows_cos)
            if rows_cos
            else torch.empty(0, rope, device=self.device, dtype=dtype)
        )
        self.rope_sin = (
            torch.cat(rows_sin)
            if rows_sin
            else torch.empty(0, rope, device=self.device, dtype=dtype)
        )
        self.ref_out = (
            torch.cat(rows_ref)
            if rows_ref
            else torch.empty(0, dm, device=self.device, dtype=dtype)
        )
        # pages: NULL page 0, then a permutation of the rest; each live sequence takes ceil(L / P) pages
        need = sum(-(-L // page_size) for L in self.kv_lens)
        self.num_pages = need + 1 + extra_pages
        perm = (
            torch.randperm(self.num_pages - 1, generator=g, device=self.device) + 1
        ).tolist()
        max_pages = max(-(-L_max // page_size), 1)
        width = max_pages if table_width is None else table_width
        assert width >= max_pages
        self.block_table = torch.zeros(
            self.b, width, dtype=torch.int32, device=self.device
        )
        self.max_pages = max_pages
        self.kv_cache = new_kv_cache(
            self.num_pages,
            geometry.num_kv_heads,
            page_size,
            d,
            self.layout,
            device=self.device,
            dtype=dtype,
        )
        slots = torch.full((max(self.t, 1),), -1, dtype=torch.int32, device=self.device)
        self.pages: List[List[int]] = []
        it = iter(perm)
        row = 0
        for b, (L, q) in enumerate(zip(self.kv_lens, self.q_lens, strict=False)):
            if L == 0:
                self.pages.append([])
                continue
            pages = [next(it) for _ in range(-(-L // page_size))]
            self.pages.append(pages)
            self.block_table[b, : len(pages)] = torch.tensor(
                pages, dtype=torch.int32, device=self.device
            )
            pos = torch.arange(L, device=self.device)
            flat = (
                torch.tensor(pages, device=self.device)[pos // page_size] * page_size
                + pos % page_size
            )
            if (
                L - q > 0
            ):  # the prefix: the oracle's post-RoPE K / V rows into the pools
                write_cache_rows(
                    self.kv_cache,
                    self.layout,
                    flat[: L - q],
                    page_size,
                    self.refs[b].k[0, : L - q],
                    self.refs[b].v[0, : L - q],
                )
            slots[row : row + q] = flat[L - q :].to(torch.int32)
            row += q
        self.slot_mapping = slots[: self.t] if self.t else slots[:0]
        self.seq_lens = torch.tensor(
            self.kv_lens, dtype=torch.int32, device=self.device
        )
        cu = [0]
        for q in self.q_lens:
            cu.append(cu[-1] + q)
        self.query_start_loc = torch.tensor(cu, dtype=torch.int32, device=self.device)
        self.live_rows = (
            torch.cat(
                [
                    torch.full((q,), L > 0, dtype=torch.bool, device=self.device)
                    for L, q in zip(self.kv_lens, self.q_lens, strict=False)
                ]
            )
            if self.t
            else None
        )

    def check_out(
        self, out: torch.Tensor, *, label: str = "out", min_cos: float = OUT_COS
    ) -> Tuple[float, float]:
        """``out`` rows (``[t, d_model]``) against the per-sequence oracle: finite, cosine over the live rows above the budget."""
        out = out.reshape(-1, self.geometry.d_model)[: self.t]
        assert torch.isfinite(out.float()).all(), f"{label}: non-finite rows"
        c = cosine(out, self.ref_out)
        md = float((out.float() - self.ref_out.float()).abs().max())
        print(
            f"[{label}] vs oracle: cos {c:.6f} max|diff| {md:.3e} (|ref| max {float(self.ref_out.float().abs().max()):.3f})"
        )
        assert c > min_cos, f"{label}: cos {c} <= {min_cos}"
        return c, md

    def check_written_rows(
        self, *, label: str = "pools", rtol: float = 2.0**-7, atol: float = 1e-2
    ) -> None:
        """The step's new tokens' K / V rows in the pools against the oracle's post-RoPE K / V (one rounding of the GEMM output,
        the fused norm + RoPE's single rounding)."""
        k_rows, v_rows = cache_rows(
            self.kv_cache,
            self.layout,
            self.slot_mapping.to(torch.long).clamp_min(0),
            self.page_size,
        )
        want_k, want_v = [], []
        for b, (L, q) in enumerate(zip(self.kv_lens, self.q_lens, strict=False)):
            if q == 0:
                continue
            want_k.append(self.refs[b].k[0, L - q :])
            want_v.append(self.refs[b].v[0, L - q :])
        want_k, want_v = torch.cat(want_k), torch.cat(want_v)
        live = self.slot_mapping >= 0
        ck, cv = cosine(k_rows[live], want_k[live]), cosine(v_rows[live], want_v[live])
        print(f"[{label}] written K rows vs oracle cos {ck:.6f}; V rows cos {cv:.6f}")
        torch.testing.assert_close(
            v_rows[live].float(), want_v[live].float(), rtol=rtol, atol=atol
        )
        torch.testing.assert_close(
            k_rows[live].float(), want_k[live].float(), rtol=rtol, atol=atol
        )

    def decode_args(self) -> Dict[str, torch.Tensor]:
        return dict(
            kv_cache=self.kv_cache,
            slot_mapping=self.slot_mapping,
            block_table=self.block_table,
            seq_lens=self.seq_lens,
        )

    def extend_args(self) -> Dict[str, torch.Tensor]:
        return dict(
            query_start_loc=self.query_start_loc,
            kv_cache=self.kv_cache,
            slot_mapping=self.slot_mapping,
            block_table=self.block_table,
            seq_lens=self.seq_lens,
        )
