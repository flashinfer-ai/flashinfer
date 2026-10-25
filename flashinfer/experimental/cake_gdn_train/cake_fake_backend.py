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

Fake backend of the GDN KCP route: a torch (FP32 accumulate, BF16 tapes)
implementation of the SEMANTICS of the nine programs of :mod:`.cake_jit`, on
the same tapes, pools and tables the kernels take, behind the same call
``run_module(name, values, grid=..., device_index=...)``.  The host swaps the
kernel backend for this one by one switch and exercises its whole path --
exchange schedule, saved set, replay, autograd -- before the kernels exist, on
any device (the tests use the CPU).

Semantics (FLA ``chunk_gated_delta_rule`` conventions; chunk
size 64, head dimension 128, value head ``hv`` reads Q/K head ``hv // (HV // H_qk)``):

  #1 fwd_prep        G = cumsum(g) per chunk; gamma = e^{G_L}; rowscale = e^{G_L - G_t}; qscale = scale e^{G_t};
                     A = (I + tril_(beta_i e^{G_i - G_j} k_i.k_j))^-1 (FP32) -> Akk = bf16(A);
                     w = bf16(Akk . bf16(beta e^G k)), u = bf16(Akk . bf16(beta v)); Aqk = bf16(scale (q k^T) e^{G_t - G_s}), s <= t
  #2 kcp_fwdh        per chunk: h_tape = bf16(h); b_v = u - W bf16(h); v_new = bf16(b_v); h = gamma h + K^T bf16(rowscale b_v)
  #3 kcp_transfer    per chunk: V = [u -] W M; M = gamma M + K^T (rowscale V)   (FP32: the TF32x3 class of the kernel)
  #4 kcp_affine_merge X <- Mop_e X + C_e along a chain (FP32), states published through the chain tables
  #5 fwd_o           o = bf16(qscale (q h_tape) + Aqk v_new)
  #6 bwd_prep        do_g = bf16(qscale do); dv_intra = bf16(Aqk^T do)
  #7 kcp_dhu         per chunk (reverse): dh_tape = bf16(dh); b_dv = dv_intra + rowscale (K bf16(dh)); dv_new = bf16(b_dv)
                     (the host passes one buffer for dv_intra and dv_new: each row is read before it is stored);
                     dh = gamma dh + Q^T do_g - W^T bf16(b_dv)
  #8 bwd_dqkg        dq, dk (FP32 over the value heads of the group, one BF16 rounding), dv (WY pullback), dbeta,
                     dg partials and the chunk gate term gamma_dot (all FP32)
  #9 gate_epilogue   dg = reverse cumsum within the chunk of (dg partials + gamma_dot at the last token), FP32

Every rounding point is a call of :func:`bf16` marked ``# ROUND``; everything
else is FP32 (``torch.matmul`` of FP32-widened operands = the FP32-accumulate
MMA).  The arithmetic is deterministic for one device and thread count, so a
replay of the same programs on the same inputs reproduces the forward's bits
and ``boundary_mode="saved"`` / ``"replay"`` give bitwise equal gradients
(G3).  Agreement with the kernels is NOT bitwise (different accumulation
orders); agreement with the FP64 reference is FLA-class (BF16 tapes).

The second half of the module emulates ``W`` ranks in one process on the real
:mod:`.cake_kcp_plan` contexts and :mod:`.cake_tables` tables (dictionary
``all_to_all_single`` over the published split sizes; forward c1 -> owner
chains -> c2 -> tape walk -> output; backward b1 -> transposed chains -> b2 ->
adjoint tape walk -> gradients) for the CPU tests of this package; the host's
own orchestration (:mod:`.cake_kcp`) runs the same programs on NCCL.
Segment plans with ``P > 1`` are not emulated here (build the contexts with
``sm_count=1`` for ``P = 1``); the programs themselves take any segment list.

The dictionary collective is adapted from ``tests/test_helpers/cake_kcp_fake_backend.py`` at FlashInfer
13d7fcf4041635f5b7f573bb0dbab23570ae1034 (sha256 eb733b4e8e15556f023379332939454c4fa5b15a2d258168e3c9daa510a4b006).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import torch

from . import cake_jit, cake_tables
from .cake_kcp_plan import (
    CHUNK,
    HEAD_DIM,
    TILE,
    KcpContext,
    KcpPartition,
    build_kcp_context,
)

F32, BF16 = torch.float32, torch.bfloat16
SCALE = HEAD_DIM**-0.5
GRAD_NAMES = (
    "dq",
    "dk",
    "dv",
    "dbeta",
    "dg",
)  # the autograd backward's output order
OPERANDS = ("q", "k", "v", "g", "beta", "dout")


def bf16(x: torch.Tensor) -> torch.Tensor:
    """The one rounding primitive: FP32 -> BF16 (round to nearest even), as the kernels' ``cvt.rn.bf16.f32``."""
    return x.to(BF16)


def f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(F32)


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP32-accumulate product of FP32-widened operands (the MMA)."""
    return torch.matmul(f32(a), f32(b))


def _ints(t) -> list[int]:
    if isinstance(t, torch.Tensor):
        return [int(x) for x in t.tolist()]
    return [int(x) for x in t]


def _tile_elem(index: int, hv: int, hg: int, n: int, tile_stride: int) -> int:
    """Element offset of (slot ``index``, head ``hv``) in a pool: tile ``((hv // hg) * n + index) * hg + hv % hg``,
    ``tile_stride`` tiles apart (the base pointer the host passes already selects the half of a pair)."""
    return (((hv // hg) * n + index) * hg + hv % hg) * tile_stride * TILE


def _pool_tile(
    pool: torch.Tensor, index: int, hv: int, hg: int, n: int, tile_stride: int
) -> torch.Tensor:
    e = _tile_elem(index, hv, hg, n, tile_stride)
    return pool[e : e + TILE].view(HEAD_DIM, HEAD_DIM)


def _group(values: Mapping[str, Any]) -> tuple[int, int, int]:
    HV = int(values["num_heads"])
    H_qk = int(values.get("num_qk_heads", HV))
    if HV % H_qk:
        raise ValueError(f"num_heads={HV} must be a multiple of num_qk_heads={H_qk}")
    return HV, H_qk, HV // H_qk


def _chunks(values: Mapping[str, Any]):
    """``(c, bos, L)`` of the real chunks ``c < num_chunks``."""
    bos, ln = _ints(values["chunk_bos"]), _ints(values["chunk_len"])
    for c in range(int(values["num_chunks"])):
        if ln[c] > 0:
            yield c, bos[c], ln[c]


def _segment_chunks(values: Mapping[str, Any], seg: int) -> range:
    table = _ints(values["seg_chunk_start"])  # the walks' segment table (logical name)
    return range(table[seg], table[seg + 1])


def _decay(G: torch.Tensor) -> torch.Tensor:
    """``e^{G_i - G_j}`` for ``i >= j`` (``<= 1``), zero above the diagonal; the difference is masked before the
    exponential so that no overflow path exists (no kernel forms ``e^{-G_s}``)."""
    L = G.shape[0]
    diff = G[:, None] - G[None, :]
    mask = torch.ones(L, L, dtype=torch.bool, device=G.device).tril()
    return torch.where(mask, diff, torch.full_like(diff, float("-inf"))).exp()


# ----------------------------------------------------------------------------------------------------------------
# #1 fwd_prep


def fwd_prep(values: Mapping[str, Any]) -> None:
    q, k, v, g, beta = (values[n] for n in ("q", "k", "v", "g", "beta"))
    HV, _H_qk, r = _group(values)
    scale = float(values["scale"])
    w, u, aqk, akk = (values[n] for n in ("w", "u", "aqk", "akk"))
    g_cum, rowscale, qscale, gamma = (
        values[n] for n in ("g_cum", "rowscale", "qscale", "gamma")
    )
    eye = torch.eye(CHUNK, dtype=F32, device=g.device)
    n_real = 0
    for c, bos, L in _chunks(values):
        n_real = max(n_real, c + 1)
        tok = slice(bos, bos + L)
        rows = slice(c * CHUNK, c * CHUNK + L)
        G = torch.cumsum(f32(g[tok]), 0)  # [L, HV] FP32 chunk-local cumsum
        g_cum[rows] = G
        gamma[c] = G[L - 1].exp()
        rowscale[rows] = (G[L - 1][None, :] - G).exp()
        qscale[rows] = scale * G.exp()
        bet = f32(beta[tok])  # [L, HV]
        for hv in range(HV):
            hq = hv // r
            kk, qq = f32(k[tok, hq]), f32(q[tok, hq])  # [L, 128]
            D = _decay(G[:, hv])  # e^{G_i - G_j}, i >= j
            KK = mm(kk, kk.transpose(0, 1))
            A_pre = (bet[:, hv][:, None] * KK * D).tril(
                -1
            )  # strictly lower: beta_i e^{G_i - G_j} k_i . k_j
            Ainv = torch.linalg.solve_triangular(
                eye[:L, :L] + A_pre, eye[:L, :L], upper=False, unitriangular=True
            )  # FP32 block inverse
            A_b = bf16(Ainv)  # ROUND #1: Akk tape
            akk[rows, hv, :L] = A_b
            kb = bf16(
                kk * (bet[:, hv] * G[:, hv].exp())[:, None]
            )  # ROUND: MMA operand beta e^G k (FLA b_kb)
            w[rows, hv] = bf16(mm(A_b, kb))  # ROUND #1: w tape
            vb = bf16(
                f32(v[tok, hv]) * bet[:, hv][:, None]
            )  # ROUND: MMA operand beta v (FLA b_vb)
            u[rows, hv] = bf16(mm(A_b, vb))  # ROUND #1: u tape
            aqk[rows, hv, :L] = bf16(
                (scale * mm(qq, kk.transpose(0, 1)) * D).tril()
            )  # ROUND #1: Aqk tape
    gamma[n_real:] = (
        1.0  # pad chunks: gamma = 1, their rows stay 0 (the #1 output contract); never walked
    )

    # ----------------------------------------------------------------------------------------------------------------
    # #2 / #3 / #7 walks over the listed segments


def _walk_io(values: Mapping[str, Any], prefix: str):
    """``(pool, index table, hg, n)`` of the pool ``prefix`` (``h0`` / ``dht`` / ``out``)."""
    return (
        values[prefix],
        _ints(values[f"{prefix}_index"]),
        int(values[f"{prefix}_hg"]),
        int(values[f"{prefix}_n"]),
    )


def kcp_fwdh(values: Mapping[str, Any], *, tape: bool) -> None:
    k, w, u, rowscale, gamma = (values[n] for n in ("k", "w", "u", "rowscale", "gamma"))
    HV, _H_qk, r = _group(values)
    bos_t, len_t = _ints(values["chunk_bos"]), _ints(values["chunk_len"])
    stride = int(values["tile_stride"])
    h0, h0_index, h0_hg, h0_n = _walk_io(values, "h0")
    out, out_index, out_hg, out_n = _walk_io(values, "out")
    h_tape, v_new = values.get("h_tape"), values.get("v_new")
    for seg in _ints(values["walk_seg"]):
        for hv in range(HV):
            hq = hv // r
            if h0_index[seg] >= 0:
                h = _pool_tile(h0, h0_index[seg], hv, h0_hg, h0_n, stride).clone()
            else:
                h = torch.zeros(HEAD_DIM, HEAD_DIM, dtype=F32, device=w.device)
            for c in _segment_chunks(values, seg):
                bos, L = bos_t[c], len_t[c]
                if L == 0:
                    continue
                tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
                hb = bf16(h)  # ROUND #2: h tape = the W-side operand bf16(h_c)
                if tape:
                    h_tape[c, hv] = hb
                b_v = f32(u[rows, hv]) - mm(w[rows, hv], hb)  # FP32
                if tape:
                    v_new[rows, hv] = bf16(b_v)  # ROUND #2: v_new tape
                kop = bf16(
                    rowscale[rows, hv][:, None] * b_v
                )  # ROUND #2: K-side operand bf16(rowscale b_v)
                h = gamma[c, hv] * h + mm(f32(k[tok, hq]).transpose(0, 1), kop)
            if out_index[seg] >= 0:
                _pool_tile(out, out_index[seg], hv, out_hg, out_n, stride).copy_(h)


def kcp_transfer_walk(values: Mapping[str, Any]) -> None:
    k, w, u, rowscale, gamma = (values[n] for n in ("k", "w", "u", "rowscale", "gamma"))
    HV, _H_qk, r = _group(values)
    bos_t, len_t = _ints(values["chunk_bos"]), _ints(values["chunk_len"])
    stride = int(values["tile_stride"])
    out, out_index, out_hg, out_n = _walk_io(values, "out")
    init_mode, use_u = int(values["init_mode"]), int(values["use_u"])
    transposed = int(values["write_transposed"])
    eye = torch.eye(HEAD_DIM, dtype=F32, device=w.device)
    for seg in _ints(values["walk_seg"]):
        for hv in range(HV):
            hq = hv // r
            M = torch.zeros_like(eye) if init_mode else eye.clone()
            for c in _segment_chunks(values, seg):
                bos, L = bos_t[c], len_t[c]
                if L == 0:
                    continue
                tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
                V = -mm(w[rows, hv], M)  # FP32 (TF32 hi/lo products in the kernel)
                if use_u:
                    V = f32(u[rows, hv]) + V
                M = gamma[c, hv] * M + mm(
                    f32(k[tok, hq]).transpose(0, 1), rowscale[rows, hv][:, None] * V
                )
            if out_index[seg] >= 0:
                dst = _pool_tile(out, out_index[seg], hv, out_hg, out_n, stride)
                dst.copy_(
                    M if transposed else M.transpose(0, 1)
                )  # write_transposed=0 writes M^T (merge operand)


def kcp_dhu(values: Mapping[str, Any], *, tape: bool) -> None:
    q, k, w, do_g, dv_intra = (values[n] for n in ("q", "k", "w", "do_g", "dv_intra"))
    rowscale, gamma = values["rowscale"], values["gamma"]
    HV, _H_qk, r = _group(values)
    bos_t, len_t = _ints(values["chunk_bos"]), _ints(values["chunk_len"])
    stride = int(values["tile_stride"])
    dht, dht_index, dht_hg, dht_n = _walk_io(values, "dht")
    out, out_index, out_hg, out_n = _walk_io(values, "out")
    dh_tape, dv_new = values.get("dh_tape"), values.get("dv_new")
    for seg in _ints(values["walk_seg"]):
        for hv in range(HV):
            hq = hv // r
            if dht_index[seg] >= 0:
                dh = _pool_tile(dht, dht_index[seg], hv, dht_hg, dht_n, stride).clone()
            else:
                dh = torch.zeros(HEAD_DIM, HEAD_DIM, dtype=F32, device=w.device)
            for c in reversed(_segment_chunks(values, seg)):
                bos, L = bos_t[c], len_t[c]
                if L == 0:
                    continue
                tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
                dhb = bf16(dh)  # ROUND #7: dh tape = the K-side operand bf16(dh_{c+1})
                if tape:
                    dh_tape[c, hv] = dhb
                b_dv = f32(dv_intra[rows, hv]) + rowscale[rows, hv][:, None] * mm(
                    k[tok, hq], dhb
                )  # FP32
                dvb = bf16(b_dv)  # ROUND #7: dv_new tape (and the W-side operand)
                if tape:
                    dv_new[rows, hv] = dvb
                dh = (
                    gamma[c, hv] * dh
                    + mm(f32(q[tok, hq]).transpose(0, 1), do_g[rows, hv])
                    - mm(f32(w[rows, hv]).transpose(0, 1), dvb)
                )
            if out_index[seg] >= 0:
                _pool_tile(out, out_index[seg], hv, out_hg, out_n, stride).copy_(dh)


# ----------------------------------------------------------------------------------------------------------------
# #4 affine merge (``kda_chunk_kcp_affine_merge_fp32_v1`` semantics)


def _state_view(pool: torch.Tensor, elem: int, ncols: int) -> list[torch.Tensor]:
    """The ``[128, 128]`` tile views of a ``ncols``-column state at element ``elem`` (one tile, or the pair)."""
    return [
        pool[elem + i * TILE : elem + (i + 1) * TILE].view(HEAD_DIM, HEAD_DIM)
        for i in range(ncols // HEAD_DIM)
    ]


def kcp_affine_merge(values: Mapping[str, Any]) -> None:
    m, c, x0, y = (values[n] for n in ("m", "c", "x0", "y"))
    chain_start = _ints(values["chain_start"])
    elem_m, elem_c, elem_y = (
        _ints(values["elem_m"]),
        _ints(values["elem_c"]),
        _ints(values["elem_y"]),
    )
    x0_index, y_final = _ints(values["x0_index"]), _ints(values["y_final"])
    ncols, transpose_m, write_states = (
        int(values["ncols"]),
        int(values["transpose_m"]),
        int(values["write_states"]),
    )
    if ncols not in (HEAD_DIM, 2 * HEAD_DIM):
        raise ValueError(f"ncols must be 128 or 256, got {ncols}")
    n_tiles = ncols // HEAD_DIM
    eye = torch.eye(HEAD_DIM, dtype=F32, device=m.device)

    def publish(pool, elem, X):
        for i, view in enumerate(_state_view(pool, elem, ncols)):
            view.copy_(X[:, i * HEAD_DIM : (i + 1) * HEAD_DIM])

    for j in range(len(x0_index)):
        if x0_index[j] >= 0:
            X = torch.cat(_state_view(x0, x0_index[j], ncols), 1).clone()
        elif n_tiles == 1:
            X = torch.zeros(HEAD_DIM, ncols, dtype=F32, device=m.device)
        else:
            X = torch.cat(
                [torch.zeros_like(eye), eye], 1
            )  # map mode starts from [0 | I]
        for e in range(chain_start[j], chain_start[j + 1]):
            if write_states and elem_y[e] >= 0:
                publish(y, elem_y[e], X)
            tile = m[elem_m[e] : elem_m[e] + TILE].view(HEAD_DIM, HEAD_DIM)
            Mop = (
                tile if transpose_m else tile.transpose(0, 1)
            )  # transpose_m=0: the tile holds Mop^T
            X = torch.matmul(
                Mop, X
            )  # FP32 (the kernel: one FMA chain per output in fixed order)
            if elem_c[e] >= 0:
                X[:, :HEAD_DIM] += c[elem_c[e] : elem_c[e] + TILE].view(
                    HEAD_DIM, HEAD_DIM
                )
        if y_final[j] >= 0:
            publish(y, y_final[j], X)


# ----------------------------------------------------------------------------------------------------------------
# #5 fwd_o, #6 bwd_prep, #8 bwd_dqkg, #9 gate_epilogue


def fwd_o(values: Mapping[str, Any]) -> None:
    q, h_tape, v_new, aqk, qscale, o = (
        values[n] for n in ("q", "h_tape", "v_new", "aqk", "qscale", "o")
    )
    HV, _H_qk, r = _group(values)
    for c, bos, L in _chunks(values):
        tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
        for hv in range(HV):
            hq = hv // r
            acc = qscale[rows, hv][:, None] * mm(q[tok, hq], h_tape[c, hv]) + mm(
                aqk[rows, hv, :L], v_new[rows, hv]
            )
            o[tok, hv] = bf16(acc)  # ROUND #5: out


def bwd_prep(values: Mapping[str, Any]) -> None:
    do, aqk, qscale, do_g, dv_intra = (
        values[n] for n in ("do", "aqk", "qscale", "do_g", "dv_intra")
    )
    HV = int(values["num_heads"])
    for c, bos, L in _chunks(values):
        tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
        for hv in range(HV):
            do_g[rows, hv] = bf16(
                qscale[rows, hv][:, None] * f32(do[tok, hv])
            )  # ROUND #6: do_g
            dv_intra[rows, hv] = bf16(
                mm(aqk[rows, hv, :L].transpose(0, 1), do[tok, hv])
            )  # ROUND #6: dv_intra


def bwd_dqkg(values: Mapping[str, Any]) -> None:
    q, k, v, do = (values[n] for n in ("q", "k", "v", "do"))
    h_tape, dh_tape, v_new, dv_new, w, u, aqk, akk, do_g = (
        values[n]
        for n in (
            "h_tape",
            "dh_tape",
            "v_new",
            "dv_new",
            "w",
            "u",
            "aqk",
            "akk",
            "do_g",
        )
    )
    beta, g_cum, rowscale, gamma = (
        values[n] for n in ("beta", "g_cum", "rowscale", "gamma")
    )
    dq, dk, dv, dbeta, dg_part, gamma_dot = (
        values[n] for n in ("dq", "dk", "dv", "dbeta", "dg_part", "gamma_dot")
    )
    scale = float(values["scale"])
    HV, H_qk, r = _group(values)
    for c, bos, L in _chunks(values):
        tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
        for hq in range(H_qk):
            kk, qq = f32(k[tok, hq]), f32(q[tok, hq])
            KK = mm(kk, kk.transpose(0, 1))
            dq_acc = torch.zeros(
                L, HEAD_DIM, dtype=F32, device=q.device
            )  # FP32 over the group's value heads
            dk_acc = torch.zeros_like(dq_acc)
            for hv in range(hq * r, (hq + 1) * r):
                vv, dd = f32(v[tok, hv]), f32(do[tok, hv])
                h, dh = f32(h_tape[c, hv]), f32(dh_tape[c, hv])
                vn, dvn, ww, dog = (f32(t[rows, hv]) for t in (v_new, dv_new, w, do_g))
                A, Ainv = f32(aqk[rows, hv, :L]), f32(akk[rows, hv, :L])
                rs, G = rowscale[rows, hv], g_cum[rows, hv]
                bet = f32(beta[tok, hv])
                eG = G.exp()
                D = _decay(G)
                # attention-like part: o = qscale (q h) + Aqk v_new
                dAqk = mm(
                    dd, vn.transpose(0, 1)
                ).tril()  # dAqk[t, s] = do_t . v_new_s, s <= t
                dS = bf16(
                    scale * dAqk * D
                )  # ROUND #8: the dq / dk MMA operand (FLA b_ds)
                dq_acc += mm(dS, kk) + mm(dog, h.transpose(0, 1))
                dk_acc += mm(dS.transpose(0, 1), qq)
                dkh = rs[:, None] * mm(
                    vn, dh.transpose(0, 1)
                )  # rowscale_t (v_new_t dh_{c+1}^T)
                dk_acc += dkh
                P = dAqk * A
                dG = P.sum(1) - P.sum(0)  # gate factors of Aqk
                dG += (dog * mm(qq, h)).sum(1)  # qscale of the q h term
                dG -= (dkh * kk).sum(1)  # rowscale_t of the K-side update
                gamma_dot[c, hv] = (
                    gamma[c, hv] * (dh * h).sum() + (dkh * kk).sum()
                )  # chunk term at G_L
                # WY pullback: w = A (beta e^G k), u = A (beta v), A = (I + tril_(beta_i e^{G_i - G_j} k_i . k_j))^-1
                dw = bf16(
                    -mm(dvn, h.transpose(0, 1))
                )  # ROUND #8: dW = -dv_new h^T (FLA stores dw bf16)
                kb = bf16(kk * (bet * eG)[:, None])  # ROUND: the #1 operand beta e^G k
                dkb = mm(Ainv.transpose(0, 1), dw)  # d(beta e^G k) = A^T dW
                dk_acc += dkb * (bet * eG)[:, None]
                dbeta_hv = (dkb * kk * eG[:, None]).sum(1)
                dG += (dkb * f32(kb)).sum(1)
                dvb_ = mm(Ainv.transpose(0, 1), dvn)  # d(beta v) = A^T du, du = dv_new
                dv[tok, hv] = bf16(dvb_ * bet[:, None])  # ROUND #8: dv
                dbeta_hv += (dvb_ * vv).sum(1)
                vb = bf16(vv * bet[:, None])  # ROUND: the #1 operand beta v
                dA = bf16(
                    mm(dw, kb.transpose(0, 1)) + mm(dvn, vb.transpose(0, 1))
                )  # ROUND #8: dA operand
                dLm = bf16(
                    -mm(mm(Ainv.transpose(0, 1), dA), Ainv.transpose(0, 1)).tril(-1)
                )  # ROUND #8: dL = -A^T dA A^T
                dL = f32(dLm)
                dLd = dL * D  # dL ⊙ e^{G_i - G_j}
                dk_acc += bet[:, None] * mm(
                    bf16(dLd), kk
                )  # ROUND: operand; dk_i += beta_i sum_j dLd[i, j] k_j
                dk_acc += mm(
                    bf16(dLd).transpose(0, 1), bf16(bet[:, None] * kk)
                )  # ROUND: operands; dk_j
                dbeta_hv += (dLd * KK).sum(1)
                PL = dL * (bet[:, None] * KK * D).tril(-1)  # dL ⊙ L
                dG += PL.sum(1) - PL.sum(0)
                dbeta[tok, hv] = dbeta_hv
                dg_part[rows, hv] = dG
            dq[tok, hq] = bf16(
                dq_acc
            )  # ROUND #8: one BF16 rounding of the FP32 group sum
            dk[tok, hq] = bf16(dk_acc)  # ROUND #8


def gate_epilogue(values: Mapping[str, Any]) -> None:
    dg_part, gamma_dot, dg = values["dg_part"], values["gamma_dot"], values["dg"]
    for c, bos, L in _chunks(values):
        tok, rows = slice(bos, bos + L), slice(c * CHUNK, c * CHUNK + L)
        part = dg_part[rows].clone()  # [L, HV] FP32
        part[L - 1] += gamma_dot[c]
        dg[tok] = torch.flip(
            torch.cumsum(torch.flip(part, [0]), 0), [0]
        )  # reverse cumsum, fixed order


# ----------------------------------------------------------------------------------------------------------------
# The backend entry: the kernels' call


def run_module(
    name: str,
    values: Mapping[str, Any],
    *,
    grid: tuple[int, int, int] | None = None,
    device_index: int | None = None,
    module: Any | None = None,
) -> None:
    """Run record ``name`` of :data:`cake_jit.MODULES` (or ``TEST_MODULES``) on logical ``values`` -- the same call
    as :func:`cake_jit.run_module`; ``grid`` / ``device_index`` / ``module`` are accepted and ignored (the fake
    backend derives its work from the tables).  Every logical argument of the record's plan must be present."""
    record = cake_jit.MODULES.get(name) or cake_jit.TEST_MODULES.get(name)
    if record is None:
        raise KeyError(f"{name!r} is not a GDN KCP record")
    missing = sorted(set(record["logical"].values()) - set(values))
    if missing:
        raise ValueError(f"record {name} needs logical argument(s) {missing}")
    program = record["program"]
    if program == "fwd_prep":
        fwd_prep(values)
    elif program == "kcp_fwdh_offset":
        kcp_fwdh(values, tape=False)
    elif program == "kcp_fwdh_tape":
        kcp_fwdh(values, tape=True)
    elif program == "kcp_transfer_walk":
        kcp_transfer_walk(values)
    elif program == "kcp_affine_merge":
        kcp_affine_merge(values)
    elif program == "fwd_o":
        fwd_o(values)
    elif program == "bwd_prep":
        bwd_prep(values)
    elif program == "kcp_dhu_offset":
        kcp_dhu(values, tape=False)
    elif program == "kcp_dhu_tape":
        kcp_dhu(values, tape=True)
    elif program == "bwd_dqkg":
        bwd_dqkg(values)
    elif program == "gate_epilogue":
        gate_epilogue(values)
    else:  # pragma: no cover - the registry lists exactly these programs
        raise ValueError(
            f"record {name}: no fake implementation of program {program!r}"
        )


# ----------------------------------------------------------------------------------------------------------------
# Tapes and pools of one rank (the host's allocation shapes)


def tape_shapes(
    T_r: int, NC_pad: int, R: int, H_qk: int, HV: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shape and dtype of every tape / output of one rank (the frozen tape set)."""
    return {
        "w": ((R, HV, HEAD_DIM), BF16),
        "u": ((R, HV, HEAD_DIM), BF16),
        "aqk": ((R, HV, CHUNK), BF16),
        "akk": ((R, HV, CHUNK), BF16),
        "g_cum": ((R, HV), F32),
        "rowscale": ((R, HV), F32),
        "qscale": ((R, HV), F32),
        "gamma": ((NC_pad, HV), F32),
        "h_tape": ((NC_pad, HV, HEAD_DIM, HEAD_DIM), BF16),
        "v_new": ((R, HV, HEAD_DIM), BF16),
        "do_g": ((R, HV, HEAD_DIM), BF16),
        "dv_intra": ((R, HV, HEAD_DIM), BF16),
        "dh_tape": ((NC_pad, HV, HEAD_DIM, HEAD_DIM), BF16),
        "dv_new": ((R, HV, HEAD_DIM), BF16),
        "dg_part": ((R, HV), F32),
        "gamma_dot": ((NC_pad, HV), F32),
        "o": ((T_r, HV, HEAD_DIM), BF16),
        "dq": ((T_r, H_qk, HEAD_DIM), BF16),
        "dk": ((T_r, H_qk, HEAD_DIM), BF16),
        "dv": ((T_r, HV, HEAD_DIM), BF16),
        "dbeta": ((T_r, HV), F32),
        "dg": ((T_r, HV), F32),
    }


def new_tapes(
    ctx: KcpContext, H_qk: int, HV: int, names, device=None
) -> dict[str, torch.Tensor]:
    dev = ctx.device if device is None else device
    shapes = tape_shapes(ctx.T_r, ctx.NC_pad, ctx.R, H_qk, HV)
    return {
        n: torch.zeros(*shapes[n][0], dtype=shapes[n][1], device=dev) for n in names
    }


# ----------------------------------------------------------------------------------------------------------------
# Emulation of W ranks in one process (CPU tests): dictionary collectives on the published splits


def all_to_all(
    sends: list[torch.Tensor],
    send_splits: list[tuple[int, ...]],
    recv_splits: list[tuple[int, ...]],
) -> list[torch.Tensor]:
    """Dictionary ``all_to_all_single`` over ``W`` ranks: ``recv[j]`` is the concatenation over ``i`` of the
    ``j``-th segment of ``sends[i]``.  Checks the split tables of every pair and the buffer sizes."""
    W = len(sends)
    for i in range(W):
        if sends[i].numel() != sum(send_splits[i]):
            raise AssertionError(
                f"rank {i}: send buffer {sends[i].numel()} != sum of its splits {sum(send_splits[i])}"
            )
        for j in range(W):
            if send_splits[i][j] != recv_splits[j][i]:
                raise AssertionError(
                    f"split mismatch {i}->{j}: sender says {send_splits[i][j]}, receiver says {recv_splits[j][i]}"
                )
    offsets = [[0] for _ in range(W)]
    for i in range(W):
        for j in range(W):
            offsets[i].append(offsets[i][-1] + send_splits[i][j])
    out = []
    for j in range(W):
        parts = [sends[i][offsets[i][j] : offsets[i][j + 1]] for i in range(W)]
        buf = torch.cat(parts) if parts else sends[0].new_zeros(0)
        if buf.numel() != sum(recv_splits[j]):
            raise AssertionError(
                f"rank {j}: receive buffer {buf.numel()} != sum of its splits {sum(recv_splits[j])}"
            )
        out.append(buf)
    return out


@dataclasses.dataclass
class FakeRank:
    """One emulated rank: its context, tables, operands (local order, no batch axis) and tapes."""

    rank: int
    ctx: KcpContext
    lt: cake_tables.LocalTables
    wt: cake_tables.WalkTables
    ops: dict[str, torch.Tensor]
    H_qk: int
    HV: int
    scale: float

    def common(self) -> dict[str, Any]:
        """The logical arguments every program of this rank shares (operands, tables, head counts)."""
        return {
            "q": self.ops["q"],
            "k": self.ops["k"],
            "v": self.ops["v"],
            "do": self.ops["dout"],
            "g": self.ops["g"],
            "beta": self.ops["beta"],
            "num_heads": self.HV,
            "num_qk_heads": self.H_qk,
            "num_chunks": self.lt.NC,
            "scale": self.scale,
            "chunk_bos": self.lt.device["chunk_bos"],
            "chunk_len": self.lt.device["chunk_len"],
            "seq_chunk_start": self.lt.device["seq_chunk_start"],
            # the persistent programs' ticket word (a zero u32 slot; the emulation derives its work from the
            # tables and never reads it)
            "ticket": torch.zeros(
                (32,), dtype=torch.uint32, device=self.ops["q"].device
            ),
        }

    def walk_common(self, segments: str) -> dict[str, Any]:
        """The segment arguments of a walk: ``segments`` = ``"offset"`` (the crossing-out / staged segments) or
        ``"tape"`` (all)."""
        wt = self.wt
        return {
            "seg_chunk_start": wt.device["seg_chunk_start"],
            "num_seqs": wt.NSEG,
            "walk_seg": wt.device[
                "walk_seg_offset" if segments == "offset" else "walk_seg_tape"
            ],
        }

    def pool_args(
        self, prefix: str, pool_name: str, buffer: torch.Tensor, index: str
    ) -> dict[str, Any]:
        """``prefix`` (``h0`` / ``dht`` / ``out``) arguments of pool ``pool_name`` (:data:`cake_tables.POOLS`) on
        ``buffer`` with the segment index table ``index``: the base pointer selects the half of a pair."""
        pa = self.wt.pools[pool_name]
        return {
            prefix: buffer[pa.base_elem :],
            f"{prefix}_index": self.wt.device[index],
            f"{prefix}_hg": pa.hg,
            f"{prefix}_n": pa.n,
            "tile_stride": pa.tile_stride,
        }

    def zero_pool(self, prefix: str) -> dict[str, Any]:
        """A never-read pool: every index ``-1``."""
        return {
            prefix: torch.zeros(TILE, dtype=F32, device=self.ctx.device),
            f"{prefix}_index": torch.full(
                (self.wt.NSEG,), -1, dtype=torch.int32, device=self.ctx.device
            ),
            f"{prefix}_hg": self.HV,
            f"{prefix}_n": 1,
        }


def build_ranks(
    part: KcpPartition,
    ops_by_rank: list[dict[str, torch.Tensor]],
    *,
    H_qk: int,
    HV: int,
    scale: float = SCALE,
    sm_count: int = 1,
    conv_kernel_size: int = 4,
    p_rule: str = "wave_fill1",
    device="cpu",
) -> list[FakeRank]:
    """The emulated ranks of ``part``: one context per rank (``sm_count=1`` keeps ``P = 1``), the tables and the
    rank's operands (``q, k, v, g, beta, dout`` in local order without the batch axis)."""
    ranks = []
    for r in range(part.world_size):
        ctx = build_kcp_context(
            part, None, r, device, conv_kernel_size, sm_count=sm_count, p_rule=p_rule
        )
        pt = ctx.prepare(H_qk, HV)  # first use: the device-resident table bundle
        wt = pt.walk
        if wt.P != 1:
            raise NotImplementedError(
                f"rank {r}: the emulation takes P = 1 segment plans, got P = {wt.P}"
            )
        ops = ops_by_rank[r]
        for name in OPERANDS:
            if ops[name].shape[0] != ctx.T_r:
                raise ValueError(
                    f"rank {r}: {name} has {ops[name].shape[0]} tokens, the partition says {ctx.T_r}"
                )
        ranks.append(
            FakeRank(
                r,
                ctx,
                pt.local,
                wt,
                ops,
                int(H_qk),
                int(HV),
                float(scale),
            )
        )
    return ranks


def slice_local(
    global_ops: dict[str, torch.Tensor], part: KcpPartition, rank: int
) -> dict[str, torch.Tensor]:
    """The rank's local operands (fragments in local order) from the global pack (no batch axis)."""
    cu = part.cu_seqlens
    out = {}
    for name in OPERANDS:
        t = global_ops[name]
        pieces = [
            t[cu[f.doc] + f.off : cu[f.doc] + f.off + f.length]
            for f in part.fragments[rank]
            if f.length > 0
        ]
        out[name] = torch.cat(pieces, 0) if pieces else t[:0].clone()
    return out


def gather_global(part: KcpPartition, per_rank: list[torch.Tensor]) -> torch.Tensor:
    """Per-rank ``[T_r, ...]`` tensors -> the pack's global token order ``[T, ...]`` (the partition inverse)."""
    cu = part.cu_seqlens
    any_t = per_rank[0]
    out = torch.zeros(cu[-1], *any_t.shape[1:], dtype=any_t.dtype, device=any_t.device)
    for r, t in enumerate(per_rank):
        local = part.local_cu_seqlens(r)
        for f in part.fragments[r]:
            if f.length:
                g0 = cu[f.doc] + f.off
                out[g0 : g0 + f.length] = t[local[f.slot] : local[f.slot + 1]]
    return out


@dataclasses.dataclass
class FakeSaved:
    """The saved set of the emulated forward: per rank ``S_in`` (the c2 receive buffer), the owner's c1 receive
    buffer (``save_transfer``) and the light tapes (``gamma`` / ``g_cum`` / ``akk``; ``save_mode="full"`` adds the
    forward's ``w`` / ``u`` / ``aqk`` / ``h_tape`` / ``v_new``)."""

    s_in: list[torch.Tensor]
    c1_recv: list[torch.Tensor] | None
    tapes: list[dict[str, torch.Tensor]]
    save_mode: str
    save_transfer: bool


FWD_TAPES = (
    "w",
    "u",
    "aqk",
    "akk",
    "g_cum",
    "rowscale",
    "qscale",
    "gamma",
    "h_tape",
    "v_new",
)
LIGHT_SAVED = ("gamma", "g_cum", "akk")
FULL_SAVED = LIGHT_SAVED + ("w", "u", "aqk", "h_tape", "v_new")


def _buffers(ranks: list[FakeRank], split: str) -> list[torch.Tensor]:
    return [
        torch.zeros(sum(rk.wt.splits[split]), dtype=F32, device=rk.ctx.device)
        for rk in ranks
    ]


def _prep(rk: FakeRank) -> dict[str, torch.Tensor]:
    tapes = new_tapes(rk.ctx, rk.H_qk, rk.HV, FWD_TAPES)
    if rk.lt.NC:
        run_module("fwd_prep", {**rk.common(), **tapes})
    return tapes


def _exchange_forward(
    ranks: list[FakeRank], tapes: list[dict[str, torch.Tensor]], log: list[str] | None
):
    """Offset + transfer walks -> c1 -> owner chains -> c2.  Returns ``(s_in, c1_recv)`` per rank."""
    c1_send = _buffers(ranks, "c1_send")
    for rk, tp in zip(ranks, tapes, strict=True):
        if rk.wt.n_walk_offset == 0:
            continue
        common = {**rk.common(), **tp, **rk.walk_common("offset")}
        run_module(
            "kcp_fwdh_offset",
            {
                **common,
                **rk.zero_pool("h0"),
                **rk.pool_args("out", "c1_b", c1_send[rk.rank], "out_index"),
            },
        )
        run_module(
            "kcp_transfer_walk",
            {
                **common,
                **rk.pool_args("out", "c1_m", c1_send[rk.rank], "out_index"),
                "init_mode": 0,
                "use_u": 0,
                "write_transposed": 0,
            },
        )
    if log is not None:
        log.append("c1")
    c1_recv = all_to_all(
        c1_send,
        [rk.wt.splits["c1_send"] for rk in ranks],
        [rk.wt.splits["c1_recv"] for rk in ranks],
    )
    c2_send = _buffers(ranks, "c2_send")
    for rk in ranks:
        ch = rk.wt.chains["fwd"]
        if ch.n_chains:
            run_module(
                "kcp_affine_merge_c8",
                {
                    "m": c1_recv[rk.rank],
                    "c": c1_recv[rk.rank],
                    "x0": c1_recv[rk.rank],
                    "y": c2_send[rk.rank],
                    **{
                        k[len("fwd.") :]: v
                        for k, v in rk.wt.device.items()
                        if k.startswith("fwd.")
                    },
                    "ncols": 128,
                    "transpose_m": 0,
                    "write_states": 1,
                    "cols_per_cta": 8,
                },
            )
    if log is not None:
        log.append("c2")
    s_in = all_to_all(
        c2_send,
        [rk.wt.splits["c2_send"] for rk in ranks],
        [rk.wt.splits["c2_recv"] for rk in ranks],
    )
    return s_in, c1_recv


def _tape_walk(rk: FakeRank, tp: dict[str, torch.Tensor], s_in: torch.Tensor) -> None:
    if rk.wt.n_walk_tape == 0:
        return
    run_module(
        "kcp_fwdh_tape",
        {
            **rk.common(),
            **tp,
            **rk.walk_common("tape"),
            **rk.pool_args("h0", "s_in", s_in, "h0_index"),
            **rk.zero_pool("out"),
        },
    )


def emulate_forward(
    ranks: list[FakeRank],
    *,
    save_mode: str = "light",
    save_transfer: bool = True,
    log: list[str] | None = None,
) -> tuple[list[torch.Tensor], FakeSaved]:
    """The forward of every rank: #1 -> (#2 offset, #3) -> c1 -> #4 -> c2 -> #2 tape -> #5.  Returns the outputs
    (local order) and the saved set."""
    if save_mode not in ("light", "full"):
        raise ValueError(f"save_mode must be 'light' or 'full', got {save_mode!r}")
    tapes = [_prep(rk) for rk in ranks]
    s_in, c1_recv = _exchange_forward(ranks, tapes, log)
    outs = []
    for rk, tp in zip(ranks, tapes, strict=True):
        _tape_walk(rk, tp, s_in[rk.rank])
        o = new_tapes(rk.ctx, rk.H_qk, rk.HV, ("o",))["o"]
        if rk.lt.NC:
            run_module("fwd_o", {**rk.common(), **tp, "o": o, "num_seqs": rk.wt.NSEG})
        outs.append(o)
    keep = FULL_SAVED if save_mode == "full" else LIGHT_SAVED
    saved = FakeSaved(
        s_in=[t.clone() for t in s_in],
        c1_recv=[t.clone() for t in c1_recv] if save_transfer else None,
        tapes=[{n: tp[n].clone() for n in keep} for tp in tapes],
        save_mode=save_mode,
        save_transfer=save_transfer,
    )
    return outs, saved


def emulate_backward(
    ranks: list[FakeRank],
    saved: FakeSaved,
    *,
    boundary_mode: str = "saved",
    log: list[str] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """The backward of every rank from the saved set (``boundary_mode="saved"``) or from the inputs alone
    (``"replay"``: the forward exchange is re-run, the saved ``S_in`` / c1 tensors unused): replay of #1 (+ #2 tape
    in ``save_mode="light"``) -> #6 -> #7 offset -> b1 -> #4 transposed -> b2 -> #7 tape -> #8 -> #9.  Returns the
    gradients ``dq, dk, dv, dbeta, dg`` per rank (local order)."""
    if boundary_mode not in ("saved", "replay"):
        raise ValueError(
            f"boundary_mode must be 'saved' or 'replay', got {boundary_mode!r}"
        )
    # tapes: the forward's recompute (#1 and #2 are deterministic: the same bits as the forward's; the saved
    # tapes of either save_mode are checked against the replay here)
    tapes = [_prep(rk) for rk in ranks]
    if boundary_mode == "replay":
        s_in, c1_recv = _exchange_forward(ranks, tapes, log)
        with_maps = False
    else:
        s_in, c1_recv = saved.s_in, saved.c1_recv
        with_maps = not saved.save_transfer
    for rk, tp, kept in zip(ranks, tapes, saved.tapes, strict=True):
        _tape_walk(
            rk, tp, s_in[rk.rank]
        )  # light: the replay of #2 tape; full: the saved tapes' bits
        for n, t in kept.items():
            if not torch.equal(tp[n], t):
                raise AssertionError(
                    f"rank {rk.rank}: the replayed tape {n} differs from the saved one"
                )
    # #6 and the adjoint offsets d_f -> b1
    key = "dm" if with_maps else "d"
    b1_send = _buffers(ranks, f"b1_send_{key}")
    adj = []
    for rk, tp in zip(ranks, tapes, strict=True):
        bw = new_tapes(
            rk.ctx,
            rk.H_qk,
            rk.HV,
            ("do_g", "dv_intra", "dh_tape", "dv_new", "dg_part", "gamma_dot"),
        )
        adj.append(bw)
        if rk.lt.NC:
            run_module(
                "bwd_prep",
                {
                    **rk.common(),
                    **tp,
                    **bw,
                    "do_tstride": int(rk.ops["dout"].stride(0)),
                    "num_seqs": rk.wt.NSEG,
                },
            )
        # the adjoint offset walk visits the crossing-in segments (the segments whose d_f the owners merge)
        in_segs = [s for s in range(rk.wt.NSEG) if rk.wt.h0_index[s] >= 0]
        if in_segs:
            pool = "b1_d_dm" if with_maps else "b1_d"
            run_module(
                "kcp_dhu_offset",
                {
                    **rk.common(),
                    **tp,
                    **bw,
                    "seg_chunk_start": rk.wt.device["seg_chunk_start"],
                    "num_seqs": rk.wt.NSEG,
                    "walk_seg": torch.tensor(
                        in_segs, dtype=torch.int32, device=rk.ctx.device
                    ),
                    **rk.zero_pool("dht"),
                    **rk.pool_args("out", pool, b1_send[rk.rank], "h0_index"),
                },
            )
            if (
                with_maps
            ):  # the re-sent M_f^T of the crossing-in fragments that also cross out
                out_segs = [s for s in in_segs if rk.wt.dht_index[s] >= 0]
                if out_segs:
                    run_module(
                        "kcp_transfer_walk",
                        {
                            **rk.common(),
                            **tp,
                            "seg_chunk_start": rk.wt.device["seg_chunk_start"],
                            "num_seqs": rk.wt.NSEG,
                            "walk_seg": torch.tensor(
                                out_segs, dtype=torch.int32, device=rk.ctx.device
                            ),
                            **rk.pool_args(
                                "out", "b1_m_dm", b1_send[rk.rank], "h0_index"
                            ),
                            "init_mode": 0,
                            "use_u": 0,
                            "write_transposed": 0,
                        },
                    )
    if log is not None:
        log.append("b1")
    b1_recv = all_to_all(
        b1_send,
        [rk.wt.splits[f"b1_send_{key}"] for rk in ranks],
        [rk.wt.splits[f"b1_recv_{key}"] for rk in ranks],
    )
    b2_send = _buffers(ranks, "b2_send")
    family = "adj_b1" if with_maps else "adj_c1"
    for rk in ranks:
        ch = rk.wt.chains[family]
        if ch.n_chains:
            if not with_maps and c1_recv is None:
                raise ValueError(
                    "saved mode with save_transfer needs the forward's c1 receive tensors"
                )
            run_module(
                "kcp_affine_merge_c8",
                {
                    "m": b1_recv[rk.rank] if with_maps else c1_recv[rk.rank],
                    "c": b1_recv[rk.rank],
                    "x0": b1_recv[rk.rank],
                    "y": b2_send[rk.rank],
                    **{
                        k[len(family) + 1 :]: v
                        for k, v in rk.wt.device.items()
                        if k.startswith(family + ".")
                    },
                    "ncols": 128,
                    "transpose_m": 1,
                    "write_states": 1,
                    "cols_per_cta": 8,
                },
            )
    if log is not None:
        log.append("b2")
    ds_out = all_to_all(
        b2_send,
        [rk.wt.splits["b2_send"] for rk in ranks],
        [rk.wt.splits["b2_recv"] for rk in ranks],
    )
    grads = []
    for rk, tp, bw in zip(ranks, tapes, adj, strict=True):
        if rk.wt.n_walk_tape:
            run_module(
                "kcp_dhu_tape",
                {
                    **rk.common(),
                    **tp,
                    **bw,
                    **rk.walk_common("tape"),
                    **rk.pool_args("dht", "ds_out", ds_out[rk.rank], "dht_index"),
                    **rk.zero_pool("out"),
                },
            )
        g_r = new_tapes(rk.ctx, rk.H_qk, rk.HV, GRAD_NAMES)
        if rk.lt.NC:
            # the record's launch logicals the emulation does not read: in-kernel group reduce (the partials
            # dq_part / dk_part are written only for group_reduce == 0, K4 ABI) and the token strides
            run_module(
                "bwd_dqkg",
                {
                    **rk.common(),
                    **tp,
                    **bw,
                    **g_r,
                    "num_seqs": rk.wt.NSEG,
                    "dqk_tstride": int(g_r["dq"].stride(0)),
                    "dv_tstride": int(g_r["dv"].stride(0)),
                    "group_reduce": 1,
                    "dq_part": g_r["dq"],
                    "dk_part": g_r["dk"],
                },
            )
            run_module("gate_epilogue", {**rk.common(), **bw, "dg": g_r["dg"]})
        grads.append(g_r)
    return grads


def emulate(
    part: KcpPartition,
    global_ops: dict[str, torch.Tensor],
    *,
    H_qk: int,
    HV: int,
    scale: float = SCALE,
    save_mode: str = "light",
    save_transfer: bool = True,
    sm_count: int = 1,
    device="cpu",
) -> dict[str, Any]:
    """Forward + both backward modes of the pack ``global_ops`` (``q, k, v, g, beta, dout`` without the batch axis,
    global token order) under partition ``part``.  Returns the global-order ``out`` and gradients of the saved-mode
    backward (``saved``), the replay-mode gradients (``replay``), the collective logs and the ranks."""
    ranks = build_ranks(
        part,
        [slice_local(global_ops, part, r) for r in range(part.world_size)],
        H_qk=H_qk,
        HV=HV,
        scale=scale,
        sm_count=sm_count,
        device=device,
    )
    log_fwd: list[str] = []
    outs, saved = emulate_forward(
        ranks, save_mode=save_mode, save_transfer=save_transfer, log=log_fwd
    )
    log_saved: list[str] = []
    grads_saved = emulate_backward(ranks, saved, boundary_mode="saved", log=log_saved)
    log_replay: list[str] = []
    grads_replay = emulate_backward(
        ranks, saved, boundary_mode="replay", log=log_replay
    )
    result = {"out": gather_global(part, outs)}
    for n in GRAD_NAMES:
        result[n] = gather_global(part, [g[n] for g in grads_saved])
    replay = {n: gather_global(part, [g[n] for g in grads_replay]) for n in GRAD_NAMES}
    return {
        "ranks": ranks,
        "saved": result,
        "replay": replay,
        "logs": {
            "forward": log_fwd,
            "backward_saved": log_saved,
            "backward_replay": log_replay,
        },
        "save_mode": save_mode,
        "save_transfer": save_transfer,
    }


def replay_equals_saved(result: dict[str, Any]) -> bool:
    return all(torch.equal(result["saved"][n], result["replay"][n]) for n in GRAD_NAMES)


__all__ = [
    "FULL_SAVED",
    "FWD_TAPES",
    "GRAD_NAMES",
    "LIGHT_SAVED",
    "OPERANDS",
    "SCALE",
    "FakeRank",
    "FakeSaved",
    "all_to_all",
    "bf16",
    "build_ranks",
    "bwd_dqkg",
    "bwd_prep",
    "emulate",
    "emulate_backward",
    "emulate_forward",
    "fwd_o",
    "fwd_prep",
    "gate_epilogue",
    "gather_global",
    "kcp_affine_merge",
    "kcp_dhu",
    "kcp_fwdh",
    "kcp_transfer_walk",
    "new_tapes",
    "replay_equals_saved",
    "run_module",
    "slice_local",
    "tape_shapes",
]
