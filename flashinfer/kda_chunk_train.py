"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Chunked KDA training backward (SM100a / SM103a).

Deterministic backward pass for the chunked KDA training algorithm
(``chunk_kda`` with in-kernel Q/K L2 normalization, the lower-bound safe gate,
sigmoid beta and ``dt_bias``; chunk size 64, K = V = 128).  The kernels
reproduce the reference chunked dataflow stage by stage -- gate prefix scan
and WY inputs, WY recompute, forward state recurrence, dAv, adjoint state
recurrence, fused dq/dk/dg, intra-chunk backward, and the gate / norm
epilogue -- with fp32 inter-chunk state carriers and fixed-order reductions
(no atomics), so repeated backward passes on identical inputs are
bit-identical.

The forward pass is not part of this module: :func:`chunk_kda_backward`
consumes the tensors the reference forward saves for backward (normalized
q/k with their inverse norms, sigmoid(beta), and the per-chunk ``Aqk`` /
``Akk`` matrices).
"""

from __future__ import annotations

import torch

from .jit.cake_kda_chunk_train import device_arch, kernel

CHUNK = 64
HEAD_DIM = 128
PAIR_ROWS = 2 * CHUNK


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


def _contiguous(name: str, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    _check(t.dtype == dtype, f"{name} must be {dtype}, got {t.dtype}")
    return t.contiguous()


def _rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    return x.reshape(rows, *x.shape[2:])


# --------------------------------------------------------------------------- #
# stage launchers (thin host wrappers over the generated kernels)
# --------------------------------------------------------------------------- #
def _launch_prep(
    arch, g_raw, q_norm, k_norm, v, beta, A_log, dt_bias, aqk, *, lower_bound
):
    rows, hv, kd = g_raw.shape
    h = q_norm.shape[1]
    _check(
        kd == HEAD_DIM
        and rows % CHUNK == 0
        and hv % h == 0
        and tuple(aqk.shape) == (rows, hv, CHUNK),
        "prep: expected [R,HV,128] g/v, [R,H,128] q/k, [R,HV,64] Aqk, R % 64 == 0",
    )
    dev, bf = g_raw.device, torch.bfloat16
    out = {"gk": torch.empty(rows, hv, HEAD_DIM, dtype=torch.float32, device=dev)}
    for name in ("vb", "kb", "qg", "kg", "ke", "qe"):
        out[name] = torch.empty(rows, hv, HEAD_DIM, dtype=bf, device=dev)
    out["aqk_tril"] = torch.empty(rows, hv, CHUNK, dtype=bf, device=dev)
    kernel("prep", arch).launch(
        grid=(rows // CHUNK, hv, 1),
        g_raw=g_raw,
        q_norm=q_norm,
        k_norm=k_norm,
        v=v,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        aqk=aqk,
        gk_out=out["gk"],
        vb_out=out["vb"],
        kb_out=out["kb"],
        qg_out=out["qg"],
        kg_out=out["kg"],
        ke_out=out["ke"],
        qe_out=out["qe"],
        aqk_tril=out["aqk_tril"],
        num_qk_heads=h,
        num_heads=hv,
        group=hv // h,
        lower_bound=float(lower_bound),
    )
    return out


def _launch_wy(arch, akk, vb, kb):
    rows, hv, _ = vb.shape
    _check(rows % PAIR_ROWS == 0, "wy: R % 128 == 0 required")
    u = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=vb.device)
    w = torch.empty_like(u)
    kernel("wy", arch).launch(
        grid=(rows // PAIR_ROWS, hv, 1),
        akk_tma=akk,
        vb_tma=vb,
        kb_tma=kb,
        u_out=u,
        w_out=w,
        num_heads=hv,
    )
    return u, w


def _launch_fwdh(arch, w, kg, u, gk, *, batch, seq_len):
    rows, hv, _ = w.shape
    nt = seq_len // CHUNK
    h_out = torch.empty(
        batch, nt, hv, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=w.device
    )
    v_new = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=w.device)
    kernel("fwdh", arch).launch(
        grid=(batch * hv * 2, 1, 1),
        w_tma=w,
        kg_tma=kg,
        u=u,
        gk=gk,
        h_out=h_out,
        v_new=v_new,
        num_heads=hv,
        seq_len=seq_len,
        num_chunks=nt,
    )
    return h_out, v_new


def _launch_dav(arch, do, v_new, aqk_tril, *, scale):
    rows, hv, _ = do.shape
    dAqk = torch.empty(rows, hv, CHUNK, dtype=torch.float32, device=do.device)
    dv1 = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=do.device)
    kernel("dav", arch).launch(
        grid=(rows // PAIR_ROWS, hv, 1),
        do_tma=do,
        vnew_tma=v_new,
        aqk_tma=aqk_tril,
        dAqk=dAqk,
        dv1=dv1,
        num_heads=hv,
        scale=float(scale),
    )
    return dAqk, dv1


def _launch_dhu(arch, kg, qg, w, do, dv1, gk, *, batch, seq_len, scale):
    rows, hv, _ = kg.shape
    nt = seq_len // CHUNK
    dh_out = torch.empty(
        batch, nt, hv, HEAD_DIM, HEAD_DIM, dtype=torch.bfloat16, device=kg.device
    )
    dv2 = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=kg.device)
    kernel("dhu", arch).launch(
        grid=(batch * hv * 2, 1, 1),
        kg_tma=kg,
        qg_tma=qg,
        w_tma=w,
        do_tma=do,
        dv1=dv1,
        gk=gk,
        dh_out=dh_out,
        dv2=dv2,
        num_heads=hv,
        seq_len=seq_len,
        num_chunks=nt,
        scale=float(scale),
    )
    return dh_out, dv2


def _launch_dqkg(
    arch, do, v_new, dv2, v, k_e, q_e, h, dh, akk, gk, beta, *, batch, seq_len, scale
):
    rows, hv, _ = do.shape
    nt = seq_len // CHUNK
    dev = do.device
    h2 = h.reshape(batch * nt * hv * HEAD_DIM, HEAD_DIM)
    dh2 = dh.reshape(batch * nt * hv * HEAD_DIM, HEAD_DIM)
    dq = torch.empty(rows, hv, HEAD_DIM, dtype=torch.float32, device=dev)
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    db = torch.empty(rows, hv, dtype=torch.float32, device=dev)
    dAkk = torch.empty(rows, hv, CHUNK, dtype=torch.float32, device=dev)
    dv = torch.empty(rows, hv, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kernel("dqkg", arch).launch(
        grid=(rows // CHUNK, hv, 1),
        do_tma=do,
        vn_tma=v_new,
        dv2_tma=dv2,
        v_tma=v,
        h_tma=h2,
        dh_tma=dh2,
        akk_tma=akk,
        h_ptr=h2,
        dh_ptr=dh2,
        k_ptr=k_e,
        q_ptr=q_e,
        v_ptr=v,
        gk=gk,
        beta=beta,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        db_out=db,
        dAkk_out=dAkk,
        dv_out=dv,
        num_heads=hv,
        num_chunks=nt,
        scale=float(scale),
    )
    return dq, dk, dg, db, dAkk, dv


def _launch_intra(arch, dAqk, dAkk, gk, k_e, q_e, beta, dq_f, dk_f, dg_f, db_f):
    rows, hv, _ = gk.shape
    hq = k_e.shape[1]
    dq, dk, dg, db = (torch.empty_like(t) for t in (dq_f, dk_f, dg_f, db_f))
    kernel("intra", arch).launch(
        grid=(rows // CHUNK, hv, 1),
        dAqk=dAqk,
        dAkk=dAkk,
        gk=gk,
        k_e=k_e,
        q_e=q_e,
        beta=beta,
        dq_f=dq_f,
        dk_f=dk_f,
        dg_f=dg_f,
        db_f=db_f,
        dq_out=dq,
        dk_out=dk,
        dg_out=dg,
        db_out=db,
        num_heads=hv,
        num_qk_heads=hq,
        group=hv // hq,
    )
    return dq, dk, dg, db


def _launch_epilogue(
    arch,
    dq_intra,
    dk_intra,
    dg_intra,
    db_total,
    *,
    q_norm,
    k_norm,
    q_rstd,
    k_rstd,
    g_raw,
    beta_raw,
    A_log,
    dt_bias,
    lower_bound,
):
    rows, hv, kd = dg_intra.shape
    h = q_norm.shape[1]
    nc = rows // CHUNK
    dev, bf = dg_intra.device, torch.bfloat16
    dg_out = torch.empty(rows, hv, kd, dtype=bf, device=dev)
    dbeta = torch.empty(rows, hv, dtype=bf, device=dev)
    dA_part = torch.empty(nc, hv, kd, dtype=torch.float32, device=dev)
    dbias_part = torch.empty(nc, hv, kd, dtype=torch.float32, device=dev)
    dq_out = torch.empty(rows, h, kd, dtype=bf, device=dev)
    dk_out = torch.empty(rows, h, kd, dtype=bf, device=dev)
    dA_log = torch.empty(hv, dtype=torch.float32, device=dev)
    dt_bias_grad = torch.empty(hv * kd, dtype=torch.float32, device=dev)
    kernel("gate_epilogue", arch).launch(
        grid=(nc, hv, 1),
        dg_intra=dg_intra,
        g_raw=g_raw,
        db_total=db_total,
        beta_raw=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        dg_out=dg_out,
        dbeta=dbeta,
        dA_part=dA_part,
        dbias_part=dbias_part,
        num_heads=hv,
        lower_bound=float(lower_bound),
    )
    kernel("qk_epilogue", arch).launch(
        grid=(nc, h, 1),
        dq_intra=dq_intra,
        dk_intra=dk_intra,
        q_norm=q_norm,
        k_norm=k_norm,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        dq_out=dq_out,
        dk_out=dk_out,
        num_qk_heads=h,
        num_v_heads=hv,
        group=hv // h,
    )
    kernel("finalize", arch).launch(
        grid=(hv, 1, 1),
        dA_part=dA_part,
        dbias_part=dbias_part,
        dA_log=dA_log,
        dt_bias_grad=dt_bias_grad,
        num_chunks=nc,
        num_heads=hv,
    )
    return dq_out, dk_out, dg_out, dbeta, dA_log, dt_bias_grad


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def chunk_kda_backward(
    *,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    q_rstd: torch.Tensor,
    k_rstd: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_logits: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    lower_bound: float,
    cu_seqlens: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministic chunked KDA training backward.

    Arguments follow the chunked KDA forward's saved-for-backward set:

    - ``q_norm``, ``k_norm``: L2-normalized q/k, bf16 ``[B, T, H, 128]``; ``q_rstd``, ``k_rstd``:
      their inverse norms, fp32 ``[B, T, H]``.
    - ``v``, ``g``: bf16 ``[B, T, HV, 128]`` (``g`` are the raw gate pre-activations).
    - ``beta_logits``: bf16 ``[B, T, HV]`` raw beta; ``beta``: fp32 ``[B, T, HV]`` = sigmoid(beta_logits).
    - ``A_log``: fp32 ``[HV]``; ``dt_bias``: fp32 ``[HV * 128]``.
    - ``Aqk``, ``Akk``: bf16 ``[B, T, HV, 64]`` per-chunk matrices saved by the forward.
    - ``do``: bf16 ``[B, T, HV, 128]`` output gradient; ``scale``: attention scale;
      ``lower_bound``: the safe-gate lower bound (negative).
    - ``cu_seqlens``: optional int32 ``[N + 1]`` packed-sequence offsets; all packed
      sequences must share one length that is a multiple of 128 (``B`` must be 1).

    ``T`` must be a multiple of 128.  ``HV`` must be a positive multiple of ``H``
    (grouped value heads).  Returns ``dq``/``dk`` bf16 ``[B, T, H, 128]``, ``dv`` bf16
    ``[B, T, HV, 128]``, ``dbeta`` bf16 ``[B, T, HV]``, ``dg`` bf16 ``[B, T, HV, 128]``,
    ``dA_log`` fp32 ``[HV]``, ``dt_bias`` fp32 ``[HV * 128]``.  Repeated calls on identical
    inputs are bit-identical.
    """
    arch = device_arch(v.device)
    batch0, seq0, h, kd = q_norm.shape
    hv = v.shape[2]
    _check(kd == HEAD_DIM and v.shape[-1] == HEAD_DIM, "K = V = 128 is required")
    _check(hv % h == 0, "HV must be a multiple of H")
    if cu_seqlens is not None:
        _check(batch0 == 1, "packed sequences require B == 1")
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        seq_len = int(lengths[0])
        _check(
            all(int(x) == seq_len for x in lengths) and seq_len % PAIR_ROWS == 0,
            "packed sequences must share one length that is a multiple of 128",
        )
        batch = len(lengths)
    else:
        batch, seq_len = batch0, seq0
    _check(seq_len % PAIR_ROWS == 0, "T must be a multiple of 128")
    rows = batch * seq_len
    bf, f32 = torch.bfloat16, torch.float32

    qn = _contiguous("q_norm", _rows(q_norm, rows), bf)
    kn = _contiguous("k_norm", _rows(k_norm, rows), bf)
    qr = _rows(q_rstd, rows).contiguous().float()
    kr = _rows(k_rstd, rows).contiguous().float()
    v_r = _contiguous("v", _rows(v, rows), bf)
    g_r = _contiguous("g", _rows(g, rows), bf)
    beta_raw = _contiguous("beta_logits", _rows(beta_logits, rows), bf)
    beta_s = _contiguous("beta", _rows(beta, rows), f32)
    do_r = _contiguous("do", _rows(do, rows), bf)
    aqk = _contiguous("Aqk", _rows(Aqk, rows), bf)
    akk = _contiguous("Akk", _rows(Akk, rows), bf)
    A_log = A_log.contiguous().float()
    dt_bias = dt_bias.contiguous().float()

    pre = _launch_prep(
        arch, g_r, qn, kn, v_r, beta_s, A_log, dt_bias, aqk, lower_bound=lower_bound
    )
    u, w = _launch_wy(arch, akk, pre["vb"], pre["kb"])
    h_state, v_new = _launch_fwdh(
        arch, w, pre["kg"], u, pre["gk"], batch=batch, seq_len=seq_len
    )
    dAqk, dv1 = _launch_dav(arch, do_r, v_new, pre["aqk_tril"], scale=scale)
    dh_state, dv2 = _launch_dhu(
        arch,
        pre["kg"],
        pre["qg"],
        w,
        do_r,
        dv1,
        pre["gk"],
        batch=batch,
        seq_len=seq_len,
        scale=scale,
    )
    dq_f, dk_f, dg_f, db_f, dAkk, dv = _launch_dqkg(
        arch,
        do_r,
        v_new,
        dv2,
        v_r,
        pre["ke"],
        pre["qe"],
        h_state,
        dh_state,
        akk,
        pre["gk"],
        beta_s,
        batch=batch,
        seq_len=seq_len,
        scale=scale,
    )
    dq_i, dk_i, dg_i, db_i = _launch_intra(
        arch,
        dAqk,
        dAkk,
        pre["gk"],
        pre["ke"],
        pre["qe"],
        beta_s,
        dq_f,
        dk_f,
        dg_f,
        db_f,
    )
    dq, dk, dg, dbeta, dA_log_grad, dt_bias_grad = _launch_epilogue(
        arch,
        dq_i,
        dk_i,
        dg_i,
        db_i,
        q_norm=qn,
        k_norm=kn,
        q_rstd=qr,
        k_rstd=kr,
        g_raw=g_r,
        beta_raw=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
    )
    return {
        "dq": dq.reshape(batch0, seq0, h, kd),
        "dk": dk.reshape(batch0, seq0, h, kd),
        "dv": dv.reshape(batch0, seq0, hv, kd),
        "dbeta": dbeta.reshape(batch0, seq0, hv),
        "dg": dg.reshape(batch0, seq0, hv, kd),
        "dA_log": dA_log_grad,
        "dt_bias": dt_bias_grad,
    }


__all__ = ["chunk_kda_backward", "CHUNK", "HEAD_DIM"]
