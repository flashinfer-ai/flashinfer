# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for the numeric MSA ops; references mirror the torch oracles
in ``tests/msa_ops/``. Metadata ops (topk select, union builders) produce
indices, not numeric tensors, so they are not traced."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

_BLK_KV = 128


def _msa_score_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    from flashinfer.trace import default_check

    # Matches test_msa_proxy_score: abs < 1e-2 on finite entries; -inf masked
    # blocks agree exactly (isclose(-inf, -inf) is True).
    rtol = 1e-2 if rtol is None else rtol
    atol = 1e-2 if atol is None else atol
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


def _msa_attention_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    from flashinfer.trace import default_check

    # bf16 output tolerance, matching test_sparse_attention.
    rtol = 2e-2 if rtol is None else rtol
    atol = 2e-2 if atol is None else atol
    return default_check(
        reference_outputs,
        actual_outputs,
        rtol=rtol,
        atol=atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    )


# ── msa_proxy_score (bf16 indexer) ────────────────────────────────────────────


@torch.no_grad()
def _msa_proxy_score_reference(q, k, cu_seqlens_q, cu_seqlens_k, causal=True):
    """Per-KV-block max of unscaled causal Q K^T logits."""
    BLK_KV = 128
    total_q, Hq, _ = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    cu_q = cu_seqlens_q.to(torch.long)
    cu_k = cu_seqlens_k.to(torch.long)
    seqlens_k = (cu_k[1:] - cu_k[:-1]).tolist()
    mkt = max((s + BLK_KV - 1) // BLK_KV for s in seqlens_k) if seqlens_k else 0
    out = torch.full(
        (Hq, mkt, total_q), float("-inf"), dtype=torch.float32, device=q.device
    )
    for b in range(cu_q.numel() - 1):
        qlo, qhi = int(cu_q[b]), int(cu_q[b + 1])
        klo, khi = int(cu_k[b]), int(cu_k[b + 1])
        sq, sk = qhi - qlo, khi - klo
        nb = (sk + BLK_KV - 1) // BLK_KV
        for h in range(Hq):
            s = q[qlo:qhi, h].float() @ k[klo:khi, h // G].float().T  # unscaled
            if causal:
                qi = torch.arange(sq, device=q.device).unsqueeze(1) + (sk - sq)
                ki = torch.arange(sk, device=q.device).unsqueeze(0)
                s = s.masked_fill(ki > qi, float("-inf"))
            for t in range(nb):
                out[h, t, qlo:qhi] = s[:, t * BLK_KV : (t + 1) * BLK_KV].amax(dim=1)
    return out


def _msa_varlen_cu_seqlens(total, batch_size, device):
    """Split ``total`` tokens across ``batch_size`` sequences; int32 cu_seqlens."""
    base = total // max(1, batch_size)
    rem = total % max(1, batch_size)
    cum = [0]
    for i in range(batch_size):
        cum.append(cum[-1] + base + (1 if i < rem else 0))
    return torch.tensor(cum, dtype=torch.int32, device=device)


def _msa_proxy_score_init(
    *,
    total_q: int,
    total_k: int = 4096,
    len_indptr: int = 0,  # derived: batch_size + 1
    max_k_tiles: int = 0,  # derived from total_k
    num_qo_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    batch_size: int = 2,
    causal: bool = True,
    device: str = "cuda",
    seed: int = 0,
):
    """Build msa_proxy_score inputs; defaults follow MiniMax-M3's indexer config."""
    del len_indptr, max_k_tiles  # derived from cu_seqlens / total_k
    torch.manual_seed(seed)
    cu_q = _msa_varlen_cu_seqlens(total_q, batch_size, device)
    cu_k = _msa_varlen_cu_seqlens(total_k, batch_size, device)
    q = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    return {
        "q": q,
        "k": k,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "causal": causal,
    }


msa_proxy_score_trace = TraceTemplate(
    op_type="msa_proxy",
    name_prefix="msa_proxy_score",
    description=(
        "MSA dense proxy / indexer (bf16): per-KV-block max of the unscaled, "
        "causally-masked Q K^T logits, feeding msa_topk_select. No softmax, no V."
    ),
    axes={
        "total_q": Var(description="Total query tokens across the batch."),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "total_k": Var(description="Total key tokens across the batch."),
        "len_indptr": Var(description="Length of cu_seqlens arrays (batch_size + 1)."),
        "max_k_tiles": Var(description="Number of KV-block columns in the output."),
    },
    inputs={
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k": Tensor(["total_k", "num_kv_heads", "head_dim"]),
        "cu_seqlens_q": Tensor(["len_indptr"], dtype="int32"),
        "cu_seqlens_k": Tensor(["len_indptr"], dtype="int32"),
        "causal": Scalar("int32", optional=True, description="Bool: causal masking."),
    },
    outputs={
        "max_score": Tensor(
            ["num_qo_heads", "max_k_tiles", "total_q"],
            dtype="float32",
            description="Per-(head, KV-block, query) max logit; -inf where masked.",
        ),
    },
    constraints=[
        "cu_seqlens_q.shape[0] == len_indptr",
        "total_q == cu_seqlens_q[-1].item()",
        "total_k == cu_seqlens_k[-1].item()",
    ],
    tags=["status:verified", "stage:indexer"],
    reference=_msa_proxy_score_reference,
    check=_msa_score_check,
    init=_msa_proxy_score_init,
)

# ── msa_proxy_score_fp4 (NVFP4 indexer) ───────────────────────────────────────


@torch.no_grad()
def _msa_proxy_score_fp4_reference(
    q_fp4,
    k_fp4,
    q_scale,
    k_scale,
    q_global_scale,
    k_global_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    causal=True,
):
    """Block-max proxy over a torch dequant of packed NVFP4 Q/K."""
    BLK_KV = 128
    HEAD_DIM = 128
    e2m1 = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=q_fp4.device,
    )

    def dequant(xq, sf_flat, mul, rows, d=HEAD_DIM):
        lo = (xq & 0xF).long()
        hi = (xq >> 4).long()
        vals = torch.empty(rows, d, dtype=torch.float32, device=xq.device)
        vals[:, 0::2] = e2m1[lo]
        vals[:, 1::2] = e2m1[hi]
        cols = d // 16
        r = torch.arange(rows, device=xq.device)[:, None]
        c = torch.arange(cols, device=xq.device)[None, :]
        off = (
            ((r // 128) * (-(-cols // 4)) + c // 4) * 512
            + (r % 128 % 32) * 16
            + (r % 128 // 32) * 4
            + c % 4
        )
        sc = (
            sf_flat.view(torch.float8_e4m3fn)[off.reshape(-1)]
            .reshape(rows, cols)
            .float()
        )
        return vals * sc.repeat_interleave(16, dim=1) * mul

    total_q, Hq, _ = q_fp4.shape
    Hkv = k_fp4.shape[1]
    G = Hq // Hkv
    inv_q = float(q_global_scale)
    inv_k = float(k_global_scale)
    # Fold both global scales into Q (matches the kernel's logit scaling).
    q_deq = dequant(
        q_fp4.reshape(-1, HEAD_DIM // 2), q_scale, inv_q * inv_k, total_q * Hq
    ).reshape(total_q, Hq, HEAD_DIM)
    total_k = k_fp4.shape[0]
    k_deq = dequant(
        k_fp4.reshape(-1, HEAD_DIM // 2), k_scale, 1.0, total_k * Hkv
    ).reshape(total_k, Hkv, HEAD_DIM)

    cu_q = cu_seqlens_q.to(torch.long)
    cu_k = cu_seqlens_k.to(torch.long)
    seqlens_k = (cu_k[1:] - cu_k[:-1]).tolist()
    mkt = max((s + BLK_KV - 1) // BLK_KV for s in seqlens_k) if seqlens_k else 0
    out = torch.full(
        (Hq, mkt, total_q), float("-inf"), dtype=torch.float32, device=q_fp4.device
    )
    for b in range(cu_q.numel() - 1):
        qlo, qhi = int(cu_q[b]), int(cu_q[b + 1])
        klo, khi = int(cu_k[b]), int(cu_k[b + 1])
        sq, sk = qhi - qlo, khi - klo
        nb = (sk + BLK_KV - 1) // BLK_KV
        for h in range(Hq):
            s = q_deq[qlo:qhi, h] @ k_deq[klo:khi, h // G].T
            if causal:
                qi = torch.arange(sq, device=q_fp4.device).unsqueeze(1) + (sk - sq)
                ki = torch.arange(sk, device=q_fp4.device).unsqueeze(0)
                s = s.masked_fill(ki > qi, float("-inf"))
            for t in range(nb):
                out[h, t, qlo:qhi] = s[:, t * BLK_KV : (t + 1) * BLK_KV].amax(dim=1)
    return out


def _msa_proxy_score_fp4_init(
    *,
    total_q: int,
    total_k: int = 4096,
    len_indptr: int = 0,
    max_k_tiles: int = 0,
    q_sf_numel: int = 0,
    k_sf_numel: int = 0,
    num_qo_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim_half: int = 64,
    batch_size: int = 2,
    causal: bool = True,
    device: str = "cuda",
    seed: int = 0,
):
    """Build msa_proxy_score_fp4 inputs: random bf16 Q/K quantized to NVFP4."""
    del len_indptr, max_k_tiles, q_sf_numel, k_sf_numel, head_dim_half
    from flashinfer.msa_ops.proxy_score import _quantize_qk_to_nvfp4

    torch.manual_seed(seed)
    cu_q = _msa_varlen_cu_seqlens(total_q, batch_size, device)
    cu_k = _msa_varlen_cu_seqlens(total_k, batch_size, device)
    q = torch.randn(total_q, num_qo_heads, 128, dtype=torch.bfloat16, device=device) * 2
    k = torch.randn(total_k, num_kv_heads, 128, dtype=torch.bfloat16, device=device) * 2
    q_fp4, q_scale, inv_q = _quantize_qk_to_nvfp4(q)
    k_fp4, k_scale, inv_k = _quantize_qk_to_nvfp4(k)
    return {
        "q_fp4": q_fp4,
        "k_fp4": k_fp4,
        "q_scale": q_scale,
        "k_scale": k_scale,
        "q_global_scale": inv_q,
        "k_global_scale": inv_k,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "causal": causal,
    }


msa_proxy_score_fp4_trace = TraceTemplate(
    op_type="msa_proxy",
    name_prefix="msa_proxy_score_fp4",
    description=(
        "NVFP4 MSA dense proxy / indexer: same per-KV-block max as "
        "msa_proxy_score, but Q/K arrive pre-quantized as packed NVFP4 (e2m1 + "
        "per-16 e4m3 block scales + per-tensor global scales), so the index K is "
        "read from HBM at ~4 bits/elem."
    ),
    axes={
        "total_q": Var(description="Total query tokens across the batch."),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim_half": Const(abbrev="", description="head_dim/2 (packed e2m1)."),
        "total_k": Var(description="Total key tokens across the batch."),
        "len_indptr": Var(description="Length of cu_seqlens arrays (batch_size + 1)."),
        "max_k_tiles": Var(description="Number of KV-block columns in the output."),
        "q_sf_numel": Var(description="Flat length of the Q block-scale buffer."),
        "k_sf_numel": Var(description="Flat length of the K block-scale buffer."),
    },
    inputs={
        "q_fp4": Tensor(
            ["total_q", "num_qo_heads", "head_dim_half"],
            dtype="uint8",
            description="Packed e2m1 Q (2 nibbles/byte).",
        ),
        "k_fp4": Tensor(
            ["total_k", "num_kv_heads", "head_dim_half"],
            dtype="uint8",
            description="Packed e2m1 K (2 nibbles/byte).",
        ),
        "q_scale": Tensor(
            ["q_sf_numel"],
            dtype="uint8",
            description="Flat e4m3 Q block scales, cuBLAS 128x4 tiled layout.",
        ),
        "k_scale": Tensor(
            ["k_sf_numel"],
            dtype="uint8",
            description="Flat e4m3 K block scales, cuBLAS 128x4 tiled layout.",
        ),
        "q_global_scale": Scalar(
            "float32", description="Per-tensor inverse global scale for Q."
        ),
        "k_global_scale": Scalar(
            "float32", description="Per-tensor inverse global scale for K."
        ),
        "cu_seqlens_q": Tensor(["len_indptr"], dtype="int32"),
        "cu_seqlens_k": Tensor(["len_indptr"], dtype="int32"),
        "causal": Scalar("int32", optional=True, description="Bool: causal masking."),
    },
    outputs={
        "max_score": Tensor(
            ["num_qo_heads", "max_k_tiles", "total_q"],
            dtype="float32",
            description="Per-(head, KV-block, query) max logit; -inf where masked.",
        ),
    },
    constraints=[
        "total_q == cu_seqlens_q[-1].item()",
        "total_k == cu_seqlens_k[-1].item()",
    ],
    tags=["status:verified", "stage:indexer", "quantize:fp4"],
    reference=_msa_proxy_score_fp4_reference,
    check=_msa_score_check,
    init=_msa_proxy_score_fp4_init,
)

# ── msa_sparse_attention (prefill) ─────────────────────────────────────────────


@torch.no_grad()
def _msa_sparse_attention_reference(
    q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k, causal=False, softmax_scale=None
):
    """Top-K block sparse attention reference."""
    BLK_KV = 128
    total_q, Hq, head_dim = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    scale = (head_dim**-0.5) if softmax_scale is None else float(softmax_scale)
    cu_q = cu_seqlens_q.to(torch.long)
    cu_k = cu_seqlens_k.to(torch.long)
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(cu_q.numel() - 1):
        q_lo, q_hi = int(cu_q[b]), int(cu_q[b + 1])
        k_lo, k_hi = int(cu_k[b]), int(cu_k[b + 1])
        seqlen_k, seqlen_q = k_hi - k_lo, q_hi - q_lo
        nb = (seqlen_k + BLK_KV - 1) // BLK_KV
        for qi in range(q_lo, q_hi):
            q_pos = qi - q_lo
            for hq in range(Hq):
                hkv = hq // G
                sel = q2k_indices[hkv, qi]
                sel = sel[(sel >= 0) & (sel < nb)].unique()
                cols = []
                for blk in sel.tolist():
                    lo = blk * BLK_KV
                    hi = min(lo + BLK_KV, seqlen_k)
                    cols.extend(range(lo, hi))
                if causal:
                    limit = q_pos + seqlen_k - seqlen_q
                    cols = [c for c in cols if c <= limit]
                if not cols:
                    continue
                col_t = torch.tensor(cols, device=q.device)
                kk = k[k_lo + col_t, hkv].float()
                vv = v[k_lo + col_t, hkv].float()
                p = torch.softmax((q[qi, hq].float() @ kk.T) * scale, dim=-1)
                out[qi, hq] = p @ vv
    return out.to(q.dtype)


def _msa_sparse_attention_init(
    *,
    total_q: int,
    total_k: int = 4096,
    len_indptr: int = 0,
    topk: int = 16,
    num_qo_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    batch_size: int = 2,
    causal: bool = False,
    device: str = "cuda",
    seed: int = 0,
):
    """Random varlen Q/K/V plus ascending in-range q2k_indices (msa_topk_select
    format)."""
    del len_indptr
    BLK_KV = 128
    torch.manual_seed(seed)
    cu_q = _msa_varlen_cu_seqlens(total_q, batch_size, device)
    cu_k = _msa_varlen_cu_seqlens(total_k, batch_size, device)
    q = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=device
    )
    cu_q_l, cu_k_l = cu_q.tolist(), cu_k.tolist()
    for b in range(batch_size):
        nb = (cu_k_l[b + 1] - cu_k_l[b] + BLK_KV - 1) // BLK_KV
        keep = min(topk, nb)
        if keep <= 0:
            continue
        blocks = torch.arange(keep, dtype=torch.int32, device=device)
        idx[:, cu_q_l[b] : cu_q_l[b + 1], :keep] = blocks
    return {
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": idx,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "causal": causal,
    }


def _msa_sparse_attention_inputs():
    return {
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k": Tensor(["total_k", "num_kv_heads", "head_dim"]),
        "v": Tensor(["total_k", "num_kv_heads", "head_dim"]),
        "q2k_indices": Tensor(
            ["num_kv_heads", "total_q", "topk"],
            dtype="int32",
            description="Per-(kv-head, query) selected KV-block ids, -1 padded.",
        ),
        "cu_seqlens_q": Tensor(["len_indptr"], dtype="int32"),
        "cu_seqlens_k": Tensor(["len_indptr"], dtype="int32"),
        "causal": Scalar("int32", optional=True, description="Bool: causal masking."),
        "softmax_scale": Scalar("float32", optional=True),
    }


def _msa_sparse_attention_axes():
    return {
        "total_q": Var(description="Total query tokens across the batch."),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "total_k": Var(description="Total key/value tokens across the batch."),
        "topk": Const(abbrev="topk"),
        "len_indptr": Var(description="Length of cu_seqlens arrays (batch_size + 1)."),
    }


msa_sparse_attention_trace = TraceTemplate(
    op_type="msa_sparse",
    name_prefix="msa_sparse_attention",
    description=(
        "MSA sparse attention (prefill): each query attends only its top-K "
        "selected KV blocks (128 tokens each) from q2k_indices. Query tiles "
        "process the union of their queries' selected blocks, so a block loads "
        "once per tile that references it."
    ),
    axes=_msa_sparse_attention_axes(),
    inputs=_msa_sparse_attention_inputs(),
    outputs={
        "output": Tensor(["total_q", "num_qo_heads", "head_dim"], dtype_from="q"),
    },
    constraints=[
        "q2k_indices.shape[0] == num_kv_heads",
        "q2k_indices.shape[1] == total_q",
        "q2k_indices.shape[-1] == topk",
        "total_q == cu_seqlens_q[-1].item()",
    ],
    tags=["status:verified", "stage:prefill", "sparse:topk"],
    reference=_msa_sparse_attention_reference,
    check=_msa_attention_check,
    init=_msa_sparse_attention_init,
)

# ── msa_sparse_decode_attention (decode) ──────────────────────────────────────


@torch.no_grad()
def _msa_sparse_decode_attention_reference(
    q, k, v, q2k_indices, cu_seqlens_k, seqlen_q=1, causal=True, softmax_scale=None
):
    """Decode reference: prefill math with right-aligned uniform seqlen_q tokens."""
    BLK_KV = 128
    total_q, Hq, head_dim = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    scale = (head_dim**-0.5) if softmax_scale is None else float(softmax_scale)
    batch_size = total_q // int(seqlen_q)
    cu_q = torch.arange(batch_size + 1, device=q.device, dtype=torch.long) * int(
        seqlen_q
    )
    cu_k = cu_seqlens_k.to(torch.long)
    out = torch.zeros_like(q, dtype=torch.float32)
    for b in range(batch_size):
        q_lo, q_hi = int(cu_q[b]), int(cu_q[b + 1])
        k_lo, k_hi = int(cu_k[b]), int(cu_k[b + 1])
        seqlen_k = k_hi - k_lo
        sq = q_hi - q_lo
        nb = (seqlen_k + BLK_KV - 1) // BLK_KV
        for qi in range(q_lo, q_hi):
            q_pos = qi - q_lo
            for hq in range(Hq):
                hkv = hq // G
                sel = q2k_indices[hkv, qi]
                sel = sel[(sel >= 0) & (sel < nb)].unique()
                cols = []
                for blk in sel.tolist():
                    lo = blk * BLK_KV
                    hi = min(lo + BLK_KV, seqlen_k)
                    cols.extend(range(lo, hi))
                if causal:
                    limit = q_pos + seqlen_k - sq
                    cols = [c for c in cols if c <= limit]
                if not cols:
                    continue
                col_t = torch.tensor(cols, device=q.device)
                kk = k[k_lo + col_t, hkv].float()
                vv = v[k_lo + col_t, hkv].float()
                p = torch.softmax((q[qi, hq].float() @ kk.T) * scale, dim=-1)
                out[qi, hq] = p @ vv
    return out.to(q.dtype)


def _msa_sparse_decode_attention_init(
    *,
    total_q: int,
    total_k: int = 4096,
    len_indptr: int = 0,
    topk: int = 16,
    num_qo_heads: int = 4,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    seqlen_q: int = 1,
    causal: bool = True,
    device: str = "cuda",
    seed: int = 0,
):
    """Decode inputs (flat KV): ``batch_size = total_q // seqlen_q`` requests."""
    del len_indptr
    BLK_KV = 128
    torch.manual_seed(seed)
    batch_size = max(1, total_q // int(seqlen_q))
    total_q = batch_size * int(seqlen_q)
    cu_k = _msa_varlen_cu_seqlens(total_k, batch_size, device)
    q = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn(
        total_k, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    idx = torch.full(
        (num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=device
    )
    cu_k_l = cu_k.tolist()
    for b in range(batch_size):
        nb = (cu_k_l[b + 1] - cu_k_l[b] + BLK_KV - 1) // BLK_KV
        keep = min(topk, nb)
        if keep <= 0:
            continue
        blocks = torch.arange(keep, dtype=torch.int32, device=device)
        idx[:, b * int(seqlen_q) : (b + 1) * int(seqlen_q), :keep] = blocks
    return {
        "q": q,
        "k": k,
        "v": v,
        "q2k_indices": idx,
        "cu_seqlens_k": cu_k,
        "seqlen_q": int(seqlen_q),
        "causal": causal,
    }


msa_sparse_decode_attention_trace = TraceTemplate(
    op_type="msa_sparse",
    name_prefix="msa_sparse_decode_attention",
    description=(
        "MSA sparse decode attention: each request contributes seqlen_q "
        "right-aligned query tokens (1, or >1 for speculative decoding) that "
        "attend only the top-K KV blocks in q2k_indices. Flat varlen KV."
    ),
    axes={
        "total_q": Var(description="batch_size * seqlen_q query tokens."),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "total_k": Var(description="Total key/value tokens across the batch."),
        "topk": Const(abbrev="topk"),
        "len_indptr": Var(description="Length of cu_seqlens_k (batch_size + 1)."),
    },
    inputs={
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k": Tensor(["total_k", "num_kv_heads", "head_dim"]),
        "v": Tensor(["total_k", "num_kv_heads", "head_dim"]),
        "q2k_indices": Tensor(
            ["num_kv_heads", "total_q", "topk"],
            dtype="int32",
            description="Per-(kv-head, query) selected KV-block ids, -1 padded.",
        ),
        "cu_seqlens_k": Tensor(["len_indptr"], dtype="int32"),
        "seqlen_q": Scalar(
            "int32", description="Uniform query length per request (1 for decode)."
        ),
        "causal": Scalar("int32", optional=True, description="Bool: causal masking."),
        "softmax_scale": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(["total_q", "num_qo_heads", "head_dim"], dtype_from="q"),
    },
    constraints=[
        "q2k_indices.shape[0] == num_kv_heads",
        "q2k_indices.shape[1] == total_q",
        "q2k_indices.shape[-1] == topk",
        "total_k == cu_seqlens_k[-1].item()",
    ],
    tags=["status:verified", "stage:decode", "sparse:topk"],
    reference=_msa_sparse_decode_attention_reference,
    check=_msa_attention_check,
    init=_msa_sparse_decode_attention_init,
)


def _bind_init_dependency(init_fn):
    # Inline _msa_varlen_cu_seqlens into each dumped JSON's "init" field so it
    # stays self-contained; unlike moe.py, __signature__ is left alone to keep
    # the Var-axis init params.
    init_fn._trace_init_dependencies = (_msa_varlen_cu_seqlens,)
    return init_fn


_msa_proxy_score_init = _bind_init_dependency(_msa_proxy_score_init)
_msa_proxy_score_fp4_init = _bind_init_dependency(_msa_proxy_score_fp4_init)
_msa_sparse_attention_init = _bind_init_dependency(_msa_sparse_attention_init)
_msa_sparse_decode_attention_init = _bind_init_dependency(
    _msa_sparse_decode_attention_init
)
