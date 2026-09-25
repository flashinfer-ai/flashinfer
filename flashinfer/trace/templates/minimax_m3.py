# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _minimax_m3_check(
    reference_outputs,
    actual_outputs,
    *,
    rtol=None,
    atol=None,
    max_mismatch_pct=0.0,
    min_cos_sim=None,
):
    import torch
    from flashinfer.trace import default_check

    reference = (
        reference_outputs[0]
        if isinstance(reference_outputs, (tuple, list))
        else reference_outputs
    )
    actual = (
        actual_outputs[0]
        if isinstance(actual_outputs, (tuple, list))
        else actual_outputs
    )
    if not default_check(
        reference,
        actual,
        rtol=0.02 if rtol is None else rtol,
        atol=0.02 if atol is None else atol,
        max_mismatch_pct=max_mismatch_pct,
        min_cos_sim=min_cos_sim,
    ):
        return False
    error_rms = (actual.float() - reference.float()).square().mean(dim=-1).sqrt()
    reference_rms = reference.float().square().mean(dim=-1).sqrt()
    return bool(torch.all(error_rms <= 0.015 * reference_rms + 1e-5))


def _minimax_m3_reference(
    q,
    kv_cache,
    topk_idx,
    block_table,
    seq_lens,
    k_scale,
    v_scale,
    sm_scale=None,
):
    import torch

    total_q, hq, dim = q.shape
    hkv = kv_cache.shape[1]
    qlen = total_q // seq_lens.numel()
    group = hq // hkv
    scale = dim**-0.5 if sm_scale is None else sm_scale
    output = torch.zeros_like(q)
    for t in range(total_q):
        n = max(0, int(seq_lens[t // qlen]) - qlen + t % qlen + 1)
        count = min(16, (n + 127) // 128)
        if count == 0:
            continue
        for h in range(hkv):
            selected = topk_idx[h, t, :count].long()
            physical = block_table[t // qlen, selected].long()
            cache = (
                kv_cache.view(torch.uint8)[physical, h]
                .view(torch.float8_e4m3fn)
                .float()
            )
            valid = (
                selected[:, None] * 128 + torch.arange(128, device=q.device)[None, :]
            ) < n
            cache = cache.reshape(-1, 256)[valid.flatten()]
            k = (cache[:, :128] * k_scale).to(q.dtype).float()
            v = (cache[:, 128:] * v_scale).to(q.dtype).float()
            scores = q[t, h * group : (h + 1) * group].float() @ k.T * scale
            output[t, h * group : (h + 1) * group] = (scores.softmax(-1) @ v).to(
                q.dtype
            )
    return output


def _minimax_m3_init(
    *,
    total_q=64,
    num_qo_heads=16,
    num_kv_heads=1,
    head_dim=128,
    num_pages=512,
    page_size=128,
    packed_dim=256,
    topk=16,
    batch_size=16,
    max_pages=32,
    scale_size=1,
    device="cuda",
    seed=0,
):
    import torch
    from flashinfer.msa_ops import MiniMaxM3SparseDecodeWorkspace

    assert head_dim == page_size == 128 and packed_dim == 256 and topk == 16
    assert scale_size == 1 and total_q % batch_size == 0
    assert num_pages >= max_pages
    assert num_kv_heads in (1, 4) and num_qo_heads == num_kv_heads * 16
    torch.manual_seed(seed)
    qlen = total_q // batch_size
    q = torch.randn(total_q, num_qo_heads, 128, dtype=torch.bfloat16, device=device)
    kv = torch.randn(
        num_pages, num_kv_heads, 128, 256, dtype=torch.bfloat16, device=device
    ).to(torch.float8_e4m3fn)
    table = torch.stack(
        [
            torch.randperm(num_pages, device=device)[:max_pages]
            for _ in range(batch_size)
        ]
    ).to(torch.int32)
    lengths = torch.full(
        (batch_size,), max_pages * 128, dtype=torch.int32, device=device
    )
    indices = torch.full(
        (num_kv_heads, total_q, 16), -1, dtype=torch.int32, device=device
    )
    for h in range(num_kv_heads):
        for t in range(total_q):
            indices[h, t, : min(16, max_pages)] = torch.randperm(
                max_pages, device=device
            )[:16].to(torch.int32)
    return dict(
        q=q,
        kv_cache=kv,
        topk_idx=indices,
        block_table=table,
        seq_lens=lengths,
        k_scale=torch.tensor([0.7], device=device),
        v_scale=torch.tensor([1.3], device=device),
        out=torch.empty_like(q),
        workspace=(
            MiniMaxM3SparseDecodeWorkspace(
                batch_size, num_qo_heads, num_kv_heads, qlen, device=device
            )
            if torch.device(device).type == "cuda"
            else None
        ),
    )


def _minimax_m3_init_k0v0(
    *,
    total_q=64,
    num_pages=512,
    batch_size=16,
    max_pages=32,
    scale_size=1,
    device="cuda",
    seed=0,
    **kwargs,
):
    inputs = _minimax_m3_init(
        total_q=total_q,
        num_pages=num_pages,
        batch_size=batch_size,
        max_pages=max_pages,
        scale_size=scale_size,
        device=device,
        seed=seed,
        **kwargs,
    )
    inputs["k_scale"] = inputs["k_scale"].reshape(())
    inputs["v_scale"] = inputs["v_scale"].reshape(())
    return inputs


def _minimax_m3_init_k0v1(
    *,
    total_q=64,
    num_pages=512,
    batch_size=16,
    max_pages=32,
    scale_size=1,
    device="cuda",
    seed=0,
    **kwargs,
):
    inputs = _minimax_m3_init(
        total_q=total_q,
        num_pages=num_pages,
        batch_size=batch_size,
        max_pages=max_pages,
        scale_size=scale_size,
        device=device,
        seed=seed,
        **kwargs,
    )
    inputs["k_scale"] = inputs["k_scale"].reshape(())
    return inputs


def _minimax_m3_init_k1v0(
    *,
    total_q=64,
    num_pages=512,
    batch_size=16,
    max_pages=32,
    scale_size=1,
    device="cuda",
    seed=0,
    **kwargs,
):
    inputs = _minimax_m3_init(
        total_q=total_q,
        num_pages=num_pages,
        batch_size=batch_size,
        max_pages=max_pages,
        scale_size=scale_size,
        device=device,
        seed=seed,
        **kwargs,
    )
    inputs["v_scale"] = inputs["v_scale"].reshape(())
    return inputs


# Keep rank-specific initializers runnable from the emitted, standalone JSON.
for _init in (_minimax_m3_init_k0v0, _minimax_m3_init_k0v1, _minimax_m3_init_k1v0):
    _init._trace_init_dependencies = (_minimax_m3_init,)  # type: ignore[attr-defined]


def _make_minimax_m3_trace(k_rank, v_rank, init):
    axes = {
        "total_q": Var(),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "page_size": Const(abbrev="p"),
        "packed_dim": Const(abbrev="packed"),
        "topk": Const(abbrev="topk"),
        "batch_size": Var(),
        "max_pages": Var(),
    }
    if k_rank or v_rank:
        axes["scale_size"] = Var()
    suffix = "" if (k_rank, v_rank) == (1, 1) else f"_k{k_rank}v{v_rank}"
    return TraceTemplate(
        op_type="msa_sparse_attention",
        name_prefix="minimax_m3_sparse_attn_decode" + suffix,
        description=(
            "MiniMax-M3 speculative decode with independent per-query top-16 pages, "
            "packed FP8 K/V, scalar K/V dequantization scales and BF16 Q/O. "
            "The workspace fixes the uniform query length and owns reusable metadata. "
            f"K/V scale ranks are {k_rank}/{v_rank}."
        ),
        axes=axes,
        inputs={
            "q": Tensor(["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16"),
            "kv_cache": Tensor(
                ["num_pages", "num_kv_heads", "page_size", "packed_dim"],
                dtype="float8_e4m3fn",
            ),
            "topk_idx": Tensor(["num_kv_heads", "total_q", "topk"], dtype="int32"),
            "block_table": Tensor(["batch_size", "max_pages"], dtype="int32"),
            "seq_lens": Tensor(["batch_size"], dtype="int32"),
            "k_scale": Tensor(["scale_size"] if k_rank else [], dtype="float32"),
            "v_scale": Tensor(["scale_size"] if v_rank else [], dtype="float32"),
            "sm_scale": Scalar("float32", optional=True),
        },
        outputs={
            "out": Tensor(["total_q", "num_qo_heads", "head_dim"], dtype="bfloat16"),
        },
        tags=["stage:decode"],
        check=_minimax_m3_check,
        reference=_minimax_m3_reference,
        init=init,
    )


_MINIMAX_M3_TRACES = {
    (1, 1): _make_minimax_m3_trace(1, 1, _minimax_m3_init),
    (0, 0): _make_minimax_m3_trace(0, 0, _minimax_m3_init_k0v0),
    (0, 1): _make_minimax_m3_trace(0, 1, _minimax_m3_init_k0v1),
    (1, 0): _make_minimax_m3_trace(1, 0, _minimax_m3_init_k1v0),
}
minimax_m3_sparse_attn_decode_trace = _MINIMAX_M3_TRACES[(1, 1)]


def minimax_m3_sparse_attn_decode_trace_dispatch(**kwargs):
    """Trace actual 0D/1D scale layouts without copying or reading device data.

    The decode API also accepts higher-rank singleton views. Their numerical
    behavior is unchanged, but tracing them is unsupported: return no schema
    rather than silently describing a different rank.
    """
    k_rank = getattr(kwargs.get("k_scale"), "ndim", 1)
    v_rank = getattr(kwargs.get("v_scale"), "ndim", 1)
    return _MINIMAX_M3_TRACES.get((k_rank, v_rank))


minimax_m3_sparse_attn_decode_trace_dispatch.templates = tuple(  # type: ignore[attr-defined]
    _MINIMAX_M3_TRACES.values()
)
