"""Runtime coverage check (#3): prove ``flashinfer.msa_ops`` serves the full
MiniMax-M3 call surface, so that an M3 deployment (as vLLM wires it) runs
end-to-end on flashinfer kernels with no unsupported error.

How it works
------------
We run a MiniMax-M3 *text* model whose **attention** config is pinned to full
M3 — 64 query heads, 4 KV heads (GQA group 16), ``head_dim=128``, lightning
indexer ``index_n_heads=4`` / ``index_head_dim=128`` / ``index_block_size=128``
/ ``index_topk_blocks=16``, causal — and only the memory-heavy,
*support-irrelevant* axes (``hidden_size``, layer count, MLP width, vocab) are
shrunk so the model fits in a single consumer GPU. Its sparse-attention layer is
routed into ``flashinfer.msa_ops`` through a registered custom
``attn_implementation``, so the forward genuinely executes through flashinfer.

Why a reduced model still proves *full*-M3 coverage
---------------------------------------------------
Every "unsupported" gate in ``flashinfer/msa_ops`` keys off a per-call **config
tuple** — ``head_dim``, block size, ``topk``, the dtype matrix, ``num_qo_heads %
num_kv_heads`` and the decode ``GQA group <= 16`` bound, the paged-vs-flat
contract, device. None gate on ``total_q`` / ``total_k`` / layer count /
``hidden_size`` / batch / sequence length — the kernels loop over those. So
support is invariant to the shrunk axes and fixed by the config tuple, which we
pin to full M3 and assert against :data:`M3_CANONICAL`.

Note on routing reality: transformers M3 today dispatches attention to the
HF-Hub ``MiniMaxAI/msa`` kernel (SM100-gated, not flashinfer). This test
substitutes a flashinfer-backed interface — demonstrating the port's coverage
and forming the basis for a future flashinfer attention backend.
"""

import math

import pytest
import torch

from flashinfer.utils import is_sm12x_supported

transformers = pytest.importorskip("transformers")
# The MiniMax-M3 model only exists in recent transformers builds; skip cleanly
# (rather than error) on releases that predate it.
pytest.importorskip("transformers.models.minimax_m3_vl.modeling_minimax_m3_vl")


# --- full-M3 canonical attention config (the surface a deployment must serve) ---
M3_CANONICAL = dict(
    head_dim=128,
    num_qo_heads=64,
    num_kv_heads=4,
    topk=16,
    block_size=128,
    causal=True,
)
M3_INDEX_N_HEADS = 4
M3_INDEX_HEAD_DIM = 128
MSA_SUPPORTED_TOPK = (4, 8, 16, 32)

pytestmark = pytest.mark.skipif(
    not is_sm12x_supported(torch.device("cuda")),
    reason="MSA SM12x kernels require SM120 / SM121 (Blackwell) and CUDA >= 12.8",
)


def _build_reduced_m3(dtype: torch.dtype):
    """Build a reduced-depth M3 text model with full-fidelity attention config."""
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig,
    )
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import (
        MiniMaxM3VLTextModel,
    )

    cfg = MiniMaxM3VLTextConfig(
        # pinned to full M3 -> decides MSA kernel support
        num_attention_heads=M3_CANONICAL["num_qo_heads"],
        num_key_value_heads=M3_CANONICAL["num_kv_heads"],
        head_dim=M3_CANONICAL["head_dim"],
        index_n_heads=M3_INDEX_N_HEADS,
        index_head_dim=M3_INDEX_HEAD_DIM,
        index_block_size=M3_CANONICAL["block_size"],
        index_topk_blocks=M3_CANONICAL["topk"],
        index_local_blocks=1,
        rotary_dim=64,
        attention_dropout=0.0,
        # shrunk -> support-irrelevant (kernels loop over these)
        hidden_size=512,
        intermediate_size=256,
        dense_intermediate_size=256,
        shared_intermediate_size=128,
        num_hidden_layers=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=512,
        max_position_embeddings=16384,
        layer_types=["full_attention", "minimax_m3_sparse"],
        mlp_layer_types=["dense", "dense"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5000000.0,
            "partial_rotary_factor": 0.5,
        },
        _attn_implementation="sdpa",
    )
    model = MiniMaxM3VLTextModel(cfg).to("cuda").to(dtype).eval()
    return model


def _register_flashinfer_msa(captured: list):
    """Register a flashinfer-backed attention interface that records each MSA
    call's config tuple and re-dispatches it through ``flashinfer.msa_ops``.

    Mirrors the packing in transformers' MSA integration
    (``integrations/msa_attention.py::_sparse_attention``).
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from flashinfer.msa_ops import (
        msa_sparse_attention_kvmajor,
        msa_sparse_decode_attention,
    )

    def flashinfer_msa_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
        block_indices=None,
        **kwargs,
    ):
        if scaling is None:
            scaling = query.shape[-1] ** -0.5
        # Dense / full-attention layers (no indexer selection) -> plain SDPA.
        if block_indices is None:
            return sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                **kwargs,
            )

        bsz, num_q_heads, q_len, head_dim = query.shape
        num_kv_heads, k_len = key.shape[1], key.shape[2]
        topk = block_indices.shape[-1]
        padded_topk = next(t for t in MSA_SUPPORTED_TOPK if t >= topk)
        if padded_topk != topk:
            pad = block_indices.new_full(
                (*block_indices.shape[:-1], padded_topk - topk), -1
            )
            block_indices = torch.cat([block_indices, pad], dim=-1)
            topk = padded_topk

        q = (
            query.transpose(1, 2)
            .reshape(bsz * q_len, num_q_heads, head_dim)
            .contiguous()
        )
        k = (
            key.transpose(1, 2)
            .reshape(bsz * k_len, num_kv_heads, head_dim)
            .contiguous()
        )
        v = (
            value.transpose(1, 2)
            .reshape(bsz * k_len, num_kv_heads, head_dim)
            .contiguous()
        )
        cu_seqlens_q = torch.arange(
            0, (bsz + 1) * q_len, q_len, device=q.device, dtype=torch.int32
        )
        cu_seqlens_k = torch.arange(
            0, (bsz + 1) * k_len, k_len, device=q.device, dtype=torch.int32
        )
        q2k = block_indices.to(torch.int32).reshape(bsz * q_len, topk)
        q2k = q2k.unsqueeze(0).expand(num_kv_heads, -1, -1).contiguous()

        captured.append(
            dict(
                kind="decode" if q_len == 1 else "prefill",
                head_dim=head_dim,
                num_qo_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                topk=topk,
                block_size=module.indexer.block_size,
                causal=True,
                q_dtype=q.dtype,
            )
        )

        if q_len == 1:
            out = msa_sparse_decode_attention(
                q,
                k,
                v,
                q2k,
                cu_seqlens_k=cu_seqlens_k,
                seqlen_q=1,
                causal=True,
                softmax_scale=scaling,
            )
        else:
            out = msa_sparse_attention_kvmajor(
                q,
                k,
                v,
                q2k,
                cu_seqlens_q,
                cu_seqlens_k,
                causal=True,
                softmax_scale=scaling,
            )
        return out.reshape(bsz, q_len, num_q_heads, head_dim), None

    ALL_ATTENTION_FUNCTIONS.register("flashinfer_msa", flashinfer_msa_forward)


def _assert_in_surface(call: dict):
    """Assert a captured MSA call falls inside flashinfer's supported surface."""
    assert call["head_dim"] == 128, call
    assert call["block_size"] == 128, call
    assert call["topk"] in MSA_SUPPORTED_TOPK, call
    assert call["num_qo_heads"] % call["num_kv_heads"] == 0, call
    group = call["num_qo_heads"] // call["num_kv_heads"]
    if call["kind"] == "decode":
        assert group <= 16, f"decode GQA group must be <= 16, got {group}"
    assert call["q_dtype"] in (torch.bfloat16, torch.float16), call


def _canonical_tuple(call: dict):
    return (
        call["head_dim"],
        call["num_qo_heads"],
        call["num_kv_heads"],
        call["topk"],
        call["block_size"],
        call["causal"],
    )


M3_TUPLE = (
    M3_CANONICAL["head_dim"],
    M3_CANONICAL["num_qo_heads"],
    M3_CANONICAL["num_kv_heads"],
    M3_CANONICAL["topk"],
    M3_CANONICAL["block_size"],
    M3_CANONICAL["causal"],
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_m3_prefill_decode_routes_through_flashinfer(dtype):
    """A real M3 prefill + decode forward runs end-to-end through
    ``flashinfer.msa_ops`` (kvmajor prefill + paged-free decode, which auto-build
    the CSR + schedule + combine), with every MSA call inside the supported
    surface and matching full-M3's canonical config tuple."""
    torch.manual_seed(0)
    captured: list = []
    _register_flashinfer_msa(captured)

    model = _build_reduced_m3(dtype)
    model.config._attn_implementation = "flashinfer_msa"

    seq = 4096  # > topk*block = 2048 -> the indexer actually selects a sparse subset
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq), device="cuda")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    assert out.last_hidden_state.shape == (1, seq, model.config.hidden_size)

    # decode step: 1 new token against the populated KV cache (q_len == 1)
    with torch.no_grad():
        next_ids = torch.randint(0, model.config.vocab_size, (1, 1), device="cuda")
        cache_position = torch.tensor([seq], device="cuda")
        out2 = model(
            input_ids=next_ids,
            past_key_values=out.past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
    assert out2.last_hidden_state.shape == (1, 1, model.config.hidden_size)

    kinds = {c["kind"] for c in captured}
    assert "prefill" in kinds, f"no prefill MSA call captured: {captured}"
    assert "decode" in kinds, f"no decode MSA call captured: {captured}"

    for call in captured:
        _assert_in_surface(call)

    # Bridge to full M3: the only distinct config tuple exercised IS full-M3's,
    # so "no unsupported error here" transfers to the full-size deployment.
    distinct = {_canonical_tuple(c) for c in captured}
    assert distinct == {M3_TUPLE}, f"captured tuples {distinct} != full-M3 {M3_TUPLE}"


def test_m3_proxy_topk_accept_m3_proxy_config():
    """flashinfer's stage-1 ops (``msa_proxy_score`` + ``msa_topk_select``)
    accept M3's lightning-indexer proxy config (``index_n_heads`` query heads,
    1 KV head, ``index_head_dim=128``) with no unsupported error.

    This is an op-*acceptance* (coverage) check, not a semantic-equivalence one:
    M3's indexer max-pools per-key scores over its ``index_n_heads`` proxy heads
    *before* the top-k, whereas ``msa_proxy_score`` emits a per-head ``max_score``
    that ``msa_topk_select`` ranks per head. Mapping the proxy-head reduction onto
    the flashinfer ops (the handoff's open "proxy head config" item) is a separate
    correctness question tracked outside this coverage test."""
    from flashinfer.msa_ops import msa_proxy_score, msa_topk_select

    torch.manual_seed(0)
    dev = "cuda"
    total = 4096
    idx_q = torch.randn(
        total, M3_INDEX_N_HEADS, M3_INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16
    )
    idx_k = torch.randn(total, 1, M3_INDEX_HEAD_DIM, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0, total], device=dev, dtype=torch.int32)

    max_score = msa_proxy_score(idx_q, idx_k, cu, cu, causal=True)
    assert max_score.shape[0] == M3_INDEX_N_HEADS
    assert max_score.shape[2] == total
    assert max_score.dtype == torch.float32

    sel = msa_topk_select(max_score, M3_CANONICAL["topk"])
    assert sel.shape == (total, M3_INDEX_N_HEADS, M3_CANONICAL["topk"])
    assert sel.dtype == torch.int32


@pytest.mark.parametrize("kv_kind", ["bf16", "fp16", "fp8", "nvfp4"])
def test_m3_kv_dtype_matrix_at_m3_shapes(kv_kind):
    """The kvmajor prefill + decode kernels accept every M3 deployment KV dtype
    (bf16 / fp16 / fp8-E4M3 / NVFP4) at M3's exact head config (64 q / 4 kv,
    head_dim 128, topk 16) — the quantized paths a vLLM deployment would use."""
    from flashinfer.msa_ops import (
        msa_sparse_attention_kvmajor,
        msa_sparse_decode_attention,
    )

    torch.manual_seed(0)
    dev = "cuda"
    Hq, Hkv, topk = M3_CANONICAL["num_qo_heads"], M3_CANONICAL["num_kv_heads"], 16
    seq_q, seq_k = 512, 2048
    cu_q = torch.tensor([0, seq_q], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, seq_k], device=dev, dtype=torch.int32)
    nb = seq_k // 128
    scale = 1.0 / math.sqrt(128)

    q_dtype = torch.float16 if kv_kind == "fp16" else torch.bfloat16
    q = torch.randn(seq_q, Hq, 128, device=dev, dtype=q_dtype) / 3
    k = torch.randn(seq_k, Hkv, 128, device=dev, dtype=q_dtype) / 3
    v = torch.randn(seq_k, Hkv, 128, device=dev, dtype=q_dtype) / 3

    # ascending, -1-padded selection in the msa_topk_select output format
    idx = torch.full((Hkv, seq_q, topk), -1, dtype=torch.int32, device=dev)
    for h in range(Hkv):
        for qi in range(seq_q):
            nsel = min(topk, nb)
            sel = torch.randperm(nb)[:nsel].sort().values.to(torch.int32)
            idx[h, qi, :nsel] = sel.to(dev)

    extra = {}
    if kv_kind in ("bf16", "fp16"):
        k_in, v_in = k, v
    elif kv_kind == "fp8":
        k_in, v_in = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    else:  # nvfp4
        from flashinfer import nvfp4_quantize

        def _q(x2d):
            gsf = (448.0 * 6.0) / x2d.float().abs().max()
            xq, sf = nvfp4_quantize(x2d, gsf.to(x2d.device), sf_vec_size=16)
            return xq.view(torch.uint8), sf.view(torch.uint8), float(1.0 / gsf)

        kq, ksf, kg = _q(k.reshape(-1, 128))
        vq, vsf, vg = _q(v.reshape(-1, 128))
        k_in = kq.reshape(seq_k, Hkv, 64)
        v_in = vq.reshape(seq_k, Hkv, 64)
        extra = dict(k_scale=ksf, v_scale=vsf, k_global_scale=kg, v_global_scale=vg)

    # prefill (kvmajor) accepts the M3-shaped quantized inputs
    out = msa_sparse_attention_kvmajor(
        q, k_in, v_in, idx, cu_q, cu_k, causal=True, softmax_scale=scale, **extra
    )
    assert out.shape == (seq_q, Hq, 128)
    assert torch.isfinite(out.float()).all()

    # decode (single query slot) at the same head config + KV dtype
    q1 = torch.randn(1, Hq, 128, device=dev, dtype=q_dtype) / 3
    idx1 = idx[:, :1, :].contiguous()
    cu_k1 = torch.tensor([0, seq_k], device=dev, dtype=torch.int32)
    out1 = msa_sparse_decode_attention(
        q1,
        k_in,
        v_in,
        idx1,
        cu_seqlens_k=cu_k1,
        seqlen_q=1,
        causal=True,
        softmax_scale=scale,
        **extra,
    )
    assert out1.shape == (1, Hq, 128)
    assert torch.isfinite(out1.float()).all()
