"""cuDNN paged prefill via the direct mixed-form sequence-length path.

The paged direct path binds the token-unit ``qo_indptr`` as ``cu_seq_len_q``
(doubling as the Q/O ragged offset) and the per-request KV lengths as
``seq_len_kv`` -- a *mixed* form (cumulative on Q, per-batch on KV) that cuDNN
supports from backend 9.25+. KV is addressed through the page tables, so no
KV-side cumulative prefix sum (and no element-unit offset conversion) is needed.

Each case compares the direct path against the legacy element-conversion paged
path (already covered against the fa2 reference by ``test_cudnn_prefill``), so a
match here validates the new plumbing.
"""

import functools

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill


@functools.lru_cache(maxsize=1)
def _mixed_paged_supported() -> bool:
    """Probe whether this cuDNN/frontend can build a mixed-form paged graph.

    The shipping gate is a version compare, but the FE package version is an
    unreliable proxy on dev builds, so tests probe the real capability instead.
    """
    if not cudnn_prefill.CUDNN_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        import cudnn

        return cudnn.backend_version() >= 92500 and hasattr(
            cudnn, "attention_implementation"
        )
    except Exception:
        return False


def _make_paged_inputs(
    batch_size, s_qo, s_kv, page_size, num_qo_heads, num_kv_heads, head_dim, device
):
    torch.manual_seed(1)
    seq_q = torch.randint(1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device)
    seq_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(seq_q, 0)]).int()  # token-unit

    q = torch.randn(
        int(seq_q.sum()), num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )

    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache = torch.randn(
        total_num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    strides = (
        2 * page_size * num_kv_heads * head_dim,
        head_dim,
        num_kv_heads * head_dim,
        1,
    )
    k_cache = kv_cache[:, 0].as_strided(kv_cache[:, 0].shape, strides)
    v_cache = kv_cache[:, 1].as_strided(kv_cache[:, 1].shape, strides)
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_tables=block_tables,
        qo_indptr=qo_indptr,
        actual_seq_lens_q=seq_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=seq_kv.view(batch_size, 1, 1, 1),
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(64, 512), (17, 200)])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy(
    monkeypatch, batch_size, s_qo, s_kv, page_size, num_kv_heads, causal
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if not _mixed_paged_supported():
        pytest.skip("cuDNN backend/frontend too old for mixed-form paged seqlens")

    device = "cuda:0"
    num_qo_heads, head_dim = 8, 128
    inp = _make_paged_inputs(
        batch_size, s_qo, s_kv, page_size, num_qo_heads, num_kv_heads, head_dim, device
    )
    scale = float(head_dim**-0.5)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    common = dict(
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=inp["actual_seq_lens_q"],
        actual_seq_lens_kv=inp["actual_seq_lens_kv"],
        block_tables=inp["block_tables"],
        causal=causal,
        return_lse=False,
        batch_offsets_q=inp["qo_indptr"],
        batch_offsets_units="tokens",
    )

    def run():
        return cudnn_batch_prefill_with_kv_cache(
            inp["q"], inp["k_cache"], inp["v_cache"], scale, ws, **common
        )[0]

    # Legacy paged path: force the gate off -> element-offset conversion.
    monkeypatch.setattr(
        cudnn_prefill,
        "_cudnn_supports_direct_seqlens",
        lambda dtype, *, mixed=False: False,
    )
    out_legacy = run()

    # Direct paged path: force the gate on -> cu_seq_len_q + seq_len_kv (mixed).
    monkeypatch.setattr(
        cudnn_prefill,
        "_cudnn_supports_direct_seqlens",
        lambda dtype, *, mixed=False: True,
    )
    out_direct = run()

    torch.testing.assert_close(out_direct, out_legacy, atol=1e-2, rtol=1e-2)
