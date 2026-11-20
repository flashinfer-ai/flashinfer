"""cuDNN paged prefill with token offsets and explicit or cumulative Q lengths.

Explicit Q and KV lengths support token offsets from cuDNN 9.24. Omitting Q
lengths uses cumulative Q masks with per-request KV lengths, requiring 9.25.

Each case compares the direct path against the legacy element-conversion paged
path (already covered against the fa2 reference by ``test_cudnn_prefill``), so a
match here validates the new plumbing.
"""

import functools

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill


@functools.lru_cache(maxsize=2)
def _paged_token_offsets_supported(explicit_q: bool) -> bool:
    if not torch.cuda.is_available():
        return False
    return cudnn_prefill._cudnn_supports_paged_token_offsets(
        torch.bfloat16, explicit_q=explicit_q, explicit_kv=True
    )


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
@pytest.mark.parametrize(
    "explicit_q", [False, True], ids=["cumulative-q", "explicit-q"]
)
def test_cudnn_paged_prefill_token_offsets_match_legacy(
    monkeypatch, batch_size, s_qo, s_kv, page_size, num_kv_heads, causal, explicit_q
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if not _paged_token_offsets_supported(explicit_q):
        pytest.skip("cuDNN backend/frontend does not support these paged token offsets")

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
        actual_seq_lens_q=inp["actual_seq_lens_q"] if explicit_q else None,
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

    # Explicit lengths need the ordinary gate; cumulative Q needs the mixed gate.
    expected_mixed = not explicit_q
    mixed_calls = []

    def _gate_direct(dtype, *, mixed=False):
        mixed_calls.append(mixed)
        return mixed == expected_mixed

    monkeypatch.setattr(cudnn_prefill, "_cudnn_supports_direct_seqlens", _gate_direct)
    out_direct = run()

    assert expected_mixed in mixed_calls, (
        "paged dispatch did not consult _cudnn_supports_direct_seqlens with "
        f"mixed={expected_mixed}, so the direct token-offset path was not exercised"
    )
    torch.testing.assert_close(out_direct, out_legacy, atol=1e-2, rtol=1e-2)
