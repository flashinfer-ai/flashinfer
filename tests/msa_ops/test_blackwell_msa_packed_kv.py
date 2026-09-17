"""Packed-HND KV coverage for the SM100/SM103 uniform-FP8 MSA route."""

import pytest
import torch

from flashinfer.msa_ops._blackwell_sm100 import _is_packed_hnd_kv
from flashinfer.utils import get_compute_capability


_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK = 16


def _packed_cache(
    *, num_pages: int, num_kv_heads: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = torch.randn(
        num_pages,
        num_kv_heads,
        _BLOCK_SIZE,
        2 * _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    key, value = packed.split(_HEAD_DIM, dim=-1)
    return packed, key, value


def _require_sm10x() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if get_compute_capability(device) not in {(10, 0), (10, 3)}:
        pytest.skip("requires SM100 or SM103")
    return device


def test_packed_hnd_geometry_detection() -> None:
    packed, key, value = _packed_cache(num_pages=2, num_kv_heads=2, device="cpu")
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    assert key.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert _is_packed_hnd_kv(key, value)

    wrong_offset = packed[..., 127:255]
    assert not _is_packed_hnd_kv(key, wrong_offset)

    nhd = torch.empty(2, _BLOCK_SIZE, 2, 2 * _HEAD_DIM).permute(0, 2, 1, 3)
    nhd_key, nhd_value = nhd.split(_HEAD_DIM, dim=-1)
    assert not _is_packed_hnd_kv(nhd_key, nhd_value)

    other = torch.empty_like(packed)
    other_value = other[..., _HEAD_DIM:]
    assert not _is_packed_hnd_kv(key, other_value)


@pytest.mark.parametrize(("seqlen_q", "num_kv_heads"), [(1, 1), (4, 4)])
def test_uniform_fp8_packed_hnd_decode_matches_contiguous(
    seqlen_q: int, num_kv_heads: int
) -> None:
    device = _require_sm10x()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    torch.manual_seed(0)
    batch_size = 4
    pages_per_request = 20
    num_q_heads = 16
    num_pages = batch_size * pages_per_request
    packed, key, value = _packed_cache(
        num_pages=num_pages,
        num_kv_heads=num_kv_heads,
        device=device,
    )
    query = torch.randn(
        batch_size * seqlen_q,
        num_q_heads,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (batch_size,),
        pages_per_request * _BLOCK_SIZE,
        dtype=torch.int32,
        device=device,
    )
    selected = torch.arange(_TOPK, dtype=torch.int32, device=device)
    q2k_token_major = (
        selected.view(1, 1, _TOPK)
        .expand(batch_size * seqlen_q, num_kv_heads, _TOPK)
        .contiguous()
    )
    q2k_indices = q2k_token_major.transpose(0, 1)
    q2k_contiguous = q2k_indices.contiguous()
    if num_kv_heads > 1:
        assert not q2k_indices.is_contiguous()
        assert q2k_indices.untyped_storage().data_ptr() == (
            q2k_token_major.untyped_storage().data_ptr()
        )
    kwargs = {
        "page_table": page_table,
        "seqused_k": seq_lens,
        "seqlen_q": seqlen_q,
        "causal": True,
        "softmax_scale": 0.7 * _HEAD_DIM**-0.5,
        "return_softmax_lse": True,
        "k_global_scale": 1.25,
        "v_global_scale": 0.75,
        "force_fused": True,
    }

    expected_out, expected_lse = msa_sparse_decode_attention(
        query,
        key.contiguous(),
        value.contiguous(),
        q2k_contiguous,
        **kwargs,
    )
    provided_out = torch.empty_like(expected_out)
    actual_out, actual_lse = msa_sparse_decode_attention(
        query,
        key,
        value,
        q2k_indices,
        out=provided_out,
        **kwargs,
    )

    assert key.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert value.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert actual_out.data_ptr() == provided_out.data_ptr()
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)


def test_uniform_fp8_packed_hnd_decode_cuda_graph() -> None:
    device = _require_sm10x()
    from flashinfer.msa_ops import msa_sparse_decode_attention

    torch.manual_seed(1)
    batch_size = 4
    seqlen_q = 4
    pages_per_request = 16
    num_pages = batch_size * pages_per_request
    num_kv_heads = 4
    _, key, value = _packed_cache(
        num_pages=num_pages, num_kv_heads=num_kv_heads, device=device
    )
    query = torch.randn(
        batch_size * seqlen_q,
        16,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    ).to(torch.float8_e4m3fn)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        batch_size, pages_per_request
    )
    seq_lens = torch.full(
        (batch_size,),
        pages_per_request * _BLOCK_SIZE,
        dtype=torch.int32,
        device=device,
    )
    q2k_indices = (
        torch.arange(_TOPK, dtype=torch.int32, device=device)
        .view(1, 1, _TOPK)
        .expand(batch_size * seqlen_q, num_kv_heads, _TOPK)
        .contiguous()
        .transpose(0, 1)
    )
    assert not q2k_indices.is_contiguous()
    provided_out = torch.empty(
        batch_size * seqlen_q,
        16,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    provided_lse = torch.empty(
        batch_size * seqlen_q,
        16,
        dtype=torch.float32,
        device=device,
    )

    def run():
        return msa_sparse_decode_attention(
            query,
            key,
            value,
            q2k_indices,
            page_table=page_table,
            seqused_k=seq_lens,
            seqlen_q=seqlen_q,
            causal=True,
            return_softmax_lse=True,
            force_fused=True,
            out=provided_out,
            lse_out=provided_lse,
        )

    capture_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(capture_stream):
        expected_out, expected_lse = run()
    capture_stream.synchronize()
    expected_out = expected_out.clone()
    expected_lse = expected_lse.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_out, graph_lse = run()
    assert graph_out.data_ptr() == provided_out.data_ptr()
    assert graph_lse.data_ptr() == provided_lse.data_ptr()
    graph.replay()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(graph_out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(graph_lse, expected_lse, rtol=0, atol=0)
