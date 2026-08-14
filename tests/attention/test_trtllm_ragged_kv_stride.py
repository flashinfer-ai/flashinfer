import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


def _require_trtllm_ragged(device: torch.device) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not hasattr(flashinfer.prefill, "trtllm_ragged_attention_deepseek"):
        pytest.skip("trtllm_ragged_attention_deepseek is not available")

    compute_capability = get_compute_capability(device)
    if compute_capability[0] != 10:
        pytest.skip(
            "TRTLLM-gen ragged attention requires SM100 or SM103 GPUs, "
            f"got sm{compute_capability[0]}{compute_capability[1]}"
        )


def _indptr(seq_lens: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(1, device=seq_lens.device, dtype=torch.int32),
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
        ],
        dim=0,
    )


def _workspace(device: torch.device) -> torch.Tensor:
    return torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)


def _run_trtllm_ragged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    max_q_len: int | None = None,
    max_kv_len: int | None = None,
    return_lse: bool = True,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    q_lens_cpu: torch.Tensor | None = None,
    kv_lens_cpu: torch.Tensor | None = None,
):
    head_dim_qk = q.shape[2]
    return flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=_workspace(q.device),
        seq_lens=kv_lens,
        max_q_len=(
            max_q_len
            if max_q_len is not None
            else int(torch.diff(q_indptr).max().item())
        ),
        max_kv_len=(
            max_kv_len if max_kv_len is not None else int(torch.diff(kv_indptr).max())
        ),
        bmm1_scale=float(1.0 / (head_dim_qk**0.5)),
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=kv_lens.shape[0],
        window_left=-1,
        cum_seq_lens_q=q_indptr,
        cum_seq_lens_kv=kv_indptr,
        enable_pdl=False,
        is_causal=False,
        return_lse=return_lse,
        out=out,
        lse=lse,
        q_seq_lens_cpu=q_lens_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def _pack_rows(
    tensor: torch.Tensor,
    indptr: torch.Tensor,
    row_indices: list[int],
) -> torch.Tensor:
    return torch.cat(
        [
            tensor[int(indptr[row].item()) : int(indptr[row + 1].item())]
            for row in row_indices
        ],
        dim=0,
    )


@pytest.mark.cuda
def test_trtllm_ragged_kv_large_stride_overflow():
    """Ragged KV with numel > int32 max should not poison the TMA batch stride."""
    device = torch.device("cuda")
    _require_trtllm_ragged(device)
    torch.manual_seed(42)

    batch_size = 16
    max_kv_len = 8192
    num_heads = 128
    head_dim_qk = 192
    head_dim_vo = 128

    q_lens = torch.randint(
        low=50, high=150, size=(batch_size,), device=device, dtype=torch.int32
    )
    q_indptr = _indptr(q_lens)
    kv_lens = torch.full((batch_size,), max_kv_len, device=device, dtype=torch.int32)
    kv_indptr = _indptr(kv_lens)

    total_q = int(q_indptr[-1].item())
    total_kv = int(kv_indptr[-1].item())
    assert total_kv * num_heads * head_dim_qk > 2**31

    q = torch.randn(
        total_q, num_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        total_kv, num_heads, head_dim_qk, device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        total_kv, num_heads, head_dim_vo, device=device, dtype=torch.bfloat16
    )

    output = _run_trtllm_ragged(
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        kv_lens,
        max_q_len=int(q_lens.max().item()),
        max_kv_len=max_kv_len,
        return_lse=False,
    )

    assert output.shape == (total_q, num_heads, head_dim_vo)


@pytest.mark.cuda
def test_trtllm_ragged_kv_positive_stride_wrap_matches_cutlass():
    """A positive int32 wraparound must not become a fake ragged batch stride."""
    device = torch.device("cuda")
    _require_trtllm_ragged(device)

    free_mem, _ = torch.cuda.mem_get_info(device)
    required_mem = 24 * 1024**3
    if free_mem < required_mem:
        pytest.skip(
            f"requires at least {required_mem / 1024**3:.0f} GiB free GPU memory"
        )

    torch.manual_seed(42)
    batch_size = 8
    max_kv_len = 24576
    num_heads = 128
    head_dim_qk = 192
    head_dim_vo = 128

    q_lens = torch.ones(batch_size, device=device, dtype=torch.int32)
    q_indptr = _indptr(q_lens)
    kv_lens = torch.full((batch_size,), max_kv_len, device=device, dtype=torch.int32)
    kv_indptr = _indptr(kv_lens)

    total_kv = int(kv_indptr[-1].item())
    key_numel = total_kv * num_heads * head_dim_qk
    assert key_numel > 2**32
    assert key_numel % 2**32 < 2**31

    q = (torch.randn(batch_size, num_heads, head_dim_qk, device=device) * 0.1).to(
        torch.bfloat16
    )
    k = (torch.randn(total_kv, num_heads, head_dim_qk, device=device) * 0.1).to(
        torch.bfloat16
    )
    v = (torch.randn(total_kv, num_heads, head_dim_vo, device=device) * 0.1).to(
        torch.bfloat16
    )

    ref_workspace = torch.empty_like(_workspace(device))
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        ref_workspace,
        kv_layout="NHD",
        backend="cutlass",
    )
    scale = float(1.0 / (head_dim_qk**0.5))
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=False,
        sm_scale=scale,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    output_ref, _ = wrapper.run(q, k, v, return_lse=True)

    output, lse = _run_trtllm_ragged(
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        kv_lens,
        max_q_len=1,
        max_kv_len=max_kv_len,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
    torch.testing.assert_close(output, output_ref, atol=2e-2, rtol=2e-2)


def _empty_kv_case(device: torch.device):
    num_heads = 16
    head_dim_qk = 128
    head_dim_vo = 128
    q_lens = torch.tensor([4, 5, 3, 2, 1], device=device, dtype=torch.int32)
    kv_lens = torch.tensor([0, 7, 0, 2, 0], device=device, dtype=torch.int32)
    q_indptr = _indptr(q_lens)
    kv_indptr = _indptr(kv_lens)
    q = torch.randn(
        int(q_indptr[-1].item()),
        num_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        int(kv_indptr[-1].item()),
        num_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        int(kv_indptr[-1].item()),
        num_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )
    return q, k, v, q_lens, kv_lens, q_indptr, kv_indptr


@pytest.mark.cuda
def test_trtllm_ragged_empty_kv_rows_are_neutral_and_match_compacted_call():
    """Rows with query tokens and no KV tokens must contribute out=0, lse=-inf."""
    device = torch.device("cuda")
    _require_trtllm_ragged(device)
    torch.manual_seed(42)

    q, k, v, q_lens, kv_lens, q_indptr, kv_indptr = _empty_kv_case(device)
    out = torch.full(
        (q.shape[0], q.shape[1], v.shape[2]),
        7.0,
        device=device,
        dtype=torch.bfloat16,
    )
    lse = torch.full((q.shape[0], q.shape[1]), 3.0, device=device)

    output, lse_out = _run_trtllm_ragged(
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        kv_lens,
        out=out,
        lse=lse,
        q_lens_cpu=q_lens.cpu(),
        kv_lens_cpu=kv_lens.cpu(),
    )

    assert output.data_ptr() == out.data_ptr()
    assert lse_out.data_ptr() == lse.data_ptr()

    active_rows = [row for row, length in enumerate(kv_lens.tolist()) if length > 0]
    for row in range(kv_lens.shape[0]):
        q_start = int(q_indptr[row].item())
        q_end = int(q_indptr[row + 1].item())
        if row not in active_rows:
            assert torch.all(output[q_start:q_end] == 0)
            assert torch.isneginf(lse_out[q_start:q_end]).all()

    compact_q_lens = q_lens[active_rows]
    compact_kv_lens = kv_lens[active_rows]
    compact_q_indptr = _indptr(compact_q_lens)
    compact_kv_indptr = _indptr(compact_kv_lens)
    compact_output, compact_lse = _run_trtllm_ragged(
        _pack_rows(q, q_indptr, active_rows),
        _pack_rows(k, kv_indptr, active_rows),
        _pack_rows(v, kv_indptr, active_rows),
        compact_q_indptr,
        compact_kv_indptr,
        compact_kv_lens,
        q_lens_cpu=compact_q_lens.cpu(),
        kv_lens_cpu=compact_kv_lens.cpu(),
    )

    for compact_row, row in enumerate(active_rows):
        q_start = int(q_indptr[row].item())
        q_end = int(q_indptr[row + 1].item())
        compact_start = int(compact_q_indptr[compact_row].item())
        compact_end = int(compact_q_indptr[compact_row + 1].item())
        torch.testing.assert_close(
            output[q_start:q_end],
            compact_output[compact_start:compact_end],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            lse_out[q_start:q_end],
            compact_lse[compact_start:compact_end],
            atol=2e-2,
            rtol=2e-2,
        )


@pytest.mark.cuda
def test_trtllm_ragged_empty_kv_rows_direct_fallback_matches_cpu_mirror_path():
    """Direct callers without CPU mirrors still get the same neutral semantics."""
    device = torch.device("cuda")
    _require_trtllm_ragged(device)
    torch.manual_seed(42)

    q, k, v, q_lens, kv_lens, q_indptr, kv_indptr = _empty_kv_case(device)
    mirror_output, mirror_lse = _run_trtllm_ragged(
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        kv_lens,
        q_lens_cpu=q_lens.cpu(),
        kv_lens_cpu=kv_lens.cpu(),
    )
    fallback_output, fallback_lse = _run_trtllm_ragged(
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        kv_lens,
    )

    torch.testing.assert_close(fallback_output, mirror_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(fallback_lse, mirror_lse, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("use_cpu_lens", [True, False])
@pytest.mark.cuda
def test_trtllm_ragged_all_empty_kv_rows_with_queries_are_neutral(use_cpu_lens):
    device = torch.device("cuda")
    _require_trtllm_ragged(device)
    torch.manual_seed(42)

    batch_size = 3
    num_heads = 16
    head_dim_qk = 128
    head_dim_vo = 128
    q_lens = torch.tensor([4, 2, 3], device=device, dtype=torch.int32)
    kv_lens = torch.zeros(batch_size, device=device, dtype=torch.int32)
    q_indptr = _indptr(q_lens)
    kv_indptr = _indptr(kv_lens)
    q = torch.randn(
        int(q_indptr[-1].item()),
        num_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.empty((0, num_heads, head_dim_qk), device=device, dtype=torch.bfloat16)
    v = torch.empty((0, num_heads, head_dim_vo), device=device, dtype=torch.bfloat16)
    out = torch.full(
        (q.shape[0], num_heads, head_dim_vo),
        7.0,
        device=device,
        dtype=torch.bfloat16,
    )
    lse = torch.full((q.shape[0], num_heads), 3.0, device=device)

    cpu_lens_kwargs = (
        {"q_lens_cpu": q_lens.cpu(), "kv_lens_cpu": kv_lens.cpu()}
        if use_cpu_lens
        else {}
    )
    output, lse_out = _run_trtllm_ragged(
        q,
        k,
        v,
        q_indptr,
        kv_indptr,
        kv_lens,
        max_kv_len=0,
        out=out,
        lse=lse,
        **cpu_lens_kwargs,
    )

    assert output.data_ptr() == out.data_ptr()
    assert lse_out.data_ptr() == lse.data_ptr()
    assert torch.all(output == 0)
    assert torch.isneginf(lse_out).all()


@pytest.mark.cuda
def test_trtllm_ragged_requires_cpu_lens_for_cuda_graph_capture(monkeypatch):
    device = torch.device("cuda")
    _require_trtllm_ragged(device)
    torch.manual_seed(42)

    q, k, v, _, kv_lens, q_indptr, kv_indptr = _empty_kv_case(device)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True, raising=False
    )

    with pytest.raises(ValueError, match="must be provided during CUDA graph capture"):
        _run_trtllm_ragged(q, k, v, q_indptr, kv_indptr, kv_lens)
