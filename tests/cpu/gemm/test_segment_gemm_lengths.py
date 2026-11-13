"""Regressions for integer segment lengths in SegmentGEMM metadata."""

import pytest
import torch

from flashinfer.gemm import gemm_base


class _ArgumentsPrepared(Exception):
    """Stop before launching a GPU kernel after checking its metadata."""


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["sm80", "sm90"])
@pytest.mark.parametrize("explicit_indptr", [False, True])
@pytest.mark.parametrize(
    "dtype,lengths,expected_indptr",
    [
        pytest.param(torch.float16, [2049, 1], [0, 2049, 2050], id="fp16-round-down"),
        pytest.param(torch.bfloat16, [257, 1], [0, 257, 258], id="bf16-round-down"),
        pytest.param(torch.bfloat16, [259, 1], [0, 259, 260], id="bf16-round-up"),
        pytest.param(torch.float16, [3, 10, 99], [0, 3, 13, 112], id="fp16-control"),
        pytest.param(torch.bfloat16, [3, 10, 99], [0, 3, 13, 112], id="bf16-control"),
    ],
)
def test_segment_gemm_preserves_integer_lengths(
    monkeypatch, backend, explicit_indptr, dtype, lengths, expected_indptr
):
    """Check exact segment boundaries passed to both argument-preparation backends."""
    expected = torch.tensor(expected_indptr, dtype=torch.int64)

    def capture_args(x, weights, out, column_major, batch_size, seg_indptr, indices):
        """Check real wrapper metadata without compiling or launching CUTLASS."""
        assert seg_indptr.device == x.device
        assert batch_size == len(lengths)
        torch.testing.assert_close(seg_indptr, expected, rtol=0, atol=0)
        raise _ArgumentsPrepared

    monkeypatch.setattr(
        gemm_base, f"launch_compute_{backend}_group_gemm_args", capture_args
    )
    wrapper = gemm_base.SegmentGEMMWrapper(torch.empty(1, dtype=torch.int8), backend)
    x = torch.empty(sum(lengths), 16, dtype=dtype)
    weights = torch.empty(len(lengths), 16, 16, dtype=dtype)
    with pytest.raises(_ArgumentsPrepared):
        wrapper.run(
            x,
            weights,
            len(lengths),
            weight_column_major=False,
            seg_lens=torch.tensor(lengths, dtype=torch.int64),
            seg_indptr=expected if explicit_indptr else None,
        )
