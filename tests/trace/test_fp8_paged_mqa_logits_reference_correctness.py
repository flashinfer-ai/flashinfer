"""Reference correctness test for the fp8_paged_mqa_logits trace API."""

import pytest
import torch

from tests.trace.reference_utils import _cc, _check


def _skip_if_no_paged_mqa_support():
    if _cc() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("paged MQA logits requires an SM100-class GPU (SM100/SM103/SM107)")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=4, next_n=2, max_seq_len=4096, block_size=64),
        dict(batch_size=8, next_n=1, max_seq_len=2048, block_size=128),
        # max_seq_len is a free axis and need not be block-aligned: the last
        # physical block then runs past the output width (257 -> 5*64 = 320).
        dict(batch_size=2, next_n=2, max_seq_len=257, block_size=64),
        dict(batch_size=2, next_n=1, max_seq_len=513, block_size=128),
    ],
)
def test_fp8_paged_mqa_logits_reference_correctness(shape_kwargs):
    """flashinfer.fp8_paged_mqa_logits vs the trace template reference."""
    _skip_if_no_paged_mqa_support()
    import flashinfer
    from flashinfer.trace.templates.attn_scores import fp8_paged_mqa_logits_trace

    inputs = fp8_paged_mqa_logits_trace.init(**shape_kwargs)
    out = flashinfer.fp8_paged_mqa_logits(
        inputs["q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["block_tables"],
        inputs["seq_lens"],
        inputs["max_seq_len"],
    )
    ref = fp8_paged_mqa_logits_trace.reference(
        inputs["q"],
        inputs["kv_fused"],
        inputs["weights"],
        inputs["block_tables"],
        inputs["seq_lens"],
        inputs["max_seq_len"],
    )
    _check(
        fp8_paged_mqa_logits_trace,
        ref,
        out,
        seq_lens=inputs["seq_lens"],
        next_n=shape_kwargs["next_n"],
    )
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("math_mode", ["relu_fast", "relu_strict", None])
def test_fp8_trace_fp16_output_preserves_fp32_intermediates(math_mode):
    """Near-cancelling weights distinguish final FP16 storage from FP16 math."""
    _skip_if_no_paged_mqa_support()
    import flashinfer
    from flashinfer.trace.templates.attn_scores import (
        _pack_fused_kv_fp8,
        fp8_paged_mqa_logits_trace,
    )

    q = torch.ones((1, 1, 8, 128), device="cuda").to(torch.float8_e4m3fn)
    kv = torch.ones((1, 32, 128), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((1, 32), 1.0001, device="cuda")
    weights = torch.zeros((1, 8), device="cuda")
    weights[0, 0] = 1.0001
    weights[0, 1] = -1.0
    inputs = dict(
        q=q,
        kv_fused=_pack_fused_kv_fp8(kv, scales, 32, 128),
        weights=weights,
        block_tables=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([32], dtype=torch.int32, device="cuda"),
        max_seq_len=32,
        output_dtype=torch.float16,
    )
    if math_mode is not None:
        inputs["math_mode"] = math_mode
    # Every dot product is exactly 128. Rounding the first weight to FP16
    # would cancel the second head entirely and incorrectly produce zero.
    expected = (128.0 * weights.sum(dim=1)[:, None] * scales).to(torch.float16)
    assert expected.abs().min().item() > 1e-3
    reference = fp8_paged_mqa_logits_trace.reference(**inputs)
    torch.testing.assert_close(reference, expected, atol=0, rtol=0)
    actual = flashinfer.fp8_paged_mqa_logits(**inputs)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    definition = flashinfer.fi_trace(flashinfer.fp8_paged_mqa_logits, **inputs)
    assert definition["outputs"]["logits"]["dtype"] == "float16"
    assert definition["inputs"]["output_dtype"]["dtype"] == "dtype"
    namespace = {}
    exec(definition["reference"], namespace)
    exported = namespace[fp8_paged_mqa_logits_trace.reference.__name__](**inputs)
    torch.testing.assert_close(exported, expected, atol=0, rtol=0)
    torch.cuda.synchronize()
