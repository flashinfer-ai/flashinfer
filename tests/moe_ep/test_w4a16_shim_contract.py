"""Host launch contracts for the W4A16-only MegaMoE shim."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig


@pytest.fixture
def shim():
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import w4a16_mega_moe

    return w4a16_mega_moe


@pytest.fixture
def buffer():
    frontend = mock.Mock()
    frontend.config.in_kernel_fc2_reduce = False
    frontend._reduce = object()
    combined = torch.empty(4, 2, 64, dtype=torch.bfloat16)
    frontend.run.return_value = combined[:1]
    return SimpleNamespace(
        _destroyed=False,
        num_max_tokens=4,
        hidden=64,
        x=torch.empty(4, 64, dtype=torch.bfloat16),
        topk_idx=torch.empty(4, 2, dtype=torch.int64),
        topk_weights=torch.empty(4, 2, dtype=torch.float32),
        combine_output=combined,
        _frontend=frontend,
    )


def _output(num_tokens=1, *, contiguous=True):
    # Only CUDA ownership/launch ordering is under test. No device allocation
    # or kernel execution is needed to exercise the host wrapper.
    output = mock.Mock(spec=torch.Tensor)
    output.shape = (num_tokens, 64)
    output.dtype = torch.bfloat16
    output.is_cuda = True
    output.is_contiguous.return_value = contiguous
    return output


def _call(shim, buffer, output, **kwargs):
    shim(
        output,
        (None, None, None),
        (None, None, None),
        buffer,
        num_tokens=output.shape[0],
        **kwargs,
    )


def test_config_rejects_ignored_fast_math_option():
    with pytest.raises(TypeError, match="fast_math"):
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=64, top_k=2, fast_math=False
        )


def test_shim_rejects_ignored_fast_math_option(shim, buffer):
    with pytest.raises(TypeError, match="fast_math"):
        _call(shim, buffer, _output(), fast_math=False)
    buffer._frontend.run.assert_not_called()


@pytest.mark.parametrize("kind", ("cpu", "noncontiguous"))
def test_output_rejected_before_collective_launch(shim, buffer, kind):
    output = (
        torch.empty(1, 64, dtype=torch.bfloat16)
        if kind == "cpu"
        else _output(contiguous=False)
    )
    with pytest.raises(ValueError, match="contiguous CUDA"):
        _call(shim, buffer, output)
    buffer._frontend.run.assert_not_called()


@pytest.mark.parametrize("sync", (False, True))
def test_sync_covers_final_reduction(shim, buffer, sync):
    events = []

    def fused(*args, **kwargs):
        events.append("fused")
        return buffer.combine_output[:1]

    buffer._frontend.run.side_effect = fused
    buffer._frontend.reduce_topk.side_effect = lambda *args: events.append("reduce")
    with mock.patch(
        "torch.cuda.synchronize", side_effect=lambda: events.append("sync")
    ):
        _call(shim, buffer, _output(), sync=sync)
    assert events == (["fused", "reduce", "sync"] if sync else ["fused", "reduce"])


def test_nonempty_capture_after_empty_warmup_fails_before_collective(shim, buffer):
    buffer._frontend._reduce = None
    with (
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        pytest.raises(RuntimeError, match="warmup.*default batch"),
    ):
        _call(shim, buffer, _output())
    buffer._frontend.run.assert_not_called()
    buffer._frontend.reduce_topk.assert_not_called()


def test_empty_capture_does_not_require_a_compiled_reducer(shim, buffer):
    buffer._frontend._reduce = None
    buffer._frontend.run.return_value = buffer.combine_output[:0]
    with mock.patch("torch.cuda.is_current_stream_capturing", return_value=True):
        _call(shim, buffer, _output(0))
    buffer._frontend.run.assert_called_once()
