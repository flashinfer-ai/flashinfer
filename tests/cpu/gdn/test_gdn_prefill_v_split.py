"""CPU tests of the ``v_split`` option of ``chunk_gated_delta_rule``.

``v_split="auto"`` splits the value rows of the SM100-family non-CP prefill
kernel when the split grid fits on the SMs or when the stock grid needs more
than one wave; argument checks reject values and routes the kernel cannot serve
before any device work.
"""

import pytest
import torch

from flashinfer.gdn_prefill import _gdn_prefill_sm100_v_split, chunk_gated_delta_rule


@pytest.mark.cpu_only
@pytest.mark.parametrize("v_split", [1, 2])
def test_explicit_v_split_passes_through(v_split):
    for parallel_states in (1, 100, 148, 1000):
        assert (
            _gdn_prefill_sm100_v_split(
                v_split, num_parallel_states=parallel_states, num_sms=148
            )
            == v_split
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "num_sms,parallel_states,expected",
    [
        # The split grid fits on the SMs.
        (148, 1, 2),
        (148, 74, 2),
        (212, 96, 2),
        (212, 106, 2),
        # The split grid would spill into a second wave: stock kernel.
        (148, 75, 1),
        (148, 148, 1),
        (212, 107, 1),
        (212, 192, 1),
        (212, 212, 1),
        # The stock grid already needs more than one wave.
        (148, 149, 2),
        (212, 213, 2),
        (212, 1312, 2),
    ],
)
def test_auto_v_split_rule(num_sms, parallel_states, expected):
    assert (
        _gdn_prefill_sm100_v_split(
            "auto", num_parallel_states=parallel_states, num_sms=num_sms
        )
        == expected
    )


def _cpu_inputs():
    q = torch.zeros(64, 4, 128, dtype=torch.bfloat16)
    k = torch.zeros(64, 4, 128, dtype=torch.bfloat16)
    v = torch.zeros(64, 8, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 64], dtype=torch.int32)
    return q, k, v, cu_seqlens


@pytest.mark.cpu_only
@pytest.mark.parametrize("v_split", [0, 3, "2", 2.0, True, None])
def test_rejects_unsupported_v_split_values(v_split):
    q, k, v, cu_seqlens = _cpu_inputs()
    with pytest.raises(ValueError, match="v_split must be"):
        chunk_gated_delta_rule(q, k, v, cu_seqlens=cu_seqlens, v_split=v_split)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(dict(use_cp=True), id="use_cp"),
        pytest.param(dict(backend="cake_gdn"), id="cake_gdn"),
        pytest.param(dict(backend="cudnn"), id="cudnn"),
        pytest.param(dict(backend="sm120_cute"), id="sm120_cute"),
    ],
)
def test_explicit_v_split_rejects_other_kernels(kwargs):
    q, k, v, cu_seqlens = _cpu_inputs()
    with pytest.raises(ValueError, match="v_split=2"):
        chunk_gated_delta_rule(q, k, v, cu_seqlens=cu_seqlens, v_split=2, **kwargs)
