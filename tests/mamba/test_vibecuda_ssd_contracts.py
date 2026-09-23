"""Negative launch-contract tests for the VibeCUDA SSD backend."""

import pytest
import torch

from flashinfer.mamba.ssd_vibecuda import VibeCUDASSDCombined


def _case(*, varlen=False):
    batch, seqlen, nheads, ngroups = 1, 128, 8, 8
    tensors = [
        torch.randn(batch, seqlen, nheads, 64, device="cuda").bfloat16(),
        torch.randn(batch, seqlen, nheads, device="cuda"),
        -torch.rand(nheads, device="cuda"),
        torch.randn(batch, seqlen, ngroups, 128, device="cuda").bfloat16(),
        torch.randn(batch, seqlen, ngroups, 128, device="cuda").bfloat16(),
    ]
    arguments = {
        "D": torch.randn(nheads, 64, device="cuda").bfloat16(),
        "z": torch.randn_like(tensors[0]),
        "dt_bias": torch.randn(nheads, device="cuda"),
        "initial_states": torch.randn(batch, nheads, 64, 128, device="cuda").bfloat16(),
    }
    runner = VibeCUDASSDCombined(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        has_initial_states=True,
        has_varlen=varlen,
        has_z=True,
    )
    return runner, tensors, arguments


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("x_rank", "x must be a 4D"),
        ("dt_shape", "dt must have shape"),
        ("a_dtype", "dtype float32"),
        ("b_shape", "B must have shape"),
        ("c_dtype", "x, B, and C must be bfloat16"),
        ("d_shape", "D must have shape"),
        ("d_noncontiguous", "D must be contiguous"),
        ("z_shape", "z must have the same shape"),
        ("initial_shape", "initial_states must have shape"),
        ("out_shape", "out must have shape"),
    ],
)
def test_rejects_invalid_tensor_contracts(mutation, match):
    runner, tensors, arguments = _case()
    if mutation == "x_rank":
        tensors[0] = tensors[0].flatten(0, 1)
    elif mutation == "dt_shape":
        tensors[1] = tensors[1][..., :-1]
    elif mutation == "a_dtype":
        tensors[2] = tensors[2].bfloat16()
    elif mutation == "b_shape":
        tensors[3] = tensors[3][..., :-1]
    elif mutation == "c_dtype":
        tensors[4] = tensors[4].float()
    elif mutation == "d_shape":
        arguments["D"] = arguments["D"][:-1]
    elif mutation == "d_noncontiguous":
        arguments["D"] = torch.randn(8, 128, device="cuda").bfloat16()[:, ::2]
    elif mutation == "z_shape":
        arguments["z"] = arguments["z"][..., :-1]
    elif mutation == "initial_shape":
        arguments["initial_states"] = arguments["initial_states"][:-1]
    elif mutation == "out_shape":
        arguments["out"] = torch.empty(
            (1, 8, 64, 2, 128), dtype=torch.bfloat16, device="cuda"
        )
    with pytest.raises(ValueError, match=match):
        runner.run(*tensors, **arguments)


def test_rejects_unpaired_optional_metadata():
    runner, tensors, arguments = _case(varlen=True)
    arguments.update(
        seq_idx=torch.zeros((1, 128), dtype=torch.int32, device="cuda"),
        chunk_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    with pytest.raises(ValueError, match="must be supplied together"):
        runner.run(*tensors, **arguments)


def test_rejects_cross_device_input():
    runner, tensors, arguments = _case()
    tensors[2] = tensors[2].cpu()
    with pytest.raises(ValueError, match="A must be on"):
        runner.run(*tensors, **arguments)


def test_two_streams_do_not_share_first_column_d_storage():
    runner, tensors, arguments = _case()
    d_first = torch.full_like(arguments["D"], 0.25)
    d_second = torch.full_like(arguments["D"], -0.5)

    expected_first = runner.run(*tensors, **{**arguments, "D": d_first})
    expected_second = runner.run(*tensors, **{**arguments, "D": d_second})
    torch.cuda.synchronize()

    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    with torch.cuda.stream(first_stream):
        actual_first = runner.run(*tensors, **{**arguments, "D": d_first})
    with torch.cuda.stream(second_stream):
        actual_second = runner.run(*tensors, **{**arguments, "D": d_second})
    first_stream.synchronize()
    second_stream.synchronize()

    torch.testing.assert_close(actual_first[0], expected_first[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_second[0], expected_second[0], rtol=0, atol=0)
