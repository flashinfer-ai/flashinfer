"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib
import importlib.util
import inspect
import re
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.mamba import SSDCombined


def _load_cake_benchmark_module():
    path = Path(__file__).parents[2] / "benchmarks" / "bench_cake_mamba_ssd_combined.py"
    spec = importlib.util.spec_from_file_location("bench_cake_mamba_ssd_combined", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_cute_parity(actual, expected):
    for index in (0, 1):
        reference = expected[index]
        # Cancellation ties the error to the head's magnitude rather than the
        # entry's; the tensor max is a coarse bound on that.
        atol = max(1e-2, 5e-4 * reference.abs().amax().item())
        torch.testing.assert_close(actual[index], reference, atol=atol, rtol=1e-2)


def _varlen_metadata(lengths, dtype):
    """Packed-varlen ``seq_idx`` / logical-chunk metadata; ``sum(lengths)`` may
    end inside a physical chunk (the trailing chunk is then partial)."""

    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=dtype, device="cuda")
    start = 0
    for sequence, length in enumerate(lengths):
        seq_idx[0, start : start + length] = sequence
        start += length
    chunk_indices = []
    chunk_offsets = []
    for chunk in range(-(-total // 128)):
        values = seq_idx[0, chunk * 128 : (chunk + 1) * 128]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)
    return (
        seq_idx,
        torch.tensor(chunk_indices, dtype=torch.int32, device="cuda"),
        torch.tensor(chunk_offsets, dtype=torch.int32, device="cuda"),
    )


def _seq_chunk_cumsum(lengths):
    """Exclusive prefix sum of per-sequence logical chunk counts."""

    cumsum = [0]
    start = 0
    for length in lengths:
        end = start + length
        cumsum.append(cumsum[-1] + (-(-end // 128) - start // 128))
        start = end
    return torch.tensor(cumsum, dtype=torch.int32, device="cuda")


def _case(
    *,
    nheads=8,
    ngroups=8,
    state_dtype=torch.bfloat16,
    varlen=False,
    seq_idx_dtype=torch.int32,
    preprocess_dtype=torch.float32,
    d_has_hdim=True,
    seqlen=None,
    lengths=(96, 160),
    initial_states=True,
    seed=7,
):
    torch.manual_seed(seed)
    if varlen:
        batch, seqlen = 1, sum(lengths)
    else:
        batch, seqlen = 2, 128 if seqlen is None else seqlen
    x = torch.randn(batch, seqlen, nheads, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(batch, seqlen, nheads, device="cuda").to(preprocess_dtype)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    d_shape = (nheads, 64) if d_has_hdim else (nheads,)
    D = torch.randn(*d_shape, device="cuda").to(torch.bfloat16)
    z = torch.randn_like(x)
    dt_bias = (torch.rand(nheads, device="cuda", dtype=torch.float32) - 4.0).to(
        preprocess_dtype
    )
    state_batch = len(lengths) if varlen else batch
    initial_states_tensor = (
        torch.randn(state_batch, nheads, 64, 128, device="cuda").to(state_dtype)
        if initial_states
        else None
    )
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, seq_idx_dtype)
        seq_chunk_cumsum = _seq_chunk_cumsum(lengths)
    else:
        seq_idx = chunk_indices = chunk_offsets = seq_chunk_cumsum = None

    constructor = dict(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=state_dtype,
        has_d=True,
        d_has_hdim=d_has_hdim,
        has_initial_states=initial_states,
        has_varlen=varlen,
        has_z=True,
        seq_idx_dtype=seq_idx_dtype,
    )
    arguments = dict(
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.001, 0.1),
        initial_states=initial_states_tensor,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        return_final_states=True,
    )
    return constructor, (x, dt, A, B, C), arguments


def _cute_padded_reference(constructor, tensors, arguments, lengths):
    """CuTe reference for a Cake problem the CuTe backend cannot run directly.

    The Cake inputs (batched ``[B, S]`` or packed ``[1, T]``) are flattened
    into one packed stream of ``lengths`` sequences, zero-padded with one
    extra sequence to a multiple of 128 tokens, and run through CuTe varlen
    with explicit (zero when absent) initial states.  Returns the token-major
    output restricted to the real tokens and the per-sequence final states.
    """

    x, dt, A, B, C = tensors
    nheads = constructor["nheads"]
    total = sum(lengths)
    assert x.shape[0] * x.shape[1] == total
    padded = -(-total // 128) * 128
    pad = padded - total

    def stream(value):
        flat = value.reshape(1, total, *value.shape[2:])
        if pad == 0:
            return flat.contiguous()
        padding = torch.zeros(
            (1, pad, *value.shape[2:]), dtype=value.dtype, device=value.device
        )
        return torch.cat((flat, padding), dim=1).contiguous()

    initial_states = arguments["initial_states"]
    if initial_states is None:
        initial_states = torch.zeros(
            (len(lengths), nheads, 64, 128),
            dtype=constructor["state_dtype"],
            device="cuda",
        )
    if pad:
        initial_states = torch.cat(
            (initial_states, torch.zeros_like(initial_states[:1])), dim=0
        )
    padded_lengths = [*lengths, pad] if pad else list(lengths)
    seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(
        padded_lengths, constructor["seq_idx_dtype"]
    )
    cute_constructor = {
        **constructor,
        "has_initial_states": True,
        "has_varlen": True,
    }
    cute_arguments = {
        **arguments,
        "z": stream(arguments["z"]),
        "initial_states": initial_states,
        "seq_idx": seq_idx,
        "chunk_indices": chunk_indices,
        "chunk_offsets": chunk_offsets,
        "seq_chunk_cumsum": _seq_chunk_cumsum(padded_lengths),
    }
    out, final = SSDCombined(**cute_constructor, backend="cute").run(
        stream(x), stream(dt), A, stream(B), stream(C), **cute_arguments
    )
    return out[:, :total].reshape(x.shape), final[: len(lengths)]


def test_cake_benchmark_validation_policy():
    module = _load_cake_benchmark_module()

    def report(*, out=True, final_states=True, speedup=1.01):
        return {
            "out": {"tolerance_passed": out},
            "final_states": {"tolerance_passed": final_states},
            "speedup": speedup,
        }

    module._validate_report(report(), require_qualified_row=False)
    module._validate_report(report(speedup=0.99), require_qualified_row=False)
    with pytest.raises(AssertionError, match="output failed BF16 parity"):
        module._validate_report(report(out=False), require_qualified_row=False)
    with pytest.raises(AssertionError, match="final state failed BF16 parity"):
        module._validate_report(report(final_states=False), require_qualified_row=False)
    with pytest.raises(AssertionError, match="must be faster than CuTe"):
        module._validate_report(report(speedup=0.99), require_qualified_row=True)


def _strided_last_dim(value):
    storage = torch.empty(
        (*value.shape[:-1], value.shape[-1] + 1),
        dtype=value.dtype,
        device=value.device,
    )
    view = storage[..., : value.shape[-1]]
    view.copy_(value)
    assert not view.is_contiguous()
    return view


def _sglang_projection_view(value):
    active_width = value.numel() // (value.shape[0] * value.shape[1])
    storage = torch.empty(
        (value.shape[0], value.shape[1], active_width + 8),
        dtype=value.dtype,
        device=value.device,
    )
    view = storage[..., :active_width].view(value.shape)
    view.copy_(value)
    assert not view.is_contiguous() and view.stride(-1) == 1
    return view


@pytest.mark.parametrize(
    "state_dtype,varlen,seq_idx_dtype,nheads,ngroups,preprocess_dtype,d_has_hdim",
    [
        (torch.bfloat16, False, torch.int32, 8, 8, torch.float32, True),
        (torch.float16, False, torch.int32, 8, 8, torch.float32, False),
        (torch.bfloat16, True, torch.int32, 8, 8, torch.float32, True),
        (torch.float16, True, torch.int64, 8, 8, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 8, 8, torch.bfloat16, False),
        # Dynamic public-API boundaries: minimum head/group and one group/head.
        (torch.bfloat16, False, torch.int32, 1, 1, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 12, 3, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 16, 4, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 1, torch.float32, False),
        (torch.bfloat16, False, torch.int32, 128, 128, torch.float32, False),
        # NVIDIA Nemotron-H-8B-Base-8K single-GPU local Mamba shape.
        (torch.bfloat16, False, torch.int32, 128, 8, torch.float32, False),
        (torch.bfloat16, True, torch.int32, 128, 8, torch.float32, False),
    ],
)
def test_cake_ssd_combined_route_matrix(
    state_dtype,
    varlen,
    seq_idx_dtype,
    nheads,
    ngroups,
    preprocess_dtype,
    d_has_hdim,
):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(
        nheads=nheads,
        ngroups=ngroups,
        state_dtype=state_dtype,
        varlen=varlen,
        seq_idx_dtype=seq_idx_dtype,
        preprocess_dtype=preprocess_dtype,
        d_has_hdim=d_has_hdim,
    )
    if nheads == 128 and ngroups == 8:
        # Nemotron-H prefill starts from zero state and uses the unbounded
        # positive-dt interval in both batched and variable-length modes.
        # Finite-clamp and nonzero-initial-state feature rows remain covered
        # independently above; do not invent their Cartesian product with the
        # model-derived head shape.
        arguments["initial_states"].zero_()
        arguments["dt_limit"] = (0.0, float("inf"))
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    _assert_cute_parity(actual, expected)


def test_cake_ssd_combined_accepts_framework_strided_input_views():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(varlen=True)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    x, dt, A, B, C = tensors
    tensors = (
        _sglang_projection_view(x),
        _sglang_projection_view(dt),
        A,
        _sglang_projection_view(B),
        _sglang_projection_view(C),
    )
    arguments = {
        **arguments,
        "z": _strided_last_dim(arguments["z"]),
        "initial_states": _strided_last_dim(arguments["initial_states"]),
    }

    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    _assert_cute_parity(actual, expected)


@pytest.mark.parametrize(
    "d_has_hdim,runtime_d_has_hdim", [(True, False), (False, True)]
)
def test_cake_ssd_combined_matches_cute_d_shape_coercion(
    d_has_hdim, runtime_d_has_hdim
):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(d_has_hdim=d_has_hdim)
    d_shape = (8, 64) if runtime_d_has_hdim else (8,)
    arguments["D"] = torch.randn(*d_shape, device="cuda").to(torch.bfloat16)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    _assert_cute_parity(actual, expected)


def test_cake_ssd_combined_updates_caller_buffers():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(varlen=True)
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    expected_cumsum = arguments["seq_chunk_cumsum"]
    actual_cumsum = torch.full_like(expected_cumsum, -1)
    # Token-major caller storage [batch, seqlen, nheads, headdim], written by
    # the kernels and returned as-is.
    out = torch.empty((1, 256, 8, 64), dtype=torch.bfloat16, device="cuda")
    runner = SSDCombined(**constructor, backend="cake")
    actual = runner.run(
        *tensors,
        **{
            **arguments,
            "seq_chunk_cumsum": actual_cumsum,
            "update_seq_chunk_cumsum": True,
            "out": out,
        },
    )

    torch.testing.assert_close(actual_cumsum, expected_cumsum, rtol=0, atol=0)
    assert actual[0] is out
    _assert_cute_parity(actual, expected)

    preserved_cumsum = actual_cumsum.clone()
    runner.run(
        *tensors,
        **{
            **arguments,
            "seq_chunk_cumsum": actual_cumsum,
            "update_seq_chunk_cumsum": False,
        },
    )
    torch.testing.assert_close(actual_cumsum, preserved_cumsum, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen"))
def test_cake_ssd_combined_writes_selected_checkpoint_state(varlen, state_dtype):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(
        varlen=varlen,
        state_dtype=state_dtype,
    )
    sequence_index = 1 if varlen else 0
    sequence_start = 96 if varlen else 0
    checkpoint_length = 128
    checkpoint_token = (
        sequence_start + checkpoint_length if varlen else checkpoint_length
    )
    checkpoint_states = torch.full(
        (3, *arguments["initial_states"].shape[1:]),
        torch.nan,
        dtype=arguments["initial_states"].dtype,
        device="cuda",
    )
    checkpoint_state = checkpoint_states[2:3]
    if varlen:
        full_arguments = {
            **arguments,
            # Expose sequence 1's checkpoint inside physical chunk 1 as a logical
            # segment boundary. This is the packed shape used by SGLang.
            "chunk_indices": torch.tensor(
                [0, 0, 1, 1], dtype=torch.int32, device="cuda"
            ),
            "chunk_offsets": torch.tensor(
                [0, 96, 0, 96], dtype=torch.int32, device="cuda"
            ),
            "seq_chunk_cumsum": torch.tensor(
                [0, 1, 4], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_token_indices": torch.tensor(
                [-1, checkpoint_token], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_state_slots": torch.tensor(
                [-1, 2], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_states": checkpoint_states,
        }
    else:
        full_arguments = {
            **arguments,
            "checkpoint_token_indices": torch.tensor(
                [checkpoint_token, -1], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_state_slots": torch.tensor(
                [2, -1], dtype=torch.int32, device="cuda"
            ),
            "checkpoint_states": checkpoint_states,
        }
    SSDCombined(**constructor, backend="cake").run(*tensors, **full_arguments)

    x, dt, A, B, C = tensors
    packed_batch_index = 0 if varlen else sequence_index
    prefix_tensors = (
        x[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        dt[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        A,
        B[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        C[
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
    )
    prefix_arguments = {
        **arguments,
        "z": arguments["z"][
            packed_batch_index : packed_batch_index + 1,
            sequence_start:checkpoint_token,
        ].contiguous(),
        "initial_states": arguments["initial_states"][
            sequence_index : sequence_index + 1
        ].contiguous(),
    }
    prefix_constructor = {**constructor, "has_varlen": varlen}
    if varlen:
        prefix_arguments.update(
            seq_idx=torch.zeros(
                (1, checkpoint_length), dtype=torch.int32, device="cuda"
            ),
            chunk_indices=torch.zeros(1, dtype=torch.int32, device="cuda"),
            chunk_offsets=torch.zeros(1, dtype=torch.int32, device="cuda"),
            seq_chunk_cumsum=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        )
    _, expected_state = SSDCombined(**prefix_constructor, backend="cute").run(
        *prefix_tensors, **prefix_arguments
    )
    torch.testing.assert_close(
        checkpoint_state,
        expected_state,
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.isnan(checkpoint_states[:2]).all()


def test_cake_ssd_combined_allocation_output_lifetime():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case()
    runner = SSDCombined(**constructor, backend="cake")
    first, first_final = runner.run(*tensors, **arguments)
    retained = first.clone()
    retained_final = first_final.clone()
    second, second_final = runner.run(*tensors, **arguments)

    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert (
        first_final.untyped_storage().data_ptr()
        != second_final.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(first, retained, rtol=0, atol=0)
    torch.testing.assert_close(first_final, retained_final, rtol=0, atol=0)

    without_final = runner.run(*tensors, **{**arguments, "return_final_states": False})
    assert isinstance(without_final, tuple)
    assert without_final[1] is None


@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen"))
def test_cake_ssd_combined_f32_state_matches_cute_on_bf16_representable_states(
    varlen,
):
    """FP32 state programs: CuTe has no FP32 state, so feed both backends the
    same bf16-representable initial states (exactly the same values in either
    dtype) and compare the outputs plus the final states rounded to bf16."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(varlen=varlen, state_dtype=torch.float32)
    bf16_states = arguments["initial_states"].to(torch.bfloat16)
    cute_constructor = {**constructor, "state_dtype": torch.bfloat16}
    cute_arguments = {**arguments, "initial_states": bf16_states}
    arguments["initial_states"] = bf16_states.to(torch.float32)

    expected = SSDCombined(**cute_constructor, backend="cute").run(
        *tensors, **cute_arguments
    )
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert actual[1].dtype == torch.float32
    _assert_cute_parity((actual[0], actual[1].to(torch.bfloat16)), expected)


@pytest.mark.parametrize(
    "varlen,lengths",
    [
        (False, (1000, 1000)),
        (True, (1000,)),
        (True, (128, 900)),
        (True, (300, 300)),
    ],
    ids=("batched_2x1000", "varlen_1000", "varlen_128_900", "varlen_300_300"),
)
def test_cake_ssd_combined_accepts_unaligned_seqlen(varlen, lengths):
    """``seqlen % 128 != 0``: the partial trailing physical chunk is handled
    in-kernel; the reference is CuTe on a zero-padded packed stream."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    if varlen:
        constructor, tensors, arguments = _case(varlen=True, lengths=lengths)
    else:
        constructor, tensors, arguments = _case(seqlen=lengths[0])
    assert tensors[0].shape[1] % 128 != 0 or sum(lengths) % 128 != 0
    expected = _cute_padded_reference(constructor, tensors, arguments, lengths)

    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert tuple(actual[0].shape) == tuple(tensors[0].shape)
    _assert_cute_parity(actual, expected)


@pytest.mark.parametrize("count_source", ("num_seqs", "seq_chunk_cumsum"))
@pytest.mark.parametrize(
    "lengths", [(128,), (128, 128), (1000,)], ids=("1x128", "2x128", "1x1000")
)
def test_cake_ssd_combined_varlen_without_initial_states(lengths, count_source):
    """``initial_states=None`` in varlen mode starts from zero state; the
    sequence count comes from ``num_seqs`` or ``seq_chunk_cumsum``."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(
        varlen=True, lengths=lengths, initial_states=False
    )
    assert constructor["has_initial_states"] is False
    assert arguments["initial_states"] is None
    expected = _cute_padded_reference(constructor, tensors, arguments, lengths)
    if count_source == "num_seqs":
        arguments["seq_chunk_cumsum"] = None
        arguments["num_seqs"] = len(lengths)

    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert tuple(actual[1].shape) == (len(lengths), 8, 64, 128)
    _assert_cute_parity(actual, expected)


def test_cake_ssd_combined_varlen_without_initial_states_needs_a_count():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(
        varlen=True, lengths=(128, 128), initial_states=False
    )
    arguments["seq_chunk_cumsum"] = None

    with pytest.raises(ValueError, match="requires seq_chunk_cumsum or num_seqs"):
        SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)


def _realistic_decay_inputs(lengths, seed, *, nheads=128, ngroups=8, varlen=True):
    """CAKE-950's repro distribution: bf16 ``dt ~ N(-2, 0.5)`` before
    softplus, ``dt_bias = 0.5``, ``A = -exp(N(0, 0.5))``, D = 1, no z."""

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total = sum(lengths)
    batch, seqlen = (1, total) if varlen else (len(lengths), lengths[0])

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    x = randn(batch, seqlen, nheads, 64).to(torch.bfloat16)
    dt = (randn(batch, seqlen, nheads) * 0.5 - 2.0).to(torch.bfloat16)
    A = -torch.exp(randn(nheads) * 0.5)
    # Pin the last heads to the Mamba2 ``A_log`` init extremes (|A| up to 16)
    # so every seed reaches the per-chunk overflow band of CAKE-950.
    A[-4:] = torch.tensor([-4.0, -8.0, -12.0, -16.0], device="cuda")
    B = randn(batch, seqlen, ngroups, 128).to(torch.bfloat16)
    C = randn(batch, seqlen, ngroups, 128).to(torch.bfloat16)
    D = torch.ones(nheads, device="cuda", dtype=torch.bfloat16)
    dt_bias = torch.full((nheads,), 0.5, device="cuda", dtype=torch.bfloat16)
    initial_states = torch.zeros(
        len(lengths), nheads, 64, 128, device="cuda", dtype=torch.bfloat16
    )
    constructor = dict(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=True,
        d_has_hdim=False,
        has_initial_states=True,
        has_varlen=varlen,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )
    arguments = dict(
        D=D,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.0, float("inf")),
        initial_states=initial_states,
        return_final_states=True,
    )
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(lengths, torch.int32)
        arguments.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            seq_chunk_cumsum=_seq_chunk_cumsum(lengths),
        )
    return constructor, (x, dt, A, B, C), arguments


@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize(
    "varlen,lengths",
    [(True, (128,)), (True, (128, 128)), (True, (128,) * 8), (False, (128, 128))],
    ids=("varlen_1x128", "varlen_2x128", "varlen_8x128", "batched_2x128"),
)
def test_cake_ssd_combined_single_chunk_realistic_decay_has_no_nan(
    varlen, lengths, seed
):
    """CAKE-950 regression: every sequence is one 128-token chunk on the
    Nemotron-H head geometry with realistic decay (heads whose per-chunk
    ``A * dt`` cumsum passes ``-126 / log2(e)``).  The retired prefix route
    produced ``0 * inf`` NaNs here; the output and states must be finite and
    match CuTe."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _realistic_decay_inputs(
        lengths, seed, varlen=varlen
    )
    dt_processed = torch.nn.functional.softplus(
        tensors[1].float() + arguments["dt_bias"].float()
    )
    batch, seqlen, nheads = dt_processed.shape
    chunk_log2_decay = (dt_processed * tensors[2]).reshape(
        batch, seqlen // 128, 128, nheads
    ).sum(2) * 1.4426950408889634
    assert (chunk_log2_decay < -126.0).any(), "inputs must reach the overflow band"

    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)

    assert torch.isfinite(actual[0].float()).all()
    assert torch.isfinite(actual[1].float()).all()
    _assert_cute_parity(actual, expected)


def _fp32_recurrent_reference(tensors, arguments, lengths):
    """Per-token fp32 SSM recurrence for a packed ``[1, T]`` stream; returns
    the token-major output and the per-sequence final states."""

    x, dt, A, B, C = tensors
    nheads = x.shape[2]
    rep = nheads // B.shape[2]
    x = x[0].float()
    dt_processed = torch.nn.functional.softplus(
        dt[0].float() + arguments["dt_bias"].float()
    )
    B = B[0].float().repeat_interleave(rep, dim=1)
    C = C[0].float().repeat_interleave(rep, dim=1)
    D = arguments["D"].float()
    y = torch.empty(x.shape, device="cuda", dtype=torch.float32)
    states = torch.empty(
        (len(lengths), nheads, 64, 128), device="cuda", dtype=torch.float32
    )
    start = 0
    for sequence, length in enumerate(lengths):
        state = arguments["initial_states"][sequence].float().clone()
        for token in range(start, start + length):
            decay = torch.exp(A * dt_processed[token])
            state = state * decay[:, None, None] + (
                (dt_processed[token][:, None] * x[token])[:, :, None]
                * B[token][:, None, :]
            )
            y[token] = (
                torch.einsum("hdn,hn->hd", state, C[token]) + D[:, None] * x[token]
            )
        states[sequence] = state
        start += length
    return y.unsqueeze(0), states


def test_cake_ssd_combined_nemotron_accuracy_vs_fp32_recurrent_reference():
    """CAKE-942: with FP16 ``delta`` the fraction of outputs outside
    atol = rtol = 1e-2 of an fp32 recurrent reference on the Nemotron-H
    geometry (T = 1024, realistic decay, zero initial state) is about 0.66 %
    (bf16 delta: 1.56 %, stock Triton: 0.62 %)."""

    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    lengths = (1024,)
    constructor, tensors, arguments = _realistic_decay_inputs(lengths, 11)
    reference_out, reference_states = _fp32_recurrent_reference(
        tensors, arguments, lengths
    )

    out, final_states = SSDCombined(**constructor, backend="cake").run(
        *tensors, **arguments
    )

    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(final_states.float()).all()
    assert torch.isfinite(reference_states).all()
    outside = (out.float() - reference_out).abs() > 1e-2 + 1e-2 * reference_out.abs()
    fraction_outside = outside.float().mean().item()
    assert fraction_outside <= 0.010, f"{fraction_outside:.4%} of outputs outside 1e-2"


@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("dt_softplus", (False, True))
def test_cake_ssd_combined_exact_scan_softplus_parity(state_dtype, dt_softplus):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(state_dtype=state_dtype)
    arguments["dt_softplus"] = dt_softplus

    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    _assert_cute_parity(actual, expected)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("varlen", (False, True), ids=("batched", "varlen_metadata"))
def test_cake_ssd_combined_program_cache_is_multi_device_safe(varlen):
    if any(
        torch.cuda.get_device_capability(index) not in ((10, 0), (10, 3))
        for index in (0, 1)
    ):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    runners = []
    cases = []
    expected = []
    for device_index in (0, 1):
        with torch.cuda.device(device_index):
            constructor, tensors, arguments = _case(
                nheads=1,
                ngroups=1,
                varlen=varlen,
            )
            expected.append(
                SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
            )
            runners.append(SSDCombined(**constructor, backend="cake"))
            cases.append((tensors, arguments))

    torch.cuda.set_device(0)
    for device_index in (0, 1, 0, 1):
        assert torch.cuda.current_device() == 0
        tensors, arguments = cases[device_index]
        actual = runners[device_index].run(*tensors, **arguments)
        assert actual[0].device.index == device_index
        _assert_cute_parity(actual, expected[device_index])
        assert torch.cuda.current_device() == 0


def test_cake_ssd_combined_public_seq_chunk_cumsum_helpers():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, _, arguments = _case(varlen=True)
    runner = SSDCombined(**constructor, backend="cake")
    seq_idx = arguments["seq_idx"]
    chunk_indices = arguments["chunk_indices"]
    chunk_offsets = arguments["chunk_offsets"]
    expected = arguments["seq_chunk_cumsum"]
    actual = torch.full_like(expected, -1)
    tile_state_bytes = runner.tile_state_size(2)
    tile_state = (
        torch.empty(tile_state_bytes, dtype=torch.uint8, device="cuda")
        if tile_state_bytes
        else None
    )

    returned = runner.compute_seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        128,
        2,
        seq_chunk_cumsum=actual,
        tile_state=tile_state,
    )

    assert returned is actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    allocated = runner.compute_seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        128,
        2,
        seq_chunk_cumsum=None,
        tile_state=None,
    )
    torch.testing.assert_close(allocated, expected, rtol=0, atol=0)

    num_seqs = 2048
    multiblock_seq_idx = (
        torch.arange(num_seqs, dtype=torch.int32, device="cuda")
        .repeat_interleave(128)
        .unsqueeze(0)
    )
    multiblock_chunk_indices = torch.arange(num_seqs, dtype=torch.int32, device="cuda")
    multiblock_chunk_offsets = torch.zeros(num_seqs, dtype=torch.int32, device="cuda")
    multiblock_expected = torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda")
    multiblock_actual = torch.full_like(multiblock_expected, -1)
    multiblock_tile_state_bytes = runner.tile_state_size(num_seqs)
    assert multiblock_tile_state_bytes > 0
    multiblock_tile_state = torch.empty(
        multiblock_tile_state_bytes, dtype=torch.uint8, device="cuda"
    )

    multiblock_returned = runner.compute_seq_chunk_cumsum(
        multiblock_seq_idx,
        multiblock_chunk_indices,
        multiblock_chunk_offsets,
        128,
        num_seqs,
        seq_chunk_cumsum=multiblock_actual,
        tile_state=multiblock_tile_state,
    )

    assert multiblock_returned is multiblock_actual
    torch.testing.assert_close(multiblock_actual, multiblock_expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["a_dtype", "out_shape"])
def test_cake_ssd_combined_rejects_invalid_public_inputs_like_cute(invalid):
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case()
    if invalid == "a_dtype":
        tensors = (*tensors[:2], tensors[2].to(torch.bfloat16), *tensors[3:])
    else:
        arguments = {
            **arguments,
            "out": torch.empty((1,), dtype=torch.bfloat16, device="cuda"),
        }

    errors = {}
    for backend in ("cute", "cake"):
        runner = SSDCombined(**constructor, backend=backend)
        with pytest.raises(AssertionError) as exc_info:
            runner.run(*tensors, **arguments)
        errors[backend] = (type(exc_info.value), str(exc_info.value))

    if invalid == "a_dtype":
        assert errors["cake"] == errors["cute"]
    else:
        # Same exception and message form; each backend names its own kernel
        # output layout (CuTe chunked, Cake token-major).
        assert "out shape torch.Size([1]) doesn't match expected" in errors["cute"][1]
        assert "out shape torch.Size([1]) doesn't match expected" in errors["cake"][1]
        assert errors["cute"][1].endswith("(2, 8, 64, 1, 128)")
        assert errors["cake"][1].endswith("(2, 128, 8, 64)")


def test_ssd_combined_fwd_caches_by_device_stream_and_config(monkeypatch, request):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    runners = []
    active_stream = {"handle": 0x1000}

    class Runner:
        def __init__(self, *args, **kwargs):
            self.constructor_args = args
            self.constructor_kwargs = kwargs
            self.run_calls = []
            runners.append(self)

        def run(self, *args, **kwargs):
            self.run_calls.append((args, kwargs))
            return (self, None)

    module._get_ssd_combined_runner.cache_clear()
    request.addfinalizer(module._get_ssd_combined_runner.cache_clear)
    monkeypatch.setattr(module, "SSDCombined", Runner)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=active_stream["handle"]),
    )
    x = SimpleNamespace(
        shape=(1, 128, 8, 64),
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
    )
    B = SimpleNamespace(shape=(1, 128, 8, 128))
    checkpoint_states = SimpleNamespace(dtype=torch.float16)
    D = SimpleNamespace(ndim=2)
    initial_states = SimpleNamespace(dtype=torch.float16)
    seq_idx = SimpleNamespace(dtype=torch.int32)
    optional = {
        "D": D,
        "z": object(),
        "dt_bias": object(),
        "dt_softplus": True,
        "dt_limit": (-0.5, 0.75),
        "initial_states": initial_states,
        "seq_idx": seq_idx,
        "chunk_indices": object(),
        "chunk_offsets": object(),
        "seq_chunk_cumsum": object(),
        "update_seq_chunk_cumsum": True,
        "checkpoint_token_indices": object(),
        "checkpoint_state_slots": object(),
        "checkpoint_states": checkpoint_states,
        "out": object(),
        "return_final_states": False,
        "num_seqs": 3,
    }
    positional = (x, object(), object(), B, object())

    first = module.ssd_combined_fwd(
        *positional,
        **optional,
    )
    repeated = module.ssd_combined_fwd(*positional, **optional)

    active_stream["handle"] = 0x2000
    different_stream = module.ssd_combined_fwd(*positional, **optional)

    active_stream["handle"] = 0x1000
    x_device_one = SimpleNamespace(
        shape=x.shape,
        dtype=x.dtype,
        device=torch.device("cuda:1"),
    )
    different_device = module.ssd_combined_fwd(
        x_device_one,
        *positional[1:],
        **optional,
    )
    second = module.ssd_combined_fwd(
        x,
        object(),
        object(),
        B,
        object(),
        checkpoint_states=checkpoint_states,
    )

    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0] is repeated[0]
    assert first[0] is not different_stream[0]
    assert first[0] is not different_device[0]
    assert first[0] is not second[0]
    assert len(runners) == 4
    assert all(runner.constructor_kwargs["backend"] == "cake" for runner in runners)
    assert all(
        runner.constructor_kwargs["state_dtype"] == torch.float16 for runner in runners
    )
    assert runners[0].constructor_args == (128, 8, 64, 128, 8)
    assert runners[0].constructor_kwargs == {
        "io_dtype": torch.bfloat16,
        "state_dtype": torch.float16,
        "has_d": True,
        "d_has_hdim": True,
        "has_initial_states": True,
        "has_varlen": True,
        "has_z": True,
        "seq_idx_dtype": torch.int32,
        "backend": "cake",
    }
    assert runners[0].run_calls == [(positional, optional), (positional, optional)]
    assert runners[-1].run_calls[0][1]["dt_softplus"] is False


def _signature_contract(callable_, *, drop_self=False):
    parameters = tuple(inspect.signature(callable_).parameters.values())
    if drop_self:
        assert parameters[0].name == "self"
        parameters = parameters[1:]
    return tuple(
        (parameter.name, parameter.kind, parameter.default) for parameter in parameters
    )


def test_source_public_api_signatures_are_stable():
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    cake_module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    empty = inspect.Parameter.empty

    constructor_names = (
        "chunk_size",
        "nheads",
        "headdim",
        "dstate",
        "ngroups",
        "io_dtype",
        "state_dtype",
        "has_d",
        "d_has_hdim",
        "has_initial_states",
        "has_varlen",
        "has_z",
        "seq_idx_dtype",
        "backend",
    )
    constructor_defaults = (
        empty,
        empty,
        empty,
        empty,
        empty,
        torch.bfloat16,
        torch.bfloat16,
        True,
        False,
        False,
        False,
        False,
        torch.int64,
        "cute",
    )
    assert _signature_contract(module.SSDCombined) == tuple(
        zip(
            constructor_names,
            (positional,) * len(constructor_names),
            constructor_defaults,
            strict=True,
        )
    )

    run_names = (
        "x",
        "dt",
        "A",
        "B",
        "C",
        "D",
        "z",
        "dt_bias",
        "dt_softplus",
        "dt_limit",
        "initial_states",
        "seq_idx",
        "chunk_indices",
        "chunk_offsets",
        "seq_chunk_cumsum",
        "update_seq_chunk_cumsum",
        "checkpoint_token_indices",
        "checkpoint_state_slots",
        "checkpoint_states",
        "out",
        "return_final_states",
        "num_seqs",
    )
    run_defaults = (
        empty,
        empty,
        empty,
        empty,
        empty,
        None,
        None,
        None,
        False,
        (0.0, float("inf")),
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        None,
        None,
        True,
        None,
    )
    expected_run = tuple(
        zip(
            run_names,
            (positional,) * len(run_names),
            run_defaults,
            strict=True,
        )
    )
    assert _signature_contract(module.SSDCombined.run, drop_self=True) == expected_run
    assert (
        _signature_contract(cake_module.CakeSSDCombined.run, drop_self=True)
        == expected_run
    )
    assert _signature_contract(module.ssd_combined_fwd) == expected_run

    helper_names = (
        "seq_idx",
        "chunk_indices",
        "chunk_offsets",
        "chunk_size",
        "num_seqs",
        "seq_chunk_cumsum",
        "tile_state",
    )
    helper_defaults = (empty, empty, empty, empty, empty, None, None)
    assert _signature_contract(
        module.SSDCombined.compute_seq_chunk_cumsum, drop_self=True
    ) == tuple(
        zip(
            helper_names,
            (positional,) * len(helper_names),
            helper_defaults,
            strict=True,
        )
    )
    assert tuple(inspect.signature(module.SSDCombined.tile_state_size).parameters) == (
        "num_seqs",
    )


def test_source_public_constructor_forwards_complete_cake_contract(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    cake_module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    calls = []

    class CakeRunner:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (10, 3))
    monkeypatch.setattr(cake_module, "CakeSSDCombined", CakeRunner)
    runner = module.SSDCombined(
        128,
        128,
        64,
        128,
        8,
        io_dtype=torch.bfloat16,
        state_dtype=torch.float16,
        has_d=False,
        d_has_hdim=True,
        has_initial_states=True,
        has_varlen=True,
        has_z=True,
        seq_idx_dtype=torch.int32,
        backend="cake",
    )

    assert calls == [
        (
            (128, 128, 64, 128, 8),
            {
                "io_dtype": torch.bfloat16,
                "state_dtype": torch.float16,
                "has_d": False,
                "d_has_hdim": True,
                "has_initial_states": True,
                "has_varlen": True,
                "has_z": True,
                "seq_idx_dtype": torch.int32,
            },
        )
    ]
    assert runner._backend == "cake"
    assert runner._cake_runner.__class__ is CakeRunner

    with pytest.raises(ValueError, match="backend must be 'cute' or 'cake'"):
        module.SSDCombined(128, 8, 64, 128, 8, backend="unknown")


def test_source_public_constructor_hardware_error_parity_without_gpu(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (12, 0))
    errors = {}

    for backend in ("cute", "cake"):
        with pytest.raises(ValueError) as exc_info:
            module.SSDCombined(128, 2, 64, 128, 1, backend=backend)
        errors[backend] = (type(exc_info.value), str(exc_info.value))

    assert errors["cake"] == errors["cute"]


def test_source_public_cake_constructor_rejects_non_exported_arch_without_gpu(
    monkeypatch,
):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (11, 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (11, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cake_module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    cake_module._target_arch.cache_clear()
    try:
        with pytest.raises(ValueError, match="requires SM100 or SM103, got SM110"):
            module.SSDCombined(128, 2, 64, 128, 1, backend="cake")
    finally:
        cake_module._target_arch.cache_clear()


@pytest.mark.parametrize(
    "backend,invalid,exception,match",
    (
        ("cute", "io_dtype", AssertionError, "io_dtype must be bfloat16"),
        ("cute", "state_dtype", AssertionError, "state_dtype must be one of"),
        ("cake", "chunk_size", ValueError, "requires chunk_size=128"),
        ("cake", "headdim", ValueError, "requires chunk_size=128"),
        ("cake", "dstate", ValueError, "requires chunk_size=128"),
        ("cake", "nheads", ValueError, "requires positive nheads"),
        ("cake", "ngroups", ValueError, "requires positive nheads"),
        ("cake", "head_group_ratio", ValueError, "requires positive nheads"),
        ("cake", "io_dtype", ValueError, "requires bfloat16 IO"),
        ("cake", "state_dtype", ValueError, "state dtype must be"),
        ("cake", "seq_idx_dtype", ValueError, "seq_idx dtype must be"),
    ),
)
def test_source_public_backend_constructor_validation_without_gpu(
    monkeypatch, backend, invalid, exception, match
):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    utils = importlib.import_module("flashinfer.utils")
    monkeypatch.setattr(utils, "get_compute_capability", lambda *_: (10, 3))
    constructor = {
        "chunk_size": 128,
        "nheads": 2,
        "headdim": 64,
        "dstate": 128,
        "ngroups": 1,
        "io_dtype": torch.bfloat16,
        "state_dtype": torch.bfloat16,
        "seq_idx_dtype": torch.int32,
        "backend": backend,
    }
    replacements = {
        "chunk_size": 64,
        "headdim": 32,
        "dstate": 64,
        "nheads": 0,
        "ngroups": 0,
        "io_dtype": torch.float16,
        # float32 states are a Cake-only feature; float64 is invalid for both.
        "state_dtype": torch.float64,
        "seq_idx_dtype": torch.float32,
    }
    if invalid == "head_group_ratio":
        constructor.update(nheads=3, ngroups=2)
    else:
        constructor[invalid] = replacements[invalid]

    with pytest.raises(exception, match=match):
        module.SSDCombined(**constructor)


def _public_runner_without_constructor(backend, cake_result=None):
    runner = object.__new__(SSDCombined)
    runner.chunk_size = 128
    runner._backend = backend
    runner._io_torch_dtype = torch.bfloat16
    runner._state_torch_dtype = torch.bfloat16
    runner._has_d = False
    runner._has_init_states = False

    class CakeRunner:
        def __init__(self):
            self.calls = []

        def run(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return cake_result

    runner._cake_runner = CakeRunner()
    return runner


def _cpu_public_run_inputs(batch=1):
    x = torch.empty((batch, 128, 2, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, 128, 2), dtype=torch.float32)
    A = torch.empty((2,), dtype=torch.float32)
    B = torch.empty((batch, 128, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    return x, dt, A, B, C


def test_source_public_cake_dispatch_preserves_full_run_contract():
    result = (object(), None)
    runner = _public_runner_without_constructor("cake", result)
    tensors = _cpu_public_run_inputs()
    sentinels = {
        "D": torch.empty(2, dtype=torch.bfloat16),
        "z": torch.empty((1, 128, 2, 64), dtype=torch.bfloat16),
        "dt_bias": object(),
        "initial_states": torch.empty((1, 2, 64, 128), dtype=torch.bfloat16),
        "seq_idx": torch.zeros((1, 128), dtype=torch.int32),
        "chunk_indices": torch.zeros(1, dtype=torch.int32),
        "chunk_offsets": torch.zeros(1, dtype=torch.int32),
        "seq_chunk_cumsum": object(),
        "checkpoint_token_indices": object(),
        "checkpoint_state_slots": object(),
        "checkpoint_states": object(),
    }
    out = torch.empty((1, 128, 2, 64), dtype=torch.bfloat16)
    kwargs = {
        **sentinels,
        "dt_softplus": True,
        "dt_limit": (-0.25, 0.75),
        "update_seq_chunk_cumsum": True,
        "out": out,
        "return_final_states": False,
        "num_seqs": 1,
    }

    actual = runner.run(*tensors, **kwargs)

    assert actual is result
    assert runner._cake_runner.calls == [(tensors, kwargs)]


@pytest.mark.parametrize(
    "relaxed", ("unaligned_seqlen", "varlen_without_initial_states")
)
def test_source_public_cake_dispatch_relaxed_domain_without_gpu(relaxed):
    """The Cake-only relaxations pass the public pre-dispatch validation."""

    result = (object(), None)
    runner = _public_runner_without_constructor("cake", result)
    x, dt, A, B, C = _cpu_public_run_inputs()
    kwargs = {}
    if relaxed == "unaligned_seqlen":
        tensors = (x[:, :100], dt[:, :100], A, B[:, :100], C[:, :100])
    else:
        tensors = (x, dt, A, B, C)
        kwargs = {
            "seq_idx": torch.zeros((1, 128), dtype=torch.int32),
            "chunk_indices": torch.zeros(1, dtype=torch.int32),
            "chunk_offsets": torch.zeros(1, dtype=torch.int32),
            "num_seqs": 1,
        }

    actual = runner.run(*tensors, **kwargs)

    assert actual is result
    assert len(runner._cake_runner.calls) == 1
    assert runner._cake_runner.calls[0][0] == tensors
    for name, value in kwargs.items():
        assert runner._cake_runner.calls[0][1][name] is value


@pytest.mark.parametrize("backend", ("cute", "cake"))
@pytest.mark.parametrize("invalid", ("shape", "dtype", "contiguous"))
def test_source_public_out_contract_per_backend_without_gpu(backend, invalid):
    """``out`` is validated against the backend's kernel layout: CuTe chunked
    ``[B, EH, D, C, L]``, Cake token-major ``[B, S, EH, D]``."""

    runner = _public_runner_without_constructor(backend)
    tensors = _cpu_public_run_inputs()
    expected = (1, 128, 2, 64) if backend == "cake" else (1, 2, 64, 1, 128)
    layout = "(B, S, EH, D)" if backend == "cake" else "(B, EH, D, C, L)"
    if invalid == "shape":
        out = torch.empty((1,), dtype=torch.bfloat16)
        match = re.escape(
            f"out shape torch.Size([1]) doesn't match expected {expected}"
        )
    elif invalid == "dtype":
        out = torch.empty(expected, dtype=torch.float16)
        match = "out dtype torch.float16 doesn't match x dtype torch.bfloat16"
    else:
        storage = torch.empty((*expected[:-1], expected[-1] + 1), dtype=torch.bfloat16)
        out = storage[..., : expected[-1]]
        assert not out.is_contiguous()
        match = rf"out must be contiguous in {re.escape(layout)} layout"

    with pytest.raises(AssertionError, match=match):
        runner.run(*tensors, out=out)
    assert runner._cake_runner.calls == []


@pytest.mark.parametrize(
    "invalid,exception",
    (
        ("x_rank", ValueError),
        ("a_dtype", AssertionError),
        ("x_dtype", AssertionError),
        ("b_dtype", AssertionError),
        ("c_dtype", AssertionError),
        ("d_dtype", AssertionError),
        ("z_dtype", AssertionError),
        ("initial_dtype", AssertionError),
        ("seq_idx_shape", AssertionError),
        ("seq_idx_dtype", AssertionError),
        ("chunk_indices_ndim", AssertionError),
        ("chunk_indices_dtype", AssertionError),
        ("chunk_offsets_ndim", AssertionError),
        ("chunk_offsets_dtype", AssertionError),
        ("chunk_vector_shape", AssertionError),
    ),
)
def test_source_public_shared_validation_error_parity_without_gpu(invalid, exception):
    tensors = _cpu_public_run_inputs()
    kwargs = {}
    if invalid == "x_rank":
        tensors = (torch.empty((128, 2, 64), dtype=torch.bfloat16), *tensors[1:])
    elif invalid == "a_dtype":
        tensors = (*tensors[:2], tensors[2].to(torch.bfloat16), *tensors[3:])
    elif invalid in {"x_dtype", "b_dtype", "c_dtype"}:
        tensor_index = {"x_dtype": 0, "b_dtype": 3, "c_dtype": 4}[invalid]
        tensors = (
            *tensors[:tensor_index],
            tensors[tensor_index].to(torch.float16),
            *tensors[tensor_index + 1 :],
        )
    elif invalid == "d_dtype":
        kwargs["D"] = torch.empty(2, dtype=torch.float16)
    elif invalid == "z_dtype":
        kwargs["z"] = torch.empty_like(tensors[0], dtype=torch.float16)
    elif invalid == "initial_dtype":
        kwargs["initial_states"] = torch.empty((1, 2, 64, 128), dtype=torch.float16)
    else:
        seq_idx = torch.empty((1, 128), dtype=torch.int32)
        chunk_indices = torch.zeros(1, dtype=torch.int32)
        chunk_offsets = torch.zeros(1, dtype=torch.int32)
        if invalid == "seq_idx_shape":
            seq_idx = torch.empty((2, 128), dtype=torch.int32)
        elif invalid == "seq_idx_dtype":
            seq_idx = torch.empty((1, 128), dtype=torch.float32)
        elif invalid == "chunk_indices_ndim":
            chunk_indices = torch.zeros((1, 1), dtype=torch.int32)
        elif invalid == "chunk_indices_dtype":
            chunk_indices = torch.zeros(1, dtype=torch.int64)
        elif invalid == "chunk_offsets_ndim":
            chunk_offsets = torch.zeros((1, 1), dtype=torch.int32)
        elif invalid == "chunk_offsets_dtype":
            chunk_offsets = torch.zeros(1, dtype=torch.int64)
        elif invalid == "chunk_vector_shape":
            chunk_offsets = torch.zeros(2, dtype=torch.int32)
        kwargs.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_states=torch.empty((1, 2, 64, 128), dtype=torch.bfloat16),
        )

    errors = {}
    for backend in ("cute", "cake"):
        runner = _public_runner_without_constructor(backend)
        runner._has_d = invalid == "d_dtype"
        runner._has_init_states = invalid == "initial_dtype"
        with pytest.raises(exception) as exc_info:
            runner.run(*tensors, **kwargs)
        errors[backend] = (type(exc_info.value), str(exc_info.value))
        assert runner._cake_runner.calls == []

    assert errors["cake"] == errors["cute"]


def test_source_public_seq_cumsum_helper_contract_without_gpu(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    calls = []

    class SeqCumsumModule:
        @staticmethod
        def seq_chunk_cumsum_tile_state_size(num_seqs):
            calls.append(("tile_state_size", num_seqs))
            return 19

        @staticmethod
        def seq_chunk_cumsum(*args):
            calls.append(("seq_chunk_cumsum", args))

    seq_module = SeqCumsumModule()
    monkeypatch.setattr(module, "_get_seq_chunk_cumsum_module", lambda: seq_module)
    runner = object.__new__(module.SSDCombined)
    runner._seq_cumsum_key = None
    runner._seq_cumsum_buf = None
    seq_idx = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    chunk_indices = torch.tensor([0, 0], dtype=torch.int32)
    chunk_offsets = torch.tensor([0, 2], dtype=torch.int32)
    output = torch.full((3,), -1, dtype=torch.int32)
    tile_state = torch.empty(19, dtype=torch.uint8)

    returned = runner.compute_seq_chunk_cumsum(
        seq_idx,
        chunk_indices,
        chunk_offsets,
        128,
        2,
        seq_chunk_cumsum=output,
        tile_state=tile_state,
    )

    assert returned is output
    assert calls == [
        (
            "seq_chunk_cumsum",
            (
                seq_idx,
                chunk_indices,
                chunk_offsets,
                output,
                tile_state,
                128,
                2,
                2,
            ),
        )
    ]
    assert runner.tile_state_size(7) == 19
    assert calls[-1] == ("tile_state_size", 7)


def _source_cake_runner_without_constructor():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    runner = object.__new__(module.CakeSSDCombined)
    runner.nheads = 2
    runner.ngroups = 1
    runner.state_dtype = torch.bfloat16
    runner.has_d = False
    runner.d_has_hdim = False
    runner.has_initial_states = False
    runner.has_varlen = False
    runner.has_z = False
    runner.seq_idx_dtype = torch.int32
    return runner


def _source_cake_varlen_arguments(runner, tensors):
    batch, seqlen = tensors[0].shape[:2]
    runner.has_initial_states = True
    runner.has_varlen = True
    return {
        "initial_states": torch.empty(
            (batch, runner.nheads, 64, 128), dtype=runner.state_dtype
        ),
        "seq_idx": torch.empty((batch, seqlen), dtype=runner.seq_idx_dtype),
        "chunk_indices": torch.arange(batch, dtype=torch.int32),
        "chunk_offsets": torch.zeros(batch, dtype=torch.int32),
        "seq_chunk_cumsum": torch.arange(batch + 1, dtype=torch.int32),
    }


# Validation below dispatch is backend-specific: lock each backend's complete
# rejection surface separately while the pre-dispatch test above enforces exact
# exception-type/message parity for the shared public contract.
@pytest.mark.parametrize(
    "invalid,match",
    (
        ("x_shape", "x must have shape"),
        ("b_shape", "B must have shape"),
        ("c_shape", "C must have the same shape as B"),
        ("x_dtype", "x, B, and C must be bfloat16"),
        ("b_dtype", "x, B, and C must be bfloat16"),
        ("c_dtype", "x, B, and C must be bfloat16"),
        ("dt_shape", "dt must have shape"),
        ("a_shape", "A must have shape"),
        ("dt_dtype", "dt must be bfloat16 or float32"),
        ("d_presence", "runtime D/z presence must match"),
        ("z_presence", "runtime D/z presence must match"),
        ("initial_presence", "runtime initial_states presence must match"),
        ("varlen_metadata", "varlen mode requires seq_idx"),
        ("batched_metadata", "batched mode does not accept varlen metadata"),
        ("batched_cumsum", "batched mode does not accept varlen metadata"),
        ("batched_num_seqs", "batched mode does not accept varlen metadata"),
        (
            "varlen_sequence_count",
            "varlen mode without initial_states requires seq_chunk_cumsum or num_seqs",
        ),
        ("num_seqs_conflict", r"num_seqs \(3\) does not match the sequence count"),
        ("initial_dtype", "initial_states dtype must match state_dtype"),
        ("out_shape", "out must have shape"),
        ("out_dtype", "out must have shape"),
        ("out_contiguous", "out must be contiguous"),
        ("d_shape", "D must have shape"),
        ("d_dtype", "D must have shape"),
        ("z_shape", "z must have the same shape and dtype as x"),
        ("z_dtype", "z must have the same shape and dtype as x"),
        ("initial_shape", "initial_states must have shape"),
        ("seq_idx_shape", "seq_idx shape or dtype"),
        ("seq_idx_dtype", "seq_idx shape or dtype"),
        ("chunk_indices_dtype", "matching int32 vectors"),
        ("chunk_offsets_dtype", "matching int32 vectors"),
        ("chunk_indices_ndim", "matching int32 vectors"),
        ("chunk_vector_shape", "matching int32 vectors"),
        ("seq_cumsum_shape", "seq_chunk_cumsum shape or dtype"),
        ("seq_cumsum_dtype", "seq_chunk_cumsum shape or dtype"),
    ),
)
def test_source_public_cake_domain_validation_without_gpu(invalid, match):
    cake_runner = _source_cake_runner_without_constructor()
    tensors = list(_cpu_public_run_inputs(batch=2))
    kwargs = {}

    if invalid == "x_shape":
        tensors[0] = torch.empty((2, 128, 3, 64), dtype=torch.bfloat16)
    elif invalid == "b_shape":
        tensors[3] = torch.empty((2, 128, 2, 128), dtype=torch.bfloat16)
    elif invalid == "c_shape":
        tensors[4] = torch.empty((2, 128, 1, 127), dtype=torch.bfloat16)
    elif invalid in {"x_dtype", "b_dtype", "c_dtype"}:
        tensor_index = {"x_dtype": 0, "b_dtype": 3, "c_dtype": 4}[invalid]
        tensors[tensor_index] = tensors[tensor_index].to(torch.float16)
    elif invalid == "dt_shape":
        tensors[1] = torch.empty((2, 128, 3), dtype=torch.float32)
    elif invalid == "a_shape":
        tensors[2] = torch.empty((3,), dtype=torch.float32)
    elif invalid == "dt_dtype":
        tensors[1] = tensors[1].to(torch.float16)
    elif invalid == "d_presence":
        cake_runner.has_d = True
    elif invalid == "z_presence":
        cake_runner.has_z = True
    elif invalid == "initial_presence":
        cake_runner.has_initial_states = True
    elif invalid == "varlen_metadata":
        cake_runner.has_initial_states = True
        cake_runner.has_varlen = True
        kwargs["initial_states"] = torch.empty((2, 2, 64, 128), dtype=torch.bfloat16)
    elif invalid == "batched_metadata":
        kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.int32)
    elif invalid == "batched_cumsum":
        kwargs["seq_chunk_cumsum"] = torch.empty(3, dtype=torch.int32)
    elif invalid == "batched_num_seqs":
        kwargs["num_seqs"] = 2
    elif invalid == "varlen_sequence_count":
        cake_runner.has_varlen = True
        kwargs.update(
            seq_idx=torch.empty((2, 128), dtype=torch.int32),
            chunk_indices=torch.arange(2, dtype=torch.int32),
            chunk_offsets=torch.zeros(2, dtype=torch.int32),
        )
    elif invalid == "num_seqs_conflict":
        kwargs.update(_source_cake_varlen_arguments(cake_runner, tensors))
        kwargs["num_seqs"] = 3
    elif invalid == "out_shape":
        kwargs["out"] = torch.empty((2, 2, 64, 1, 128), dtype=torch.bfloat16)
    elif invalid == "out_dtype":
        kwargs["out"] = torch.empty((2, 128, 2, 64), dtype=torch.float16)
    elif invalid == "out_contiguous":
        kwargs["out"] = torch.empty((2, 128, 2, 65), dtype=torch.bfloat16)[..., :64]
    elif invalid == "initial_dtype":
        cake_runner.has_initial_states = True
        kwargs["initial_states"] = torch.empty((2, 2, 64, 128), dtype=torch.float16)
    elif invalid in {"d_shape", "d_dtype"}:
        cake_runner.has_d = True
        kwargs["D"] = torch.empty(
            (2, 63),
            dtype=torch.bfloat16 if invalid == "d_shape" else torch.float16,
        )
        if invalid == "d_dtype":
            kwargs["D"] = torch.empty(2, dtype=torch.float16)
    elif invalid in {"z_shape", "z_dtype"}:
        cake_runner.has_z = True
        kwargs["z"] = torch.empty(
            (2, 127, 2, 64) if invalid == "z_shape" else tensors[0].shape,
            dtype=torch.bfloat16 if invalid == "z_shape" else torch.float16,
        )
    else:
        kwargs.update(_source_cake_varlen_arguments(cake_runner, tensors))
        if invalid == "initial_shape":
            kwargs["initial_states"] = torch.empty(
                (2, 2, 64, 127), dtype=torch.bfloat16
            )
        elif invalid == "seq_idx_shape":
            kwargs["seq_idx"] = torch.empty((1, 128), dtype=torch.int32)
        elif invalid == "seq_idx_dtype":
            kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.int64)
        elif invalid == "chunk_indices_dtype":
            kwargs["chunk_indices"] = torch.arange(2, dtype=torch.int64)
        elif invalid == "chunk_offsets_dtype":
            kwargs["chunk_offsets"] = torch.zeros(2, dtype=torch.int64)
        elif invalid == "chunk_indices_ndim":
            kwargs["chunk_indices"] = torch.zeros((1, 2), dtype=torch.int32)
        elif invalid == "chunk_vector_shape":
            kwargs["chunk_offsets"] = torch.zeros(3, dtype=torch.int32)
        elif invalid == "seq_cumsum_shape":
            kwargs["seq_chunk_cumsum"] = torch.empty(2, dtype=torch.int32)
        elif invalid == "seq_cumsum_dtype":
            kwargs["seq_chunk_cumsum"] = torch.empty(3, dtype=torch.int64)

    with pytest.raises(ValueError, match=match):
        cake_runner.run(*tensors, **kwargs)


def _source_cute_runner_without_constructor():
    runner = _public_runner_without_constructor("cute")
    runner._io_torch_dtype = torch.bfloat16
    runner._cumsum_dtype = object()
    runner._state_torch_dtype = torch.bfloat16
    runner._has_d = False
    runner._d_has_hdim = False
    runner._has_init_states = False
    runner._has_varlen = False
    runner._has_z = False
    runner._get_or_alloc_fstate = lambda batch: torch.empty(
        (batch, 2, 64, 128), dtype=runner._state_torch_dtype
    )
    return runner


@pytest.mark.parametrize(
    "invalid,exception,match",
    (
        ("checkpoint", ValueError, "require SSDCombined backend='cake'"),
        ("seq_idx_shape", AssertionError, "seq_idx shape"),
        ("seq_idx_dtype", AssertionError, "seq_idx must be int32 or int64"),
        ("chunk_indices_ndim", AssertionError, "chunk_indices must be 1D"),
        ("chunk_indices_dtype", AssertionError, "chunk_indices must be int32"),
        ("chunk_offsets_ndim", AssertionError, "chunk_offsets must be 1D"),
        ("chunk_offsets_dtype", AssertionError, "chunk_offsets must be int32"),
        ("chunk_vector_shape", AssertionError, "must have the same shape"),
        ("x_dtype", AssertionError, "x dtype"),
        ("b_dtype", AssertionError, "B dtype"),
        ("c_dtype", AssertionError, "C dtype"),
        ("d_dtype", AssertionError, "D dtype"),
        ("z_dtype", AssertionError, "z dtype"),
        ("initial_dtype", AssertionError, "init_states dtype"),
        ("varlen_initial", ValueError, "initial_states must be provided"),
        ("seqlen", AssertionError, "must be divisible by chunk_size"),
        ("num_seqs", ValueError, "num_seqs requires SSDCombined backend='cake'"),
    ),
)
def test_source_public_cute_backend_validation_without_gpu(
    monkeypatch, invalid, exception, match
):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")

    def chunk_cumsum(dt, _a, chunk_size, **_kwargs):
        batch, seqlen, nheads = dt.shape
        shape = (batch, nheads, seqlen // chunk_size, chunk_size)
        return torch.empty(shape, dtype=torch.float32), torch.empty(
            shape, dtype=torch.bfloat16
        )

    monkeypatch.setattr(module, "chunk_cumsum_fwd", chunk_cumsum)
    monkeypatch.setattr(module.cutlass_torch, "dtype", lambda _: torch.float32)
    runner = _source_cute_runner_without_constructor()
    tensors = list(_cpu_public_run_inputs(batch=2))
    kwargs = {}
    seq_idx = torch.empty((2, 128), dtype=torch.int32)
    chunk_indices = torch.arange(2, dtype=torch.int32)
    chunk_offsets = torch.zeros(2, dtype=torch.int32)

    if invalid == "checkpoint":
        kwargs.update(
            checkpoint_token_indices=torch.zeros(2, dtype=torch.int32),
            checkpoint_state_slots=torch.zeros(2, dtype=torch.int32),
            checkpoint_states=torch.empty((1, 2, 64, 128), dtype=torch.bfloat16),
        )
    elif invalid == "seq_idx_shape":
        kwargs["seq_idx"] = torch.empty((1, 128), dtype=torch.int32)
    elif invalid == "seq_idx_dtype":
        kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.float32)
    elif invalid == "chunk_indices_ndim":
        kwargs["chunk_indices"] = torch.empty((1, 2), dtype=torch.int32)
    elif invalid == "chunk_indices_dtype":
        kwargs["chunk_indices"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "chunk_offsets_ndim":
        kwargs["chunk_offsets"] = torch.empty((1, 2), dtype=torch.int32)
    elif invalid == "chunk_offsets_dtype":
        kwargs["chunk_offsets"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "chunk_vector_shape":
        kwargs.update(
            chunk_indices=chunk_indices,
            chunk_offsets=torch.empty(3, dtype=torch.int32),
        )
    elif invalid in {"x_dtype", "b_dtype", "c_dtype"}:
        tensor_index = {"x_dtype": 0, "b_dtype": 3, "c_dtype": 4}[invalid]
        tensors[tensor_index] = tensors[tensor_index].to(torch.float16)
    elif invalid == "d_dtype":
        runner._has_d = True
        kwargs["D"] = torch.empty(2, dtype=torch.float16)
    elif invalid == "z_dtype":
        kwargs["z"] = torch.empty_like(tensors[0], dtype=torch.float16)
    elif invalid == "initial_dtype":
        runner._has_init_states = True
        kwargs["initial_states"] = torch.empty((2, 2, 64, 128), dtype=torch.float16)
    elif invalid == "varlen_initial":
        kwargs.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )
    elif invalid == "seqlen":
        x, dt, A, B, C = tensors
        tensors = [x[:, :-1], dt[:, :-1], A, B[:, :-1], C[:, :-1]]
    elif invalid == "num_seqs":
        kwargs["num_seqs"] = 2

    with pytest.raises(exception, match=match):
        runner.run(*tensors, **kwargs)


@pytest.mark.parametrize("invalid", ("dt_bias_shape", "dt_bias_dtype"))
def test_source_public_cake_dt_bias_validation_without_gpu(monkeypatch, invalid):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    cake_runner = _source_cake_runner_without_constructor()
    cake_runner._get_workspace = lambda **_: {
        "final": torch.empty((2, 2, 64, 128), dtype=torch.bfloat16)
    }
    cake_runner._dummy = lambda device, dtype: torch.empty(
        1, dtype=dtype, device=device
    )
    runner = _public_runner_without_constructor("cake")
    runner._cake_runner = cake_runner
    tensors = _cpu_public_run_inputs(batch=2)
    dt_bias = torch.empty(
        3 if invalid == "dt_bias_shape" else 2,
        dtype=torch.float32 if invalid == "dt_bias_shape" else torch.float16,
    )

    with pytest.raises(ValueError, match="dt_bias must have shape"):
        runner.run(*tensors, dt_bias=dt_bias)


def test_source_public_cake_rejects_non_cuda_inputs_without_gpu(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    cake_runner = _source_cake_runner_without_constructor()
    cake_runner._get_workspace = lambda **_: {
        "final": torch.empty((1, 2, 64, 128), dtype=torch.bfloat16)
    }
    runner = _public_runner_without_constructor("cake")
    runner._cake_runner = cake_runner

    with pytest.raises(ValueError, match="inputs must be on a CUDA device"):
        runner.run(*_cpu_public_run_inputs())


@pytest.mark.parametrize(
    "invalid,match",
    (
        ("partial", "must be provided together"),
        ("token_shape", "checkpoint_token_indices must be"),
        ("token_dtype", "checkpoint_token_indices must be"),
        ("token_contiguous", "checkpoint_token_indices must be"),
        ("slot_shape", "checkpoint_state_slots must be"),
        ("slot_dtype", "checkpoint_state_slots must be"),
        ("slot_contiguous", "checkpoint_state_slots must be"),
        ("state_shape", "checkpoint_states must be contiguous"),
        ("state_dtype", "checkpoint_states must be contiguous"),
        ("state_contiguous", "checkpoint_states must be contiguous"),
    ),
)
def test_source_cake_checkpoint_validation_without_gpu(invalid, match):
    runner = _source_cake_runner_without_constructor()
    tensors = _cpu_public_run_inputs(batch=2)
    token_storage = torch.tensor([16, -1, 32, -1], dtype=torch.int32)
    slot_storage = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    kwargs = {
        "checkpoint_token_indices": token_storage[:2].clone(),
        "checkpoint_state_slots": slot_storage[:2].clone(),
        "checkpoint_states": torch.empty((2, 2, 64, 128), dtype=torch.bfloat16),
    }
    if invalid == "partial":
        kwargs["checkpoint_state_slots"] = None
    elif invalid == "token_shape":
        kwargs["checkpoint_token_indices"] = torch.empty(1, dtype=torch.int32)
    elif invalid == "token_dtype":
        kwargs["checkpoint_token_indices"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "token_contiguous":
        kwargs["checkpoint_token_indices"] = token_storage[::2]
    elif invalid == "slot_shape":
        kwargs["checkpoint_state_slots"] = torch.empty(1, dtype=torch.int32)
    elif invalid == "slot_dtype":
        kwargs["checkpoint_state_slots"] = torch.empty(2, dtype=torch.int64)
    elif invalid == "slot_contiguous":
        kwargs["checkpoint_state_slots"] = slot_storage[::2]
    elif invalid == "state_shape":
        kwargs["checkpoint_states"] = torch.empty((2, 2, 64, 127), dtype=torch.bfloat16)
    elif invalid == "state_dtype":
        kwargs["checkpoint_states"] = torch.empty((2, 2, 64, 128), dtype=torch.float16)
    else:
        state_storage = torch.empty((2, 2, 64, 129), dtype=torch.bfloat16)
        kwargs["checkpoint_states"] = state_storage[..., :128]
        assert not kwargs["checkpoint_states"].is_contiguous()

    with pytest.raises(ValueError, match=match):
        runner.run(*tensors, **kwargs)


def test_source_runner_forwards_softplus_and_checkpoint_count(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = {}

    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(module, "_sm_count", lambda _: 1)
    monkeypatch.setattr(
        module,
        "_launch_program",
        lambda name, _arch, **kwargs: calls.__setitem__(name, kwargs),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=0x1234),
    )

    batch, seqlen, nheads, ngroups = 2, 128, 1, 1
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, ngroups, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    checkpoint_token_indices = torch.tensor([32, 64], dtype=torch.int32)
    checkpoint_state_slots = torch.tensor([0, 2], dtype=torch.int32)
    checkpoint_states = torch.empty((3, nheads, 64, 128), dtype=torch.bfloat16)
    runner = module.CakeSSDCombined(
        128,
        nheads,
        64,
        128,
        ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=False,
        d_has_hdim=False,
        has_initial_states=False,
        has_varlen=False,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )

    first = runner.run(
        x,
        dt,
        A,
        B,
        C,
        dt_softplus=False,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
    )
    second = runner.run(
        x,
        dt,
        A,
        B,
        C,
        dt_softplus=False,
        checkpoint_token_indices=checkpoint_token_indices,
        checkpoint_state_slots=checkpoint_state_slots,
        checkpoint_states=checkpoint_states,
    )

    exact = calls["exact_bf16_batched"]
    assert exact["preprocess"]["dt_softplus"] == 0
    assert exact["preprocess"]["write_seq_chunk_cumsum"] == 0
    assert exact["preprocess_grid"] == (1, 1, 1)
    main = exact["main"]
    assert main["dt_softplus"] == 0
    assert main["has_seq_chunk_cumsum"] == 0
    assert main["checkpoint_state_count"] == checkpoint_states.shape[0]
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first[1].data_ptr() != second[1].data_ptr()


@pytest.mark.parametrize(
    "state_dtype,mode_varlen,expected",
    [
        (torch.bfloat16, False, "exact_bf16_batched"),
        (torch.bfloat16, True, "exact_bf16_varlen"),
        (torch.float16, False, "exact_f16_batched"),
        (torch.float16, True, "exact_f16_varlen"),
        (torch.float32, False, "exact_f32_batched"),
        (torch.float32, True, "exact_f32_varlen"),
    ],
)
def test_source_program_name_covers_every_state_dtype(
    state_dtype, mode_varlen, expected
):
    """One kernel family serves every admitted input: the program follows
    from the state dtype and the batched/packed mode alone."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")

    actual = module._program_name(state_dtype, mode_varlen)

    assert actual == expected
    assert actual in module._PROGRAMS
    assert module._STATE_DTYPE_CODES == {
        "bf16": (4, 16),
        "f16": (2, 16),
        "f32": (2, 32),
    }
    assert not hasattr(module, "_select_scan_route")


def _cpu_forwarding_runner(module, monkeypatch, calls, **constructor):
    """A real runner on CPU tensors whose launcher call is captured."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(module, "_sm_count", lambda _: 1)
    monkeypatch.setattr(
        module,
        "_launch_program",
        lambda name, _arch, **kwargs: calls.append((name, kwargs)),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=0x1234),
    )
    return module.CakeSSDCombined(
        128,
        constructor.pop("nheads", 1),
        64,
        128,
        constructor.pop("ngroups", 1),
        io_dtype=torch.bfloat16,
        state_dtype=constructor.pop("state_dtype", torch.bfloat16),
        has_d=False,
        d_has_hdim=False,
        has_initial_states=constructor.pop("has_initial_states", False),
        has_varlen=constructor.pop("has_varlen", False),
        has_z=False,
        seq_idx_dtype=constructor.pop("seq_idx_dtype", torch.int32),
    )


@pytest.mark.parametrize(
    "case",
    (
        "precomputed",
        "update_caller_buffer",
        "runner_buffer_from_initial_states",
        "runner_buffer_from_num_seqs",
    ),
)
def test_source_varlen_cumsum_binding_without_gpu(monkeypatch, case):
    """The preprocess owns ``seq_chunk_cumsum``: one launcher call binds the
    same vector to both stages and ``write_seq_chunk_cumsum`` says whether
    the preprocess fills it; ``num_seqs`` supplies the count when there are
    no initial states."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    has_initial_states = case != "runner_buffer_from_num_seqs"
    runner = _cpu_forwarding_runner(
        module,
        monkeypatch,
        calls,
        has_varlen=True,
        has_initial_states=has_initial_states,
    )
    seqlen, num_seqs = 300, 2
    x = torch.empty((1, seqlen, 1, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, 1), dtype=torch.float32)
    A = torch.empty((1,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    kwargs = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
    }
    caller_vector = torch.tensor([0, 1, 3], dtype=torch.int32)
    if case == "precomputed":
        kwargs["seq_chunk_cumsum"] = caller_vector
    elif case == "update_caller_buffer":
        kwargs["seq_chunk_cumsum"] = caller_vector
        kwargs["update_seq_chunk_cumsum"] = True
    else:
        kwargs["num_seqs"] = num_seqs
    if has_initial_states:
        kwargs["initial_states"] = torch.empty(
            (num_seqs, 1, 64, 128), dtype=torch.bfloat16
        )

    out, final = runner.run(x, dt, A, B, C, **kwargs)

    assert tuple(out.shape) == (1, seqlen, 1, 64)
    assert tuple(final.shape) == (num_seqs, 1, 64, 128)
    ((name, launch),) = calls
    assert name == "exact_bf16_varlen"
    preprocess, main = launch["preprocess"], launch["main"]
    assert preprocess["seq_chunk_cumsum"] is main["seq_chunk_cumsum"]
    assert preprocess["num_sequences"] == main["sequence_count"] == num_seqs
    assert preprocess["seq_idx_i32"] is main["seq_idx_i32"] is kwargs["seq_idx"]
    assert preprocess["seq_idx_int64"] == main["seq_idx_int64"] == 0
    assert main["has_seq_chunk_cumsum"] == 1
    assert main["has_initial"] == int(has_initial_states)
    assert main["nchunks"] == 3 and main["seqlen"] == seqlen
    if case == "precomputed":
        assert preprocess["seq_chunk_cumsum"] is caller_vector
        assert preprocess["write_seq_chunk_cumsum"] == 0
    elif case == "update_caller_buffer":
        assert preprocess["seq_chunk_cumsum"] is caller_vector
        assert preprocess["write_seq_chunk_cumsum"] == 1
    else:
        assert preprocess["write_seq_chunk_cumsum"] == 1
        bound = preprocess["seq_chunk_cumsum"]
        assert bound.dtype == torch.int32 and bound.numel() == num_seqs + 1
        # The runner-owned vector is reused across calls of the same count.
        runner.run(x, dt, A, B, C, **kwargs)
        assert calls[-1][1]["preprocess"]["seq_chunk_cumsum"] is bound


def test_source_batched_unaligned_seqlen_binding_without_gpu(monkeypatch):
    """Batched mode: ceil'd chunk count, partial trailing segment length, the
    cumsum dummy with ``write_seq_chunk_cumsum=0``, FP16 delta workspace."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []
    runner = _cpu_forwarding_runner(module, monkeypatch, calls, nheads=2, ngroups=1)
    batch, seqlen = 2, 1000
    x = torch.empty((batch, seqlen, 2, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, 2), dtype=torch.bfloat16)
    A = torch.empty((2,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 1, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)

    out, final = runner.run(x, dt, A, B, C)

    assert tuple(out.shape) == (batch, seqlen, 2, 64)
    assert tuple(final.shape) == (batch, 2, 64, 128)
    ((name, launch),) = calls
    assert name == "exact_bf16_batched"
    preprocess, main = launch["preprocess"], launch["main"]
    assert main["nchunks"] == 8 and main["num_logical_chunks"] == 8
    assert preprocess["num_segments"] == 16
    assert preprocess["segment_lengths"].tolist() == [128] * 7 + [104] + [128] * 7 + [
        104
    ]
    assert preprocess["segment_starts"].tolist() == [
        *(chunk * 128 for chunk in range(8)),
        *(1000 + chunk * 128 for chunk in range(8)),
    ]
    assert preprocess["delta"].dtype == torch.float16
    assert preprocess["delta"].shape == (32, 128)
    assert preprocess["write_seq_chunk_cumsum"] == 0
    assert preprocess["num_sequences"] == batch
    assert main["has_seq_chunk_cumsum"] == 0 and main["mode_varlen"] == 0
    # bf16 dt is widened into the FP32 workspace both stages read.
    assert preprocess["dt"] is main["dt"] and preprocess["dt"].dtype == torch.float32
    # No dt_bias: the zero vector allocated with the workspace is bound.
    assert preprocess["dt_bias"] is main["dt_bias"]
    assert preprocess["dt_bias"].dtype == torch.float32
    assert not preprocess["dt_bias"].any()


def test_source_direct_preprocess_and_sequence_argument_order():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    sentinels = {
        name: object()
        for name in (
            "dt",
            "A",
            "dt_bias",
            "starts",
            "lengths",
            "chunk_indices",
            "chunk_offsets",
            "delta",
            "cumsum",
            "seq_idx_i32",
            "seq_idx_i64",
            "seq_chunk_cumsum",
        )
    }
    preprocess, preprocess_grid = module._direct_preprocess_inputs(
        dt=sentinels["dt"],
        A=sentinels["A"],
        dt_bias=sentinels["dt_bias"],
        segment_starts=sentinels["starts"],
        segment_lengths=sentinels["lengths"],
        chunk_indices=sentinels["chunk_indices"],
        chunk_offsets=sentinels["chunk_offsets"],
        delta=sentinels["delta"],
        cumsum=sentinels["cumsum"],
        num_segments=3,
        nheads=128,
        seqlen=256,
        mode_varlen=True,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        threads=32,
        seq_idx_i32=sentinels["seq_idx_i32"],
        seq_idx_i64=sentinels["seq_idx_i64"],
        seq_idx_int64=True,
        seq_chunk_cumsum=sentinels["seq_chunk_cumsum"],
        num_sequences=2,
        write_seq_chunk_cumsum=True,
    )
    main = {name: object() for name in module._MAIN_ARGS}

    bound = module._sequence_arguments(
        preprocess,
        preprocess_grid,
        main,
        (148, 1, 1),
        cuda_stream=0x1234,
    )

    assert preprocess["chunk_indices"] is sentinels["chunk_indices"]
    assert preprocess["chunk_offsets"] is sentinels["chunk_offsets"]
    assert preprocess["direct_varlen_metadata"] == 1
    assert preprocess["dt_softplus"] == 0
    assert preprocess["seq_idx_i32"] is sentinels["seq_idx_i32"]
    assert preprocess["seq_idx_i64"] is sentinels["seq_idx_i64"]
    assert preprocess["seq_idx_int64"] == 1
    assert preprocess["seq_chunk_cumsum"] is sentinels["seq_chunk_cumsum"]
    assert preprocess["num_sequences"] == 2
    assert preprocess["write_seq_chunk_cumsum"] == 1
    assert preprocess_grid == (12, 1, 1)
    assert set(preprocess) == set(module._PREPROCESS_ARGS)
    assert module._PREPROCESS_ARGS[-6:] == (
        "seq_idx_i32",
        "seq_idx_i64",
        "seq_idx_int64",
        "seq_chunk_cumsum",
        "num_sequences",
        "write_seq_chunk_cumsum",
    )
    assert bound == (
        *(preprocess[name] for name in module._PREPROCESS_ARGS),
        12,
        1,
        1,
        *(main[name] for name in module._MAIN_ARGS),
        148,
        1,
        1,
        0x1234,
    )
    assert len(bound) == 22 + 3 + 43 + 3 + 1

    assert module._persistent_grid_size(total_work=256, sm_count=148) == 128
    assert module._persistent_grid_size(total_work=384, sm_count=148) == 128
    assert module._persistent_grid_size(total_work=129, sm_count=148) == 129


def test_source_sequence_arguments_fail_closed_on_missing_values():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    preprocess = {name: object() for name in module._PREPROCESS_ARGS}
    main = {name: object() for name in module._MAIN_ARGS}
    del main["checkpoint_state_count"]

    with pytest.raises(KeyError, match="checkpoint_state_count"):
        module._sequence_arguments(preprocess, (1, 1, 1), main, (1, 1, 1), 0)


def test_source_program_launch_orders_stage_arguments(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []

    class Generated:
        def run(self, *args):
            calls.append(args)

    monkeypatch.setattr(module, "_load_generated_program", lambda *_: Generated())
    preprocess = {name: f"pre:{name}" for name in module._PREPROCESS_ARGS}
    main = {name: f"main:{name}" for name in module._MAIN_ARGS}

    module._launch_program(
        "exact_bf16_varlen",
        "sm_103a",
        preprocess=preprocess,
        preprocess_grid=(32, 1, 1),
        main=main,
        main_grid=(148, 1, 1),
        cuda_stream=0x1234,
    )

    assert calls == [
        (
            *(f"pre:{name}" for name in module._PREPROCESS_ARGS),
            32,
            1,
            1,
            *(f"main:{name}" for name in module._MAIN_ARGS),
            148,
            1,
            1,
            0x1234,
        )
    ]


def test_source_program_table_names_shipped_sources():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_dir = module._source_dir()

    assert set(module._PROGRAMS) == {
        "exact_bf16_batched",
        "exact_bf16_varlen",
        "exact_f16_batched",
        "exact_f16_varlen",
        "exact_f32_batched",
        "exact_f32_varlen",
    }
    assert "PENDINGEXPORT" not in "".join(module._SCAN_MODULES.values())
    template = (source_dir / module._HOST_TEMPLATE).read_text(encoding="utf-8")
    device_sources = set()
    for name, program in module._PROGRAMS.items():
        family, state_key, mode = name.split("_")
        assert family == "exact"
        assert (program.state_dtype_code, program.state_dtype_bits) == (
            module._STATE_DTYPE_CODES[state_key]
        )
        assert program.preprocess is module._SEGMENT_PREPROCESS
        assert program.main_smem_bytes == module._EXACT_SMEM_BYTES
        assert program.main.module.startswith(
            program.main.kernel.removeprefix("kernel_") + "_"
        )
        assert program.main.kernel.endswith(f"_{state_key}_{mode}")
        assert program.preprocess.threads > 0 and program.main.threads == 512
        assert not program.preprocess.fast_math and program.main.fast_math
        for kernel in program.kernels:
            source = source_dir / module._DEVICE_DIR / kernel.source
            assert source.is_file(), source
            assert re.search(
                rf"\b{kernel.kernel}\(", source.read_text(encoding="utf-8")
            )
            device_sources.add(source)
        rendered = module._render_host_source(template, name, program)
        placeholders = (
            "CAKE_SSD_PROGRAM",
            "CAKE_SSD_PREPROCESS_MODULE",
            "CAKE_SSD_PREPROCESS_KERNEL",
            "CAKE_SSD_PREPROCESS_THREADS",
            "CAKE_SSD_MAIN_MODULE",
            "CAKE_SSD_MAIN_KERNEL",
            "CAKE_SSD_STATE_DTYPE_CODE",
            "CAKE_SSD_STATE_DTYPE_BITS",
            "CAKE_SSD_MAIN_SMEM_BYTES",
        )
        assert all(placeholder in template for placeholder in placeholders)
        assert not any(placeholder in rendered for placeholder in placeholders)
        assert f"TVM_FFI_EMBED_CUBIN({program.preprocess.module});" in rendered
        assert f"TVM_FFI_EMBED_CUBIN({program.main.module});" in rendered
        assert f'"{program.main.kernel}"' in rendered
        assert f"namespace cake_mamba_ssd_combined_host_{name} {{" in rendered
        assert f"stream, {program.main_smem_bytes}u)" in rendered
        state_check = (
            f'"initial_states", {program.state_dtype_code}, '
            f"{program.state_dtype_bits}, 1)"
        )
        assert state_check in rendered
    # One shared source per physical kernel: six scan sources plus the one
    # preprocess kernel, no architecture copies.
    assert len(device_sources) == 7
    assert sorted(
        path.name for path in (source_dir / module._DEVICE_DIR).glob("*.cu")
    ) == sorted(path.name for path in device_sources)


def test_active_source_package_declares_cuda_half_types_explicitly():
    """Every program carries ``delta`` in FP16 (the state may also be FP16),
    so each shipped device source must declare the CUDA half types itself."""

    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    source_root = module._source_dir() / module._DEVICE_DIR
    sources = {
        kernel.source
        for program in module._PROGRAMS.values()
        for kernel in program.kernels
    }
    for name in sorted(sources):
        source = (source_root / name).read_text(encoding="utf-8")
        assert source.count("#include <cuda_fp16.h>") == 1, name
        assert "#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n" in source, name


def test_source_program_loader_builds_one_module_per_arch(monkeypatch, tmp_path):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    program = module._PROGRAMS["exact_bf16_varlen"]
    device_dir = tmp_path / module._DEVICE_DIR
    device_dir.mkdir(parents=True)
    for kernel in program.kernels:
        (device_dir / kernel.source).write_text(
            f"{kernel.kernel} source\n", encoding="utf-8"
        )
    host = tmp_path / module._HOST_TEMPLATE
    host.parent.mkdir(parents=True)
    host.write_text(
        "namespace host_CAKE_SSD_PROGRAM {}\n"
        "TVM_FFI_EMBED_CUBIN(CAKE_SSD_PREPROCESS_MODULE);\n"
        "TVM_FFI_EMBED_CUBIN(CAKE_SSD_MAIN_MODULE);\n"
        "CAKE_SSD_PREPROCESS_KERNEL CAKE_SSD_PREPROCESS_THREADS CAKE_SSD_MAIN_KERNEL "
        "CAKE_SSD_STATE_DTYPE_CODE CAKE_SSD_STATE_DTYPE_BITS CAKE_SSD_MAIN_SMEM_BYTES\n",
        encoding="utf-8",
    )
    nvcc = tmp_path / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = module.Path(command[-1])
        source_path = module.Path(command[-3])
        output.write_bytes(source_path.read_bytes())
        return SimpleNamespace(returncode=0, stderr="")

    loaded = object()
    load_calls = []

    def load_inline(*args, **kwargs):
        load_calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "_nvcc", lambda: nvcc)
    monkeypatch.setattr(module.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "cpp", SimpleNamespace(load_inline=load_inline))
    module._load_generated_program.cache_clear()

    actual = module._load_generated_program("exact_bf16_varlen", "sm_103a")
    again = module._load_generated_program("exact_bf16_varlen", "sm_103a")
    module._load_generated_program.cache_clear()

    assert actual is loaded and again is loaded
    assert len(calls) == 2
    assert all("-arch=sm_103a" in command for command, _ in calls)
    assert "--use_fast_math" not in calls[0][0]
    assert "--use_fast_math" in calls[1][0]
    assert len(load_calls) == 1
    rendered = load_calls[0][1]["cpp_sources"]
    assert "CAKE_SSD_" not in rendered
    assert "namespace host_exact_bf16_varlen {}" in rendered
    assert f"TVM_FFI_EMBED_CUBIN({program.preprocess.module});" in rendered
    assert (
        f"{program.preprocess.kernel} 128 {program.main.kernel} 4 16 231936" in rendered
    )
    assert set(load_calls[0][1]["embed_cubin"]) == {
        program.preprocess.module,
        program.main.module,
    }
    assert load_calls[0][0][0].startswith("cake_mamba_ssd_exact_bf16_varlen_sm_103a_")
