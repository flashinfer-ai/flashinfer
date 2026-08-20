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
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flashinfer.mamba import SSDCombined


def _assert_cute_parity(actual, expected, *, nheads, ngroups):
    torch.testing.assert_close(actual[0], expected[0], atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-2, rtol=1e-2)


def _varlen_metadata(lengths, dtype):
    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=dtype, device="cuda")
    start = 0
    for sequence, length in enumerate(lengths):
        seq_idx[0, start : start + length] = sequence
        start += length
    chunk_indices = []
    chunk_offsets = []
    for chunk in range(total // 128):
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


def _case(
    *,
    nheads=8,
    ngroups=8,
    state_dtype=torch.bfloat16,
    varlen=False,
    seq_idx_dtype=torch.int32,
    preprocess_dtype=torch.float32,
    d_has_hdim=True,
):
    torch.manual_seed(7)
    batch, seqlen = (1, 256) if varlen else (2, 128)
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
    state_batch = 2 if varlen else batch
    initial_states = torch.randn(state_batch, nheads, 64, 128, device="cuda").to(
        state_dtype
    )
    if varlen:
        seq_idx, chunk_indices, chunk_offsets = _varlen_metadata(
            (96, 160), seq_idx_dtype
        )
        seq_chunk_cumsum = torch.tensor([0, 1, 3], dtype=torch.int32, device="cuda")
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
        has_initial_states=True,
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
        initial_states=initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        seq_chunk_cumsum=seq_chunk_cumsum,
        return_final_states=True,
    )
    return constructor, (x, dt, A, B, C), arguments


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
    if varlen and nheads == 128 and ngroups == 8:
        # Nemotron-H prefill starts from zero state and uses the unbounded
        # positive-dt interval. Finite-clamp and nonzero-initial-state feature
        # rows remain covered independently above; do not invent their
        # Cartesian product with the model-derived head shape.
        arguments["initial_states"].zero_()
        arguments["dt_limit"] = (0.0, float("inf"))
    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    _assert_cute_parity(actual, expected, nheads=nheads, ngroups=ngroups)


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
    _assert_cute_parity(actual, expected, nheads=8, ngroups=8)


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

    _assert_cute_parity(actual, expected, nheads=8, ngroups=8)


def test_cake_ssd_combined_updates_caller_buffers():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(varlen=True)
    expected_cumsum = arguments["seq_chunk_cumsum"]
    actual_cumsum = torch.full_like(expected_cumsum, -1)
    out = torch.empty((1, 8, 64, 2, 128), dtype=torch.bfloat16, device="cuda")
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
    assert actual[0].untyped_storage().data_ptr() == out.untyped_storage().data_ptr()

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


def test_cake_ssd_combined_writes_selected_checkpoint_state():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case(varlen=True)
    sequence_start = 96
    checkpoint_length = 128
    checkpoint_token = sequence_start + checkpoint_length
    checkpoint_states = torch.full(
        (3, *arguments["initial_states"].shape[1:]),
        torch.nan,
        dtype=arguments["initial_states"].dtype,
        device="cuda",
    )
    checkpoint_state = checkpoint_states[2:3]
    full_arguments = {
        **arguments,
        # Expose sequence 1's checkpoint inside physical chunk 1 as a logical
        # segment boundary. This is the packed shape used by SGLang.
        "chunk_indices": torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda"),
        "chunk_offsets": torch.tensor([0, 96, 0, 96], dtype=torch.int32, device="cuda"),
        "seq_chunk_cumsum": torch.tensor([0, 1, 4], dtype=torch.int32, device="cuda"),
        "checkpoint_token_indices": torch.tensor(
            [-1, checkpoint_token], dtype=torch.int32, device="cuda"
        ),
        "checkpoint_state_slots": torch.tensor(
            [-1, 2], dtype=torch.int32, device="cuda"
        ),
        "checkpoint_states": checkpoint_states,
    }
    SSDCombined(**constructor, backend="cake").run(*tensors, **full_arguments)

    x, dt, A, B, C = tensors
    prefix_tensors = (
        x[:, sequence_start:checkpoint_token].contiguous(),
        dt[:, sequence_start:checkpoint_token].contiguous(),
        A,
        B[:, sequence_start:checkpoint_token].contiguous(),
        C[:, sequence_start:checkpoint_token].contiguous(),
    )
    prefix_arguments = {
        **arguments,
        "z": arguments["z"][:, sequence_start:checkpoint_token].contiguous(),
        "initial_states": arguments["initial_states"][1:2].contiguous(),
        "seq_idx": torch.zeros(
            (1, checkpoint_length), dtype=torch.int32, device="cuda"
        ),
        "chunk_indices": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "chunk_offsets": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "seq_chunk_cumsum": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
    }
    _, expected_state = SSDCombined(**constructor, backend="cute").run(
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

    without_final = runner.run(
        *tensors, **{**arguments, "return_final_states": False}
    )
    assert isinstance(without_final, tuple)
    assert without_final[1] is None


def test_cake_ssd_combined_supports_disabled_softplus():
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        pytest.skip("Cake SSDCombined requires SM100 or SM103")

    constructor, tensors, arguments = _case()
    arguments["dt_softplus"] = False

    expected = SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
    actual = SSDCombined(**constructor, backend="cake").run(*tensors, **arguments)
    _assert_cute_parity(actual, expected, nheads=8, ngroups=8)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cake_ssd_combined_program_cache_is_cuda_context_scoped():
    runners = []
    cases = []
    expected = []
    for device_index in (0, 1):
        with torch.cuda.device(device_index):
            constructor, tensors, arguments = _case(nheads=1, ngroups=1)
            expected.append(
                SSDCombined(**constructor, backend="cute").run(*tensors, **arguments)
            )
            runners.append(SSDCombined(**constructor, backend="cake"))
            cases.append((tensors, arguments))

    torch.cuda.set_device(0)
    for device_index, (runner, case, reference) in enumerate(
        zip(runners, cases, expected, strict=True)
    ):
        tensors, arguments = case
        actual = runner.run(*tensors, **arguments)
        assert actual[0].device.index == device_index
        _assert_cute_parity(actual, reference, nheads=1, ngroups=1)


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

    assert errors["cake"] == errors["cute"]


def test_ssd_combined_fwd_keeps_default_and_runner_ownership(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.ssd_combined")
    runners = []

    class Runner:
        def __init__(self, *args, **kwargs):
            self.constructor_args = args
            self.constructor_kwargs = kwargs
            self.run_kwargs = None
            runners.append(self)

        def run(self, *args, **kwargs):
            self.run_kwargs = kwargs
            return (object(), None)

    monkeypatch.setattr(module, "SSDCombined", Runner)
    x = SimpleNamespace(
        shape=(1, 128, 8, 64),
        dtype=torch.bfloat16,
    )
    B = SimpleNamespace(shape=(1, 128, 8, 128))
    checkpoint_states = SimpleNamespace(dtype=torch.float16)

    first = module.ssd_combined_fwd(
        x,
        object(),
        object(),
        B,
        object(),
        checkpoint_states=checkpoint_states,
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
    assert len(runners) == 2 and runners[0] is not runners[1]
    assert all(runner.constructor_kwargs["backend"] == "cake" for runner in runners)
    assert all(
        runner.constructor_kwargs["state_dtype"] == torch.float16
        for runner in runners
    )
    assert all(runner.run_kwargs["dt_softplus"] is False for runner in runners)


def test_source_runner_forwards_softplus_and_checkpoint_count(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = {}

    class Program:
        def __init__(self, name):
            self.name = name

        def run(self, *args):
            calls[self.name] = args

    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(
        module,
        "_load_program",
        lambda name, _arch, _device_index: Program(name),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *_: SimpleNamespace(multi_processor_count=1),
    )

    batch, seqlen, nheads, ngroups = 2, 128, 1, 1
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, ngroups, 128), dtype=torch.bfloat16)
    C = torch.empty_like(B)
    checkpoint_token_indices = torch.tensor([32, 64], dtype=torch.int32)
    checkpoint_state_slots = torch.tensor([0, 2], dtype=torch.int32)
    checkpoint_states = torch.empty(
        (3, nheads, 64, 128), dtype=torch.bfloat16
    )
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

    assert calls["preprocess"][-6] == 0
    assert calls["bf16_batched"][-8] == 0
    assert calls["bf16_batched"][-4] == checkpoint_states.shape[0]
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first[1].data_ptr() != second[1].data_ptr()
