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

import hashlib
import importlib
import inspect
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

    without_final = runner.run(*tensors, **{**arguments, "return_final_states": False})
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
            self.run_args = None
            self.run_kwargs = None
            runners.append(self)

        def run(self, *args, **kwargs):
            self.run_args = args
            self.run_kwargs = kwargs
            return (object(), None)

    monkeypatch.setattr(module, "SSDCombined", Runner)
    x = SimpleNamespace(
        shape=(1, 128, 8, 64),
        dtype=torch.bfloat16,
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
    }
    positional = (x, object(), object(), B, object())

    first = module.ssd_combined_fwd(
        *positional,
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
    assert len(runners) == 2 and runners[0] is not runners[1]
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
    assert runners[0].run_args == positional
    assert runners[0].run_kwargs == optional
    assert runners[1].run_kwargs["dt_softplus"] is False


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

    with pytest.raises(ValueError, match="requires SM100 or SM103, got SM110"):
        module.SSDCombined(128, 2, 64, 128, 1, backend="cake")


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
        "state_dtype": torch.float32,
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
        name: object()
        for name in (
            "D",
            "z",
            "dt_bias",
            "initial_states",
            "seq_idx",
            "chunk_indices",
            "chunk_offsets",
            "seq_chunk_cumsum",
            "checkpoint_token_indices",
            "checkpoint_state_slots",
            "checkpoint_states",
        )
    }
    out = torch.empty((1, 2, 64, 1, 128), dtype=torch.bfloat16)
    kwargs = {
        **sentinels,
        "dt_softplus": True,
        "dt_limit": (-0.25, 0.75),
        "update_seq_chunk_cumsum": True,
        "out": out,
        "return_final_states": False,
    }

    actual = runner.run(*tensors, **kwargs)

    assert actual is result
    assert runner._cake_runner.calls == [(tensors, kwargs)]


@pytest.mark.parametrize(
    "invalid,exception",
    (
        ("x_rank", ValueError),
        ("seqlen", AssertionError),
        ("a_dtype", AssertionError),
        ("out_shape", AssertionError),
        ("out_dtype", AssertionError),
        ("out_contiguous", AssertionError),
    ),
)
def test_source_public_shared_validation_error_parity_without_gpu(invalid, exception):
    tensors = _cpu_public_run_inputs()
    kwargs = {}
    if invalid == "x_rank":
        tensors = (torch.empty((128, 2, 64), dtype=torch.bfloat16), *tensors[1:])
    elif invalid == "seqlen":
        x, dt, A, B, C = tensors
        tensors = (x[:, :-1], dt[:, :-1], A, B[:, :-1], C[:, :-1])
    elif invalid == "a_dtype":
        tensors = (*tensors[:2], tensors[2].to(torch.bfloat16), *tensors[3:])
    elif invalid == "out_shape":
        kwargs["out"] = torch.empty((1,), dtype=torch.bfloat16)
    elif invalid == "out_dtype":
        kwargs["out"] = torch.empty((1, 2, 64, 1, 128), dtype=torch.float16)
    else:
        storage = torch.empty((1, 2, 64, 1, 129), dtype=torch.bfloat16)
        kwargs["out"] = storage[..., :128]
        assert not kwargs["out"].is_contiguous()

    errors = {}
    for backend in ("cute", "cake"):
        runner = _public_runner_without_constructor(backend)
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
        ("varlen_initial", "varlen mode requires initial_states"),
        ("initial_dtype", "initial_states dtype must match state_dtype"),
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
def test_source_public_cake_domain_validation_without_gpu(monkeypatch, invalid, match):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_select_scan_route", lambda **_: "test")
    monkeypatch.setattr(module, "_prefix_route_selected", lambda: False)
    cake_runner = _source_cake_runner_without_constructor()
    runner = _public_runner_without_constructor("cake")
    runner._cake_runner = cake_runner
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
        kwargs["initial_states"] = torch.empty(
            (2, 2, 64, 128), dtype=torch.bfloat16
        )
    elif invalid == "batched_metadata":
        kwargs["seq_idx"] = torch.empty((2, 128), dtype=torch.int32)
    elif invalid == "batched_cumsum":
        kwargs["seq_chunk_cumsum"] = torch.empty(3, dtype=torch.int32)
    elif invalid == "varlen_initial":
        cake_runner.has_varlen = True
        kwargs.update(
            seq_idx=torch.empty((2, 128), dtype=torch.int32),
            chunk_indices=torch.arange(2, dtype=torch.int32),
            chunk_offsets=torch.zeros(2, dtype=torch.int32),
        )
    elif invalid == "initial_dtype":
        cake_runner.has_initial_states = True
        kwargs["initial_states"] = torch.empty(
            (2, 2, 64, 128), dtype=torch.float16
        )
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
        runner.run(*tensors, **kwargs)


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
        kwargs["initial_states"] = torch.empty(
            (2, 2, 64, 128), dtype=torch.float16
        )
    elif invalid == "varlen_initial":
        kwargs.update(
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    with pytest.raises(exception, match=match):
        runner.run(*tensors, **kwargs)


@pytest.mark.parametrize("invalid", ("dt_bias_shape", "dt_bias_dtype"))
def test_source_public_cake_dt_bias_validation_without_gpu(monkeypatch, invalid):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(module, "_select_scan_route", lambda **_: "test")
    monkeypatch.setattr(module, "_prefix_route_selected", lambda: False)
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
    monkeypatch.setattr(module, "_select_scan_route", lambda **_: "test")
    monkeypatch.setattr(module, "_prefix_route_selected", lambda: False)
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

    assert calls["preprocess"][-6] == 0
    assert calls["bf16_batched"][-8] == 0
    assert calls["bf16_batched"][-4] == checkpoint_states.shape[0]
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first[1].data_ptr() != second[1].data_ptr()


@pytest.mark.parametrize(
    (
        "mode_varlen",
        "num_logical_chunks",
        "num_sequences",
        "nheads",
        "ngroups",
        "dt_min",
        "prefix_route_selected",
        "expected",
    ),
    [
        (False, 2, 2, 128, 8, 0.0, True, "exact_scan"),
        (True, 3, 2, 128, 8, 0.0, True, "exact_scan"),
        (True, 2, 2, 128, 8, -0.001, True, "exact_scan"),
        (True, 2, 2, 128, 8, 0.0, False, "shallow_varlen"),
        (True, 0, 2, 128, 8, 0.0, False, "shallow_varlen"),
        (True, 2, 2, 8, 8, 0.0, True, "shallow_varlen"),
        (True, 2, 2, 128, 8, 0.0, True, "prefix_varlen"),
    ],
)
def test_source_route_predicates_do_not_bind_program_symbols(
    mode_varlen,
    num_logical_chunks,
    num_sequences,
    nheads,
    ngroups,
    dt_min,
    prefix_route_selected,
    expected,
):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")

    actual = module._select_scan_route(
        mode_varlen=mode_varlen,
        num_logical_chunks=num_logical_chunks,
        num_sequences=num_sequences,
        nheads=nheads,
        ngroups=ngroups,
        dt_min=dt_min,
        prefix_route_selected=prefix_route_selected,
    )

    assert actual == expected


def test_source_direct_preprocess_and_prepared_sequence_binding():
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
            "main_x",
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
    )
    main = {"x_map": sentinels["main_x"], "nheads": 128}
    stage_plans = (
        (
            "preprocess",
            (
                ("buffer", "dt"),
                ("buffer", "chunk_indices"),
                ("buffer", "chunk_offsets"),
                ("parameter", "direct_varlen_metadata"),
                ("parameter", "dt_softplus"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
        ),
        (
            "main",
            (
                ("tma_buffer", "x_map"),
                ("parameter", "nheads"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
        ),
    )

    bound = module._bind_prepared_sequence_arguments(
        stage_plans,
        {"preprocess": preprocess, "main": main},
        {"preprocess": preprocess_grid, "main": (148, 1, 1)},
        cuda_stream=0x1234,
    )

    assert preprocess["chunk_indices"] is sentinels["chunk_indices"]
    assert preprocess["chunk_offsets"] is sentinels["chunk_offsets"]
    assert preprocess["direct_varlen_metadata"] == 1
    assert preprocess["dt_softplus"] == 0
    assert preprocess_grid == (12, 1, 1)
    assert bound == (
        sentinels["dt"],
        sentinels["chunk_indices"],
        sentinels["chunk_offsets"],
        1,
        0,
        12,
        1,
        1,
        sentinels["main_x"],
        128,
        148,
        1,
        1,
        0x1234,
    )

    assert module._persistent_grid_size(total_work=256, sm_count=148) == 128
    assert module._persistent_grid_size(total_work=384, sm_count=148) == 128
    assert module._persistent_grid_size(total_work=129, sm_count=148) == 129


def test_source_generated_argument_binding_fails_closed():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")

    with pytest.raises(ValueError, match="buffer:missing is unresolved"):
        module._bind_generated_arguments(
            (("buffer", "missing"),),
            {},
            (1, 1, 1),
        )
    with pytest.raises(ValueError, match="kind is unsupported"):
        module._bind_generated_arguments(
            (("unknown", "value"),),
            {"value": object()},
            (1, 1, 1),
        )


def test_source_generated_program_runs_catalog_entry(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    calls = []

    class Generated:
        def run(self, *args):
            calls.append(args)

    monkeypatch.setattr(
        module,
        "_generated_program_profile",
        lambda *_: {
            "entry": "run",
            "launch_count": 2,
            "stage_order": ["preprocess", "main"],
            "stages": {
                "preprocess": {
                    "arg_plan": [
                        ["buffer", "dt"],
                        ["grid", "grid_x"],
                    ]
                },
                "main": {
                    "arg_plan": [
                        ["tma_buffer", "x_map"],
                        ["parameter", "nheads"],
                        ["grid", "grid_x"],
                    ]
                },
            },
        },
    )
    monkeypatch.setattr(module, "_load_generated_program", lambda *_: Generated())

    module._run_generated_program(
        "prefix_bf16_varlen",
        "sm_103a",
        0,
        stage_values={
            "preprocess": {"dt": "dt"},
            "main": {"x_map": "x", "nheads": 128},
        },
        stage_grids={"preprocess": (32, 1, 1), "main": (148, 1, 1)},
        cuda_stream=0x1234,
    )

    assert calls == [("dt", 32, "x", 128, 148, 0x1234)]


def test_source_generated_program_rejects_unresolved_launch_abi(monkeypatch):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    monkeypatch.setattr(
        module,
        "_generated_program_profile",
        lambda *_: {
            "entry": "run",
            "launch_count": 1,
            "stage_order": ["preprocess", "main"],
            "stages": {},
        },
    )

    with pytest.raises(RuntimeError, match="unresolved launch ABI"):
        module._run_generated_program(
            "prefix_bf16_varlen",
            "sm_103a",
            0,
            stage_values={},
            stage_grids={},
            cuda_stream=0,
        )


def test_source_generated_catalog_is_sealed_and_inactive():
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    module._source_catalog.cache_clear()

    catalog = module._source_catalog()

    assert catalog["source_status"] == "prepared_nonterminal"
    assert catalog["prefix_route_selected"] is False
    assert set(catalog["programs"]) == {
        "prefix_bf16_varlen",
        "prefix_f16_varlen",
    }
    for program in catalog["programs"].values():
        assert set(program) == {"sm_100a", "sm_103a"}
        for profile in program.values():
            assert profile["launch_count"] == 2
            assert profile["stage_order"] == ["preprocess", "main"]
            assert len(profile["device_sources"]) == 2
            sources = [profile["host_source"], *profile["device_sources"]]
            for source in sources:
                path = module._source_dir() / "generated" / source["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_source_generated_multistage_loader_binds_exact_cubins(monkeypatch, tmp_path):
    module = importlib.import_module("flashinfer.mamba.cake_ssd_combined")
    generated = tmp_path / "generated"
    generated.mkdir()
    host = generated / "host.cpp"
    first = generated / "first.cu"
    second = generated / "second.cu"
    host.write_text("host source\n", encoding="utf-8")
    first.write_text("first source\n", encoding="utf-8")
    second.write_text("second source\n", encoding="utf-8")

    def source(path, module_ident=None, compile_flags=None):
        payload = (generated / path).read_bytes()
        result = {"path": path, "sha256": hashlib.sha256(payload).hexdigest()}
        if module_ident is not None:
            result["module_ident"] = module_ident
            result["compile_flags"] = compile_flags
        return result

    profile = {
        "entry": "run_prepared",
        "host_source": source("host.cpp"),
        "device_sources": [
            source("first.cu", "first_ident", []),
            source("second.cu", "second_ident", ["--use_fast_math"]),
        ],
    }
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

    monkeypatch.setattr(module, "_generated_program_profile", lambda *_: profile)
    monkeypatch.setattr(module, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "_nvcc", lambda: nvcc)
    monkeypatch.setattr(module.jit_env, "FLASHINFER_JIT_DIR", tmp_path / "jit")
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "cpp", SimpleNamespace(load_inline=load_inline))
    module._load_generated_program.cache_clear()

    actual = module._load_generated_program("prefix_bf16_varlen", "sm_103a", 0)

    assert actual is loaded
    assert len(calls) == 2
    assert "--use_fast_math" not in calls[0][0]
    assert "--use_fast_math" in calls[1][0]
    assert len(load_calls) == 1
    assert load_calls[0][1]["cpp_sources"] == "host source\n"
    assert set(load_calls[0][1]["embed_cubin"]) == {
        "first_ident",
        "second_ident",
    }
