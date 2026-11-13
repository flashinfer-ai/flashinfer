# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the experimental VibeCUDA recurrent KDA prefill backend."""

import importlib

import pytest
import torch

from flashinfer.kda import recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.utils import get_compute_capability

from tests.kda.test_helpers.kda_prefill import (
    _make_inputs,
    _reference,
    _strict_prefill_kwargs,
    cpu_route_tensors,
)

kda_api = importlib.import_module("flashinfer.kda")
kda_vibecuda_api = importlib.import_module("flashinfer.experimental.kda_vibecuda")
kda_vibecuda_backend = importlib.import_module(
    "flashinfer.experimental.kda_vibecuda.backend"
)


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


@pytest.fixture
def flash_kda_device(cuda_device):
    if get_compute_capability(cuda_device) not in ((10, 0), (10, 3)):
        pytest.skip(
            "frozen recurrent KDA prefill requires CC 10.0 "
            "(SM100a; B200/GB200) or CC 10.3 (SM103a; B300/GB300)"
        )
    return cuda_device


@pytest.mark.parametrize(
    ("target", "expected_arch", "uses_sm103_schedule"),
    [
        ("sm100a", "compute_100a", False),
        ("sm103a", "compute_103a", True),
    ],
)
def test_vibecuda_prefill_uses_exact_arch_schedule(
    target, expected_arch, uses_sm103_schedule
):
    spec = kda_vibecuda_api.gen_vibecuda_flash_kda_module(target)
    assert any(expected_arch in flag for flag in spec.extra_cuda_cflags)
    assert "-DKDA_FUSE_FIXUP" in spec.extra_cuda_cflags
    assert ("-DKDA_SM103" in spec.extra_cuda_cflags) is uses_sm103_schedule


def test_public_prefill_vibecuda_forwards_public_contract(monkeypatch):
    calls = []
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_vibecuda_api,
        "_vibecuda_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_vibecuda_api,
        "_run_vibecuda_kda_prefill",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    assert recurrent_kda(**cpu_route_tensors(), backend="vibecuda") is sentinel
    assert len(calls) == 1
    assert calls[0]["output_final_state"] is False


def test_public_prefill_vibecuda_backend_is_strict(monkeypatch):
    monkeypatch.setattr(
        kda_vibecuda_api,
        "_vibecuda_kda_prefill_is_eligible",
        lambda **kwargs: False,
    )

    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(**cpu_route_tensors(), backend="vibecuda")


def test_public_decode_vibecuda_backend_is_prefill_only():
    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(**cpu_route_tensors(token_count=1), backend="vibecuda")


def test_vibecuda_prefill_rejects_non_sm100_arch(cuda_device, monkeypatch):
    monkeypatch.setattr(kda_api, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        kda_vibecuda_backend, "_fast_plan_device_supported", lambda device: False
    )
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4200
    )

    with pytest.raises(RuntimeError, match=r"10\.0 and 10\.3"):
        recurrent_kda(**_strict_prefill_kwargs(inputs), backend="vibecuda")


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "packed", "seed"),
    [
        ([4, 4], 96, False, 4201),
        ([3, 5], 96, True, 4202),
        ([512], 64, False, 4203),
        ([1024], 96, False, 4204),
        ([16384], 1, False, 4205),
        ([8192], 4, False, 4206),
        ([1300, 547, 2048], 96, True, 4207),
        ([128] * 8, 96, True, 4208),
        ([4096], 12, False, 4209),
        ([1024] * 4, 12, True, 4210),
    ],
)
def test_vibecuda_prefill_matches_torch_reference(
    flash_kda_device, seq_lens, num_heads, packed, seed
):
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=True,
        seed=seed,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]
    seq_order = (
        torch.tensor(
            sorted(range(len(seq_lens)), key=seq_lens.__getitem__, reverse=True),
            dtype=torch.int32,
            device=flash_kda_device,
        )
        if packed
        else None
    )

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        seq_order=seq_order,
        backend="vibecuda",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is state_identity
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state.float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_vibecuda_fast_plan_shared_buffer_different_slice_shapes(flash_kda_device):
    """Regression test: two fast-plan hits that share the same underlying base
    pointers and dispatch variant (e.g. ``fam2h64`` on ``qbuf[:, :2048]`` vs
    ``qbuf[:, :8192]``) must re-encode TMA descriptors when switching shapes.
    """
    full_inputs = _make_inputs(
        seq_lens=[8192],
        num_heads=64,
        packed=True,
        initial_state=True,
        seed=4301,
    )
    qbuf = full_inputs["q"]
    kbuf = full_inputs["k"]
    vbuf = full_inputs["v"]
    gbuf = full_inputs["g"]
    betabuf = full_inputs["beta"]
    A_log = full_inputs["A_log"]
    dt_bias = full_inputs["dt_bias"]
    state_init = full_inputs["initial_state"].clone()
    outbuf = torch.empty_like(qbuf)

    cu_2048 = torch.tensor([0, 2048], dtype=torch.int64, device=flash_kda_device)
    cu_8192 = torch.tensor([0, 8192], dtype=torch.int64, device=flash_kda_device)

    def make_slice_inputs(t_len: int, cu: torch.Tensor, state_tensor: torch.Tensor):
        return {
            "q": qbuf[:, :t_len],
            "k": kbuf[:, :t_len],
            "v": vbuf[:, :t_len],
            "g": gbuf[:, :t_len],
            "beta": betabuf[:, :t_len],
            "A_log": A_log,
            "dt_bias": dt_bias,
            "initial_state": state_tensor,
            "cu_seqlens": cu,
        }

    expected_8192, expected_state_8192 = _reference(
        make_slice_inputs(8192, cu_8192, state_init.clone())
    )

    # 1. Warm both fast plans (T=2048 and T=8192).
    for t_len, cu in ((2048, cu_2048), (8192, cu_8192)):
        s = state_init.clone()
        recurrent_kda(
            **_strict_prefill_kwargs(make_slice_inputs(t_len, cu, s)),
            output=outbuf[:, :t_len],
            output_final_state=True,
            backend="vibecuda",
        )

    # 2. Hit T=2048 fast plan first so descriptor storage holds T=2048 TMA boxes.
    s_2048 = state_init.clone()
    recurrent_kda(
        **_strict_prefill_kwargs(make_slice_inputs(2048, cu_2048, s_2048)),
        output=outbuf[:, :2048],
        output_final_state=True,
        backend="vibecuda",
    )

    # 3. Poison outbuf and immediately hit T=8192 fast plan on the same base pointers.
    outbuf.fill_(float("nan"))
    s_8192 = state_init.clone()
    actual_8192, actual_state_8192 = recurrent_kda(
        **_strict_prefill_kwargs(make_slice_inputs(8192, cu_8192, s_8192)),
        output=outbuf[:, :8192],
        output_final_state=True,
        backend="vibecuda",
    )

    torch.testing.assert_close(
        actual_8192.float(),
        expected_8192.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_state_8192.float(),
        expected_state_8192.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_vibecuda_packed_prefill_sorts_sequences_on_device(flash_kda_device):
    seq_lens = [1300, 547, 2048]
    inputs = _make_inputs(
        seq_lens=seq_lens, num_heads=96, packed=True, initial_state=True, seed=4210
    )

    expected = _reference(inputs)
    call_inputs = {**inputs, "initial_state": inputs["initial_state"].clone()}
    output = torch.empty_like(inputs["q"])
    actual = recurrent_kda(
        **_strict_prefill_kwargs(call_inputs),
        output=output,
        output_final_state=True,
        backend="vibecuda",
    )

    torch.testing.assert_close(
        actual[0].float(),
        expected[0].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual[1].float(),
        expected[1].float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_vibecuda_prefill_without_initial_state(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=False, seed=4211
    )

    expected = _reference(inputs)
    output = torch.empty_like(inputs["q"])
    actual = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="vibecuda",
    )

    torch.testing.assert_close(
        actual[0].float(),
        expected[0].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual[1].float(),
        expected[1].float(),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("packed", [False, True])
def test_vibecuda_prefill_runs_on_nondefault_stream(flash_kda_device, packed):
    seq_lens = [1300, 547] if packed else [512]
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=96,
        packed=packed,
        initial_state=True,
        seed=4212 + packed,
    )
    initial_state_seed = inputs["initial_state"].clone()
    expected = _reference({**inputs, "initial_state": initial_state_seed.clone()})
    output = torch.empty_like(inputs["q"])
    stream = torch.cuda.Stream(device=flash_kda_device)
    stream.wait_stream(torch.cuda.current_stream(flash_kda_device))

    with torch.cuda.stream(stream):
        actual = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            output_final_state=True,
            backend="vibecuda",
        )
    stream.synchronize()

    assert actual[0].data_ptr() == output.data_ptr()
    assert actual[1] is inputs["initial_state"]
    torch.testing.assert_close(
        actual[0].float(), expected[0].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual[1].float(), expected[1].float(), atol=1e-2, rtol=1e-2
    )


def test_vibecuda_prefill_without_initial_or_final_state(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=False, seed=4214
    )
    expected, _ = _reference(inputs)
    output = torch.empty_like(inputs["q"])
    # Run twice so the second invocation exercises the fast-plan hit path with
    # initial_state=None and output_final_state=False.
    for _ in range(2):
        output.zero_()
        actual, final_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            output_final_state=False,
            backend="vibecuda",
        )
        assert actual.data_ptr() == output.data_ptr()
        assert final_state is None
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=1e-2, rtol=1e-2
        )


def test_vibecuda_prefill_rejects_token_row_strided_beta(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4220
    )
    padded_beta = torch.empty(
        (1, 512, 192), dtype=torch.bfloat16, device=flash_kda_device
    )
    padded_beta[:, :, :96] = inputs["beta"]
    strided_beta = padded_beta[:, :, :96]
    assert not strided_beta.is_contiguous()

    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(
            **_strict_prefill_kwargs({**inputs, "beta": strided_beta}),
            backend="vibecuda",
        )


def test_vibecuda_prefill_rejects_state_pool_inputs(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4221
    )
    ssm_state_indices = torch.zeros((1,), dtype=torch.int32, device=flash_kda_device)

    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            ssm_state_indices=ssm_state_indices,
            backend="vibecuda",
        )


@pytest.mark.parametrize("field", ("A_log", "dt_bias", "lower_bound"))
def test_vibecuda_prefill_rejects_missing_prep_operands(flash_kda_device, field):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4223
    )
    kwargs = {**_strict_prefill_kwargs(inputs), field: None}

    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(**kwargs, backend="vibecuda")


def test_vibecuda_prefill_rejects_initial_state_source_inputs(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4224
    )
    source_pool = torch.zeros(
        (4, 96, 128, 128), dtype=torch.bfloat16, device=flash_kda_device
    )
    source_indices = torch.zeros((1,), dtype=torch.int32, device=flash_kda_device)

    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            initial_state_source=source_pool,
            initial_state_indices=source_indices,
            backend="vibecuda",
        )


def test_vibecuda_prefill_rejects_state_checkpoints(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4222
    )
    interval = 32
    num_checkpoints = (512 + interval - 1) // interval
    checkpoints = torch.empty(
        (num_checkpoints, 96, 128, 128), dtype=torch.bfloat16, device=flash_kda_device
    )
    checkpoint_cu_starts = torch.tensor(
        [0, num_checkpoints], dtype=torch.int64, device=flash_kda_device
    )

    with pytest.raises(ValueError, match="backend='vibecuda' does not support"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=interval,
            backend="vibecuda",
        )


def test_vibecuda_graph_capture_requires_explicit_workspace(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4230
    )
    output = torch.empty_like(inputs["q"])
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "backend": "vibecuda",
    }

    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(RuntimeError, match="requires an explicit"),
        torch.cuda.graph(graph),
    ):
        recurrent_kda(**call_kwargs)


def test_vibecuda_m128_prefill_cuda_graph_capture_and_replay(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512], num_heads=96, packed=False, initial_state=True, seed=4231
    )
    initial_state_seed = inputs["initial_state"].clone()
    expected = _reference({**inputs, "initial_state": initial_state_seed.clone()})
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))

    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "vibecuda",
    }
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    assert workspace._captured
    torch.testing.assert_close(
        captured_output.float(), expected[0].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected[1].float(), atol=1e-2, rtol=1e-2
    )
    with pytest.raises(RuntimeError, match="captured by another CUDA graph"):
        second_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(second_graph, stream=capture_stream):
            recurrent_kda(**call_kwargs)


def test_vibecuda_packed_prefill_cuda_graph_capture_and_replay(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[1300, 547], num_heads=96, packed=True, initial_state=True, seed=4232
    )
    initial_state_seed = inputs["initial_state"].clone()
    seq_order = torch.tensor([0, 1], dtype=torch.int32, device=flash_kda_device)
    expected = _reference({**inputs, "initial_state": initial_state_seed.clone()})
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))

    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "seq_order": seq_order,
        "prefill_workspace": workspace,
        "backend": "vibecuda",
    }
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    assert workspace._captured
    torch.testing.assert_close(
        captured_output.float(), expected[0].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected[1].float(), atol=1e-2, rtol=1e-2
    )


def test_vibecuda_cuda_graph_refreshes_padded_beta(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[32], num_heads=12, packed=False, initial_state=True, seed=4233
    )
    initial_state_seed = inputs["initial_state"].clone()
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "vibecuda",
    }
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["beta"].fill_(2.0)
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    expected = _reference({**inputs, "initial_state": initial_state_seed.clone()})
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    torch.testing.assert_close(
        captured_output.float(), expected[0].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected[1].float(), atol=1e-2, rtol=1e-2
    )


def test_vibecuda_cuda_graph_workspaces_are_isolated(flash_kda_device):
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    bundles = []
    for seed in (4234, 4235):
        inputs = _make_inputs(
            seq_lens=[512],
            num_heads=64,
            packed=False,
            initial_state=True,
            seed=seed,
        )
        state_seed = inputs["initial_state"].clone()
        expected = _reference({**inputs, "initial_state": state_seed.clone()})
        output = torch.empty_like(inputs["q"])
        workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
        call_kwargs = {
            **_strict_prefill_kwargs(inputs),
            "output": output,
            "output_final_state": True,
            "prefill_workspace": workspace,
            "backend": "vibecuda",
        }
        capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
        with torch.cuda.stream(capture_stream):
            recurrent_kda(**call_kwargs)
            inputs["initial_state"].copy_(state_seed)
            output.zero_()
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            recurrent_kda(**call_kwargs)
        bundles.append((graph, workspace, inputs, state_seed, output, expected))

    first_state = bundles[0][1]._vibecuda_state
    second_state = bundles[1][1]._vibecuda_state
    assert first_state is not None and second_state is not None
    assert (
        first_state.descriptor_storages["m64"].data_ptr()
        != second_state.descriptor_storages["m64"].data_ptr()
    )
    for bundle_index in (0, 1, 0, 1):
        graph, _workspace, inputs, state_seed, output, expected = bundles[bundle_index]
        with torch.cuda.stream(capture_stream):
            inputs["initial_state"].copy_(state_seed)
            output.fill_(float("nan"))
        capture_stream.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(), expected[0].float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            inputs["initial_state"].float(),
            expected[1].float(),
            atol=1e-2,
            rtol=1e-2,
        )


def test_vibecuda_default_stream_calls_retain_no_large_scratch(
    flash_kda_device, monkeypatch
):
    monkeypatch.setattr(kda_vibecuda_backend, "_vibecuda_stream_workspaces", {})
    split_inputs = _make_inputs(
        seq_lens=[16384], num_heads=1, packed=False, initial_state=True, seed=4250
    )
    fn16_inputs = _make_inputs(
        seq_lens=[512], num_heads=12, packed=False, initial_state=True, seed=4251
    )
    pad_inputs = _make_inputs(
        seq_lens=[16], num_heads=12, packed=False, initial_state=True, seed=4252
    )
    for inputs in (split_inputs, fn16_inputs, pad_inputs):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            output_final_state=True,
            backend="vibecuda",
        )
    torch.cuda.synchronize(flash_kda_device)
    assert len(kda_vibecuda_backend._vibecuda_stream_workspaces) == 1
    (workspace,) = kda_vibecuda_backend._vibecuda_stream_workspaces.values()
    state = workspace._vibecuda_state
    assert state is not None
    assert state.owned_by_stream_workspace is True
    for attr in (
        "_beta_padding",
        "_split_state",
        "_split_map_state",
        "_split_carry",
        "_split_out",
        "_split_map_state_bf16",
        "_split_lookback_flags",
        "_fn16_qd",
        "_fn16_kd",
        "_fn16_w",
        "_fn16_qk",
        "_fn16_diag",
    ):
        assert getattr(state, attr) is None, f"{attr} was retained on default stream"


@pytest.mark.parametrize(
    ("compute_capability", "expected_target"),
    [
        ((10, 0), "sm100a"),
        ((10, 3), "sm103a"),
    ],
)
def test_vibecuda_prefill_target_resolution(
    monkeypatch, compute_capability, expected_target
):
    monkeypatch.setattr(
        kda_vibecuda_backend,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    assert (
        kda_vibecuda_api._vibecuda_prefill_target(torch.device("cuda"))
        == expected_target
    )


def test_vibecuda_prefill_target_rejects_unsupported_arch(monkeypatch):
    monkeypatch.setattr(
        kda_vibecuda_backend,
        "get_compute_capability",
        lambda device: (9, 0),
    )
    with pytest.raises(RuntimeError, match=r"10\.0 and 10\.3"):
        kda_vibecuda_api._vibecuda_prefill_target(torch.device("cuda"))
