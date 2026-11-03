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

"""Tests for the experimental ``RecurrentKDAPrefillWrapper`` (see #4936).

Numerics for the kernels the wrapper hands off to are covered by the stable
lane in ``tests/kda/test_recurrent_kda_prefill.py``; these cover the wrapper's
own contract: that ``plan`` stages offsets without reading them on the host,
that ``run`` forwards the planned buffers, that the planned path agrees with the
eager path it wraps, and that captured graphs replay against fixed addresses.
"""

import importlib
import inspect

import flashinfer
import pytest
import torch

from flashinfer.kda import RecurrentKDAPrefillWrapper, recurrent_kda
from flashinfer.utils import get_compute_capability

from tests.test_helpers.kda_prefill import (
    _chunk16_debug_reference,
    _make_inputs,
    _reference,
    _strict_prefill_kwargs,
    cpu_route_tensors,
    packed_prefill_inputs,
)

kda_api = importlib.import_module("flashinfer.kda")


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


def test_prefill_wrapper_is_exported_from_the_top_level():
    assert flashinfer.RecurrentKDAPrefillWrapper is RecurrentKDAPrefillWrapper


def test_prefill_wrapper_is_marked_experimental():
    """The marking is the point of the change, so removing it must fail here.

    The warning itself fires once per process, so it is not assertable in a
    lane that already exercises ``plan``; the flag and the injected banner are.
    """

    assert RecurrentKDAPrefillWrapper.is_experimental
    for method in (RecurrentKDAPrefillWrapper.plan, RecurrentKDAPrefillWrapper.run):
        assert method.is_experimental
        assert "experimental" in method.__doc__


def test_prefill_wrapper_plan_builds_stable_device_metadata(cuda_device, monkeypatch):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    offsets = torch.tensor([0, 0, 7, 7, 12], device=cuda_device)
    original_to = torch.Tensor.to

    def reject_device_to_host(self, *args, **kwargs):
        if args and torch.device(args[0]).type == "cpu" and self.is_cuda:
            pytest.fail("wrapper plan must not read CUDA offsets on the host")
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", reject_device_to_host)
    wrapper.plan(offsets)

    cu_seqlens_ptr = wrapper._impl.cu_seqlens_buf.data_ptr()
    seq_order_ptr = wrapper._impl.seq_order_buf.data_ptr()
    cu_chunks_ptr = wrapper._impl.cu_chunks_buf.data_ptr()
    assert wrapper._impl.cu_seqlens_buf.dtype == torch.int64
    assert wrapper._impl.cu_seqlens_buf.tolist() == [0, 0, 7, 7, 12]
    assert wrapper._impl.workspace._cute_dsl_generate_planned_metadata is True

    wrapper.plan(torch.tensor([0, 0, 2, 2, 12], device=cuda_device))
    assert wrapper._impl.cu_seqlens_buf.data_ptr() == cu_seqlens_ptr
    assert wrapper._impl.seq_order_buf.data_ptr() == seq_order_ptr
    assert wrapper._impl.cu_chunks_buf.data_ptr() == cu_chunks_ptr

    with pytest.raises(ValueError, match="number of sequences is fixed"):
        wrapper.plan(torch.tensor([0, 2, 12], device=cuda_device))


def test_prefill_wrapper_run_forwards_planned_buffers(cuda_device, monkeypatch):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    wrapper.plan(torch.tensor([0, 1, 3], device=cuda_device))
    calls = []
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_api,
        "recurrent_kda",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )
    tensors = cpu_route_tensors(token_count=3)
    tensors = {
        key: value.to(cuda_device) if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }

    tensors["scale"] = 0.125
    tensors["output_final_state"] = True

    assert wrapper.run(**tensors) is sentinel
    assert calls[0]["cu_seqlens"] is wrapper._impl.cu_seqlens_buf
    assert calls[0]["seq_order"] is wrapper._impl.seq_order_buf
    assert calls[0]["prefill_workspace"] is wrapper._impl.workspace
    assert calls[0]["backend"] == "cute-dsl"

    # Every parameter of run must reach recurrent_kda, so that dropping one
    # from the handoff fails here rather than silently changing behaviour.
    forwarded = set(inspect.signature(RecurrentKDAPrefillWrapper.run).parameters)
    assert forwarded - {"self"} <= set(calls[0])
    for name, value in tensors.items():
        assert calls[0][name] is value or calls[0][name] == value
    assert wrapper._impl.workspace._cute_dsl_cu_chunks is wrapper._impl.cu_chunks_buf
    assert wrapper._impl.workspace._cute_dsl_generate_planned_metadata is True


@pytest.mark.parametrize("checkpointed", [False, True])
def test_prefill_wrapper_planned_path_matches_eager_reference(
    cuda_device, checkpointed
):
    """The planned path must agree with the eager packed path it wraps.

    The wrapper only reorders work and stages metadata, so the reference is the
    same kernels driven without a plan. The checkpointed case also covers the
    configuration the stable cake comparisons exercise eagerly.
    """

    if torch.cuda.get_device_capability(cuda_device) not in ((10, 0), (10, 3)):
        pytest.skip("packed CuTe DSL prefill requires CC 10.0 or 10.3")

    seq_lens = [33, 65] if checkpointed else [7, 29, 13]
    num_heads = 12 if checkpointed else 2
    inputs = packed_prefill_inputs(
        cuda_device, seq_lens=seq_lens, num_heads=num_heads, seed=4936
    )
    common = dict(inputs)
    cu_seqlens = common.pop("cu_seqlens")

    if checkpointed:
        interval = 32
        counts = [-(-length // interval) for length in seq_lens]
        starts = [0]
        for count in counts:
            starts.append(starts[-1] + count)
        common["ssm_state_indices"] = torch.arange(
            len(seq_lens), dtype=torch.int32, device=cuda_device
        )
        common["checkpoint_cu_starts"] = torch.tensor(
            starts, dtype=torch.int64, device=cuda_device
        )
        common["checkpoint_every_n_tokens"] = interval

    def per_call_buffers():
        """Fresh state pool per call: the kernel writes final state back in place."""

        if not checkpointed:
            return {}
        return {
            "initial_state": torch.zeros(
                len(seq_lens),
                num_heads,
                128,
                128,
                dtype=torch.bfloat16,
                device=cuda_device,
            ),
            "state_checkpoints": torch.empty(
                starts[-1],
                num_heads,
                128,
                128,
                dtype=torch.bfloat16,
                device=cuda_device,
            ),
        }

    eager_ckpt = per_call_buffers()
    expected = kda_api.recurrent_kda(
        **common,
        **eager_ckpt,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        backend="cute-dsl",
    )

    planned_ckpt = per_call_buffers()
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    wrapper.plan(cu_seqlens)
    actual = wrapper.run(**common, **planned_ckpt, output_final_state=True)

    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)
    if checkpointed:
        torch.testing.assert_close(
            planned_ckpt["state_checkpoints"],
            eager_ckpt["state_checkpoints"],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("num_heads", [12, 64])
def test_cute_dsl_planned_zero_length_cuda_graph_capture_and_replay(
    flash_kda_device, num_heads
):
    inputs = _make_inputs(
        seq_lens=[0, 17, 0, 33],
        num_heads=num_heads,
        packed=True,
        initial_state=True,
        seed=2040 + num_heads,
    )
    initial_state_seed = inputs["initial_state"].clone()
    reference_inputs = {
        **inputs,
        "initial_state": initial_state_seed.clone(),
    }
    reference = _chunk16_debug_reference if num_heads == 12 else _reference
    expected_output, expected_state = reference(reference_inputs)

    wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
    wrapper.plan(inputs["cu_seqlens"])
    output = torch.empty_like(inputs["q"])
    run_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
    }
    run_kwargs.pop("cu_seqlens")

    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    with torch.cuda.stream(capture_stream):
        wrapper.run(**run_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()
    assert wrapper._impl.seq_order_buf.tolist() == [3, 1, 0, 2]
    if num_heads == 12:
        assert wrapper._impl.cu_chunks_buf.tolist() == [0, 0, 2, 2, 5]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = wrapper.run(**run_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    assert wrapper._impl.workspace._captured
    torch.testing.assert_close(
        captured_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )

    # Reuse the captured graph with different device-resident offsets.  The
    # graph's metadata prepass must refresh both scheduling buffers without a
    # data-dependent host read or a changed launch geometry.
    replay_seq_lens = [8, 0, 25, 17]
    replay_offsets = torch.tensor(
        [0, 8, 8, 33, 50], dtype=torch.int64, device=flash_kda_device
    )
    replay_inputs = {**inputs, "cu_seqlens": replay_offsets}
    expected_replay_output, expected_replay_state = reference(
        {**replay_inputs, "initial_state": initial_state_seed.clone()}
    )
    with torch.cuda.stream(capture_stream):
        wrapper.plan(replay_offsets)
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
        graph.replay()
    capture_stream.synchronize()

    assert wrapper._impl.seq_order_buf.tolist() == [2, 3, 0, 1]
    if num_heads == 12:
        assert wrapper._impl.cu_chunks_buf.tolist() == [0, 1, 1, 3, 5]
    assert sum(replay_seq_lens) == output.shape[1]
    torch.testing.assert_close(
        captured_output.float(),
        expected_replay_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        captured_state.float(),
        expected_replay_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_cute_dsl_cuda_graph_replay_updates_offsets_and_indexed_checkpoints(
    flash_kda_device,
):
    inputs = _make_inputs(
        seq_lens=[64, 131],
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=4898,
    )
    initial_states = inputs["initial_state"].clone()

    def eager_control(offsets, state_indices, checkpoint_starts, checkpoint_indices):
        state_pool = torch.zeros(
            (4, 12, 128, 128),
            dtype=initial_states.dtype,
            device=flash_kda_device,
        )
        state_pool[state_indices.long()] = initial_states
        checkpoint_pool = torch.full(
            (6, 12, 128, 128),
            torch.nan,
            dtype=initial_states.dtype,
            device=flash_kda_device,
        )
        output, returned_state, returned_checkpoints = recurrent_kda(
            **_strict_prefill_kwargs(
                {
                    **inputs,
                    "cu_seqlens": offsets,
                    "initial_state": state_pool,
                }
            ),
            output=torch.empty_like(inputs["q"]),
            output_final_state=True,
            ssm_state_indices=state_indices,
            state_checkpoints=checkpoint_pool,
            checkpoint_cu_starts=checkpoint_starts,
            checkpoint_state_indices=checkpoint_indices,
            checkpoint_every_n_tokens=64,
            backend="cute-dsl",
        )
        assert returned_state is state_pool
        assert returned_checkpoints is checkpoint_pool
        return (
            output.clone(),
            state_pool[state_indices.long()].clone(),
            checkpoint_pool[checkpoint_indices.long()].clone(),
        )

    offsets = inputs["cu_seqlens"]
    state_indices = torch.tensor([0, 2], dtype=torch.int32, device=flash_kda_device)
    checkpoint_starts = torch.tensor(
        [0, 1, 3], dtype=torch.int64, device=flash_kda_device
    )
    checkpoint_indices = torch.tensor(
        [5, 1, 4], dtype=torch.int32, device=flash_kda_device
    )
    expected = eager_control(
        offsets, state_indices, checkpoint_starts, checkpoint_indices
    )

    state_pool = torch.zeros(
        (4, 12, 128, 128),
        dtype=initial_states.dtype,
        device=flash_kda_device,
    )
    checkpoint_pool = torch.full(
        (6, 12, 128, 128),
        torch.nan,
        dtype=initial_states.dtype,
        device=flash_kda_device,
    )
    output = torch.empty_like(inputs["q"])
    wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
    wrapper.plan(offsets)
    run_kwargs = {
        **_strict_prefill_kwargs({**inputs, "initial_state": state_pool}),
        "output": output,
        "output_final_state": True,
        "ssm_state_indices": state_indices,
        "state_checkpoints": checkpoint_pool,
        "checkpoint_cu_starts": checkpoint_starts,
        "checkpoint_state_indices": checkpoint_indices,
        "checkpoint_every_n_tokens": 64,
    }
    run_kwargs.pop("cu_seqlens")

    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    with torch.cuda.stream(capture_stream):
        state_pool[state_indices.long()] = initial_states
        wrapper.run(**run_kwargs)
        state_pool.zero_()
        state_pool[state_indices.long()] = initial_states
        checkpoint_pool.fill_(float("nan"))
        output.fill_(float("nan"))
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state, captured_checkpoints = wrapper.run(
            **run_kwargs
        )

    with torch.cuda.stream(capture_stream):
        state_pool.zero_()
        state_pool[state_indices.long()] = initial_states
        checkpoint_pool.fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()
    capture_stream.synchronize()

    assert captured_output is output
    assert captured_state is state_pool
    assert captured_checkpoints is checkpoint_pool
    torch.testing.assert_close(
        output.float(), expected[0].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_pool[state_indices.long()].float(),
        expected[1].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        checkpoint_pool[checkpoint_indices.long()].float(),
        expected[2].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.count_nonzero(state_pool[[1, 3]]) == 0
    assert torch.isnan(checkpoint_pool[[0, 2, 3]]).all()

    # Keep all captured addresses fixed while changing the sequence split,
    # initial-state slots, checkpoint prefix, and checkpoint destinations.
    replay_offsets = torch.tensor(
        [0, 129, 195], dtype=torch.int64, device=flash_kda_device
    )
    replay_state_indices = torch.tensor(
        [3, 1], dtype=torch.int32, device=flash_kda_device
    )
    replay_checkpoint_starts = torch.tensor(
        [0, 2, 3], dtype=torch.int64, device=flash_kda_device
    )
    replay_checkpoint_indices = torch.tensor(
        [0, 3, 2], dtype=torch.int32, device=flash_kda_device
    )
    expected_replay = eager_control(
        replay_offsets,
        replay_state_indices,
        replay_checkpoint_starts,
        replay_checkpoint_indices,
    )
    with torch.cuda.stream(capture_stream):
        wrapper.plan(replay_offsets)
        state_indices.copy_(replay_state_indices)
        checkpoint_starts.copy_(replay_checkpoint_starts)
        checkpoint_indices.copy_(replay_checkpoint_indices)
        state_pool.zero_()
        state_pool[replay_state_indices.long()] = initial_states
        checkpoint_pool.fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()
    capture_stream.synchronize()

    assert wrapper._impl.seq_order_buf.tolist() == [0, 1]
    assert wrapper._impl.cu_chunks_buf.tolist() == [0, 9, 14]
    torch.testing.assert_close(
        output.float(), expected_replay[0].float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_pool[replay_state_indices.long()].float(),
        expected_replay[1].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        checkpoint_pool[replay_checkpoint_indices.long()].float(),
        expected_replay[2].float(),
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.count_nonzero(state_pool[[0, 2]]) == 0
    assert torch.isnan(checkpoint_pool[[1, 4, 5]]).all()


def test_prefill_wrapper_rejects_invalid_plans_and_out_of_order_runs(
    cuda_device, monkeypatch
):
    """The validation that moved to the planner must still reject bad input."""

    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    tensors = {
        key: value.to(cuda_device) if isinstance(value, torch.Tensor) else value
        for key, value in cpu_route_tensors(token_count=3).items()
    }

    with pytest.raises(RuntimeError, match="call plan before run"):
        wrapper.run(**tensors)

    with pytest.raises(TypeError, match="cu_seqlens must be a torch.Tensor"):
        wrapper.plan([0, 1, 3])

    for bad in (
        torch.tensor([0.0, 1.0, 3.0], device=cuda_device),
        torch.tensor([[0, 1, 3]], device=cuda_device),
        torch.tensor([0], device=cuda_device),
        torch.tensor([0, 1, 2, 3], device=cuda_device)[::2],
    ):
        with pytest.raises(ValueError, match="at least two entries"):
            wrapper.plan(bad)

    monkeypatch.setattr(kda_api, "recurrent_kda", lambda **kwargs: None)
    wrapper.plan(torch.tensor([0, 1, 3], device=cuda_device))
    wrapper.run(**tensors)
    with pytest.raises(ValueError, match="q token count is fixed"):
        wrapper.run(**{**tensors, "q": tensors["q"][:, :2]})

    fresh = RecurrentKDAPrefillWrapper(cuda_device)
    fresh.plan(torch.tensor([0, 1, 3], device=cuda_device))
    with pytest.raises(ValueError, match="q must be a rank-4 tensor"):
        fresh.run(**{**tensors, "q": tensors["q"][0]})
