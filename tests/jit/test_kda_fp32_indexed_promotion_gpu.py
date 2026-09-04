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

from __future__ import annotations

import math
import os

import pytest
import torch

from flashinfer.jit import kda_fp32_indexed_promotion as promotion


_EXPECTED_MODE_ENV = "FLASHINFER_KDA_PROMOTION_EXPECTED_MODE"
_CAPABILITY_FACTS = {
    (10, 0): ("sm100a", "sm_100a"),
    (10, 3): ("sm103a", "sm_103a"),
}
_HEADS = 6
_HEAD_DIM = 128
_TOTAL_TOKENS = 1024
_SEQUENCE_LENGTHS = (128,) * 8
_STATE_POOL_CAPACITY = 65
_STATE_INDICES = (62, 20, 43, 1, 24, 47, 5, 28)
_SEED = 13003


def _expected_mode() -> promotion.PromotionMode:
    value = os.environ.get(_EXPECTED_MODE_ENV)
    if value is None:
        pytest.skip(f"set {_EXPECTED_MODE_ENV} to 'cubin' or 'cuda'")
    if value not in ("cubin", "cuda"):
        pytest.fail(f"{_EXPECTED_MODE_ENV} must be 'cubin' or 'cuda', got {value!r}")
    return value


def _blackwell_device() -> tuple[torch.device, tuple[int, int], str, str, int]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    capability = tuple(torch.cuda.get_device_capability(device))
    if capability not in _CAPABILITY_FACTS:
        pytest.skip("the promoted KDA payload requires exact CC 10.0 or CC 10.3")
    target, architecture = _CAPABILITY_FACTS[capability]
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    return device, capability, target, architecture, sm_count


def _make_inputs(device: torch.device) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(_SEED)
    q_shape = (1, _TOTAL_TOKENS, _HEADS, _HEAD_DIM)
    q = torch.randn(
        q_shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        q_shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn(
        q_shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    g = torch.randn(
        q_shape,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    beta = torch.randn(
        (1, _TOTAL_TOKENS, _HEADS),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    A_log = torch.rand(
        (_HEADS,),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    dt_bias = torch.rand(
        (_HEADS, _HEAD_DIM),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    state_indices = torch.tensor(
        _STATE_INDICES,
        dtype=torch.int32,
        device=device,
    )
    state_pool = torch.zeros(
        (_STATE_POOL_CAPACITY, _HEADS, _HEAD_DIM, _HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    selected_state = (
        torch.randn(
            (len(_SEQUENCE_LENGTHS), _HEADS, _HEAD_DIM, _HEAD_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        * 0.25
    )
    state_pool.index_copy_(0, state_indices.long(), selected_state)
    cu_seqlens = torch.tensor(
        (0, 128, 256, 384, 512, 640, 768, 896, 1024),
        dtype=torch.int64,
        device=device,
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": 1.0 / math.sqrt(_HEAD_DIM),
        "initial_state": state_pool,
        "output_final_state": True,
        "lower_bound": -5.0,
        "cu_seqlens": cu_seqlens,
        "output": torch.empty_like(q),
        "seq_order": None,
        "prefill_workspace": None,
        "state_indices": state_indices,
        "state_checkpoints": None,
        "checkpoint_cu_starts": None,
        "checkpoint_every_n_tokens": 0,
    }


def _assert_unselected_state_unchanged(
    state_pool: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    selected = torch.zeros(
        _STATE_POOL_CAPACITY,
        dtype=torch.bool,
        device=state_pool.device,
    )
    selected[state_indices.long()] = True
    assert torch.equal(state_pool[~selected], initial_state[~selected])


def _production_reference(
    arguments: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        import flash_kda
    except ImportError as exc:
        pytest.fail(
            "the production FlashKDA reference must be importable for promotion validation",
            pytrace=False,
        )
        raise AssertionError from exc

    state_pool = arguments["initial_state"]
    state_indices = arguments["state_indices"]
    output = arguments["output"]
    cu_seqlens = arguments["cu_seqlens"]
    assert isinstance(state_pool, torch.Tensor) and state_pool.dtype == torch.float32
    assert (
        isinstance(state_indices, torch.Tensor) and state_indices.dtype == torch.int32
    )
    assert isinstance(output, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)

    compact_initial = state_pool.index_select(0, state_indices.long()).contiguous()
    compact_final = torch.empty_like(compact_initial)
    reference_output = torch.empty_like(output)
    workspace_size = flash_kda.get_workspace_size(
        _TOTAL_TOKENS,
        _HEADS,
        len(_SEQUENCE_LENGTHS),
    )
    workspace = torch.empty(
        workspace_size,
        dtype=torch.uint8,
        device=output.device,
    )
    flash_kda._fwd_raw(
        arguments["q"],
        arguments["k"],
        arguments["v"],
        arguments["g"],
        arguments["beta"],
        arguments["scale"],
        reference_output,
        workspace,
        arguments["A_log"],
        arguments["dt_bias"],
        arguments["lower_bound"],
        initial_state=compact_initial,
        final_state=compact_final,
        cu_seqlens=cu_seqlens,
    )
    reference_state = state_pool.clone()
    reference_state.index_copy_(0, state_indices.long(), compact_final)
    torch.cuda.synchronize(output.device)
    return reference_output, reference_state


def test_checked_in_fp32_indexed_kda_promotion_gpu() -> None:
    expected_mode = _expected_mode()
    device, capability, target, architecture, sm_count = _blackwell_device()

    specs = promotion.get_module_specs()
    assert [(spec.target, spec.mode, len(spec.routes)) for spec in specs] == [
        ("sm100a", expected_mode, 48),
        ("sm103a", expected_mode, 48),
    ]
    assert promotion.selected_mode() == expected_mode

    spec = next(spec for spec in specs if spec.target == target)
    selector_arguments = {
        "gpu_arch": architecture,
        "sm_count": sm_count,
        "fixed_layout": False,
        "sequence_lengths": _SEQUENCE_LENGTHS,
        "num_heads": _HEADS,
        "use_initial_state": True,
        "store_final_state": True,
    }
    manifest_selector = {
        **selector_arguments,
        "sequence_lengths": list(_SEQUENCE_LENGTHS),
    }
    assert sum(route["selector"] == manifest_selector for route in spec.routes) == 1

    loaded = promotion.load(compute_capability=capability, mode=expected_mode)
    assert promotion.load(compute_capability=capability, mode=expected_mode) is loaded
    selected_route = loaded.dispatcher.select(**selector_arguments)
    assert isinstance(selected_route, str) and selected_route

    arguments = _make_inputs(device)
    state_pool = arguments["initial_state"]
    output = arguments["output"]
    state_indices = arguments["state_indices"]
    assert isinstance(state_pool, torch.Tensor)
    assert isinstance(output, torch.Tensor)
    assert isinstance(state_indices, torch.Tensor)
    initial_state = state_pool.clone()
    reference_output, reference_state = _production_reference(arguments)

    prepared = promotion.prepare(**arguments, compute_capability=capability)
    returned_output, returned_state = prepared.launch()
    assert returned_output is output
    assert returned_output.data_ptr() == output.data_ptr()
    assert returned_state is state_pool
    assert returned_state.data_ptr() == state_pool.data_ptr()
    assert returned_state.dtype == torch.float32
    torch.testing.assert_close(output, reference_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(state_pool, reference_state, rtol=1e-2, atol=1e-2)
    _assert_unselected_state_unchanged(state_pool, initial_state, state_indices)
    expected_output = output.clone()
    expected_selected_state = state_pool.index_select(0, state_indices.long()).clone()

    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError, match="is closed"):
        prepared.launch()

    run_state = initial_state.clone()
    run_output = torch.empty_like(output)
    run_arguments = {
        **arguments,
        "initial_state": run_state,
        "output": run_output,
    }
    returned_output, returned_state = promotion.run(
        **run_arguments,
        compute_capability=capability,
    )
    assert returned_output is run_output
    assert returned_output.data_ptr() == run_output.data_ptr()
    assert returned_state is run_state
    assert returned_state.data_ptr() == run_state.data_ptr()
    assert returned_state.dtype == torch.float32
    torch.testing.assert_close(run_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(
        run_state.index_select(0, state_indices.long()),
        expected_selected_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(run_output, reference_output, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(run_state, reference_state, rtol=1e-2, atol=1e-2)
    _assert_unselected_state_unchanged(run_state, initial_state, state_indices)
