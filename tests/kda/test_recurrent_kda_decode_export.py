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

import dataclasses
import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from flashinfer.kda_decode import recurrent_kda

recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")

_D = 128
_T = 5
_VARIANT_PREFIX = "d128_t5_precomputed_gram_split"


@pytest.fixture
def b200():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        pytest.skip("frozen FlashKDA decode tests require exact B200 / sm_100a")
    return device


@dataclasses.dataclass(frozen=True)
class _FakeTensor:
    """Small Tensor-shaped object for CPU-only dispatch-contract tests."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    strides: tuple[int, ...]
    ptr: int
    device: str = "cuda:0"
    cuda: bool = True
    contiguous: bool = True

    @property
    def is_cuda(self):
        return self.cuda

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, dim=None):
        if dim is None:
            return self.strides
        return self.strides[dim]

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return {
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.float32: 4,
            torch.int32: 4,
            torch.int64: 8,
        }[self.dtype]

    def data_ptr(self):
        return self.ptr

    def is_contiguous(self):
        return self.contiguous


def _contiguous_strides(shape):
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= size
    return tuple(reversed(strides))


def _fake_tensor(shape, dtype, ptr, *, strides=None, contiguous=True, cuda=True):
    return _FakeTensor(
        shape=shape,
        dtype=dtype,
        strides=_contiguous_strides(shape) if strides is None else strides,
        ptr=ptr,
        contiguous=contiguous,
        cuda=cuda,
    )


def _fake_selector_kwargs():
    address = iter(range(0x100000000000, 0x200000000000, 0x100000000))

    def tensor(shape, dtype=torch.bfloat16, **kwargs):
        return _fake_tensor(shape, dtype, next(address), **kwargs)

    q = tensor((1, _T, 1, _D))
    return {
        "q": q,
        "k": tensor(q.shape),
        "v": tensor((1, _T, 1, _D)),
        "g": tensor((1, _T, 1, _D)),
        "beta": tensor((1, _T, 1)),
        "state": tensor((_T, 1, _D, _D)),
        "out": tensor((1, _T, 1, _D)),
        "cu_seqlens": tensor((2,), torch.int32),
        "ssm_state_indices": tensor((_T,), torch.int32),
        "num_accepted_tokens": tensor((1,), torch.int32),
        "scale": _D**-0.5,
        "num_tokens": _T,
        "num_spec_tokens": _T - 1,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "lower_bound": None,
        "A_log": None,
        "dt_bias": None,
        "initial_state_source": None,
        "beta_is_logit": False,
    }


def _patch_cpu_selector_environment(monkeypatch, *, sm_count=148, cc=(10, 0)):
    monkeypatch.setattr(recurrent_module, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sm_count),
    )


@pytest.mark.parametrize(
    ("work", "sm_count", "expected_split"),
    [
        (55, 148, 8),
        (56, 148, 2),
        (74, 148, 2),
        (75, 148, 4),
        (111, 148, 4),
        (112, 148, 2),
        (222, 148, 2),
        (223, 148, 1),
    ],
)
def test_auto_value_split_boundaries(work, sm_count, expected_split):
    assert (
        recurrent_module._select_flash_kda_decode_value_split(work, sm_count)
        == expected_split
    )


def test_exact_cpu_contract_selects_frozen_variant(monkeypatch):
    _patch_cpu_selector_environment(monkeypatch)
    assert recurrent_module._select_flash_kda_decode_variant(
        **_fake_selector_kwargs()
    ) == (_VARIANT_PREFIX + "8")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_tokens", 4),
        ("num_spec_tokens", None),
        ("num_spec_tokens", 3),
        ("cu_seqlens", None),
        ("ssm_state_indices", None),
        ("initial_state_source", object()),
        ("beta_is_logit", True),
        ("use_qk_l2norm_in_kernel", False),
        ("use_gate_in_kernel", True),
        ("lower_bound", -5.0),
        ("A_log", object()),
        ("dt_bias", object()),
        ("scale", float("inf")),
    ],
)
def test_public_contract_mismatches_strictly_fall_back_cpu(monkeypatch, field, value):
    _patch_cpu_selector_environment(monkeypatch)
    kwargs = _fake_selector_kwargs()
    kwargs[field] = value
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) is None


def _replace_fake(kwargs, name, **changes):
    kwargs[name] = dataclasses.replace(kwargs[name], **changes)
    return kwargs


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda kw: _replace_fake(kw, "q", cuda=False),
            id="q-on-cpu",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "q", dtype=torch.float16),
            id="wrong-data-dtype",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "q", contiguous=False),
            id="noncontiguous-q",
        ),
        pytest.param(
            lambda kw: _replace_fake(
                kw,
                "g",
                strides=(_T * (_D + 8), _D + 8, _D + 1, 1),
                contiguous=False,
            ),
            id="noncompact-gate-tail",
        ),
        pytest.param(
            lambda kw: _replace_fake(
                kw,
                "g",
                strides=(_T * (_D - 4), _D - 4, _D, 1),
                contiguous=False,
            ),
            id="overlapping-gate-token-stride",
        ),
        pytest.param(
            lambda kw: _replace_fake(
                kw,
                "state",
                strides=(_D * _D - 8, _D * _D, _D, 1),
                contiguous=False,
            ),
            id="overlapping-state-slot-stride",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "cu_seqlens", dtype=torch.int64),
            id="wrong-cu-seqlens-dtype",
        ),
        pytest.param(
            lambda kw: _replace_fake(
                kw, "ssm_state_indices", shape=(_T - 1,), strides=(1,)
            ),
            id="wrong-state-indices-size",
        ),
        pytest.param(
            lambda kw: _replace_fake(
                kw, "num_accepted_tokens", shape=(2,), strides=(1,)
            ),
            id="wrong-num-accepted-size",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "q", ptr=kw["q"].ptr + 2),
            id="misaligned-q",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "out", ptr=kw["q"].ptr),
            id="output-aliases-q",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "state", ptr=kw["q"].ptr),
            id="state-aliases-q",
        ),
    ],
)
def test_tensor_layout_and_alias_mismatches_strictly_fall_back_cpu(
    monkeypatch, mutation
):
    _patch_cpu_selector_environment(monkeypatch)
    kwargs = mutation(_fake_selector_kwargs())
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) is None


def test_non_sm100a_strictly_falls_back_cpu(monkeypatch):
    _patch_cpu_selector_environment(monkeypatch, cc=(10, 3))
    assert (
        recurrent_module._select_flash_kda_decode_variant(**_fake_selector_kwargs())
        is None
    )


class _RecorderModule:
    def __init__(self):
        self.calls = []

    def run(self, *args):
        self.calls.append(args)


def test_frozen_runner_forwards_ffi_abi_and_current_stream_cpu(monkeypatch):
    module = _RecorderModule()
    loaded = []

    def get_module(variant):
        loaded.append(variant)
        return module

    monkeypatch.setattr(recurrent_module, "get_flash_kda_decode_module", get_module)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=0xCAFE),
    )
    tensors = _fake_selector_kwargs()
    dummy_f32 = _fake_tensor((1,), torch.float32, 0x300000000000)
    variant = _VARIANT_PREFIX + "4"
    recurrent_module._run_flash_kda_decode(
        variant,
        q=tensors["q"],
        k=tensors["k"],
        v=tensors["v"],
        g=tensors["g"],
        beta=tensors["beta"],
        state=tensors["state"],
        out=tensors["out"],
        cu_seqlens=tensors["cu_seqlens"],
        ssm_state_indices=tensors["ssm_state_indices"],
        num_accepted_tokens=tensors["num_accepted_tokens"],
        scale=tensors["scale"],
        dummy_f32=dummy_f32,
    )

    assert loaded == [variant]
    (args,) = module.calls
    assert len(args) == 15
    assert args[:5] == (
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["g"],
        tensors["beta"],
    )
    assert args[5] is args[6] is dummy_f32
    assert args[7:12] == (
        tensors["state"],
        tensors["out"],
        tensors["cu_seqlens"],
        tensors["ssm_state_indices"],
        tensors["num_accepted_tokens"],
    )
    assert args[12] == tensors["scale"]
    assert args[13] == 0.0
    assert args[14] == 0xCAFE


def _padded_slot_state(slots, num_value_heads, device, *, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    slot_stride = num_value_heads * _D * _D + 8
    storage = torch.randn(
        slots * slot_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    state = torch.as_strided(
        storage,
        (slots, num_value_heads, _D, _D),
        (slot_stride, _D * _D, _D, 1),
    )
    return state, storage


def _padded_token_gate(total_tokens, num_value_heads, device, *, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    token_stride = num_value_heads * _D + 8
    storage = torch.randn(
        total_tokens * token_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    gate = torch.as_strided(
        storage,
        (1, total_tokens, num_value_heads, _D),
        (total_tokens * token_stride, token_stride, _D, 1),
    )
    return gate, storage


def _make_case(
    device,
    *,
    num_sequences,
    num_heads=1,
    num_value_heads=1,
    padded_gate=False,
    padded_state=False,
    padded_last_sequence=False,
    accepted_tokens=None,
    seed=42,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    total_tokens = num_sequences * _T
    q = torch.randn(
        (1, total_tokens, num_heads, _D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        q.shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    v = torch.randn(
        (1, total_tokens, num_value_heads, _D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    if padded_gate:
        g, gate_storage = _padded_token_gate(
            total_tokens, num_value_heads, device, seed=seed + 1
        )
        g.copy_(
            F.logsigmoid(
                torch.randn(
                    g.shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
            ).to(torch.bfloat16)
        )
    else:
        g = F.logsigmoid(
            torch.randn(
                (1, total_tokens, num_value_heads, _D),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        ).to(torch.bfloat16)
        gate_storage = None
    beta = torch.rand(
        (1, total_tokens, num_value_heads),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    cu_seqlens = torch.arange(0, total_tokens + 1, _T, dtype=torch.int32, device=device)
    ssm_state_indices = torch.arange(
        num_sequences * _T, dtype=torch.int32, device=device
    ).reshape(num_sequences, _T)
    if padded_last_sequence:
        ssm_state_indices[-1].fill_(-1)
    if accepted_tokens is None:
        num_accepted_tokens = (
            torch.arange(num_sequences, dtype=torch.int32, device=device) % _T
        ) + 1
    else:
        num_accepted_tokens = torch.tensor(
            accepted_tokens, dtype=torch.int32, device=device
        )
    slots = num_sequences * _T
    if padded_state:
        state, state_storage = _padded_slot_state(
            slots, num_value_heads, device, seed=seed + 2
        )
    else:
        state = torch.randn(
            (slots, num_value_heads, _D, _D),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        state_storage = None
    output = torch.empty_like(v)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": _D**-0.5,
        "initial_state": state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "lower_bound": None,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": _T - 1,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
        "_state_storage": state_storage,
        "_gate_storage": gate_storage,
    }


def _call_kwargs(case, *, state=None, output=None):
    kwargs = {key: value for key, value in case.items() if not key.startswith("_")}
    kwargs["initial_state"] = case["initial_state"] if state is None else state
    kwargs["output"] = case["output"] if output is None else output
    return kwargs


def _clone_state_with_layout(state):
    if state.is_contiguous():
        return state.clone()
    clone, _ = _padded_slot_state(
        state.shape[0],
        state.shape[1],
        state.device,
        seed=2026,
    )
    clone.copy_(state)
    return clone


def _representative_num_sequences(sm_count, num_value_heads, split):
    if split == 8:
        target_work = 1
    elif split == 4:
        target_work = sm_count // 2 + 1
    elif split == 1:
        target_work = 3 * sm_count // 2 + 1
    elif split == 2:
        target_work = 3 * sm_count // 8 + 1
    else:
        raise ValueError(f"unexpected split: {split}")
    return max(1, math.ceil(target_work / num_value_heads))


@pytest.mark.parametrize("split", [8, 2, 4, 1])
def test_public_recurrent_kda_exact_routes_match_upstream_cute(
    b200, monkeypatch, split
):
    sm_count = torch.cuda.get_device_properties(b200).multi_processor_count
    num_value_heads = 32
    num_sequences = _representative_num_sequences(sm_count, num_value_heads, split)
    assert (
        recurrent_module._select_flash_kda_decode_value_split(
            num_sequences * num_value_heads, sm_count
        )
        == split
    )
    case = _make_case(
        b200,
        num_sequences=num_sequences,
        num_heads=16,
        num_value_heads=num_value_heads,
        seed=2040 + split,
    )
    initial = _clone_state_with_layout(case["initial_state"])
    baseline_state = _clone_state_with_layout(initial)
    actual_state = _clone_state_with_layout(initial)
    baseline_output = torch.empty_like(case["output"])
    actual_output_buffer = torch.empty_like(case["output"])

    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            recurrent_module,
            "_select_flash_kda_decode_variant",
            lambda **kwargs: None,
        )
        expected_output, expected_state = recurrent_kda(
            **_call_kwargs(case, state=baseline_state, output=baseline_output)
        )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    actual_output, actual_state_result = recurrent_kda(
        **_call_kwargs(case, state=actual_state, output=actual_output_buffer)
    )

    assert frozen_calls == [_VARIANT_PREFIX + str(split)]
    assert actual_output.data_ptr() == actual_output_buffer.data_ptr()
    assert actual_state_result is actual_state
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_public_route_supports_padded_slots_nat_and_outer_strides(b200, monkeypatch):
    accepted_tokens = [0, 1, _T, _T + 7]
    case = _make_case(
        b200,
        num_sequences=len(accepted_tokens),
        num_heads=16,
        num_value_heads=32,
        padded_gate=True,
        padded_state=True,
        padded_last_sequence=True,
        accepted_tokens=accepted_tokens,
        seed=2050,
    )
    assert not case["g"].is_contiguous()
    assert case["g"].stride(-2) == _D
    assert case["g"].stride(1) > 32 * _D
    assert not case["initial_state"].is_contiguous()
    assert case["initial_state"].stride(0) > 32 * _D * _D

    initial = _clone_state_with_layout(case["initial_state"])
    baseline_state = _clone_state_with_layout(initial)
    actual_state = _clone_state_with_layout(initial)
    baseline_output = torch.empty_like(case["output"])
    actual_output_buffer = torch.empty_like(case["output"])
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            recurrent_module,
            "_select_flash_kda_decode_variant",
            lambda **kwargs: None,
        )
        expected_output, expected_state = recurrent_kda(
            **_call_kwargs(case, state=baseline_state, output=baseline_output)
        )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    actual_output, actual_state_result = recurrent_kda(
        **_call_kwargs(case, state=actual_state, output=actual_output_buffer)
    )

    expected_split = recurrent_module._select_flash_kda_decode_value_split(
        len(accepted_tokens) * 32,
        torch.cuda.get_device_properties(b200).multi_processor_count,
    )
    assert frozen_calls == [_VARIANT_PREFIX + str(expected_split)]
    assert actual_state_result is actual_state
    assert actual_state_result.stride() == initial.stride()
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )
    padded_tokens = slice(-_T, None)
    torch.testing.assert_close(
        actual_output[:, padded_tokens],
        torch.zeros_like(actual_output[:, padded_tokens]),
        atol=0,
        rtol=0,
    )


def test_frozen_decode_cuda_graph_on_non_default_stream(b200, monkeypatch):
    case = _make_case(
        b200,
        num_sequences=1,
        num_heads=16,
        num_value_heads=32,
        seed=2060,
    )
    initial = case["initial_state"].clone()
    baseline_state = initial.clone()
    baseline_output = torch.empty_like(case["output"])
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(
            recurrent_module,
            "_select_flash_kda_decode_variant",
            lambda **kwargs: None,
        )
        expected_output, expected_state = recurrent_kda(
            **_call_kwargs(case, state=baseline_state, output=baseline_output)
        )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(
            (
                variant,
                int(torch.cuda.current_stream(b200).cuda_stream),
            )
        )
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    graph_state = initial.clone()
    graph_output = torch.empty_like(case["output"])
    graph_kwargs = _call_kwargs(case, state=graph_state, output=graph_output)
    capture_stream = torch.cuda.Stream(device=b200)
    capture_stream.wait_stream(torch.cuda.current_stream(b200))
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**graph_kwargs)
        graph_state.copy_(initial)
        graph_output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**graph_kwargs)

    expected_variant = _VARIANT_PREFIX + "8"
    assert [variant for variant, _stream in frozen_calls] == [
        expected_variant,
        expected_variant,
    ]
    assert all(
        stream == int(capture_stream.cuda_stream) for _variant, stream in frozen_calls
    )
    for _ in range(2):
        with torch.cuda.stream(capture_stream):
            graph_state.copy_(initial)
            graph_output.fill_(float("nan"))
        capture_stream.synchronize()
        with torch.cuda.stream(capture_stream):
            graph.replay()
        torch.cuda.synchronize()
        assert captured_output.data_ptr() == graph_output.data_ptr()
        assert captured_state is graph_state
        torch.testing.assert_close(
            captured_output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            captured_state.float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
