# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
from types import SimpleNamespace

import pytest
import torch


fused = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")


class _FakeTensor:
    def __init__(self, shape, strides, dtype, *, contiguous=False, data_ptr=0x100000):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self.dtype = dtype
        self.device = "cuda:0"
        self.is_cuda = True
        self._contiguous = contiguous
        self._data_ptr = data_ptr

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, index=None):
        return self._strides if index is None else self._strides[index]

    def is_contiguous(self):
        return self._contiguous

    def data_ptr(self):
        return self._data_ptr

    def element_size(self):
        return torch.empty((), dtype=self.dtype).element_size()


def _fake_inputs():
    rows = 4
    heads = 12
    hidden = heads * 128
    qkv = 3 * hidden
    slots = rows + 1
    return {
        "x": _FakeTensor((rows, qkv), (qkv + 17, 1), torch.bfloat16),
        "weight": _FakeTensor(
            (3, 4, hidden), (4 * hidden, hidden, 1), torch.float32, contiguous=True
        ),
        "conv_state": _FakeTensor(
            (slots, qkv, 3), (3 * qkv, 1, qkv), torch.bfloat16
        ),
        "raw_gate": _FakeTensor(
            (1, rows, heads, 128),
            (rows * hidden, hidden, 128, 1),
            torch.bfloat16,
            contiguous=True,
        ),
        "raw_beta": _FakeTensor(
            (1, rows, heads), (rows * (heads + 1), heads + 1, 1), torch.bfloat16
        ),
        "A_log": _FakeTensor((heads,), (1,), torch.float32, contiguous=True),
        "dt_bias": _FakeTensor((hidden,), (1,), torch.float32, contiguous=True),
        "state_indices": _FakeTensor((rows,), (1,), torch.int32, contiguous=True),
        "state": _FakeTensor(
            (slots, heads, 128, 128),
            (heads * 128 * 128, 128 * 128, 128, 1),
            torch.float32,
        ),
        "output_gate": _FakeTensor(
            (rows, heads, 128), (hidden + 7, 128, 1), torch.bfloat16
        ),
        "norm_weight": _FakeTensor((128,), (1,), torch.float32, contiguous=True),
        "output": _FakeTensor(
            (1, rows, heads, 128),
            (rows * hidden, hidden, 128, 1),
            torch.bfloat16,
            contiguous=True,
        ),
    }


def test_cake_selector_uses_only_explicit_mode_and_tensor_metadata(monkeypatch):
    inputs = _fake_inputs()
    variant = object()
    calls = []
    monkeypatch.setattr(fused, "get_cake_fused_kda_decode_variants", lambda: (variant,))
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))

    def select_variant(**kwargs):
        calls.append(kwargs)
        return variant

    monkeypatch.setattr(fused, "select_cake_fused_kda_decode_variant", select_variant)
    selected = fused._select_cake_variant(
        x=inputs["x"],
        conv_state=inputs["conv_state"],
        raw_beta=inputs["raw_beta"],
        state=inputs["state"],
        output_gate=inputs["output_gate"],
        output=inputs["output"],
        state_indices_mode="unique_or_null",
        lower_bound=-5.0,
        norm_eps=1e-5,
    )

    assert selected is variant
    assert calls == [
        {
            "target": "sm100a",
            "num_heads": 12,
            "num_rows": 4,
            "num_slots": 5,
            "state_dtype": "float32",
            "state_indices_mode": "unique_or_null",
            "lower_bound": -5.0,
            "norm_eps": 1e-5,
            "x_row_stride": 4625,
            "conv_slot_stride": 13824,
            "beta_row_stride": 13,
            "state_slot_stride": 196608,
            "output_gate_row_stride": 1543,
            "variants": (variant,),
        }
    ]


@pytest.mark.parametrize(
    ("name", "data_ptr"),
    (("conv_state", 0x100004), ("state", 0x100010), ("output", 0x100004)),
)
def test_cake_selector_rejects_misaligned_mutable_buffers(monkeypatch, name, data_ptr):
    inputs = _fake_inputs()
    original = inputs[name]
    inputs[name] = _FakeTensor(
        original.shape,
        original.stride(),
        original.dtype,
        contiguous=original.is_contiguous(),
        data_ptr=data_ptr,
    )
    monkeypatch.setattr(fused, "get_cake_fused_kda_decode_variants", lambda: (object(),))
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fused,
        "select_cake_fused_kda_decode_variant",
        lambda **kwargs: pytest.fail("misaligned inputs must be rejected before routing"),
    )

    assert (
        fused._select_cake_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state=inputs["state"],
            output_gate=inputs["output_gate"],
            output=inputs["output"],
            state_indices_mode="positive_unique",
            lower_bound=-5.0,
            norm_eps=1e-5,
        )
        is None
    )


@pytest.mark.parametrize(
    ("lower_bound", "expected_tail"),
    ((None, (0, 0.0, 1e-5)), (-5.0, (1, -5.0, 1e-5))),
)
def test_cake_submit_passes_caller_owned_tensors_without_staging(
    monkeypatch, lower_bound, expected_tail
):
    calls = []
    variant = SimpleNamespace(name="selected", target="sm100a")
    monkeypatch.setattr(
        fused,
        "load_cake_fused_kda_decode_module",
        lambda name, target: SimpleNamespace(run=lambda *args: calls.append(args)),
    )
    tensors = [object() for _ in range(12)]

    fused._run_cake_variant(
        variant,
        x=tensors[0],
        weight=tensors[1],
        conv_state=tensors[2],
        raw_gate=tensors[3],
        raw_beta=tensors[4],
        A_log=tensors[5],
        dt_bias=tensors[6],
        state_indices=tensors[7],
        state=tensors[8],
        output_gate=tensors[9],
        norm_weight=tensors[10],
        output=tensors[11],
        lower_bound=lower_bound,
        norm_eps=1e-5,
    )

    assert calls == [tuple(tensors) + expected_tail]


def _patch_validated_run(monkeypatch, cake_variant):
    cake_calls = []
    fallback_calls = []
    monkeypatch.setattr(fused, "_check_cuda_tensor", lambda *args: None)
    monkeypatch.setattr(fused, "_select_cake_variant", lambda **kwargs: cake_variant)
    monkeypatch.setattr(
        fused,
        "_run_cake_variant",
        lambda selected, **kwargs: cake_calls.append((selected, kwargs)),
    )
    monkeypatch.setattr(
        fused,
        "_get_compiled_kernel",
        lambda *args: lambda *kernel_args: fallback_calls.append(kernel_args),
    )
    return cake_calls, fallback_calls


def test_default_backend_preserves_cute_dsl_without_consulting_cake(monkeypatch):
    inputs = _fake_inputs()
    output = inputs.pop("output")
    cake_calls, fallback_calls = _patch_validated_run(monkeypatch, None)
    monkeypatch.setattr(
        fused,
        "_select_cake_variant",
        lambda **kwargs: pytest.fail("default backend must not consult Cake"),
    )

    assert fused.run_fused_kda_decode(**inputs, output=output) is output
    assert cake_calls == []
    assert len(fallback_calls) == 1


def test_strict_cake_requires_host_known_state_indices_mode():
    with pytest.raises(ValueError, match="requires the host-known state_indices_mode"):
        fused.run_fused_kda_decode(**_fake_inputs(), backend="cake")


def test_strict_cake_dispatches_without_cute_fallback(monkeypatch):
    inputs = _fake_inputs()
    output = inputs.pop("output")
    variant = object()
    cake_calls, fallback_calls = _patch_validated_run(monkeypatch, variant)

    result = fused.run_fused_kda_decode(
        **inputs,
        output=output,
        backend="cake",
        state_indices_mode="positive_unique",
    )

    assert result is output
    assert len(cake_calls) == 1
    assert cake_calls[0][0] is variant
    assert cake_calls[0][1]["output"] is output
    assert fallback_calls == []


@pytest.mark.parametrize("backend", ("cake", "auto"))
def test_no_cake_route_is_strict_or_falls_back(monkeypatch, backend):
    inputs = _fake_inputs()
    output = inputs.pop("output")
    cake_calls, fallback_calls = _patch_validated_run(monkeypatch, None)

    def call():
        return fused.run_fused_kda_decode(
            **inputs,
            output=output,
            backend=backend,
            state_indices_mode="positive_unique",
        )

    if backend == "cake":
        with pytest.raises(RuntimeError, match="does not have a route"):
            call()
        assert fallback_calls == []
    else:
        assert call() is output
        assert len(fallback_calls) == 1
    assert cake_calls == []
