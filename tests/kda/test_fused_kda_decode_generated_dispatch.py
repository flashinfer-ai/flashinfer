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
import math
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
        "conv_state": _FakeTensor((slots, qkv, 3), (3 * qkv, 1, qkv), torch.bfloat16),
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


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([3, 2, 1], (False, True)),
        ([3, 0, -1], (False, False)),
        ([3, 2, 3], (True, True)),
        ([3, 0, 3], (True, False)),
    ],
)
def test_state_index_slot_classification_cpu(values, expected):
    assert fused._classify_state_indices(values) == expected


def _clear_slot_classification_cache():
    with fused._STATE_INDICES_CLASSIFICATION_LOCK:
        fused._STATE_INDICES_CLASSIFICATION_CACHE.clear()
        fused._STATE_INDICES_CLASSIFICATION_LRU.clear()


def test_slot_classification_cache_tracks_tensor_version_cpu(monkeypatch):
    state_indices = torch.tensor([3, 2, 1], dtype=torch.int32)
    _clear_slot_classification_cache()
    monkeypatch.setattr(fused, "is_current_stream_capturing", lambda: False)
    try:
        assert fused._cached_state_indices_classification(state_indices) == (
            False,
            True,
        )
        state_indices.copy_(torch.tensor([3, 0, 3], dtype=torch.int32))
        assert fused._cached_state_indices_classification(state_indices) == (
            True,
            False,
        )
    finally:
        _clear_slot_classification_cache()


def test_slot_classification_cache_miss_fails_during_capture_cpu(monkeypatch):
    state_indices = torch.tensor([3, 2, 1], dtype=torch.int32)
    _clear_slot_classification_cache()
    monkeypatch.setattr(fused, "is_current_stream_capturing", lambda: True)
    try:
        with pytest.raises(RuntimeError, match="cache miss during CUDA Graph capture"):
            fused._cached_state_indices_classification(state_indices)
    finally:
        _clear_slot_classification_cache()


def test_pending_manifest_falls_back_without_inspecting_indices(monkeypatch):
    inputs = _fake_inputs()
    monkeypatch.setattr(fused, "load_fused_kda_decode_generated_variants", lambda: ())
    monkeypatch.setattr(
        fused,
        "_cached_state_indices_classification",
        lambda tensor: pytest.fail("pending manifests must not inspect state_indices"),
    )

    assert (
        fused._select_generated_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state_indices=inputs["state_indices"],
            state=inputs["state"],
            output_gate=inputs["output_gate"],
            lower_bound=-5.0,
            norm_eps=1e-5,
        )
        is None
    )


def test_generated_selector_receives_exact_runtime_facts(monkeypatch):
    inputs = _fake_inputs()
    variant = object()
    calls = []
    monkeypatch.setattr(
        fused, "load_fused_kda_decode_generated_variants", lambda: (variant,)
    )
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fused, "_cached_state_indices_classification", lambda tensor: (False, True)
    )

    def select_variant(**kwargs):
        calls.append(kwargs)
        return variant if kwargs["slot_class"] == "positive_unique" else None

    monkeypatch.setattr(
        fused, "select_fused_kda_decode_generated_variant", select_variant
    )

    selected = fused._select_generated_variant(
        x=inputs["x"],
        conv_state=inputs["conv_state"],
        raw_beta=inputs["raw_beta"],
        state_indices=inputs["state_indices"],
        state=inputs["state"],
        output_gate=inputs["output_gate"],
        lower_bound=-5.0,
        norm_eps=1e-5,
    )

    assert selected is variant
    assert calls == [
        {
            "target": "sm100a",
            "num_heads": 12,
            "num_rows": 4,
            "state_dtype": "float32",
            "slot_class": "positive_unique",
            "lower_bound": -5.0,
            "norm_eps": 1e-5,
            "x_row_stride": 4625,
            "conv_slot_stride": 13824,
            "beta_row_stride": 13,
            "state_slot_stride": 196608,
            "output_gate_row_stride": 1543,
            "variants": (variant,),
        },
        {
            **calls[0],
            "slot_class": "unique_or_null",
        },
        {
            **calls[0],
            "slot_class": "repeated_positive",
        },
    ]


def test_generated_selector_skips_index_copy_when_no_route_matches(monkeypatch):
    inputs = _fake_inputs()
    monkeypatch.setattr(
        fused, "load_fused_kda_decode_generated_variants", lambda: (object(),)
    )
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fused, "select_fused_kda_decode_generated_variant", lambda **kwargs: None
    )
    monkeypatch.setattr(
        fused,
        "_cached_state_indices_classification",
        lambda tensor: pytest.fail(
            "manifest gaps must fall back before classification"
        ),
    )

    assert (
        fused._select_generated_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state_indices=inputs["state_indices"],
            state=inputs["state"],
            output_gate=inputs["output_gate"],
            lower_bound=-5.0,
            norm_eps=1e-5,
        )
        is None
    )


@pytest.mark.parametrize(
    ("lower_bound", "norm_eps"),
    [(float("nan"), 1e-5), (float("-inf"), 1e-5), (-5.0, float("inf"))],
)
def test_generated_selector_falls_back_for_nonfinite_public_scalars(
    monkeypatch, lower_bound, norm_eps
):
    inputs = _fake_inputs()
    monkeypatch.setattr(
        fused, "load_fused_kda_decode_generated_variants", lambda: (object(),)
    )
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fused,
        "_cached_state_indices_classification",
        lambda tensor: pytest.fail(
            "unsupported scalars must fall back before classification"
        ),
    )

    assert (
        fused._select_generated_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state_indices=inputs["state_indices"],
            state=inputs["state"],
            output_gate=inputs["output_gate"],
            lower_bound=lower_bound,
            norm_eps=norm_eps,
        )
        is None
    )


def test_generated_selector_falls_back_for_unaligned_state(monkeypatch):
    inputs = _fake_inputs()
    inputs["state"] = _FakeTensor(
        inputs["state"].shape,
        inputs["state"].stride(),
        torch.float32,
        data_ptr=0x100010,
    )
    monkeypatch.setattr(
        fused, "load_fused_kda_decode_generated_variants", lambda: (object(),)
    )
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fused,
        "_cached_state_indices_classification",
        lambda tensor: pytest.fail(
            "unaligned inputs must fall back before classification"
        ),
    )

    assert (
        fused._select_generated_variant(
            x=inputs["x"],
            conv_state=inputs["conv_state"],
            raw_beta=inputs["raw_beta"],
            state_indices=inputs["state_indices"],
            state=inputs["state"],
            output_gate=inputs["output_gate"],
            lower_bound=-5.0,
            norm_eps=1e-5,
        )
        is None
    )


@pytest.mark.parametrize(
    ("lower_bound", "expected_tail"),
    [
        (None, (0, 0.0, 1e-5)),
        (-5.0, (1, -5.0 * math.log2(math.e), 1e-5)),
    ],
)
def test_generated_launch_uses_public_scalar_semantics(
    monkeypatch, lower_bound, expected_tail
):
    calls = []
    variant = SimpleNamespace(name="selected", target="sm100a")
    monkeypatch.setattr(
        fused,
        "load_fused_kda_decode_generated_module",
        lambda name, target: SimpleNamespace(run=lambda *args: calls.append(args)),
    )
    tensors = [object() for _ in range(12)]

    fused._run_generated_variant(
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


@pytest.mark.parametrize("use_generated", [False, True])
def test_public_backend_dispatches_or_preserves_cute_fallback_cpu(
    monkeypatch, use_generated
):
    inputs = _fake_inputs()
    output = inputs.pop("output")
    variant = object()
    generated_calls = []
    fallback_calls = []
    monkeypatch.setattr(fused, "_check_cuda_tensor", lambda *args: None)
    monkeypatch.setattr(
        fused,
        "_select_generated_variant",
        lambda **kwargs: variant if use_generated else None,
    )
    monkeypatch.setattr(
        fused,
        "_run_generated_variant",
        lambda selected, **kwargs: generated_calls.append((selected, kwargs)),
    )
    monkeypatch.setattr(
        fused,
        "_get_compiled_kernel",
        lambda *args: lambda *kernel_args: fallback_calls.append(kernel_args),
    )

    result = fused.run_fused_kda_decode(**inputs, output=output)

    assert result is output
    if use_generated:
        assert len(generated_calls) == 1
        assert generated_calls[0][0] is variant
        assert generated_calls[0][1]["output"] is output
        assert fallback_calls == []
    else:
        assert generated_calls == []
        assert len(fallback_calls) == 1
        assert fallback_calls[0][-1] is output
