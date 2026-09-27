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

import inspect
from typing import Any, cast

import pytest
import torch  # pyright: ignore[reportMissingImports]

import flashinfer.kda_decode as kda_decode


def _arguments(num_sequences=4, num_rows=4):
    return {
        "x": torch.empty((num_rows, 1)),
        "weight": torch.empty((1,)),
        "conv_state": torch.empty((1,)),
        "raw_gate": torch.empty((1,)),
        "raw_beta": torch.empty((1,)),
        "A_log": torch.empty((1,)),
        "dt_bias": torch.empty((1,)),
        "state_indices": torch.tensor(
            [[11 + 11 * index] for index in range(num_sequences)], dtype=torch.int32
        ),
        "state": torch.empty((1,)),
        "output_gate": torch.empty((1,)),
        "norm_weight": torch.empty((1,)),
        "query_start_loc": torch.arange(num_sequences + 1, dtype=torch.int32),
        "num_accepted_tokens": torch.ones(num_sequences, dtype=torch.int32),
    }


def test_packed_fused_kda_decode_signature_and_public_name():
    packed = inspect.signature(kda_decode.packed_fused_kda_decode)
    legacy = inspect.signature(kda_decode.fused_kda_decode)
    packed_parameters = list(packed.parameters.values())
    legacy_parameters = list(legacy.parameters.values())
    assert [parameter.name for parameter in packed_parameters[:14]] == [
        parameter.name for parameter in legacy_parameters[:14]
    ]
    assert [parameter.kind for parameter in packed_parameters[:14]] == [
        parameter.kind for parameter in legacy_parameters[:14]
    ]
    assert [parameter.default for parameter in packed_parameters[:14]] == [
        parameter.default for parameter in legacy_parameters[:14]
    ]
    assert [parameter.name for parameter in packed_parameters[14:]] == [
        "query_start_loc",
        "num_accepted_tokens",
        "t1_state_indices",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in packed_parameters[14:]
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in packed_parameters[14:16]
    )
    assert packed_parameters[16].default is None
    assert not hasattr(kda_decode, "fused_kda_decode_packed")
    flashinfer = __import__("flashinfer")
    assert not hasattr(flashinfer, "fused_kda_decode_packed")
    assert hasattr(flashinfer, "packed_fused_kda_decode")


def _capture_metadata(monkeypatch, arguments):
    captured = {}

    def fake_backend(**kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(kda_decode, "_run_fused_kda_decode", fake_backend)
    assert kda_decode.packed_fused_kda_decode(**arguments) == "output"
    return captured


def test_packed_fused_kda_decode_t2_rejects_t1_state_indices(monkeypatch):
    arguments = _arguments(num_sequences=4, num_rows=4)
    arguments["state_indices"] = torch.ones((4, 2), dtype=torch.int32)
    arguments["t1_state_indices"] = torch.ones((4,), dtype=torch.int32)
    monkeypatch.setattr(kda_decode, "_run_fused_kda_decode", lambda **kwargs: "bad")
    with pytest.raises(ValueError, match="only supported for T=1"):
        kda_decode.packed_fused_kda_decode(**arguments)


def test_packed_fused_kda_decode_t2_forwards_none_lower_bound(monkeypatch):
    captured = {}

    def fake_backend(**kwargs):
        captured.update(kwargs)
        return "output"

    monkeypatch.setattr(kda_decode, "_run_packed_fused_kda_decode", fake_backend)
    arguments = _arguments(num_sequences=4, num_rows=4)
    arguments["state_indices"] = torch.ones((4, 2), dtype=torch.int32)
    arguments["query_start_loc"] = torch.arange(5, dtype=torch.int32)
    arguments["num_accepted_tokens"] = torch.ones(4, dtype=torch.int32)
    assert kda_decode.packed_fused_kda_decode(**arguments, lower_bound=None) == "output"
    assert captured["lower_bound"] is None


def test_packed_fused_kda_decode_t1_forwards_metadata_without_remap(monkeypatch):
    arguments = _arguments(num_sequences=3, num_rows=5)
    arguments["state_indices"] = torch.tensor([[11], [22], [33]], dtype=torch.int32)
    arguments["query_start_loc"] = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    captured = _capture_metadata(monkeypatch, arguments)
    assert captured["query_start_loc"] is arguments["query_start_loc"]
    assert captured["packed_state_indices"] is arguments["state_indices"]
    assert (
        captured["state_indices"].data_ptr()
        == arguments["state_indices"][:, 0].data_ptr()
    )
    assert captured["state_indices"].shape == (3,)


def test_packed_fused_kda_decode_t1_forwards_pre_resolved_rows(monkeypatch):
    arguments = _arguments(num_sequences=3, num_rows=5)
    arguments["state_indices"] = torch.tensor([[11], [22], [33]], dtype=torch.int32)
    arguments["query_start_loc"] = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
    arguments["t1_state_indices"] = torch.tensor([11, 0, 33, 0, 0], dtype=torch.int32)
    captured = _capture_metadata(monkeypatch, arguments)
    assert captured["state_indices"] is arguments["t1_state_indices"]
    assert "query_start_loc" not in captured
    assert "packed_state_indices" not in captured


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "query_start_loc",
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            "dtype",
        ),
        ("state_indices", torch.ones((4, 1), dtype=torch.int64), "dtype"),
        ("query_start_loc", torch.arange(4, dtype=torch.int32), "shape"),
        (
            "query_start_loc",
            torch.arange(10, dtype=torch.int32)[::2],
            "contiguous",
        ),
    ],
)
def test_packed_fused_kda_decode_t1_validates_remap_structure(
    monkeypatch, field, value, error
):
    arguments = _arguments()
    arguments[field] = value
    monkeypatch.setattr(kda_decode, "_run_fused_kda_decode", lambda **kwargs: "output")
    with pytest.raises((TypeError, ValueError), match=error):
        kda_decode.packed_fused_kda_decode(**arguments)


def test_packed_fused_kda_decode_cache_getters_are_cached(monkeypatch):
    kernel_module = __import__(
        "flashinfer.kda_kernels.fused_kda_decode_multitoken", fromlist=["module"]
    )
    calls = []

    def fake_build(*args, **kwargs):
        del kwargs
        calls.append(args)
        return object()

    monkeypatch.setattr(kernel_module, "build_and_load_cute_dsl_kernel", fake_build)
    assert not hasattr(kernel_module, "_get_t1_remap_kernel")
    assert not hasattr(kernel_module, "remap_packed_t1_state_indices")
    assert hasattr(kernel_module._get_compiled_kernel, "cache_info")
    assert hasattr(kernel_module._get_compiled_kernel, "cache_clear")
    kernel_module._get_compiled_kernel.cache_clear()
    try:
        first = kernel_module._get_compiled_kernel(3, 12, -5.0, 1e-5, 2, 16, 1, 1, 1)
        second = kernel_module._get_compiled_kernel(3, 12, -5.0, 1e-5, 2, 16, 1, 1, 1)
        assert first is second
        assert len(calls) == 1
    finally:
        kernel_module._get_compiled_kernel.cache_clear()


def test_packed_fused_kda_decode_metadata_is_required():
    arguments = _arguments()
    positional = [
        arguments.get(name, parameter.default)
        for name, parameter in list(
            inspect.signature(kda_decode.fused_kda_decode).parameters.items()
        )[:14]
    ]
    with pytest.raises(TypeError):
        kda_decode.packed_fused_kda_decode(*positional)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        packed = cast(Any, kda_decode.packed_fused_kda_decode)
        packed(
            *positional,
            arguments["query_start_loc"],
            arguments["num_accepted_tokens"],
        )
