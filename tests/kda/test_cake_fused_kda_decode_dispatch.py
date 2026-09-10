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
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.jit import cake_fused_kda_decode as cake_jit
from flashinfer.jit import core as jit_core


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
    ("capability", "target"), (((10, 0), "sm100a"), ((10, 3), "sm103a"))
)
def test_cake_selector_uses_only_explicit_mode_and_tensor_metadata(
    monkeypatch, capability, target
):
    inputs = _fake_inputs()
    variant = object()
    calls = []
    registry_targets = []

    def get_variants(requested_target):
        registry_targets.append(requested_target)
        return (variant,)

    monkeypatch.setattr(fused, "get_cake_fused_kda_decode_variants", get_variants)
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: capability)

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
    assert registry_targets == [target]
    assert calls == [
        {
            "target": target,
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


@pytest.mark.parametrize("capability", ((9, 0), (10, 1), (10, 2), (12, 0)))
def test_cake_selector_rejects_unregistered_architectures(monkeypatch, capability):
    inputs = _fake_inputs()
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: capability)
    monkeypatch.setattr(
        fused,
        "get_cake_fused_kda_decode_variants",
        lambda target: pytest.fail(
            "unsupported architectures must not load a registry"
        ),
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


def test_cake_targets_share_sources_but_have_distinct_build_identities():
    sm100_variants = cake_jit.get_cake_fused_kda_decode_variants()
    sm103_variants = cake_jit.get_cake_fused_kda_decode_variants("sm103a")
    assert sm100_variants == cake_jit.get_cake_fused_kda_decode_variants("sm100a")
    assert len(sm100_variants) == len(sm103_variants) == 46
    for sm100, sm103 in zip(sm100_variants, sm103_variants, strict=True):
        assert replace(sm100, target="sm103a") == sm103
        assert cake_jit.get_cake_fused_kda_decode_variant(sm103.name, "sm103a") == sm103
        sm100_uri = cake_jit.get_cake_fused_kda_decode_uri(sm100.name, "sm100a")
        sm103_uri = cake_jit.get_cake_fused_kda_decode_uri(sm103.name, "sm103a")
        assert sm100_uri != sm103_uri
        assert sm100_uri.endswith("_sm100a")
        assert sm103_uri.endswith("_sm103a")
    sm100_identity = cake_jit.get_cake_fused_kda_decode_program_identity()
    assert sm100_identity == cake_jit.get_cake_fused_kda_decode_program_identity(
        "sm100a"
    )
    assert sm100_identity != cake_jit.get_cake_fused_kda_decode_program_identity(
        "sm103a"
    )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_cake_selector_resolves_requested_target_registry(target):
    variant = cake_jit.select_cake_fused_kda_decode_variant(
        target=target,
        num_heads=12,
        num_rows=4,
        num_slots=5,
        state_dtype="float32",
        state_indices_mode="unique_or_null",
        lower_bound=-5.0,
        norm_eps=1e-5,
        x_row_stride=4625,
        conv_slot_stride=13824,
        beta_row_stride=13,
        state_slot_stride=196608,
        output_gate_row_stride=1543,
    )
    assert variant is not None
    assert variant.target == target
    assert variant in cake_jit.get_cake_fused_kda_decode_variants(target)


def _select_positive_route(target, heads, rows, *, lower_bound=-5.0, wide=False):
    hidden = heads * 128
    conv_stride = 9 * hidden + 2 * heads * 128 * 128
    state_stride = conv_stride // 2
    slots = (2**31 // state_stride + 2) if wide else rows + 1
    return cake_jit.select_cake_fused_kda_decode_variant(
        target=target,
        num_heads=heads,
        num_rows=rows,
        num_slots=slots,
        state_dtype="float32",
        state_indices_mode="positive_unique",
        lower_bound=lower_bound,
        norm_eps=1e-5,
        x_row_stride=3 * hidden + 17,
        conv_slot_stride=conv_stride,
        beta_row_stride=heads + 1,
        state_slot_stride=state_stride,
        output_gate_row_stride=hidden + 7,
    )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("heads", "rows", "expected"),
    (
        (12, 18, "wide512_positive_f32"),
        (12, 19, "wide512_vector4_positive_f32"),
        (12, 24, "wide512_vector4_positive_f32"),
        (12, 25, "compact_async_positive_f32"),
        (24, 9, "wide512_positive_f32"),
        (24, 10, "wide512_vector4_positive_f32"),
        (24, 12, "wide512_vector4_positive_f32"),
        (24, 13, "compact_async_positive_f32"),
        (32, 9, "wide512_positive_f32"),
        (48, 6, "wide512_positive_f32"),
        (96, 3, "wide512_positive_f32"),
        (12, 50, "wide512_positive_f32"),
        (12, 51, "high_work_positive_pr_eval_h12_f32"),
        (12, 55, "high_work_positive_pr_eval_h12_f32"),
        (12, 56, "wide512_positive_f32"),
        (12, 74, "wide512_positive_f32"),
        (12, 75, "high_work_positive_pr_eval_h12_f32"),
        (12, 80, "high_work_positive_pr_eval_h12_f32"),
        (12, 81, "wide512_positive_f32"),
        (24, 24, "wide512_positive_f32"),
        (24, 25, "high_work_positive_pr_eval_h24_f32"),
        (24, 28, "wide512_positive_f32"),
        (32, 18, "wide512_positive_f32"),
        (32, 19, "high_work_positive_pr_eval_h32_f32"),
        (32, 21, "wide512_positive_f32"),
        (32, 31, "wide512_positive_f32"),
        (32, 32, "high_work_positive_pr_eval_h32_f32"),
        (48, 13, "high_work_positive_pr_eval_h48_f32"),
        (96, 10, "high_work_positive_h96_pr_strides_f32"),
    ),
)
def test_positive_f32_producer_and_partial_wave_routes(target, heads, rows, expected):
    variant = _select_positive_route(target, heads, rows)
    assert variant is not None
    assert variant.name == expected


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_new_positive_routes_preserve_wide_offsets_and_runtime_config(target):
    variant = _select_positive_route(target, 12, 24, wide=True)
    assert variant.name == "wide512_vector4_positive_f32_wide_slot_offsets"
    variant = _select_positive_route(target, 12, 51, lower_bound=-20.0)
    assert variant.name == "high_work_positive_f32"


@pytest.mark.parametrize(("target", "minor"), (("sm100a", 0), ("sm103a", 3)))
@pytest.mark.parametrize(
    ("name", "min_blocks"),
    (
        ("wide512_f32_wide_slot_offsets", False),
        ("compact_async_f32_wide_slot_offsets", True),
        ("compact_async_pr_eval_h96_f32_wide_slot_offsets", True),
    ),
)
def test_cake_jit_spec_uses_exact_target_flags(
    monkeypatch, tmp_path, target, minor, name, min_blocks
):
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, f"{minor}a")}
    )
    monkeypatch.setattr(cake_jit.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    cake_jit.gen_cake_fused_kda_decode_module.cache_clear()
    variant = cake_jit.get_cake_fused_kda_decode_variant(name, target)
    spec = cake_jit.gen_cake_fused_kda_decode_module(variant.name, target)
    assert spec.name == cake_jit.get_cake_fused_kda_decode_uri(variant.name, target)
    assert spec.sources[0] == variant.body_path
    assert [
        flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")
    ] == [f"-gencode=arch=compute_10{minor}a,code=sm_10{minor}a"]
    assert (
        f"-DFLASHINFER_CAKE_FUSED_KDA_DECODE_TARGET_MINOR={minor}"
        in spec.extra_cuda_cflags
    )
    expected_occupancy_flags = ["-Xptxas=--minnctapersm=3"] if min_blocks else []
    assert [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-Xptxas=--minnctapersm=")
    ] == expected_occupancy_flags
    cake_jit.gen_cake_fused_kda_decode_module.cache_clear()


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
    monkeypatch.setattr(
        fused, "get_cake_fused_kda_decode_variants", lambda target: (object(),)
    )
    monkeypatch.setattr(fused, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(
        fused,
        "select_cake_fused_kda_decode_variant",
        lambda **kwargs: pytest.fail(
            "misaligned inputs must be rejected before routing"
        ),
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
