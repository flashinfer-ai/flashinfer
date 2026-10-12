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
from flashinfer.jit import env as jit_env


fused = importlib.import_module("flashinfer.kda_kernels.fused_kda_decode")

# Multiprocessor count of the B200 device the positive-unique FP32 route bands
# were measured on. Route selection hardcodes this (the producer's own
# B200_SM_COUNT constant, never a live device query -- see
# cake_jit._ROUTE_SM_COUNT); select_cake_fused_kda_decode_variant takes no
# sm_count parameter at all. Only used here for cake_fused_kda_decode_grid
# calls (grid SIZING for the persistent stream family genuinely takes the
# launching device's live multiprocessor count) and to pin the mocked device
# query in tests that exercise fused._run_cake_variant directly.
_SM_COUNT = 148
_LOG2E = 1.4426950408889634
_ABI_NAMES = {
    kind: tuple(name for _kind, name, _dtype in abi)
    for kind, abi in cake_jit.CAKE_FUSED_KDA_DECODE_ABIS.items()
}


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
    monkeypatch.setattr(fused, "_device_index", lambda device: 0)
    # No _device_sm_count mock here: _select_cake_variant no longer queries the
    # device at all (route selection is a fixed constant, never a device
    # scalar -- see the sm_count fix comment on select_cake_fused_kda_decode_variant).

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


def test_cake_targets_share_programs_but_have_distinct_build_identities():
    sm100_variants = cake_jit.get_cake_fused_kda_decode_variants()
    sm103_variants = cake_jit.get_cake_fused_kda_decode_variants("sm103a")
    assert sm100_variants == cake_jit.get_cake_fused_kda_decode_variants("sm100a")
    # 17 FACTORIES entries (export.py) x 2 slot-offset widths (32/64-bit twins).
    assert len(sm100_variants) == len(sm103_variants) == 34
    for sm100, sm103 in zip(sm100_variants, sm103_variants, strict=True):
        assert sm100.module == sm103.module
        assert sm100.body_path == sm103.body_path
        assert (
            replace(sm100, target="sm103a", source_sha256=sm103.source_sha256) == sm103
        )
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


def test_registry_routes_every_variant_to_one_generated_program():
    variants = cake_jit.get_cake_fused_kda_decode_variants()
    assert set(cake_jit.ROUTES) == {variant.name for variant in variants}
    assert set(cake_jit.ROUTES.values()) == set(cake_jit.MODULES)
    assert len(set(cake_jit.ROUTES.values())) == len(cake_jit.ROUTES)
    for variant in variants:
        record = cake_jit.MODULES[variant.module]
        assert variant.body_path.name.startswith("cake_fused_kda_decode_")
        assert variant.binding_path.name.startswith("cake_fused_kda_decode_")
        assert variant.body_path.is_file() and variant.binding_path.is_file()
        assert variant.kernel_symbol == record["kernel_symbol"]
        assert variant.threads == record["launch"]["block"][0]
        assert variant.dynamic_smem_bytes == record["launch"]["dynamic_smem_bytes"]
        assert (list(record["launch"]["cluster"]) or [1])[0] == variant.cluster_x
        bound = {name for kind, name in variant.arg_plan if kind != "grid"}
        assert set(_ABI_NAMES[variant.abi_kind]) <= bound
        assert bound <= set(_ABI_NAMES[variant.abi_kind]) | {"device_index"}
        if variant.name in cake_jit._PRECISE_MATH_VARIANTS:
            assert not set(variant.extra_cuda_cflags) & cake_jit._FAST_MATH_FLAGS
        else:
            assert "--use_fast_math" in variant.extra_cuda_cflags
        assert len(variant.source_sha256) == 64


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


# The generic stride-taking programs serve every layout: the evaluation layout
# (padded strides, lower_bound -5, norm_eps 1e-5) selects the same kernels as any
# other layout.
@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("heads", "rows", "expected"),
    (
        (12, 18, "wide512_positive_f32"),
        (12, 19, "wide512_vector4_positive_f32"),
        (12, 22, "wide512_positive_f32"),
        (12, 24, "wide512_vector4_positive_f32"),
        (12, 25, "compact_async_positive_f32"),
        (24, 9, "wide512_positive_f32"),
        (24, 10, "wide512_positive_f32"),
        (24, 12, "wide512_vector4_positive_f32"),
        (24, 13, "compact_async_positive_f32"),
        (32, 9, "wide512_positive_f32"),
        (48, 6, "wide512_positive_f32"),
        (96, 3, "wide512_positive_f32"),
        (96, 4, "compact_async_positive_f32"),
        (12, 50, "wide512_positive_f32"),
        (12, 51, "high_work_positive_f32"),
        (12, 55, "high_work_positive_f32"),
        (12, 56, "wide512_positive_f32"),
        (12, 74, "wide512_positive_f32"),
        (12, 75, "high_work_positive_f32"),
        (12, 80, "high_work_positive_f32"),
        (12, 81, "wide512_positive_f32"),
        (24, 24, "wide512_positive_f32"),
        (24, 25, "high_work_positive_f32"),
        (24, 28, "wide512_positive_f32"),
        # heads=32 rows 5/16 fall inside the (removed) "direct_positive_f32"
        # registration's old eligibility band, but _positive_f32_variants never
        # names it: work_items=32*rows stays below the compact/high-work/vector4
        # thresholds, so the wave-arithmetic fallback (wide512_positive_f32) wins.
        (32, 5, "wide512_positive_f32"),
        (32, 16, "wide512_positive_f32"),
        (32, 18, "wide512_positive_f32"),
        (32, 19, "high_work_positive_f32"),
        (32, 21, "wide512_positive_f32"),
        (32, 31, "wide512_positive_f32"),
        (32, 32, "high_work_positive_f32"),
        (48, 13, "high_work_positive_f32"),
        (96, 10, "high_work_positive_f32"),
        (96, 13, "high_work_positive_f32"),
    ),
)
def test_positive_f32_producer_and_partial_wave_routes(target, heads, rows, expected):
    variant = _select_positive_route(target, heads, rows)
    assert variant is not None
    assert variant.name == expected
    assert variant.slot_offset_bits == 32


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_positive_routes_preserve_wide_offsets_and_runtime_config(target):
    variant = _select_positive_route(target, 12, 24, wide=True)
    assert variant.name == "wide512_vector4_positive_f32_wide_slot_offsets"
    assert variant.slot_offset_bits == 64
    for lower_bound in (-5.0, -20.0, None):
        variant = _select_positive_route(target, 12, 51, lower_bound=lower_bound)
        assert variant.name == "high_work_positive_f32"


def test_no_variant_pins_a_layout_or_a_runtime_configuration():
    for variant in cake_jit.get_cake_fused_kda_decode_variants():
        for rule in variant.eligibility:
            assert rule.lower_bound_values is None
            assert rule.norm_eps_values is None
            assert all(expected is None for _name, expected in rule.strides)


def _select_h8_route(
    target,
    rows,
    *,
    state_dtype="float32",
    state_indices_mode="positive_unique",
    lower_bound=-5.0,
    norm_eps=1e-5,
    wide=False,
):
    heads = 8
    hidden = heads * 128
    element_bytes = 2 if state_dtype == "bfloat16" else 4
    conv_stride = 9 * hidden + heads * 128 * 128 * element_bytes // 2
    state_stride = 9 * hidden * 2 // element_bytes + heads * 128 * 128
    slots = (2**31 // state_stride + 2) if wide else rows + 1
    return cake_jit.select_cake_fused_kda_decode_variant(
        target=target,
        num_heads=heads,
        num_rows=rows,
        num_slots=slots,
        state_dtype=state_dtype,
        state_indices_mode=state_indices_mode,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        x_row_stride=3 * hidden + 17,
        conv_slot_stride=conv_stride,
        beta_row_stride=heads + 1,
        state_slot_stride=state_stride,
        output_gate_row_stride=hidden + 7,
    )


# H=8 route bands mirror the Cake launcher (148 SMs): one wide512 wave ends at
# rows 18, the staged compact band spans 38..55 for nullable FP32 and 19..63 for
# BF16, high-work starts at rows 148 for nullable slots, and the measured
# positive-FP32 schedule runs the two-CTA cluster split while one wave holds
# every CTA pair (<= 9), then the LAUNCH_MIN_BLOCKS=1 instantiation of the
# wide512 schedule ("wide512_regcap128_positive_f32", 10..18), wide512
# (19..37, 56..76) and high-work (38..55, >= 77).
_H8_ROUTES = (
    ("float32", "positive_unique", 1, "cluster2_wide_positive_f32"),
    ("float32", "positive_unique", 9, "cluster2_wide_positive_f32"),
    ("float32", "positive_unique", 10, "wide512_regcap128_positive_f32"),
    ("float32", "positive_unique", 18, "wide512_regcap128_positive_f32"),
    ("float32", "positive_unique", 19, "wide512_positive_f32"),
    ("float32", "positive_unique", 37, "wide512_positive_f32"),
    ("float32", "positive_unique", 38, "high_work_positive_f32"),
    ("float32", "positive_unique", 55, "high_work_positive_f32"),
    ("float32", "positive_unique", 56, "wide512_positive_f32"),
    ("float32", "positive_unique", 63, "wide512_positive_f32"),
    ("float32", "positive_unique", 64, "wide512_positive_f32"),
    ("float32", "positive_unique", 76, "wide512_positive_f32"),
    ("float32", "positive_unique", 77, "high_work_positive_f32"),
    ("float32", "positive_unique", 148, "high_work_positive_f32"),
    ("float32", "positive_unique", 4096, "high_work_positive_f32"),
    ("float32", "unique_or_null", 1, "wide512_f32"),
    ("float32", "unique_or_null", 18, "wide512_f32"),
    ("float32", "unique_or_null", 19, "direct_f32"),
    ("float32", "unique_or_null", 37, "direct_f32"),
    ("float32", "unique_or_null", 38, "compact_async_f32"),
    ("float32", "unique_or_null", 55, "compact_async_f32"),
    ("float32", "unique_or_null", 56, "direct_f32"),
    ("float32", "unique_or_null", 63, "direct_f32"),
    ("float32", "unique_or_null", 64, "direct_f32"),
    ("float32", "unique_or_null", 76, "direct_f32"),
    ("float32", "unique_or_null", 77, "direct_f32"),
    ("float32", "unique_or_null", 147, "direct_f32"),
    ("float32", "unique_or_null", 148, "high_work_f32"),
    ("float32", "unique_or_null", 4096, "high_work_f32"),
    ("float32", "repeated_positive", 1, "repeated_safe_f32"),
    ("float32", "repeated_positive", 64, "repeated_safe_f32"),
    ("float32", "repeated_positive", 4096, "repeated_safe_f32"),
    ("bfloat16", "positive_unique", 1, "wide512_bf16"),
    ("bfloat16", "positive_unique", 18, "wide512_bf16"),
    ("bfloat16", "positive_unique", 19, "compact_async_bf16"),
    ("bfloat16", "positive_unique", 37, "compact_async_bf16"),
    ("bfloat16", "positive_unique", 38, "compact_async_bf16"),
    ("bfloat16", "positive_unique", 55, "compact_async_bf16"),
    ("bfloat16", "positive_unique", 56, "compact_async_bf16"),
    ("bfloat16", "positive_unique", 63, "compact_async_bf16"),
    ("bfloat16", "positive_unique", 64, "stream_bf16"),
    ("bfloat16", "positive_unique", 65, "stream_bf16"),
    ("bfloat16", "positive_unique", 147, "stream_bf16"),
    ("bfloat16", "positive_unique", 148, "stream_bf16"),
    ("bfloat16", "positive_unique", 4096, "stream_bf16"),
    ("bfloat16", "unique_or_null", 1, "wide512_bf16"),
    ("bfloat16", "unique_or_null", 18, "wide512_bf16"),
    ("bfloat16", "unique_or_null", 19, "compact_async_bf16"),
    ("bfloat16", "unique_or_null", 63, "compact_async_bf16"),
    ("bfloat16", "unique_or_null", 64, "direct_bf16"),
    ("bfloat16", "unique_or_null", 147, "direct_bf16"),
    ("bfloat16", "unique_or_null", 148, "high_work_bf16"),
    ("bfloat16", "unique_or_null", 4096, "high_work_bf16"),
    ("bfloat16", "repeated_positive", 1, "repeated_safe_bf16"),
    ("bfloat16", "repeated_positive", 64, "repeated_safe_bf16"),
    ("bfloat16", "repeated_positive", 4096, "repeated_safe_bf16"),
)


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    ("state_dtype", "state_indices_mode", "rows", "expected"),
    [pytest.param(*case, id=f"{case[0]}-{case[1]}-n{case[2]}") for case in _H8_ROUTES],
)
def test_h8_routes_follow_cake_bands(
    target, state_dtype, state_indices_mode, rows, expected
):
    variant = _select_h8_route(
        target, rows, state_dtype=state_dtype, state_indices_mode=state_indices_mode
    )
    assert variant is not None
    assert variant.name == expected
    assert variant.target == target
    assert variant.state_dtype == state_dtype
    assert variant.slot_offset_bits == 32
    assert any(
        8 in rule.heads and state_indices_mode in rule.state_indices_modes
        for rule in variant.eligibility
    )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_h8_routes_preserve_wide_offsets_and_runtime_config(target):
    for state_dtype, state_indices_mode, rows, expected in _H8_ROUTES:
        variant = _select_h8_route(
            target,
            rows,
            state_dtype=state_dtype,
            state_indices_mode=state_indices_mode,
            wide=True,
        )
        assert variant is not None
        assert variant.name == f"{expected}_wide_slot_offsets"
        assert variant.slot_offset_bits == 64
    for rows, expected in (
        (37, "wide512_positive_f32"),
        (100, "high_work_positive_f32"),
    ):
        variant = _select_h8_route(target, rows, lower_bound=None, norm_eps=3e-4)
        assert variant.name == expected


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_h8_never_selects_other_heads_specialisations(target):
    excluded = {
        "compact_async_positive_f32",
        "wide512_vector4_positive_f32",
    }
    for variant in cake_jit.get_cake_fused_kda_decode_variants(target):
        if variant.name.removesuffix("_wide_slot_offsets") in excluded:
            assert all(8 not in rule.heads for rule in variant.eligibility), (
                variant.name
            )


def test_unsupported_head_counts_are_rejected():
    for heads in (1, 4, 16, 64, 128):
        with pytest.raises(ValueError, match="head count"):
            cake_jit.select_cake_fused_kda_decode_variant(
                target="sm100a",
                num_heads=heads,
                num_rows=4,
                num_slots=5,
                state_dtype="float32",
                state_indices_mode="positive_unique",
                lower_bound=-5.0,
                norm_eps=1e-5,
                x_row_stride=3 * heads * 128,
                conv_slot_stride=9 * heads * 128,
                beta_row_stride=heads,
                state_slot_stride=heads * 128 * 128,
                output_gate_row_stride=heads * 128,
            )


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize(
    "name",
    ("cluster2_wide_positive_f32", "cluster2_wide_positive_f32_wide_slot_offsets"),
)
def test_cluster_variants_launch_two_ctas_per_work_item(target, name):
    variant = cake_jit.get_cake_fused_kda_decode_variant(name, target)
    assert variant.cluster_x == 2
    assert variant.threads == 512
    assert variant.state_dtype == "float32"
    assert variant.abi_kind == "standard"
    assert cake_jit.cake_fused_kda_decode_grid(
        variant, num_heads=8, num_rows=9, sm_count=_SM_COUNT
    ) == (16, 9, 1)
    rule = variant.eligibility[0]
    assert rule.heads == (8,)
    assert rule.minimum_rows == 1
    assert rule.maximum_rows == cake_jit._H8_CLUSTER2_MAX_ROWS
    assert rule.state_indices_modes == ("positive_unique",)


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_regcap128_is_the_h8_one_wave_instantiation_of_wide512(target):
    # "wide512_regcap128_positive_f32" is wide512_positive_f32's schedule with the
    # declared constant LAUNCH_MIN_BLOCKS = 1 (one template, two instantiations);
    # the H=8 rows 10..18 bucket selects it, wide512_positive_f32 starts at 19.
    regcap = cake_jit.get_cake_fused_kda_decode_variant(
        "wide512_regcap128_positive_f32", target
    )
    wide = cake_jit.get_cake_fused_kda_decode_variant("wide512_positive_f32", target)
    assert regcap.module != wide.module
    assert regcap.abi_kind == wide.abi_kind == "standard"
    assert regcap.threads == wide.threads == 512
    assert [
        (rule.heads, rule.minimum_rows, rule.maximum_rows, rule.state_indices_modes)
        for rule in regcap.eligibility
    ] == [
        (
            (8,),
            cake_jit._H8_CLUSTER2_MAX_ROWS + 1,
            cake_jit._H8_WIDE512_REGCAP128_MAX_ROWS,
            ("positive_unique",),
        )
    ]
    h8_wide = [rule for rule in wide.eligibility if rule.heads == (8,)]
    assert [(rule.minimum_rows, rule.maximum_rows) for rule in h8_wide] == [
        (cake_jit._H8_WIDE512_REGCAP128_MAX_ROWS + 1, 37),
        (56, 76),
    ]
    for rows, expected in (
        (10, "wide512_regcap128_positive_f32"),
        (18, "wide512_regcap128_positive_f32"),
        (19, "wide512_positive_f32"),
    ):
        assert cake_jit._positive_f32_variants(8, rows) == (expected,)


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
@pytest.mark.parametrize("name", ("stream_bf16", "stream_bf16_wide_slot_offsets"))
def test_persistent_stream_variants_size_the_grid_by_resident_ctas(target, name):
    variant = cake_jit.get_cake_fused_kda_decode_variant(name, target)
    assert variant.abi_kind == "persistent_rows"
    assert variant.state_dtype == "bfloat16"
    assert variant.threads == 256
    assert variant.dynamic_smem_bytes == 70272
    assert variant.slot_offset_bits == (
        64 if name.endswith("_wide_slot_offsets") else 32
    )
    assert [
        (rule.heads, rule.minimum_rows, rule.maximum_rows, rule.state_indices_modes)
        for rule in variant.eligibility
    ] == [((8,), 64, None, ("positive_unique",))]
    grid = cake_jit.cake_fused_kda_decode_grid
    assert grid(variant, num_heads=8, num_rows=1024, sm_count=148) == (444, 1, 1)
    assert grid(variant, num_heads=8, num_rows=1024, sm_count=160) == (480, 1, 1)
    assert grid(variant, num_heads=8, num_rows=10, sm_count=148) == (80, 1, 1)


@pytest.mark.parametrize("target", ("sm100a", "sm103a"))
def test_repeated_safe_variants_take_a_single_head_grid(target):
    # The program binding loops the rows with this grid (one launch per row); the host passes all rows.
    for name in ("repeated_safe_f32", "repeated_safe_bf16_wide_slot_offsets"):
        variant = cake_jit.get_cake_fused_kda_decode_variant(name, target)
        assert variant.abi_kind == "repeated_safe"
        assert cake_jit.cake_fused_kda_decode_grid(
            variant, num_heads=24, num_rows=1, sm_count=_SM_COUNT
        ) == (24, 1, 1)
    standard = cake_jit.get_cake_fused_kda_decode_variant("direct_f32", target)
    assert cake_jit.cake_fused_kda_decode_grid(
        standard, num_heads=24, num_rows=7, sm_count=_SM_COUNT
    ) == (24, 7, 1)


def test_persistent_rows_abi_extends_the_standard_argument_plan():
    standard = cake_jit.CAKE_FUSED_KDA_DECODE_ABIS["standard"]
    persistent = cake_jit.CAKE_FUSED_KDA_DECODE_ABIS["persistent_rows"]
    assert persistent == cake_jit.CAKE_FUSED_KDA_DECODE_ABIS["repeated_safe"]
    assert persistent[:18] == standard[:18]
    assert persistent[18] == ("parameter", "rows", "int32")
    assert persistent[19:] == standard[18:]


def test_lower_bound_is_passed_in_log2_units():
    assert cake_jit.cake_fused_kda_decode_lower_bound_log2(None) == 0.0
    assert cake_jit.cake_fused_kda_decode_lower_bound_log2(-5.0) == pytest.approx(
        -5.0 * _LOG2E
    )


@pytest.mark.parametrize(("target", "minor"), (("sm100a", 0), ("sm103a", 3)))
@pytest.mark.parametrize(
    ("name", "min_blocks"),
    (
        ("wide512_f32_wide_slot_offsets", False),
        ("compact_async_f32_wide_slot_offsets", True),
        ("compact_async_f32", False),
        ("stream_bf16", False),
        ("cluster2_wide_positive_f32", False),
    ),
)
def test_cake_jit_spec_uses_exact_target_flags(
    monkeypatch, tmp_path, target, minor, name, min_blocks
):
    monkeypatch.setattr(
        jit_core.current_compilation_context, "TARGET_CUDA_ARCHS", {(10, f"{minor}a")}
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    cake_jit.gen_cake_fused_kda_decode_module.cache_clear()
    variant = cake_jit.get_cake_fused_kda_decode_variant(name, target)
    spec = cake_jit.gen_cake_fused_kda_decode_module(variant.name, target)
    assert spec.name == cake_jit.get_cake_fused_kda_decode_uri(variant.name, target)
    assert list(spec.sources) == [variant.body_path, variant.binding_path]
    assert [
        flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")
    ] == [f"-gencode=arch=compute_10{minor}a,code=sm_10{minor}a"]
    if name in cake_jit._PRECISE_MATH_VARIANTS:
        assert not set(spec.extra_cuda_cflags) & cake_jit._FAST_MATH_FLAGS
    else:
        assert "--use_fast_math" in spec.extra_cuda_cflags
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


def _host_tensors(rows=4, heads=12, slots=5, state_dtype=torch.float32):
    hidden = heads * 128
    qkv = 3 * hidden
    return {
        "x": torch.empty((rows, qkv + 17), dtype=torch.bfloat16)[:, :qkv],
        "weight": torch.empty((3, 4, hidden), dtype=torch.float32),
        "conv_state": torch.empty_strided(
            (slots, qkv, 3), (3 * qkv + 8, 1, qkv), dtype=torch.bfloat16
        ),
        "raw_gate": torch.empty((1, rows, heads, 128), dtype=torch.bfloat16),
        "raw_beta": torch.empty((1, rows, heads + 1), dtype=torch.bfloat16)[
            :, :, :heads
        ],
        "A_log": torch.empty((heads,), dtype=torch.float32),
        "dt_bias": torch.empty((hidden,), dtype=torch.float32),
        "state_indices": torch.arange(rows, 0, -1, dtype=torch.int32),
        "state": torch.empty_strided(
            (slots, heads, 128, 128),
            (heads * 128 * 128 + 32, 128 * 128, 128, 1),
            dtype=state_dtype,
        ),
        "output_gate": torch.empty((rows, hidden + 7), dtype=torch.bfloat16).as_strided(
            (rows, heads, 128), (hidden + 7, 128, 1)
        ),
        "norm_weight": torch.empty((128,), dtype=torch.float32),
        "output": torch.empty((1, rows, heads, 128), dtype=torch.bfloat16),
    }


def _plan(abi_kind, *, device_index=False):
    plan = [("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z")]
    if device_index:
        plan.insert(0, ("device", "device_index"))
    plan.extend(
        (kind, name)
        for kind, name, _dtype in cake_jit.CAKE_FUSED_KDA_DECODE_ABIS[abi_kind]
    )
    return tuple(plan)


def _fake_variant(abi_kind, *, cluster_x=1, device_index=False):
    return SimpleNamespace(
        name="selected",
        target="sm100a",
        module="cake_fused_kda_decode_test",
        ffi_entry="run",
        abi_kind=abi_kind,
        cluster_x=cluster_x,
        arg_plan=_plan(abi_kind, device_index=device_index),
    )


@pytest.mark.parametrize(
    ("lower_bound", "expected_config"),
    ((None, (0, 0.0, 1e-5)), (-5.0, (1, -5.0 * _LOG2E, 1e-5))),
)
def test_cake_submit_binds_caller_owned_storage_through_the_argument_plan(
    monkeypatch, lower_bound, expected_config
):
    calls = []
    monkeypatch.setattr(
        fused,
        "load_cake_fused_kda_decode_module",
        lambda name, target: SimpleNamespace(run=lambda *args: calls.append(args)),
    )
    monkeypatch.setattr(fused, "_device_index", lambda device: 0)
    monkeypatch.setattr(fused, "_device_sm_count", lambda index: 148)
    tensors = _host_tensors()
    variant = _fake_variant("standard", device_index=True)

    fused._run_cake_variant(variant, **tensors, lower_bound=lower_bound, norm_eps=1e-5)

    assert len(calls) == 1
    (args,) = calls
    assert args[:4] == (0, 12, 4, 1)
    bound = dict(zip(_ABI_NAMES["standard"], args[4:], strict=True))
    # Every buffer reaches the binding as the caller's own tensor: the padded ones
    # (x, conv_state, raw_beta, state, output_gate) keep their strides, which the
    # generated binding checks against the stride scalars below; no per-call view.
    for name in _ABI_NAMES["standard"]:
        if name in tensors:
            assert bound[name] is tensors[name]
    hidden = 12 * 128
    assert (
        bound["x_row_stride"],
        bound["conv_slot_stride"],
        bound["beta_row_stride"],
        bound["state_slot_stride"],
        bound["output_gate_row_stride"],
        bound["H"],
    ) == (3 * hidden + 17, 9 * hidden + 8, 13, 12 * 128 * 128 + 32, hidden + 7, 12)
    assert (bound["use_lower_bound"], bound["norm_eps"]) == (
        expected_config[0],
        expected_config[2],
    )
    assert bound["lower_bound_log2"] == pytest.approx(expected_config[1])


def test_cake_repeated_submit_passes_every_row_in_one_call(monkeypatch):
    """Repeated-slot rows reach the module in one call; the binding walks the rows."""
    calls = []
    monkeypatch.setattr(
        fused,
        "load_cake_fused_kda_decode_module",
        lambda name, target: SimpleNamespace(run=lambda *args: calls.append(args)),
    )
    monkeypatch.setattr(fused, "_device_index", lambda device: 0)
    monkeypatch.setattr(fused, "_device_sm_count", lambda index: 148)
    tensors = _host_tensors()
    tensors["state_indices"] = torch.tensor([1, 2, 1, 2], dtype=torch.int32)

    fused._run_cake_variant(
        _fake_variant("repeated_safe"), **tensors, lower_bound=-5.0, norm_eps=1e-5
    )

    (args,) = calls
    assert args[:3] == (12, 1, 1)
    bound = dict(zip(_ABI_NAMES["repeated_safe"], args[3:], strict=True))
    assert bound["rows"] == 4
    assert bound["state_indices"] is tensors["state_indices"]
    assert bound["output"] is tensors["output"]
    # Padded rows travel as the caller's own strided x (no copy, no view).
    assert bound["x"] is tensors["x"]
    assert bound["x_row_stride"] == tensors["x"].stride(0)


def test_cake_persistent_submit_passes_every_row_and_sizes_the_grid(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fused,
        "load_cake_fused_kda_decode_module",
        lambda name, target: SimpleNamespace(run=lambda *args: calls.append(args)),
    )
    monkeypatch.setattr(fused, "_device_index", lambda device: 0)
    monkeypatch.setattr(fused, "_device_sm_count", lambda index: 148)
    tensors = _host_tensors(rows=100, heads=8, slots=101, state_dtype=torch.bfloat16)

    fused._run_cake_variant(
        _fake_variant("persistent_rows"), **tensors, lower_bound=-5.0, norm_eps=1e-5
    )

    (args,) = calls
    assert args[:3] == (444, 1, 1)
    bound = dict(zip(_ABI_NAMES["persistent_rows"], args[3:], strict=True))
    assert bound["rows"] == 100
    assert bound["output"] is tensors["output"]


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
        lambda *args, **kwargs: lambda *kernel_args: fallback_calls.append(kernel_args),
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
