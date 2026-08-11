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
from packaging.version import Version

from flashinfer.kda_decode import recurrent_kda

kda_decode_module = importlib.import_module("flashinfer.kda_decode")
recurrent_module = importlib.import_module("flashinfer.kda_kernels.recurrent_kda")

_D = 128
_T = 5
_VARIANT_PREFIX = "d128_t5_precomputed_gram_split"
_T3 = 3
_T3_VARIANT = "d128_t3_lower_bound_split4"
_T3_SEQUENCE_COUNTS = (1, 2, 4, 8, 16)
_PRECOMPUTED_VARIANT_PREFIXES = {
    1: "d128_t1_precomputed_direct_split",
    2: "d128_t2_precomputed_split",
    4: "d128_t4_precomputed_split",
    5: _VARIANT_PREFIX,
    6: "d128_t6_precomputed_gram_split",
}


@pytest.fixture
def flash_kda_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
        pytest.skip(
            "frozen FlashKDA decode tests require exact CC 10.0 "
            "(SM100a; B200/GB200) or CC 10.3 (SM103a; B300/GB300)"
        )
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

    num_sequences = 8
    num_heads = 16
    num_value_heads = 32
    total_tokens = num_sequences * _T
    q = tensor((1, total_tokens, num_heads, _D))
    return {
        "q": q,
        "k": tensor(q.shape),
        "v": tensor((1, total_tokens, num_value_heads, _D)),
        "g": tensor((1, total_tokens, num_value_heads, _D)),
        "beta": tensor((1, total_tokens, num_value_heads)),
        "state": tensor((total_tokens + 6, num_value_heads, _D, _D)),
        "out": tensor((1, total_tokens, num_value_heads, _D)),
        "cu_seqlens": tensor((num_sequences + 1,), torch.int32),
        "ssm_state_indices": tensor((total_tokens,), torch.int32),
        "num_accepted_tokens": tensor((num_sequences,), torch.int32),
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


def _fake_precomputed_selector_kwargs(
    num_tokens, *, num_sequences=8, num_heads=16, num_value_heads=32
):
    address = iter(range(0x200000000000, 0x300000000000, 0x100000000))

    def tensor(shape, dtype=torch.bfloat16, **kwargs):
        return _fake_tensor(shape, dtype, next(address), **kwargs)

    total_tokens = num_sequences * num_tokens
    q = tensor((1, total_tokens, num_heads, _D))
    return {
        "q": q,
        "k": tensor(q.shape),
        "v": tensor((1, total_tokens, num_value_heads, _D)),
        "g": tensor((1, total_tokens, num_value_heads, _D)),
        "beta": tensor((1, total_tokens, num_value_heads)),
        "state": tensor((total_tokens + 6, num_value_heads, _D, _D)),
        "out": tensor((1, total_tokens, num_value_heads, _D)),
        "cu_seqlens": tensor((num_sequences + 1,), torch.int32),
        "ssm_state_indices": tensor((total_tokens,), torch.int32),
        "num_accepted_tokens": tensor((num_sequences,), torch.int32),
        "scale": _D**-0.5,
        "num_tokens": num_tokens,
        "num_spec_tokens": None if num_tokens == 1 else num_tokens - 1,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": False,
        "lower_bound": None,
        "A_log": None,
        "dt_bias": None,
        "initial_state_source": None,
        "beta_is_logit": False,
    }


def _fake_t3_selector_kwargs(*, num_sequences=4, num_heads=16, num_value_heads=16):
    address = iter(range(0x400000000000, 0x500000000000, 0x100000000))

    def tensor(shape, dtype=torch.bfloat16, **kwargs):
        return _fake_tensor(shape, dtype, next(address), **kwargs)

    num_tokens = _T3
    total_tokens = num_sequences * num_tokens
    q = tensor((1, total_tokens, num_heads, _D))
    return {
        "q": q,
        "k": tensor(q.shape),
        "v": tensor((1, total_tokens, num_value_heads, _D)),
        "g": tensor((1, total_tokens, num_value_heads, _D)),
        "beta": tensor((1, total_tokens, num_value_heads)),
        "state": tensor((total_tokens, num_value_heads, _D, _D)),
        "out": tensor((1, total_tokens, num_value_heads, _D)),
        "cu_seqlens": tensor((num_sequences + 1,), torch.int32),
        "ssm_state_indices": tensor((total_tokens,), torch.int32),
        "num_accepted_tokens": tensor((num_sequences,), torch.int32),
        "scale": _D**-0.5,
        "num_tokens": num_tokens,
        "num_spec_tokens": num_tokens - 1,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "A_log": tensor((num_heads,), torch.float32),
        "dt_bias": tensor((num_heads * _D,), torch.float32),
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
    ("backend", "pass_explicitly"),
    [
        ("cute-dsl", False),
        ("cute-dsl", True),
        ("cake", True),
    ],
)
def test_public_backend_option_forwards_to_kernel_layer_cpu(
    monkeypatch, backend, pass_explicitly
):
    calls = []
    expected = (object(), object())

    def run(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(kda_decode_module, "_run_recurrent_kda", run)
    tensors = [object() for _ in range(5)]
    kwargs = {"backend": backend} if pass_explicitly else {}
    assert recurrent_kda(*tensors, **kwargs) is expected
    assert len(calls) == 1
    assert calls[0]["backend"] == backend


def test_public_backend_option_rejects_unknown_value_cpu(monkeypatch):
    monkeypatch.setattr(
        kda_decode_module,
        "_run_recurrent_kda",
        lambda **kwargs: pytest.fail(f"unexpected kernel call: {kwargs}"),
    )
    tensors = [object() for _ in range(5)]
    with pytest.raises(ValueError, match="backend must be"):
        recurrent_kda(*tensors, backend="unknown")


def test_cake_backend_rejects_empty_packed_decode_instead_of_noop_cpu():
    num_sequences = 2
    q = torch.empty(1, 0, 1, _D, dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty(1, 0, 1, _D, dtype=torch.bfloat16)
    g = torch.empty_like(v)
    beta = torch.empty(1, 0, 1, dtype=torch.bfloat16)
    state = torch.empty(num_sequences, 1, _D, _D, dtype=torch.bfloat16)
    output = torch.empty_like(v)
    cu_seqlens = torch.zeros(num_sequences + 1, dtype=torch.int32)

    default_output, default_state = recurrent_module.run_recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        output=output,
    )
    assert default_output is output
    assert default_state is state

    with pytest.raises(ValueError, match="backend='cake' does not support"):
        recurrent_module.run_recurrent_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            output=output,
            backend="cake",
        )


@pytest.mark.parametrize(
    ("num_tokens", "work", "sm_count", "expected_split"),
    [
        (1, 1, 148, 16),
        (1, 296, 148, 16),
        (1, 297, 148, 8),
        (1, 4096, 148, 8),
        (2, 1, 148, 4),
        (2, 4096, 148, 4),
        (4, 1, 148, 2),
        (4, 4096, 148, 2),
        (5, 55, 148, 8),
        (5, 56, 148, 2),
        (5, 74, 148, 2),
        (5, 75, 148, 4),
        (5, 111, 148, 4),
        (5, 112, 148, 2),
        (5, 222, 148, 2),
        (5, 223, 148, 1),
        (6, 55, 148, 8),
        (6, 56, 148, 2),
        (6, 75, 148, 4),
        (6, 112, 148, 2),
        (6, 223, 148, 1),
    ],
)
def test_cake_value_split_boundaries(num_tokens, work, sm_count, expected_split):
    assert (
        recurrent_module._select_flash_kda_decode_value_split(
            num_tokens, work, sm_count
        )
        == expected_split
    )


@pytest.mark.parametrize(
    ("num_tokens", "work", "sm_count", "expected_split"),
    [
        (1, 5120, 160, 16),
        (1, 5121, 160, 8),
        (2, 80, 160, 8),
        (2, 81, 160, 4),
        (4, 80, 160, 8),
        (4, 81, 160, 4),
        (4, 160, 160, 4),
        (4, 161, 160, 2),
        (5, 256, 160, 1),
        (6, 60, 160, 8),
        (6, 61, 160, 2),
        (6, 80, 160, 2),
        (6, 81, 160, 1),
    ],
)
def test_sm103a_cake_value_split_boundaries(num_tokens, work, sm_count, expected_split):
    assert (
        recurrent_module._select_flash_kda_decode_value_split(
            num_tokens, work, sm_count, "sm103a"
        )
        == expected_split
    )


def test_sm103a_split_policy_has_an_independent_retuning_hook(monkeypatch):
    sentinel_calls = []

    def sm103a_selector(num_tokens, work, sm_count):
        sentinel_calls.append((num_tokens, work, sm_count))
        return 8

    monkeypatch.setitem(
        recurrent_module._FLASH_KDA_DECODE_VALUE_SPLIT_SELECTOR_BY_ARCH,
        "sm103a",
        sm103a_selector,
    )

    assert (
        recurrent_module._select_flash_kda_decode_value_split(5, 256, 160, "sm103a")
        == 8
    )
    assert sentinel_calls == [(5, 256, 160)]
    assert (
        recurrent_module._select_flash_kda_decode_value_split(5, 256, 160, "sm100a")
        == 1
    )


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
def test_exact_cpu_contract_selects_frozen_variant(monkeypatch, cc):
    _patch_cpu_selector_environment(monkeypatch, cc=cc)
    assert recurrent_module._select_flash_kda_decode_variant(
        **_fake_selector_kwargs()
    ) == (_VARIANT_PREFIX + "1")


@pytest.mark.parametrize(
    ("num_tokens", "num_sequences", "expected_variant"),
    [
        (1, 8, _PRECOMPUTED_VARIANT_PREFIXES[1] + "16"),
        (1, 16, _PRECOMPUTED_VARIANT_PREFIXES[1] + "8"),
        (2, 8, _PRECOMPUTED_VARIANT_PREFIXES[2] + "4"),
        (4, 8, _PRECOMPUTED_VARIANT_PREFIXES[4] + "2"),
        (5, 8, _PRECOMPUTED_VARIANT_PREFIXES[5] + "1"),
        (6, 8, _PRECOMPUTED_VARIANT_PREFIXES[6] + "1"),
    ],
)
def test_precomputed_token_family_uses_tuned_export_dispatch(
    monkeypatch, num_tokens, num_sequences, expected_variant
):
    _patch_cpu_selector_environment(monkeypatch)
    assert (
        recurrent_module._select_flash_kda_decode_variant(
            **_fake_precomputed_selector_kwargs(
                num_tokens,
                num_sequences=num_sequences,
            )
        )
        == expected_variant
    )


@pytest.mark.parametrize(
    ("num_tokens", "num_sequences", "expected_variant"),
    [
        (1, 16, _PRECOMPUTED_VARIANT_PREFIXES[1] + "16"),
        (2, 2, _PRECOMPUTED_VARIANT_PREFIXES[2] + "8"),
        (2, 3, _PRECOMPUTED_VARIANT_PREFIXES[2] + "4"),
        (4, 2, _PRECOMPUTED_VARIANT_PREFIXES[4] + "8"),
        (4, 3, _PRECOMPUTED_VARIANT_PREFIXES[4] + "4"),
        (4, 7, _PRECOMPUTED_VARIANT_PREFIXES[4] + "2"),
        (4, 8, _PRECOMPUTED_VARIANT_PREFIXES[4] + "1"),
        (4, 16, _PRECOMPUTED_VARIANT_PREFIXES[4] + "2"),
        (5, 3, _PRECOMPUTED_VARIANT_PREFIXES[5] + "4"),
        (5, 4, _PRECOMPUTED_VARIANT_PREFIXES[5] + "1"),
        (5, 5, _PRECOMPUTED_VARIANT_PREFIXES[5] + "2"),
        (6, 1, _PRECOMPUTED_VARIANT_PREFIXES[6] + "8"),
        (6, 2, _PRECOMPUTED_VARIANT_PREFIXES[6] + "2"),
        (6, 3, _PRECOMPUTED_VARIANT_PREFIXES[6] + "1"),
    ],
)
def test_sm103a_precomputed_token_family_uses_gb300_dispatch(
    monkeypatch, num_tokens, num_sequences, expected_variant
):
    _patch_cpu_selector_environment(monkeypatch, sm_count=152, cc=(10, 3))
    assert (
        recurrent_module._select_flash_kda_decode_variant(
            **_fake_precomputed_selector_kwargs(
                num_tokens,
                num_sequences=num_sequences,
            )
        )
        == expected_variant
    )


@pytest.mark.parametrize("num_sequences", _T3_SEQUENCE_COUNTS)
def test_t3_lower_bound_measured_contract_selects_frozen_variant(
    monkeypatch, num_sequences
):
    _patch_cpu_selector_environment(monkeypatch)
    assert (
        recurrent_module._select_flash_kda_decode_variant(
            **_fake_t3_selector_kwargs(num_sequences=num_sequences)
        )
        == _T3_VARIANT
    )


@pytest.mark.parametrize("num_sequences", [3, 5, 7, 9, 15, 17])
def test_t3_lower_bound_route_rejects_excluded_sequence_count(
    monkeypatch, num_sequences
):
    _patch_cpu_selector_environment(monkeypatch)
    assert (
        recurrent_module._select_flash_kda_decode_variant(
            **_fake_t3_selector_kwargs(num_sequences=num_sequences)
        )
        is None
    )


@pytest.mark.parametrize(
    ("num_heads", "num_value_heads"),
    [
        pytest.param(8, 16, id="h8-hv16"),
        pytest.param(16, 32, id="h16-hv32"),
    ],
)
def test_t3_lower_bound_route_rejects_h_hv_mismatch(
    monkeypatch, num_heads, num_value_heads
):
    _patch_cpu_selector_environment(monkeypatch)
    assert (
        recurrent_module._select_flash_kda_decode_variant(
            **_fake_t3_selector_kwargs(
                num_heads=num_heads,
                num_value_heads=num_value_heads,
            )
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_tokens", 4),
        ("num_spec_tokens", 3),
        ("use_gate_in_kernel", False),
        ("lower_bound", None),
        ("lower_bound", 0.0),
        ("lower_bound", float("-inf")),
        ("lower_bound", float("nan")),
        ("lower_bound", -float(torch.finfo(torch.float32).max) * 2.0),
        ("A_log", None),
        ("dt_bias", None),
    ],
)
def test_t3_lower_bound_route_strictly_rejects_contract_mismatches(
    monkeypatch, field, value
):
    _patch_cpu_selector_environment(monkeypatch)
    kwargs = _fake_t3_selector_kwargs()
    kwargs[field] = value
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) is None


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        ("A_log", {"dtype": torch.bfloat16}),
        ("A_log", {"contiguous": False}),
        ("A_log", {"shape": (15,), "strides": (1,)}),
        ("dt_bias", {"dtype": torch.bfloat16}),
        ("dt_bias", {"contiguous": False}),
        ("dt_bias", {"shape": (16 * _D - 1,), "strides": (1,)}),
    ],
)
def test_t3_lower_bound_route_strictly_checks_gate_parameters(
    monkeypatch, name, changes
):
    _patch_cpu_selector_environment(monkeypatch)
    kwargs = _fake_t3_selector_kwargs()
    _replace_fake(kwargs, name, **changes)
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) is None


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda kw: _replace_fake(kw, "g", ptr=kw["out"].ptr),
            id="raw-gate-aliases-output",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "A_log", ptr=kw["state"].ptr),
            id="a-log-aliases-state",
        ),
        pytest.param(
            lambda kw: _replace_fake(kw, "dt_bias", ptr=kw["out"].ptr),
            id="dt-bias-aliases-output",
        ),
    ],
)
def test_t3_lower_bound_route_rejects_gate_tensor_aliases(monkeypatch, mutation):
    _patch_cpu_selector_environment(monkeypatch)
    kwargs = mutation(_fake_t3_selector_kwargs())
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) is None


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
def test_public_contract_mismatches_are_rejected_by_cake_selector_cpu(
    monkeypatch, field, value
):
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
def test_tensor_layout_and_alias_mismatches_are_rejected_by_cake_selector_cpu(
    monkeypatch, mutation
):
    _patch_cpu_selector_environment(monkeypatch)
    kwargs = mutation(_fake_selector_kwargs())
    assert recurrent_module._select_flash_kda_decode_variant(**kwargs) is None


@pytest.mark.parametrize("cc", [(10, 1), (12, 0)])
def test_unsupported_arch_is_rejected_by_cake_selector_cpu(monkeypatch, cc):
    _patch_cpu_selector_environment(monkeypatch, cc=cc)
    assert (
        recurrent_module._select_flash_kda_decode_variant(**_fake_selector_kwargs())
        is None
    )


class _RecorderModule:
    def __init__(self):
        self.calls = []

    def run(self, *args):
        self.calls.append(args)


@pytest.mark.parametrize(
    ("cc", "cuda_version", "variant", "expected_target"),
    [
        ((10, 0), "12.8", _VARIANT_PREFIX + "4", "sm100a"),
        (
            (10, 0),
            "12.8",
            "d128_t1_precomputed_direct_split16",
            "sm100a",
        ),
        ((10, 0), "12.9", _VARIANT_PREFIX + "4", "sm100f"),
        ((10, 0), "13.0", "d128_t1_precomputed_direct_split16", "sm100f"),
        ((10, 3), "12.9", _VARIANT_PREFIX + "4", "sm100f"),
        ((10, 3), "12.9", "d128_t1_precomputed_direct_split16", "sm103a"),
        ((10, 3), "13.0", "d128_t1_precomputed_direct_split8", "sm103a"),
    ],
)
def test_frozen_runner_selects_physical_target_and_forwards_ffi_abi_cpu(
    monkeypatch, cc, cuda_version, variant, expected_target
):
    module = _RecorderModule()
    loaded = []

    def get_module(variant, target):
        loaded.append((variant, target))
        return module

    monkeypatch.setattr(recurrent_module, "get_flash_kda_decode_module", get_module)
    monkeypatch.setattr(recurrent_module, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(
        recurrent_module,
        "is_cuda_version_at_least",
        lambda version: Version(cuda_version) >= Version(version),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=0xCAFE),
    )
    tensors = _fake_selector_kwargs()
    dummy_f32 = _fake_tensor((1,), torch.float32, 0x300000000000)
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
        A_log=dummy_f32,
        dt_bias=dummy_f32,
        lower_bound=0.0,
    )

    assert loaded == [(variant, expected_target)]
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


def test_frozen_runner_rejects_sm103a_before_cuda_12_9_cpu(monkeypatch):
    monkeypatch.setattr(
        recurrent_module, "get_compute_capability", lambda device: (10, 3)
    )
    monkeypatch.setattr(
        recurrent_module, "is_cuda_version_at_least", lambda version: False
    )
    tensors = _fake_selector_kwargs()
    dummy_f32 = _fake_tensor((1,), torch.float32, 0x300000000000)

    with pytest.raises(RuntimeError, match="requires CUDA 12.9 or newer"):
        recurrent_module._run_flash_kda_decode(
            _VARIANT_PREFIX + "4",
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
            A_log=dummy_f32,
            dt_bias=dummy_f32,
            lower_bound=0.0,
        )


def test_frozen_runner_rejects_sm100a_before_cuda_12_8_cpu(monkeypatch):
    monkeypatch.setattr(
        recurrent_module, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        recurrent_module,
        "is_cuda_version_at_least",
        lambda version: Version("12.7") >= Version(version),
    )
    tensors = _fake_selector_kwargs()
    dummy_f32 = _fake_tensor((1,), torch.float32, 0x300000000000)

    with pytest.raises(RuntimeError, match="requires CUDA 12.8 or newer"):
        recurrent_module._run_flash_kda_decode(
            _VARIANT_PREFIX + "4",
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
            A_log=dummy_f32,
            dt_bias=dummy_f32,
            lower_bound=0.0,
        )


def test_t3_frozen_runner_forwards_real_gate_parameters_cpu(monkeypatch):
    module = _RecorderModule()
    monkeypatch.setattr(
        recurrent_module,
        "get_flash_kda_decode_module",
        lambda variant, target: module,
    )
    monkeypatch.setattr(
        recurrent_module, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        recurrent_module, "is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device: SimpleNamespace(cuda_stream=0xFACE),
    )
    tensors = _fake_t3_selector_kwargs()
    recurrent_module._run_flash_kda_decode(
        "d128_t3_lower_bound_split4",
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
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        lower_bound=tensors["lower_bound"],
    )

    (args,) = module.calls
    assert args[5] is tensors["A_log"]
    assert args[6] is tensors["dt_bias"]
    assert args[13] == tensors["lower_bound"]
    assert args[14] == 0xFACE


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
    num_tokens=_T,
    seed=42,
):
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if num_tokens == 1 and accepted_tokens is not None:
        raise ValueError("standard decode does not take accepted-token metadata")
    if num_tokens == 1 and padded_last_sequence:
        raise ValueError("dense standard decode has no packed padding metadata")
    if num_tokens == 1 and padded_gate:
        raise ValueError("the padded gate helper models packed token layout")
    generator = torch.Generator(device=device).manual_seed(seed)
    total_tokens = num_sequences * num_tokens
    q_shape = (
        (num_sequences, 1, num_heads, _D)
        if num_tokens == 1
        else (1, total_tokens, num_heads, _D)
    )
    v_shape = (
        (num_sequences, 1, num_value_heads, _D)
        if num_tokens == 1
        else (1, total_tokens, num_value_heads, _D)
    )
    q = torch.randn(
        q_shape,
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
        v_shape,
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
                v_shape,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        ).to(torch.bfloat16)
        gate_storage = None
    beta = torch.rand(
        (*v_shape[:2], num_value_heads),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    if num_tokens == 1:
        cu_seqlens = None
        ssm_state_indices = None
        num_accepted_tokens = None
        num_spec_tokens = None
    else:
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            num_tokens,
            dtype=torch.int32,
            device=device,
        )
        ssm_state_indices = torch.arange(
            num_sequences * num_tokens, dtype=torch.int32, device=device
        ).reshape(num_sequences, num_tokens)
        if padded_last_sequence:
            ssm_state_indices[-1].fill_(-1)
        if accepted_tokens is None:
            num_accepted_tokens = (
                torch.arange(num_sequences, dtype=torch.int32, device=device)
                % num_tokens
            ) + 1
        else:
            if len(accepted_tokens) != num_sequences:
                raise ValueError("accepted_tokens must contain one value per sequence")
            num_accepted_tokens = torch.tensor(
                accepted_tokens, dtype=torch.int32, device=device
            )
        num_spec_tokens = num_tokens - 1
    slots = num_sequences * num_tokens
    if num_tokens == 1:
        # Dense standard decode has one caller-owned cache slot per sequence.
        slots = num_sequences
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
        "num_spec_tokens": num_spec_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
        "_state_storage": state_storage,
        "_gate_storage": gate_storage,
    }


def _make_t3_lower_bound_case(
    device,
    *,
    num_sequences,
    accepted_token,
    padded_last_sequence=False,
    seed=42,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    num_heads = 16
    num_value_heads = 16
    total_tokens = num_sequences * _T3
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
    g = torch.randn(
        v.shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    beta = torch.sigmoid(
        torch.randn(
            (1, total_tokens, num_value_heads),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    ).to(torch.bfloat16)
    A_log = torch.log(
        torch.rand(
            (num_heads,),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        + 1.0
    )
    dt_bias = torch.randn(
        (num_heads * _D,),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        _T3,
        dtype=torch.int32,
        device=device,
    )
    ssm_state_indices = torch.arange(
        total_tokens,
        dtype=torch.int32,
        device=device,
    ).reshape(num_sequences, _T3)
    if padded_last_sequence:
        ssm_state_indices[-1].fill_(-1)
    num_accepted_tokens = torch.full(
        (num_sequences,),
        accepted_token,
        dtype=torch.int32,
        device=device,
    )
    state = torch.randn(
        (total_tokens, num_value_heads, _D, _D),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    output = torch.empty_like(v)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "scale": _D**-0.5,
        "initial_state": state,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_spec_tokens": _T3 - 1,
        "num_accepted_tokens": num_accepted_tokens,
        "output": output,
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


@pytest.mark.parametrize("split", [8, 2, 4, 1])
def test_public_recurrent_kda_forced_t5_splits_match_upstream_cute(
    flash_kda_device, monkeypatch, split
):
    num_value_heads = 32
    num_sequences = 8
    variant = _VARIANT_PREFIX + str(split)
    case = _make_case(
        flash_kda_device,
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

    expected_output, expected_state = recurrent_kda(
        **_call_kwargs(case, state=baseline_state, output=baseline_output),
        backend="cute-dsl",
    )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(
        recurrent_module,
        "_select_flash_kda_decode_variant",
        lambda **kwargs: variant,
    )
    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    actual_output, actual_state_result = recurrent_kda(
        **_call_kwargs(case, state=actual_state, output=actual_output_buffer),
        backend="cake",
    )

    assert frozen_calls == [variant]
    assert actual_output.data_ptr() == actual_output_buffer.data_ptr()
    assert actual_state_result is actual_state
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize(
    ("num_tokens", "num_sequences"),
    [(1, 8), (1, 16), (2, 8), (4, 8), (5, 8), (6, 8)],
)
def test_public_recurrent_kda_precomputed_matrix_matches_cute_dsl(
    flash_kda_device, monkeypatch, num_tokens, num_sequences
):
    accepted_tokens = (
        None
        if num_tokens == 1
        else [
            0,
            1,
            num_tokens,
            num_tokens + 7,
            max(1, num_tokens - 1),
            0,
            1,
            num_tokens,
        ]
    )
    case = _make_case(
        flash_kda_device,
        num_sequences=num_sequences,
        num_heads=16,
        num_value_heads=32,
        padded_last_sequence=num_tokens != 1,
        accepted_tokens=accepted_tokens,
        num_tokens=num_tokens,
        seed=2100 + num_tokens,
    )

    if num_tokens == 1:
        assert case["num_spec_tokens"] is None
        assert case["num_accepted_tokens"] is None
        assert case["q"].shape == (num_sequences, 1, 16, _D)
        assert case["cu_seqlens"] is None
        assert case["ssm_state_indices"] is None
    else:
        assert case["num_spec_tokens"] == num_tokens - 1
        assert case["ssm_state_indices"].shape == (
            num_sequences,
            num_tokens,
        )
        assert case["num_accepted_tokens"].tolist() == accepted_tokens
        assert case["cu_seqlens"].tolist() == [
            sequence * num_tokens for sequence in range(num_sequences + 1)
        ]

    initial = _clone_state_with_layout(case["initial_state"])
    baseline_state = _clone_state_with_layout(initial)
    actual_state = _clone_state_with_layout(initial)
    baseline_output = torch.empty_like(case["output"])
    actual_output_buffer = torch.empty_like(case["output"])

    expected_output, expected_state = recurrent_kda(
        **_call_kwargs(case, state=baseline_state, output=baseline_output),
        backend="cute-dsl",
    )

    frozen_calls = []
    frozen_call_kwargs = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        frozen_call_kwargs.append(kwargs)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    actual_output, actual_state_result = recurrent_kda(
        **_call_kwargs(case, state=actual_state, output=actual_output_buffer),
        backend="cake",
    )

    arch = recurrent_module._FLASH_KDA_DECODE_ARCH_BY_COMPUTE_CAPABILITY[
        torch.cuda.get_device_capability(flash_kda_device)
    ]
    expected_split = recurrent_module._select_flash_kda_decode_value_split(
        num_tokens,
        num_sequences * 32,
        torch.cuda.get_device_properties(flash_kda_device).multi_processor_count,
        arch,
    )
    expected_variant = _PRECOMPUTED_VARIANT_PREFIXES[num_tokens] + str(expected_split)
    assert frozen_calls == [expected_variant]

    if num_tokens == 1:
        frozen_kwargs = frozen_call_kwargs[0]
        assert frozen_kwargs["q"].shape == (1, num_sequences, 16, _D)
        assert frozen_kwargs["k"].shape == (1, num_sequences, 16, _D)
        assert frozen_kwargs["v"].shape == (1, num_sequences, 32, _D)
        assert frozen_kwargs["g"].shape == (1, num_sequences, 32, _D)
        assert frozen_kwargs["beta"].shape == (1, num_sequences, 32)
        assert frozen_kwargs["out"].shape == (1, num_sequences, 32, _D)
        assert frozen_kwargs["q"].data_ptr() == case["q"].data_ptr()
        assert frozen_kwargs["k"].data_ptr() == case["k"].data_ptr()
        assert frozen_kwargs["v"].data_ptr() == case["v"].data_ptr()
        assert frozen_kwargs["g"].data_ptr() == case["g"].data_ptr()
        assert frozen_kwargs["beta"].data_ptr() == case["beta"].data_ptr()
        assert frozen_kwargs["out"].data_ptr() == actual_output_buffer.data_ptr()
        assert frozen_kwargs["cu_seqlens"].tolist() == list(range(num_sequences + 1))
        assert frozen_kwargs["ssm_state_indices"].tolist() == list(range(num_sequences))
        assert frozen_kwargs["num_accepted_tokens"].tolist() == [1] * num_sequences

    assert actual_output.data_ptr() == actual_output_buffer.data_ptr()
    assert actual_state_result is actual_state
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )

    # Check every active checkpoint explicitly, rather than relying only on
    # the whole-pool comparison above to cover speculative state writes.
    if num_tokens == 1:
        active_checkpoint_slots = range(num_sequences)
    else:
        active_checkpoint_slots = case["ssm_state_indices"][:-1].reshape(-1).tolist()
    for slot in active_checkpoint_slots:
        torch.testing.assert_close(
            actual_state[slot].float(),
            expected_state[slot].float(),
            atol=1e-2,
            rtol=1e-2,
        )

    if num_tokens > 1:
        torch.testing.assert_close(
            actual_output[:, -num_tokens:],
            torch.zeros_like(actual_output[:, -num_tokens:]),
            atol=0,
            rtol=0,
        )
        padded_slot_start = (num_sequences - 1) * num_tokens
        torch.testing.assert_close(
            actual_state[padded_slot_start:],
            initial[padded_slot_start:],
            atol=0,
            rtol=0,
        )


def test_cake_backend_rejects_unexported_precomputed_t3_without_entering_cute_dsl(
    flash_kda_device, monkeypatch
):
    case = _make_case(
        flash_kda_device,
        num_sequences=4,
        num_heads=16,
        num_value_heads=32,
        num_tokens=3,
        seed=2260,
    )

    def unexpected_cake_launch(*args, **kwargs):
        pytest.fail(f"unexpected Cake launch: args={args}, kwargs={kwargs}")

    def unexpected_cute_compile(*args, **kwargs):
        pytest.fail(f"unexpected CuTe compile: args={args}, kwargs={kwargs}")

    monkeypatch.setattr(
        recurrent_module,
        "_run_flash_kda_decode",
        unexpected_cake_launch,
    )
    monkeypatch.setattr(
        recurrent_module,
        "_get_grouped_compiled",
        unexpected_cute_compile,
    )
    monkeypatch.setattr(
        recurrent_module,
        "_get_compiled_kernel",
        unexpected_cute_compile,
    )
    with pytest.raises(ValueError, match="backend='cake' does not support"):
        recurrent_kda(**_call_kwargs(case), backend="cake")


def test_cake_backend_rejects_explicit_t1_cu_seqlens_without_launching(
    flash_kda_device, monkeypatch
):
    num_sequences = 2
    case = _make_case(
        flash_kda_device,
        num_sequences=num_sequences,
        num_heads=16,
        num_value_heads=32,
        num_tokens=1,
        seed=2270,
    )
    for name in ("q", "k", "v", "g", "beta", "output"):
        tensor = case[name]
        case[name] = tensor.reshape(1, num_sequences, *tensor.shape[2:])
    case["cu_seqlens"] = torch.arange(
        num_sequences + 1, dtype=torch.int32, device=flash_kda_device
    )
    case["ssm_state_indices"] = torch.arange(
        num_sequences, dtype=torch.int32, device=flash_kda_device
    )

    def unexpected_launch(*args, **kwargs):
        pytest.fail(f"unexpected kernel launch: args={args}, kwargs={kwargs}")

    monkeypatch.setattr(
        recurrent_module,
        "_run_flash_kda_decode",
        unexpected_launch,
    )
    monkeypatch.setattr(
        recurrent_module,
        "_get_grouped_compiled",
        unexpected_launch,
    )
    monkeypatch.setattr(
        recurrent_module,
        "_get_compiled_kernel",
        unexpected_launch,
    )
    with pytest.raises(
        ValueError,
        match="does not support explicit T=1 cu_seqlens",
    ):
        recurrent_kda(**_call_kwargs(case), backend="cake")


def test_internal_direct_t1_nonidentity_metadata_is_memory_safe(flash_kda_device):
    generator = torch.Generator(device=flash_kda_device).manual_seed(2271)
    num_sequences = 2
    num_heads = 16
    num_value_heads = 32
    shape_q = (1, num_sequences, num_heads, _D)
    shape_v = (1, num_sequences, num_value_heads, _D)
    q = torch.randn(
        shape_q, dtype=torch.bfloat16, device=flash_kda_device, generator=generator
    )
    k = torch.randn_like(q)
    v = torch.randn(
        shape_v, dtype=torch.bfloat16, device=flash_kda_device, generator=generator
    )
    g = F.logsigmoid(
        torch.randn(
            shape_v,
            dtype=torch.float32,
            device=flash_kda_device,
            generator=generator,
        )
    ).to(torch.bfloat16)
    beta = torch.rand(
        (1, num_sequences, num_value_heads),
        dtype=torch.bfloat16,
        device=flash_kda_device,
        generator=generator,
    )
    state = torch.randn(
        (num_sequences, num_value_heads, _D, _D),
        dtype=torch.bfloat16,
        device=flash_kda_device,
        generator=generator,
    )
    state_before = state.clone()
    output_sentinel = 17.0
    out = torch.full(
        shape_v, output_sentinel, dtype=torch.bfloat16, device=flash_kda_device
    )
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device=flash_kda_device)

    recurrent_module._run_flash_kda_decode(
        "d128_t1_precomputed_direct_split16",
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        state=state,
        out=out,
        cu_seqlens=torch.tensor([0, 2, 2], dtype=torch.int32, device=flash_kda_device),
        ssm_state_indices=torch.arange(
            num_sequences, dtype=torch.int32, device=flash_kda_device
        ),
        num_accepted_tokens=torch.ones(
            num_sequences, dtype=torch.int32, device=flash_kda_device
        ),
        scale=_D**-0.5,
        A_log=dummy_f32,
        dt_bias=dummy_f32,
        lower_bound=0.0,
    )
    torch.cuda.synchronize(flash_kda_device)

    torch.testing.assert_close(
        out[:, 1],
        torch.full_like(out[:, 1], output_sentinel),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(state[1], state_before[1], atol=0, rtol=0)


@pytest.mark.parametrize("num_sequences", _T3_SEQUENCE_COUNTS)
def test_public_recurrent_kda_t3_lower_bound_measured_routes_match_cute_dsl(
    flash_kda_device, monkeypatch, num_sequences
):
    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)

    def check_case(*, accepted_token, padded_last_sequence):
        case = _make_t3_lower_bound_case(
            flash_kda_device,
            num_sequences=num_sequences,
            accepted_token=accepted_token,
            padded_last_sequence=padded_last_sequence,
            seed=2300
            + 20 * num_sequences
            + accepted_token
            + 1000 * padded_last_sequence,
        )
        initial = case["initial_state"].clone()
        baseline_state = initial.clone()
        actual_state = initial.clone()
        baseline_output = torch.empty_like(case["output"])
        actual_output_buffer = torch.empty_like(case["output"])

        expected_output, expected_state = recurrent_kda(
            **_call_kwargs(
                case,
                state=baseline_state,
                output=baseline_output,
            ),
            backend="cute-dsl",
        )

        frozen_calls.clear()
        actual_output, actual_state_result = recurrent_kda(
            **_call_kwargs(
                case,
                state=actual_state,
                output=actual_output_buffer,
            ),
            backend="cake",
        )

        assert frozen_calls == [_T3_VARIANT]
        assert actual_output.data_ptr() == actual_output_buffer.data_ptr()
        assert actual_state_result is actual_state
        assert actual_output.dtype == expected_output.dtype == torch.bfloat16
        assert actual_state.dtype == expected_state.dtype == torch.bfloat16
        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            actual_state,
            expected_state,
            atol=1e-2,
            rtol=1e-2,
        )

        active_checkpoint_slots = case["ssm_state_indices"][
            case["ssm_state_indices"] >= 0
        ].tolist()
        for slot in active_checkpoint_slots:
            torch.testing.assert_close(
                actual_state[slot],
                expected_state[slot],
                atol=1e-2,
                rtol=1e-2,
            )

        if padded_last_sequence:
            padded_token_start = (num_sequences - 1) * _T3
            torch.testing.assert_close(
                actual_output[:, padded_token_start:],
                torch.zeros_like(actual_output[:, padded_token_start:]),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                actual_state[padded_token_start:],
                initial[padded_token_start:],
                atol=0,
                rtol=0,
            )

    for accepted_token in (0, 1, _T3, _T3 + 7):
        check_case(
            accepted_token=accepted_token,
            padded_last_sequence=False,
        )

    check_case(
        accepted_token=_T3 + 7,
        padded_last_sequence=True,
    )


def test_public_route_supports_padded_slots_nat_and_outer_strides(
    flash_kda_device, monkeypatch
):
    accepted_tokens = [0, 1, _T, _T + 7, _T - 1]
    case = _make_case(
        flash_kda_device,
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
    expected_output, expected_state = recurrent_kda(
        **_call_kwargs(case, state=baseline_state, output=baseline_output),
        backend="cute-dsl",
    )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(variant)
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    actual_output, actual_state_result = recurrent_kda(
        **_call_kwargs(case, state=actual_state, output=actual_output_buffer),
        backend="cake",
    )

    arch = recurrent_module._FLASH_KDA_DECODE_ARCH_BY_COMPUTE_CAPABILITY[
        torch.cuda.get_device_capability(flash_kda_device)
    ]
    expected_split = recurrent_module._select_flash_kda_decode_value_split(
        _T,
        len(accepted_tokens) * 32,
        torch.cuda.get_device_properties(flash_kda_device).multi_processor_count,
        arch,
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


def test_frozen_decode_cuda_graph_on_non_default_stream(flash_kda_device, monkeypatch):
    case = _make_case(
        flash_kda_device,
        num_sequences=5,
        num_heads=16,
        num_value_heads=32,
        seed=2060,
    )
    initial = case["initial_state"].clone()
    baseline_state = initial.clone()
    baseline_output = torch.empty_like(case["output"])
    expected_output, expected_state = recurrent_kda(
        **_call_kwargs(case, state=baseline_state, output=baseline_output),
        backend="cute-dsl",
    )

    frozen_calls = []
    run_frozen = recurrent_module._run_flash_kda_decode

    def track_frozen_call(variant, **kwargs):
        frozen_calls.append(
            (
                variant,
                int(torch.cuda.current_stream(flash_kda_device).cuda_stream),
            )
        )
        return run_frozen(variant, **kwargs)

    monkeypatch.setattr(recurrent_module, "_run_flash_kda_decode", track_frozen_call)
    graph_state = initial.clone()
    graph_output = torch.empty_like(case["output"])
    graph_kwargs = _call_kwargs(case, state=graph_state, output=graph_output)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    with torch.cuda.stream(capture_stream):
        recurrent_kda(**graph_kwargs, backend="cake")
        graph_state.copy_(initial)
        graph_output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**graph_kwargs, backend="cake")

    expected_variant = _VARIANT_PREFIX + "2"
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
