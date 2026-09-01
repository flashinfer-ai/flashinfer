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

import importlib
import json
import math
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version

import flashinfer
from flashinfer.kda import RecurrentKDAPrefillWrapper, recurrent_kda
from flashinfer.kda_prefill import RecurrentKDAPrefillWorkspace
from flashinfer.utils import get_compute_capability

kda_decode_api = importlib.import_module("flashinfer.kda_decode")
kda_api = importlib.import_module("flashinfer.kda")
kda_prefill_api = importlib.import_module("flashinfer.kda_prefill")
kda_prefill_cute_api = importlib.import_module("flashinfer.kda_prefill_cute")
cake_kda_jit_api = importlib.import_module("flashinfer.jit.cake_kda")


@pytest.fixture(autouse=True)
def _legacy_module_stubs_select_the_legacy_fallback(monkeypatch):
    """Keep legacy ABI stubs on the selector-miss fallback they exercise."""

    from flashinfer.jit.flash_kda import _GeneratedFlashKDASelectorNotFoundError

    original_legacy_resolver = kda_prefill_api._get_flash_kda_prefill_module
    original_generated_resolver = kda_prefill_api._get_flash_kda_generated_module

    def resolve_generated(selector_key):
        if (
            kda_prefill_api._get_flash_kda_prefill_module
            is not original_legacy_resolver
        ):
            raise _GeneratedFlashKDASelectorNotFoundError(
                "legacy test module stub requests the selector-miss fallback"
            )
        return original_generated_resolver(selector_key)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_generated_module",
        resolve_generated,
    )


def test_public_api_uses_phase_neutral_facade_and_prefill_workspace():
    assert flashinfer.recurrent_kda is kda_api.recurrent_kda
    assert (
        flashinfer.RecurrentKDAPrefillWorkspace
        is kda_prefill_api.RecurrentKDAPrefillWorkspace
    )
    assert flashinfer.RecurrentKDAPrefillWrapper is RecurrentKDAPrefillWrapper


def test_cake_kda_prefill_jit_surface_includes_checkpoint_aligned_bt64():
    assert cake_kda_jit_api.CAKE_KDA_VARIANTS == (
        "m128_unbounded_softplus",
        "m128_bt64_unbounded_softplus",
    )
    for target in ("sm100a", "sm103a"):
        n32_uri = cake_kda_jit_api.get_cake_kda_uri("m128_unbounded_softplus", target)
        bt64_uri = cake_kda_jit_api.get_cake_kda_uri(
            "m128_bt64_unbounded_softplus", target
        )
        assert n32_uri != bt64_uri
        assert bt64_uri.endswith(f"_8f5147c17f_{target}")
    csrc_dir = cake_kda_jit_api._get_cake_kda_csrc_dir()
    assert (csrc_dir / "cake_kda_bf16_fused_m128_bt64_unbounded_softplus.cu").is_file()
    assert (
        csrc_dir / "cake_kda_bf16_fused_m128_bt64_unbounded_softplus_binding.cu"
    ).is_file()


def test_cake_kda_affine_manifest_controls_export_availability():
    csrc_dir = cake_kda_jit_api._get_cake_kda_csrc_dir()
    manifest = json.loads(
        (
            csrc_dir / "cake_kda_bf16_affine_unbounded_softplus_import_manifest.json"
        ).read_text()
    )
    cake_kda_jit_api.get_cake_kda_affine_module_specs.cache_clear()
    specs = cake_kda_jit_api.get_cake_kda_affine_module_specs()
    if manifest["status"] == "pending_generated_sources":
        assert manifest["modules"] == []
        assert manifest["remaining_generated_inputs"]
        assert specs == ()
        assert not cake_kda_jit_api.cake_kda_affine_is_available()
    else:
        assert manifest["status"] == "complete"
        assert len(specs) == 8
        assert cake_kda_jit_api.cake_kda_affine_is_available()
        assert {spec.target for spec in specs} == {"sm100a", "sm103a"}
        assert {spec.role for spec in specs} == {
            "main",
            "map",
            "scan",
            "correction",
        }


def _valid_cake_kda_affine_selector_kwargs():
    return {
        "export_available": True,
        "compute_capability": (10, 0),
        "sm_count": 148,
        "fixed_layout": True,
        "batch_size": 1,
        "total_tokens": 8192,
        "num_heads": 32,
        "head_dim": 128,
        "qkv_shapes_equal": True,
        "qkv_dtype": torch.bfloat16,
        "beta_contiguous": True,
        "beta_dtype": torch.bfloat16,
        "indexed_state": True,
        "initial_state_dtype": torch.bfloat16,
        "has_checkpoints": False,
        "lower_bound": None,
    }


@pytest.mark.parametrize(
    ("num_heads", "expected_affine"),
    ((4, True), (8, True), (16, True), (32, False)),
)
@pytest.mark.parametrize(
    ("compute_capability", "expected_target"),
    (((10, 0), "sm100a"), ((10, 3), "sm103a")),
)
def test_cake_kda_affine_selector_builds_exact_blackwell_partition(
    num_heads, expected_affine, compute_capability, expected_target
):
    kwargs = _valid_cake_kda_affine_selector_kwargs()
    kwargs["num_heads"] = num_heads
    kwargs["compute_capability"] = compute_capability
    plan = kda_prefill_api._select_cake_kda_affine_plan(**kwargs)
    if not expected_affine:
        assert plan is None
        return
    assert plan is not None
    assert plan.target == expected_target
    assert plan.num_parts >= 2
    assert plan.token_offsets[0] == 0
    assert plan.token_offsets[-1] == kwargs["total_tokens"]
    assert all(offset % 32 == 0 for offset in plan.token_offsets)
    assert all(
        left < right
        for left, right in zip(
            plan.token_offsets[:-1], plan.token_offsets[1:], strict=True
        )
    )


@pytest.mark.parametrize(
    ("override", "value"),
    (
        ("export_available", False),
        ("compute_capability", (9, 0)),
        ("sm_count", 1),
        ("fixed_layout", False),
        ("batch_size", 2),
        ("total_tokens", 8160),
        ("total_tokens", 8193),
        ("num_heads", 33),
        ("head_dim", 64),
        ("qkv_shapes_equal", False),
        ("qkv_dtype", torch.float16),
        ("beta_contiguous", False),
        ("beta_dtype", torch.float32),
        ("indexed_state", False),
        ("initial_state_dtype", torch.float32),
        ("has_checkpoints", True),
        ("lower_bound", -1.0),
    ),
)
def test_cake_kda_affine_selector_rejects_out_of_contract_calls(override, value):
    kwargs = _valid_cake_kda_affine_selector_kwargs()
    kwargs[override] = value
    assert kda_prefill_api._select_cake_kda_affine_plan(**kwargs) is None


def test_cake_kda_affine_workspace_buffer_is_grow_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    workspace = SimpleNamespace(_cake_kda_affine_buffers={})
    first = kda_prefill_api._cake_kda_affine_workspace_buffer(
        workspace=workspace,
        name="carry",
        device=torch.device("cpu"),
        shape=(4, 8),
        dtype=torch.float32,
        zero_on_allocate=True,
    )
    assert torch.count_nonzero(first) == 0
    smaller = kda_prefill_api._cake_kda_affine_workspace_buffer(
        workspace=workspace,
        name="carry",
        device=torch.device("cpu"),
        shape=(2, 8),
        dtype=torch.float32,
    )
    assert smaller.data_ptr() == first.data_ptr()
    assert workspace._cake_kda_affine_buffers["carry"].numel() == 32

    with pytest.raises(ValueError, match="dimensions must be positive"):
        kda_prefill_api._cake_kda_affine_workspace_buffer(
            workspace=workspace,
            name="carry",
            device=torch.device("cpu"),
            shape=(0, 8),
            dtype=torch.float32,
        )


def test_prefill_wrapper_plan_builds_stable_device_metadata(cuda_device):
    wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    wrapper.plan(torch.tensor([0, 0, 7, 7, 12], device=cuda_device))

    cu_seqlens_ptr = wrapper._cu_seqlens_buf.data_ptr()
    seq_order_ptr = wrapper._seq_order_buf.data_ptr()
    cu_chunks_ptr = wrapper._cu_chunks_buf.data_ptr()
    assert wrapper._cu_seqlens_buf.dtype == torch.int64
    assert wrapper._cu_seqlens_buf.tolist() == [0, 0, 7, 7, 12]
    assert wrapper._seq_order_buf.tolist() == [1, 3, 0, 2]
    assert wrapper._cu_chunks_buf.tolist() == [0, 0, 1, 1, 2]
    assert wrapper._workspace._cute_dsl_total_chunks == 2

    wrapper.plan(torch.tensor([0, 0, 2, 2, 12], device=cuda_device))
    assert wrapper._cu_seqlens_buf.data_ptr() == cu_seqlens_ptr
    assert wrapper._seq_order_buf.data_ptr() == seq_order_ptr
    assert wrapper._cu_chunks_buf.data_ptr() == cu_chunks_ptr
    assert wrapper._seq_order_buf.tolist() == [3, 1, 0, 2]

    with pytest.raises(ValueError, match="total token count is fixed"):
        wrapper.plan(torch.tensor([0, 0, 2, 2, 13], device=cuda_device))

    with pytest.raises(ValueError, match="number of sequences is fixed"):
        wrapper.plan(torch.tensor([0, 2, 12], device=cuda_device))

    chunk_wrapper = RecurrentKDAPrefillWrapper(cuda_device)
    chunk_wrapper.plan(torch.tensor([0, 16, 16, 32], device=cuda_device))
    with pytest.raises(ValueError, match="chunk count is fixed"):
        chunk_wrapper.plan(torch.tensor([0, 1, 17, 32], device=cuda_device))

    with pytest.raises(ValueError, match="non-decreasing"):
        RecurrentKDAPrefillWrapper(cuda_device).plan(
            torch.tensor([0, 2, 1, 12], device=cuda_device)
        )


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
    tensors = _cpu_route_tensors(token_count=3)
    tensors = {
        key: value.to(cuda_device) if isinstance(value, torch.Tensor) else value
        for key, value in tensors.items()
    }

    assert wrapper.run(**tensors) is sentinel
    assert calls[0]["cu_seqlens"] is wrapper._cu_seqlens_buf
    assert calls[0]["seq_order"] is wrapper._seq_order_buf
    assert calls[0]["prefill_workspace"] is wrapper._workspace
    assert calls[0]["backend"] == "cute-dsl"
    assert wrapper._workspace._cute_dsl_cu_chunks is wrapper._cu_chunks_buf
    assert wrapper._workspace._cute_dsl_total_chunks == 2


def _cpu_route_tensors(token_count=2):
    shape = (1, token_count, 1, 128)
    return {
        "q": torch.empty(shape, dtype=torch.bfloat16),
        "k": torch.empty(shape, dtype=torch.bfloat16),
        "v": torch.empty(shape, dtype=torch.bfloat16),
        "g": torch.empty(shape, dtype=torch.bfloat16),
        "beta": torch.empty((1, token_count, 1), dtype=torch.bfloat16),
        "A_log": torch.empty(1, dtype=torch.float32),
        "dt_bias": torch.empty((1, 128), dtype=torch.float32),
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "beta_is_logit": True,
    }


def test_public_prefill_backend_option_routes_to_cute_dsl(monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert recurrent_kda(**_cpu_route_tensors(), backend="cute-dsl") is sentinel


def test_public_prefill_auto_prefers_cute_dsl(monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: sentinel,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: pytest.fail("auto should not probe Cake after a CuTe match"),
    )

    assert recurrent_kda(**_cpu_route_tensors()) is sentinel


def test_public_prefill_forwards_sequence_order_to_cute_dsl(monkeypatch):
    calls = []
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    seq_order = torch.tensor([1, 0], dtype=torch.int32)
    assert (
        recurrent_kda(
            **_cpu_route_tensors(token_count=3),
            cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int64),
            seq_order=seq_order,
        )
        is sentinel
    )
    assert calls[0]["seq_order"] is seq_order


def test_public_prefill_auto_falls_back_to_cake(monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert recurrent_kda(**_cpu_route_tensors()) is sentinel


def test_public_prefill_explicit_cake_skips_cute_dsl_probe_with_checkpoints(
    monkeypatch,
):
    sentinel = (object(), object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: pytest.fail("backend='cake' must not probe CuTe DSL"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    checkpoint_state = torch.empty((1, 1, 128, 128), dtype=torch.bfloat16)
    checkpoint_starts = torch.tensor([0, 1], dtype=torch.int64)
    assert (
        recurrent_kda(
            **_cpu_route_tensors(),
            state_checkpoints=checkpoint_state,
            checkpoint_cu_starts=checkpoint_starts,
            checkpoint_every_n_tokens=32,
            backend="cake",
        )
        is sentinel
    )


def test_public_prefill_auto_routes_supported_checkpoints_to_cute_dsl(monkeypatch):
    calls = []
    sentinel = (object(), object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    checkpoints = torch.empty((1, 1, 128, 128), dtype=torch.bfloat16)
    starts = torch.tensor([0, 1], dtype=torch.int64)
    assert (
        recurrent_kda(
            **_cpu_route_tensors(),
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=starts,
            checkpoint_every_n_tokens=32,
        )
        is sentinel
    )
    assert calls[0]["state_checkpoints"] is checkpoints
    assert calls[0]["checkpoint_cu_starts"] is starts
    assert calls[0]["checkpoint_every_n_tokens"] == 32


def test_public_prefill_cake_backend_is_strict(monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: False,
    )

    with pytest.raises(ValueError, match="backend='cake' does not support"):
        recurrent_kda(**_cpu_route_tensors(), backend="cake")


def test_public_decode_backend_option_forwards_to_decode_layer(monkeypatch):
    calls = []
    sentinel = (object(), object())

    def run(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", run)
    assert (
        recurrent_kda(**_cpu_route_tensors(token_count=1), backend="cake") is sentinel
    )
    assert calls[0]["backend"] == "cake"


def test_public_backend_option_rejects_unknown_value():
    with pytest.raises(ValueError, match="backend must be"):
        recurrent_kda(**_cpu_route_tensors(), backend="unknown")


def test_cute_dsl_prefill_adapter_preserves_indexed_in_place_state_semantics(
    monkeypatch,
):
    calls = []
    compile_args = []
    identity_order = torch.tensor([0], dtype=torch.int32)

    class Compiled:
        def workspace_size(self, cu_seqlens, heads, **kwargs):
            assert cu_seqlens is None
            assert heads == 1
            assert kwargs == {"batch": 1, "seqlen": 2}
            return 0

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))

    def get_compiled(**kwargs):
        compile_args.append(kwargs)
        return Compiled()

    monkeypatch.setattr(
        kda_prefill_cute_api, "_get_compiled_cute_dsl_kda", get_compiled
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_identity_seq_order",
        lambda **kwargs: identity_order,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )

    inputs = _cpu_route_tensors()
    state = torch.empty((3, 1, 128, 128), dtype=torch.bfloat16)
    state_indices = torch.tensor([2], dtype=torch.int32)
    output = torch.empty_like(inputs["q"])
    result = kda_prefill_cute_api._run_cute_dsl_kda_prefill(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=None,
        initial_state=state,
        output_final_state=False,
        lower_bound=-5.0,
        cu_seqlens=None,
        seq_order=None,
        output=output,
        prefill_workspace=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
        state_indices=state_indices,
    )

    assert result[0] is output
    assert result[1] is None
    assert compile_args == [
        {
            "lower_bound": -5.0,
            "has_state_in": True,
            "has_state_out": True,
            "has_state_ckpt": False,
            "has_state_indices": True,
        }
    ]
    args, kwargs = calls[0]
    assert args[8] is state
    assert args[10] is state
    assert args[11] is None
    assert args[12] == 7
    assert kwargs == {
        "seq_order": identity_order,
        "state_indices": state_indices,
        "planned_cu_chunks": None,
        "planned_total_chunks": None,
    }


@pytest.mark.parametrize("explicit_order", [False, True])
def test_cute_dsl_prefill_adapter_forwards_packed_sequence_order(
    monkeypatch, explicit_order
):
    calls = []

    class Compiled:
        def workspace_size(self, cu_seqlens, heads, **kwargs):
            assert cu_seqlens.tolist() == [0, 1, 2]
            assert heads == 1
            assert kwargs == {}
            return 0

        def __call__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_get_compiled_cute_dsl_kda",
        lambda **kwargs: Compiled(),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: SimpleNamespace(cuda_stream=7)
    )

    inputs = _cpu_route_tensors()
    output = torch.empty_like(inputs["q"])
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int64)
    seq_order = torch.tensor([1, 0], dtype=torch.int32) if explicit_order else None
    result = kda_prefill_cute_api._run_cute_dsl_kda_prefill(
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        scale=None,
        initial_state=None,
        output_final_state=False,
        lower_bound=-5.0,
        cu_seqlens=cu_seqlens,
        seq_order=seq_order,
        output=output,
        prefill_workspace=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )

    assert result[0] is output
    assert result[1] is None
    args, kwargs = calls[0]
    assert args[7] is cu_seqlens
    assert set(kwargs) == {
        "state_indices",
        "seq_order",
        "planned_cu_chunks",
        "planned_total_chunks",
    }
    assert kwargs["seq_order"] is seq_order
    assert kwargs["state_indices"] is None
    assert kwargs["planned_cu_chunks"] is None
    assert kwargs["planned_total_chunks"] is None


def test_cute_dsl_lpt_sequence_order_is_content_cached(monkeypatch):
    kernel_module = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    monkeypatch.setattr(kernel_module, "_CU_CONTENTS_MEMO", {})
    monkeypatch.setattr(kernel_module, "_LPT_SEQUENCE_ORDER_CACHE", {})
    cu_seqlens = torch.tensor(
        [0, 1300, 1847, 3895, 4858, 5129, 8192], dtype=torch.int64
    )

    first = kernel_module._lpt_sequence_order(cu_seqlens)
    second = kernel_module._lpt_sequence_order(cu_seqlens.clone())

    assert first.tolist() == [5, 2, 0, 3, 1, 4]
    assert second.data_ptr() == first.data_ptr()


def test_cute_dsl_unplanned_packed_engine_rejects_graph_capture(monkeypatch):
    kernel_module = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(kernel_module, "_device_sm_count", lambda device: 148)
    monkeypatch.setattr(
        kernel_module,
        "_route_for_workspace",
        lambda n_seq, heads, device, mode: "engine",
    )
    compiled = kernel_module._make_call(
        lambda *args, **kwargs: None,
        {
            "mode": None,
            "dtype": object(),
            "state_dtype": object(),
            "gate_dtype": object(),
            "safe_gate": True,
            "gate_lower_bound": -5.0,
            "has_state_in": False,
            "has_state_out": False,
            "has_state_ckpt": False,
            "has_state_indices": False,
        },
    )
    inputs = _cpu_route_tensors()
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int64)

    with pytest.raises(RuntimeError, match=r"Wrapper\.plan\(\)"):
        compiled(
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["g"],
            inputs["A_log"],
            inputs["dt_bias"],
            inputs["beta"],
            cu_seqlens,
            None,
            torch.empty_like(inputs["q"]),
            None,
            torch.empty(0, dtype=torch.uint8),
            0,
        )


def test_cute_dsl_engine_workspace_query_does_not_read_device_offsets(monkeypatch):
    kernel_module = importlib.import_module("flashinfer.kda_kernels.kda_chunked_bt16")
    monkeypatch.setattr(kernel_module, "_device_sm_count", lambda device: 148)
    monkeypatch.setattr(
        kernel_module,
        "_cu_seqlens_contents",
        lambda tensor: pytest.fail("engine workspace query must not read offsets"),
    )

    cu_seqlens = torch.tensor([0, 3, 7, 12, 18, 25], dtype=torch.int64)
    assert kernel_module.workspace_size(cu_seqlens, heads=64) == 0


def _strict_prefill_kwargs(inputs, *, lower_bound=-5.0):
    return {
        **inputs,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": lower_bound,
        "beta_is_logit": True,
    }


def _make_inputs(
    *,
    seq_lens,
    num_heads: int,
    packed: bool,
    initial_state: bool = False,
    state_dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    if packed:
        batch_size = 1
        seq_len = sum(seq_lens)
    else:
        if len(set(seq_lens)) != 1:
            raise ValueError("fixed test inputs require equal sequence lengths")
        batch_size = len(seq_lens)
        seq_len = seq_lens[0]
    shape = (batch_size, seq_len, num_heads, 128)
    q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = (0.1 * torch.randn(shape, dtype=torch.float32, device="cuda")).to(
        torch.bfloat16
    )
    beta = torch.randn(
        (batch_size, seq_len, num_heads),
        dtype=torch.bfloat16,
        device="cuda",
    )
    A_log = 0.1 * torch.randn(num_heads, dtype=torch.float32, device="cuda")
    dt_bias = 0.1 * torch.randn((num_heads, 128), dtype=torch.float32, device="cuda")
    offsets = [0]
    for length in seq_lens:
        offsets.append(offsets[-1] + length)
    state = None
    if initial_state:
        state = (
            0.1
            * torch.randn(
                (len(seq_lens), num_heads, 128, 128),
                dtype=torch.float32,
                device="cuda",
            )
        ).to(state_dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "initial_state": state,
        "cu_seqlens": (
            torch.tensor(offsets, dtype=torch.int64, device="cuda") if packed else None
        ),
    }


@pytest.mark.parametrize(
    "field",
    (
        "beta",
        "cu_seqlens",
        "seq_order",
        "ssm_state_indices",
        "initial_state",
        "output",
    ),
)
def test_public_prefill_auto_falls_back_for_non_tensor_arguments(
    flash_kda_device,
    monkeypatch,
    field,
):
    with torch.cuda.device(flash_kda_device):
        inputs = _make_inputs(
            seq_lens=[3, 5],
            num_heads=1,
            packed=True,
            initial_state=True,
        )
    kwargs = _strict_prefill_kwargs(inputs)
    kwargs[field] = object()
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_prefill_is_eligible",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: pytest.fail("ineligible call must not run CuTe DSL"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: sentinel,
    )

    assert recurrent_kda(**kwargs) is sentinel


def _reference(inputs, *, lower_bound=-5.0, scale=None, checkpoint_every_n_tokens=0):
    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_flat = torch.sigmoid(inputs["beta"].float().reshape(-1, num_heads))
    gate_input = g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim)
    if lower_bound is None:
        gate = -torch.exp(inputs["A_log"]).reshape(1, num_heads, 1) * F.softplus(
            gate_input
        )
    else:
        gate = lower_bound * torch.sigmoid(
            torch.exp(inputs["A_log"]).reshape(1, num_heads, 1) * gate_input
        )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    checkpoints = []
    for sequence in range(len(offsets) - 1):
        if checkpoint_every_n_tokens:
            checkpoints.append(state[sequence].clone())
        sequence_length = offsets[sequence + 1] - offsets[sequence]
        for local_token, token in enumerate(
            range(offsets[sequence], offsets[sequence + 1]), start=1
        ):
            state_f32 = state[sequence].float()
            decayed = state_f32 * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            residual = beta_flat[token].unsqueeze(-1) * (v_flat[token] - predicted)
            updated = decayed + residual.unsqueeze(-1) * k_flat[token].unsqueeze(1)
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum(
                "hk,hvk->hv", q_flat[token], state[sequence].float()
            )
            out[token] = (scale * projected).to(torch.bfloat16)
            if (
                checkpoint_every_n_tokens
                and local_token % checkpoint_every_n_tokens == 0
                and local_token < sequence_length
            ):
                checkpoints.append(state[sequence].clone())
    result = (out.reshape_as(q), state)
    if checkpoint_every_n_tokens:
        return (*result, torch.stack(checkpoints))
    return result


def _h12_bf16_residual_carriers(torch, *, value, prediction, beta_logit):
    """Apply the four BF16 residual carriers selected by the public H12 ABI."""

    prediction_carrier = prediction.to(torch.bfloat16).float()
    delta_carrier = (value - prediction_carrier).to(torch.bfloat16).float()
    beta_carrier = torch.sigmoid(beta_logit).to(torch.bfloat16).float()
    update_carrier = (
        (beta_carrier.unsqueeze(-1) * delta_carrier).to(torch.bfloat16).float()
    )
    return prediction_carrier, delta_carrier, beta_carrier, update_carrier


def test_h12_smoke_reference_residual_carriers_round_every_boundary_on_cpu():
    prediction = torch.tensor(
        [[-15.22768497, -1.95509577, 3.25501537, 0.3333]],
        dtype=torch.float32,
    )
    value = torch.tensor(
        [[3.81683922, -9.65635967, -4.79144955, 0.7123]],
        dtype=torch.float32,
    )
    beta_logit = torch.tensor([-1.02760863], dtype=torch.float32)

    prediction_carrier, delta_carrier, beta_carrier, update_carrier = (
        _h12_bf16_residual_carriers(
            torch,
            value=value,
            prediction=prediction,
            beta_logit=beta_logit,
        )
    )
    expected_prediction = prediction.to(torch.bfloat16).float()
    unrounded_delta = value - expected_prediction
    expected_delta = unrounded_delta.to(torch.bfloat16).float()
    unrounded_beta = torch.sigmoid(beta_logit)
    expected_beta = unrounded_beta.to(torch.bfloat16).float()
    unrounded_update = expected_beta.unsqueeze(-1) * expected_delta
    expected_update = unrounded_update.to(torch.bfloat16).float()

    assert torch.equal(prediction_carrier, expected_prediction)
    assert torch.equal(delta_carrier, expected_delta)
    assert torch.equal(beta_carrier, expected_beta)
    assert torch.equal(update_carrier, expected_update)
    assert not torch.equal(prediction_carrier, prediction)
    assert not torch.equal(delta_carrier, unrounded_delta)
    assert not torch.equal(beta_carrier, unrounded_beta)
    assert not torch.equal(update_carrier, unrounded_update)


def _chunk16_debug_reference(
    inputs, *, lower_bound=-5.0, scale=None, checkpoint_every_n_tokens=0
):
    """Clean-room H12 smoke reference for focused numerical diagnostics.

    The recurrent state carrier stays in FP32 within each 16-token chunk, but
    the state/K prediction, V-minus-prediction delta, sigmoid beta, and
    post-beta update carrier each round through BF16.  A BF16 state snapshot
    becomes the next chunk's carrier, while each output projects the unrounded
    FP32 state for its token.  The public benchmark separately compares output
    and complete final state against the pinned FlashKDA implementation.
    """

    q = inputs["q"]
    batch_size, seq_len, num_heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else scale
    q_flat = F.normalize(q.float(), dim=-1).reshape(-1, num_heads, head_dim)
    k_flat = F.normalize(inputs["k"].float(), dim=-1).reshape(-1, num_heads, head_dim)
    v_flat = inputs["v"].float().reshape(-1, num_heads, head_dim)
    g_flat = inputs["g"].float().reshape(-1, num_heads, head_dim)
    beta_logits_flat = inputs["beta"].float().reshape(-1, num_heads)
    gate = lower_bound * torch.sigmoid(
        torch.exp(inputs["A_log"]).reshape(1, num_heads, 1)
        * (g_flat + inputs["dt_bias"].reshape(1, num_heads, head_dim))
    )
    decay = torch.exp(gate)
    if inputs["cu_seqlens"] is None:
        offsets = [index * seq_len for index in range(batch_size + 1)]
    else:
        offsets = [int(value) for value in inputs["cu_seqlens"].tolist()]
    if inputs["initial_state"] is None:
        state = torch.zeros(
            (len(offsets) - 1, num_heads, head_dim, head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        state = inputs["initial_state"].clone()
    out = torch.empty_like(q_flat)
    checkpoints = []
    for sequence in range(len(offsets) - 1):
        if checkpoint_every_n_tokens:
            checkpoints.append(state[sequence].clone())
        carrier = state[sequence].float()
        sequence_length = offsets[sequence + 1] - offsets[sequence]
        for local_token, token in enumerate(
            range(offsets[sequence], offsets[sequence + 1]), start=1
        ):
            decayed = carrier * decay[token].unsqueeze(1)
            predicted = torch.einsum("hk,hvk->hv", k_flat[token], decayed)
            _, _, _, update_carrier = _h12_bf16_residual_carriers(
                torch,
                value=v_flat[token],
                prediction=predicted,
                beta_logit=beta_logits_flat[token],
            )
            updated = decayed + update_carrier.unsqueeze(-1) * k_flat[token].unsqueeze(
                1
            )
            state[sequence] = updated.to(torch.bfloat16)
            projected = torch.einsum("hk,hvk->hv", q_flat[token], updated)
            out[token] = (scale * projected).to(torch.bfloat16)
            carrier = state[sequence].float() if local_token % 16 == 0 else updated
            if (
                checkpoint_every_n_tokens
                and local_token % checkpoint_every_n_tokens == 0
                and local_token < sequence_length
            ):
                checkpoints.append(state[sequence].clone())
    result = (out.reshape_as(q), state)
    if checkpoint_every_n_tokens:
        return (*result, torch.stack(checkpoints))
    return result


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
    ("compute_capability", "cuda_version", "expected_target", "error_match"),
    [
        ((10, 0), "12.8", "sm100a", None),
        ((10, 0), "12.9", "sm100a", None),
        ((10, 3), "12.8", None, "10.3 requires CUDA 12.9"),
        ((10, 3), "12.9", "sm103a", None),
        ((12, 0), "13.0", None, "requires compute capability 10.0"),
        ((10, 0), "12.7", None, "10.0 requires CUDA 12.8"),
    ],
)
def test_flash_kda_target_resolution(
    monkeypatch,
    compute_capability,
    cuda_version,
    expected_target,
    error_match,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda required: Version(cuda_version) >= Version(required),
    )
    device = torch.device("cuda")
    if error_match is not None:
        with pytest.raises(RuntimeError, match=error_match):
            kda_prefill_api._select_flash_kda_prefill_target(device)
    else:
        assert (
            kda_prefill_api._select_flash_kda_prefill_target(device) == expected_target
        )


def test_flash_kda_sm_count_is_cached_per_device(monkeypatch):
    calls = []

    def get_device_properties(device):
        resolved = torch.device(device)
        calls.append(resolved)
        return SimpleNamespace(
            multi_processor_count=148 if resolved.index == 0 else 152
        )

    kda_prefill_api._flash_kda_device_sm_count.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    try:
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:0")) == 148
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:0")) == 148
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:1")) == 152
        assert kda_prefill_api._flash_kda_device_sm_count(torch.device("cuda:1")) == 152
        assert calls == [torch.device("cuda:0"), torch.device("cuda:1")]
    finally:
        kda_prefill_api._flash_kda_device_sm_count.cache_clear()


def _make_padded_state_pool(*, slots, num_heads, dtype, device):
    slot_numel = num_heads * 128 * 128
    storage = torch.empty(
        (slots, slot_numel + 64),
        dtype=dtype,
        device=device,
    )
    return storage.as_strided(
        (slots, num_heads, 128, 128),
        (storage.stride(0), 128 * 128, 128, 1),
    )


def _frozen_prefill_eligibility_kwargs(inputs, *, output, state_indices=None):
    return {
        "q": inputs["q"],
        "k": inputs["k"],
        "v": inputs["v"],
        "g": inputs["g"],
        "beta": inputs["beta"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "initial_state": inputs["initial_state"],
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "lower_bound": -5.0,
        "cu_seqlens": inputs["cu_seqlens"],
        "ssm_state_indices": state_indices,
        "num_spec_tokens": None,
        "num_accepted_tokens": None,
        "output": output,
        "initial_state_source": None,
        "initial_state_indices": None,
        "beta_is_logit": True,
        "state_checkpoints": None,
        "checkpoint_cu_starts": None,
        "checkpoint_every_n_tokens": 0,
    }


@pytest.mark.parametrize(
    ("state_dtype", "state_mode"),
    [
        (torch.bfloat16, "compact"),
        (torch.bfloat16, "indexed"),
        (torch.float32, "indexed"),
    ],
)
@pytest.mark.parametrize("packed", [False, True], ids=["fixed", "packed"])
@pytest.mark.parametrize("num_heads", [6, 12], ids=["h6", "h12"])
def test_frozen_prefill_state_pool_eligibility_accepts_supported_contracts(
    flash_kda_device,
    state_dtype,
    state_mode,
    packed,
    num_heads,
):
    seq_lens = [17, 33] if packed else [33, 33]
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=True,
        state_dtype=state_dtype,
        seed=23_000 + num_heads + int(packed),
    )
    compact_seed = inputs["initial_state"]
    state_indices = None
    if state_mode == "compact":
        state_pool = _make_padded_state_pool(
            slots=len(seq_lens),
            num_heads=num_heads,
            dtype=state_dtype,
            device=flash_kda_device,
        )
        state_pool.copy_(compact_seed)
    else:
        state_indices = torch.tensor([3, 1], dtype=torch.int32, device=flash_kda_device)
        state_pool = _make_padded_state_pool(
            slots=5,
            num_heads=num_heads,
            dtype=state_dtype,
            device=flash_kda_device,
        )
        state_pool.index_copy_(0, state_indices.to(torch.int64), compact_seed)
    inputs["initial_state"] = state_pool

    assert kda_prefill_api._flash_kda_prefill_is_eligible(
        **_frozen_prefill_eligibility_kwargs(
            inputs,
            output=torch.empty_like(inputs["q"]),
            state_indices=state_indices,
        )
    )


def test_frozen_prefill_state_pool_eligibility_rejects_compact_fp32_state(
    flash_kda_device,
):
    inputs = _make_inputs(
        seq_lens=[17, 33],
        num_heads=6,
        packed=True,
        initial_state=True,
        state_dtype=torch.float32,
        seed=23_058,
    )

    eligibility_kwargs = _frozen_prefill_eligibility_kwargs(
        inputs,
        output=torch.empty_like(inputs["q"]),
    )
    assert not kda_prefill_api._flash_kda_prefill_is_eligible(**eligibility_kwargs)
    with pytest.raises(ValueError, match="backend='cake' does not support"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=eligibility_kwargs["output"],
            output_final_state=True,
            backend="cake",
        )


@pytest.mark.parametrize(
    "invalid_contract",
    [
        "state_dtype",
        "state_inner_stride",
        "compact_slot_count",
        "index_dtype",
        "index_length",
        "index_contiguity",
        "index_without_state",
    ],
)
def test_frozen_prefill_state_pool_eligibility_rejects_invalid_contracts(
    flash_kda_device,
    invalid_contract,
):
    inputs = _make_inputs(
        seq_lens=[17, 33],
        num_heads=6,
        packed=True,
        initial_state=True,
        seed=23_106,
    )
    state_indices = None
    if invalid_contract == "state_dtype":
        inputs["initial_state"] = inputs["initial_state"].to(torch.float16)
    elif invalid_contract == "state_inner_stride":
        inputs["initial_state"] = inputs["initial_state"].transpose(-1, -2)
    elif invalid_contract == "compact_slot_count":
        inputs["initial_state"] = _make_padded_state_pool(
            slots=3,
            num_heads=6,
            dtype=torch.bfloat16,
            device=flash_kda_device,
        )
    elif invalid_contract == "index_without_state":
        inputs["initial_state"] = None
        state_indices = torch.tensor([3, 1], dtype=torch.int32, device=flash_kda_device)
    else:
        state_pool = _make_padded_state_pool(
            slots=5,
            num_heads=6,
            dtype=torch.bfloat16,
            device=flash_kda_device,
        )
        inputs["initial_state"] = state_pool
        if invalid_contract == "index_dtype":
            state_indices = torch.tensor(
                [3, 1], dtype=torch.int64, device=flash_kda_device
            )
        elif invalid_contract == "index_length":
            state_indices = torch.tensor(
                [3], dtype=torch.int32, device=flash_kda_device
            )
        else:
            state_indices = torch.tensor(
                [3, 0, 1, 0], dtype=torch.int32, device=flash_kda_device
            )[::2]
            assert not state_indices.is_contiguous()

    eligibility_kwargs = _frozen_prefill_eligibility_kwargs(
        inputs,
        output=torch.empty_like(inputs["q"]),
        state_indices=state_indices,
    )
    assert not kda_prefill_api._flash_kda_prefill_is_eligible(**eligibility_kwargs)
    with pytest.raises(ValueError, match="backend='cake' does not support"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            ssm_state_indices=state_indices,
            output=eligibility_kwargs["output"],
            backend="cake",
        )


def test_frozen_prefill_auto_falls_back_only_when_state_contract_is_ineligible(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[17, 33],
        num_heads=6,
        packed=True,
        initial_state=True,
        seed=23_206,
    )
    inputs["initial_state"] = inputs["initial_state"].to(torch.float16)
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_flash_kda_prefill",
        lambda **kwargs: pytest.fail("an ineligible state must not enter Cake prefill"),
    )
    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", lambda **kwargs: sentinel)

    assert recurrent_kda(**_strict_prefill_kwargs(inputs)) is sentinel


@pytest.mark.parametrize("backend", ["auto", "cake"])
def test_frozen_prefill_missing_selected_module_is_fail_closed(
    flash_kda_device,
    monkeypatch,
    backend,
):
    inputs = _make_inputs(
        seq_lens=[33, 33],
        num_heads=6,
        packed=False,
        initial_state=True,
        seed=23_306,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_prefill_eligible",
        lambda **kwargs: False,
    )
    monkeypatch.setattr(
        kda_decode_api,
        "_run_recurrent_kda",
        lambda **kwargs: pytest.fail("an eligible Cake route must not fall back"),
    )

    def missing_module(variant, target):
        raise FileNotFoundError(
            f"selected generated module is not materialized: {variant}/{target}"
        )

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        missing_module,
    )

    with pytest.raises(FileNotFoundError, match="not materialized"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            output_final_state=True,
            backend=backend,
        )


@pytest.mark.parametrize("packed", [False, True], ids=["fixed", "packed"])
@pytest.mark.parametrize("num_heads", [6, 12], ids=["h6", "h12"])
def test_frozen_prefill_compact_and_indexed_state_contracts_match(
    flash_kda_device,
    packed,
    num_heads,
):
    """Selected pool slots must behave exactly like compact in-place state.

    This is deliberately a public-API contract test.  It does not name a
    private launcher class or freeze a route that may legitimately differ by
    physical Blackwell SKU.  The route/source receipt tests separately pin the
    selected physical module.
    """

    seq_lens = [17, 33] if packed else [33, 33]
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=True,
        state_dtype=torch.bfloat16,
        seed=24_000 + num_heads + int(packed),
    )
    state_seed = inputs.pop("initial_state")
    compact_state = state_seed.clone()
    compact_output, compact_final = recurrent_kda(
        **_strict_prefill_kwargs({**inputs, "initial_state": compact_state}),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    state_indices = torch.tensor([3, 1], dtype=torch.int32, device=flash_kda_device)
    indexed_state = (
        0.1
        * torch.randn(
            (5, num_heads, 128, 128),
            dtype=torch.float32,
            device=flash_kda_device,
        )
    ).to(torch.bfloat16)
    indexed_state.index_copy_(0, state_indices.to(torch.int64), state_seed)
    indexed_state_before = indexed_state.clone()
    indexed_output, indexed_final = recurrent_kda(
        **_strict_prefill_kwargs({**inputs, "initial_state": indexed_state}),
        ssm_state_indices=state_indices,
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    assert compact_final is compact_state
    assert indexed_final is indexed_state
    assert compact_final.dtype == indexed_final.dtype == torch.bfloat16
    torch.testing.assert_close(
        indexed_output.float(), compact_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        indexed_state.index_select(0, state_indices.to(torch.int64)).float(),
        compact_final.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    unselected = torch.tensor([0, 2, 4], dtype=torch.int64, device=flash_kda_device)
    assert torch.equal(
        indexed_state.index_select(0, unselected),
        indexed_state_before.index_select(0, unselected),
    )


def test_frozen_prefill_rejects_fp32_state_checkpoints_explicitly(
    flash_kda_device,
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=6,
        packed=False,
        initial_state=True,
        state_dtype=torch.float32,
        seed=24_106,
    )
    checkpoints = torch.empty(
        (1, 6, 128, 128), dtype=torch.float32, device=flash_kda_device
    )
    checkpoint_cu_starts = torch.tensor(
        [0, 1], dtype=torch.int64, device=flash_kda_device
    )

    with pytest.raises(
        ValueError,
        match=r"(?i)(fp32.*checkpoint|checkpoint.*fp32)",
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            output_final_state=True,
            state_checkpoints=checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=32,
            backend="cake",
        )


def test_affine_launch_plan_reads_mutated_state_indices_each_call(
    flash_kda_device,
):
    inputs = _make_inputs(
        seq_lens=[65_536],
        num_heads=4,
        packed=False,
        initial_state=True,
        state_dtype=torch.bfloat16,
        seed=24_204,
    )
    compact_state = inputs.pop("initial_state")
    state_pool = torch.cat(
        (compact_state, compact_state + 0.25, compact_state - 0.25), dim=0
    )
    state_indices = torch.tensor([0], dtype=torch.int32, device=flash_kda_device)
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    first_output = torch.empty_like(inputs["q"])
    first_result, first_state = recurrent_kda(
        **_strict_prefill_kwargs({**inputs, "initial_state": state_pool}),
        ssm_state_indices=state_indices,
        output=first_output,
        output_final_state=True,
        prefill_workspace=workspace,
        backend="cake",
    )
    assert first_result is first_output
    assert first_state is state_pool

    state_before_second = state_pool.clone()
    state_indices.fill_(2)
    second_output = torch.empty_like(inputs["q"])
    second_result, second_state = recurrent_kda(
        **_strict_prefill_kwargs({**inputs, "initial_state": state_pool}),
        ssm_state_indices=state_indices,
        output=second_output,
        output_final_state=True,
        prefill_workspace=workspace,
        backend="cake",
    )
    assert second_result is second_output
    assert second_state is state_pool
    assert torch.equal(state_pool[0], state_before_second[0])
    assert torch.equal(state_pool[1], state_before_second[1])
    assert not torch.equal(state_pool[2], state_before_second[2])
    assert not torch.equal(first_output, second_output)


def test_persistent_policy_uses_physical_arch_and_sm_count_independently():
    for compute_capability, sm_count, expected in (
        ((10, 0), 148, True),
        ((10, 0), 152, True),
        ((10, 3), 148, False),
        ((10, 3), 152, False),
    ):
        assert (
            kda_prefill_api._uses_measured_sm100_persistent_policy(
                compute_capability=compute_capability,
                sm_count=sm_count,
            )
            is expected
        )

    uniform = kda_prefill_api._persistent_task_plan(
        (8192,) * 8,
        num_heads=96,
        sm_count=148,
    )
    assert uniform is not None
    sequence_order, task_ids, task_offsets = uniform
    assert sequence_order == tuple(range(8))
    assert sorted(task_ids) == list(range(8 * 96))
    assert len(task_offsets) == 129
    assert {
        right - left
        for left, right in zip(task_offsets, task_offsets[1:], strict=False)
    } == {6}

    mixed = kda_prefill_api._persistent_task_plan(
        (3063, 2048, 1300, 963, 547, 271),
        num_heads=96,
        sm_count=148,
    )
    assert mixed is not None
    _, mixed_ids, mixed_offsets = mixed
    assert sorted(mixed_ids) == list(range(6 * 96))
    assert len(mixed_offsets) == 149
    large_sm_count_uniform = kda_prefill_api._persistent_task_plan(
        (8192,) * 8,
        num_heads=96,
        sm_count=152,
    )
    assert large_sm_count_uniform is not None
    assert len(large_sm_count_uniform[2]) == 129

    large_sm_count_mixed = kda_prefill_api._persistent_task_plan(
        (3063, 2048, 1300, 963, 547, 271),
        num_heads=96,
        sm_count=152,
    )
    assert large_sm_count_mixed is not None
    assert len(large_sm_count_mixed[2]) == 153
    assert (
        kda_prefill_api._persistent_task_plan(
            (3063, 2048, 1300, 963, 547, 271),
            num_heads=64,
            sm_count=152,
        )
        is None
    )
    assert (
        kda_prefill_api._persistent_task_plan(
            (8192,) * 8,
            num_heads=96,
            sm_count=150,
        )
        is None
    )


@pytest.mark.parametrize(
    ("route", "route_role", "state_mode", "specialization", "abi_family"),
    (
        (
            "direct_m128",
            "main",
            "bf16",
            {
                "chunk": 32,
                "serving_native_abi": False,
                "gate_kind": "lower_bound",
                "checkpoint_tma": False,
                "pair_packed_beta": False,
                "scalar_beta": False,
                "early_n32_state_pack": False,
                "generic_register_inverse": False,
                "n32_prediction_first": False,
                "tensor_state_decay": False,
                "state_dtype_is_fp32": False,
                "n32_ft_slab": False,
                "pdl_wait_initial_state_f32": False,
                "pdl_publish_final_state": False,
                "affine_main_indexed_initial": False,
                "affine_main_indexed_initial_bf16": False,
            },
            "direct_m128",
        ),
        (
            "source599_vtile_m128",
            "main",
            "bf16",
            {
                "full_n32_chunks": True,
                "num_heads": 96,
                "use_initial_state": True,
                "store_final_state": True,
                "scale": 0.08838834764831845,
                "lower_bound": -5.0,
                "persistent_mode": True,
                "persistent_six_task_schedule": True,
                "persistent_stride_head_aligned": False,
                "state_dtype_is_fp32": False,
            },
            "vtile_m128",
        ),
        (
            "bt16_prepare_chain_m64",
            "bt16_prepare",
            "none",
            {},
            "bt16_prepare",
        ),
        (
            "bt16_prepare_chain_m64",
            "main",
            "bf16",
            {
                "bt16_stage_count": 8,
                "state_dtype_is_fp32": False,
                "serving_native_abi": False,
            },
            "bt16_chain",
        ),
        (
            "independent_dvsplit_m64",
            "main",
            "bf16",
            {
                "full_n32_chunks": True,
                "num_heads": 64,
                "use_initial_state": True,
                "store_final_state": True,
                "scale": 0.08838834764831845,
                "lower_bound": -5.0,
                "state_dtype_is_fp32": False,
            },
            "m64",
        ),
        (
            "scalar_chunk_lpt_m128",
            "main",
            "bf16",
            {
                "num_heads": 96,
                "use_initial_state": True,
                "store_final_state": True,
                "scale": 0.08838834764831845,
                "lower_bound": -5.0,
                "persistent_schedule": True,
                "state_dtype_is_fp32": False,
            },
            "scalar_lpt_m128",
        ),
        (
            "piece_persistent_m128",
            "main",
            "fp32",
            {"piece_tasks": True, "state_dtype_is_fp32": True},
            "taskized_persistent_m128",
        ),
        (
            "small_bh_owner_helper_m128",
            "main",
            "fp32",
            {"serving_native_abi": True, "state_dtype_is_fp32": True},
            "small_bh_m128",
        ),
        (
            "affine_split_m128",
            "affine_scan",
            "none",
            {"use_pdl": True},
            "affine_scan",
        ),
    ),
)
def test_generated_prefill_selector_key_uses_receipt_field_order(
    route, route_role, state_mode, specialization, abi_family
):
    selector_key = kda_prefill_api._make_flash_kda_generated_selector_key(
        target="sm103a",
        route=route,
        route_role=route_role,
        state_mode=state_mode,
        family_specialization=specialization,
    )
    assert selector_key["arch"] == "sm_103a"
    assert selector_key["route"] == route
    assert selector_key["route_role"] == route_role
    assert selector_key["abi_family"] == abi_family
    assert selector_key["state_mode"] == state_mode
    assert selector_key["family_specialization_vector"] == [
        [field, specialization[field]]
        for field in kda_prefill_api._FLASH_KDA_GENERATED_SPECIALIZATION_FIELDS[
            abi_family
        ]
    ]


def test_generated_prefill_selector_key_fails_closed():
    direct_specialization = {
        field: False
        for field in kda_prefill_api._FLASH_KDA_GENERATED_SPECIALIZATION_FIELDS[
            "direct_m128"
        ]
    }
    direct_specialization["chunk"] = 32
    with pytest.raises(ValueError, match="no exact architecture"):
        kda_prefill_api._make_flash_kda_generated_selector_key(
            target="sm100f",
            route="direct_m128",
            route_role="main",
            state_mode="bf16",
            family_specialization=direct_specialization,
        )
    with pytest.raises(ValueError, match="no receipt-backed ABI family"):
        kda_prefill_api._make_flash_kda_generated_selector_key(
            target="sm100a",
            route="benchmark_shape_0",
            route_role="main",
            state_mode="bf16",
            family_specialization={},
        )
    with pytest.raises(ValueError, match="specialization fields differ"):
        kda_prefill_api._make_flash_kda_generated_selector_key(
            target="sm100a",
            route="direct_m128",
            route_role="main",
            state_mode="bf16",
            family_specialization={"chunk": 32},
        )
    with pytest.raises(ValueError, match="requires state_mode=none"):
        kda_prefill_api._make_flash_kda_generated_selector_key(
            target="sm100a",
            route="bt16_prepare_chain_m64",
            route_role="bt16_prepare",
            state_mode="bf16",
            family_specialization={},
        )


def test_generated_affine_selector_construction_is_cached(monkeypatch):
    direct_cache = kda_prefill_api._flash_kda_generated_affine_direct_selector_key
    scan_cache = kda_prefill_api._flash_kda_generated_affine_scan_selector_key
    direct_cache.cache_clear()
    scan_cache.cache_clear()
    calls = []

    def fake_selector(**selector_fields):
        calls.append(selector_fields)
        return selector_fields

    monkeypatch.setattr(
        kda_prefill_api, "_make_flash_kda_generated_selector_key", fake_selector
    )
    direct_kwargs = {
        "target": "sm103a",
        "role": "affine_map",
        "num_heads": 4,
        "num_sequences": 4,
        "uniform_sequences": True,
        "max_sequence_length": 16384,
        "pair_packed_beta": False,
        "external_state_is_fp32": False,
    }
    try:
        first = direct_cache(**direct_kwargs)
        assert direct_cache(**direct_kwargs) is first
        assert len(calls) == 1
        assert first["state_mode"] == "bf16"

        direct_cache(**{**direct_kwargs, "max_sequence_length": 32768})
        assert len(calls) == 2

        scan = scan_cache(target="sm103a")
        assert scan_cache(target="sm103a") is scan
        assert len(calls) == 3

        bf16_main = direct_cache(
            **{
                **direct_kwargs,
                "role": "affine_main",
                "external_state_is_fp32": False,
            }
        )
        bf16_main_specialization = bf16_main["family_specialization"]
        assert bf16_main_specialization["affine_main_indexed_initial"]
        assert bf16_main_specialization["affine_main_indexed_initial_bf16"]
        assert bf16_main["state_mode"] == "bf16_f32_dependency"

        fp32_main = direct_cache(
            **{
                **direct_kwargs,
                "role": "affine_main",
                "external_state_is_fp32": True,
            }
        )
        fp32_main_specialization = fp32_main["family_specialization"]
        assert fp32_main_specialization["affine_main_indexed_initial"]
        assert not fp32_main_specialization["affine_main_indexed_initial_bf16"]
        assert fp32_main["state_mode"] == "fp32"
    finally:
        direct_cache.cache_clear()
        scan_cache.cache_clear()


def test_generated_affine_carriers_are_workspace_cached(monkeypatch):
    calls = []

    def fake_dummy(name):
        def make(_device):
            value = object()
            calls.append((name, None, value))
            return value

        return make

    def fake_empty(_device, dtype):
        value = object()
        calls.append(("empty", dtype, value))
        return value

    for name in ("bf16", "i32", "i64", "f32", "u32"):
        monkeypatch.setattr(
            kda_prefill_api,
            f"_dummy_{name}",
            fake_dummy(name),
        )
    monkeypatch.setattr(kda_prefill_api, "_empty_cuda_tensor", fake_empty)

    first_workspace = SimpleNamespace(_generated_affine_carriers=None)
    first = kda_prefill_api._generated_affine_carriers(
        workspace=first_workspace,
        device=torch.device("cuda:0"),
    )
    assert (
        kda_prefill_api._generated_affine_carriers(
            workspace=first_workspace,
            device=torch.device("cuda:0"),
        )
        is first
    )
    assert len(calls) == 9
    assert [name for name, _dtype, _value in calls] == [
        "bf16",
        "i32",
        "i64",
        "f32",
        "u32",
        "empty",
        "empty",
        "empty",
        "empty",
    ]
    assert [dtype for name, dtype, _value in calls if name == "empty"] == [
        torch.bfloat16,
        torch.float32,
        torch.int64,
        torch.uint8,
    ]

    second_workspace = SimpleNamespace(_generated_affine_carriers=None)
    second = kda_prefill_api._generated_affine_carriers(
        workspace=second_workspace,
        device=torch.device("cuda:0"),
    )
    assert second is not first
    assert len(calls) == 18


def test_generated_affine_module_bundle_resolves_cold_and_observes_hot(monkeypatch):
    roles = ("affine_main", "affine_map", "affine_scan", "affine_correction")
    selector_keys = {role: {"role": role, "shape": 32} for role in roles}
    resolver_calls = []

    def resolve(selector_key):
        resolver_calls.append(selector_key)
        role = selector_key["role"]
        return SimpleNamespace(variant_id=f"{role}_test"), _RecorderModule()

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_generated_module", resolve)
    workspace = SimpleNamespace(_generated_affine_module_bundle=None)

    def get_bundle(keys, *, capturing=False):
        return kda_prefill_api._generated_affine_module_bundle(
            workspace=workspace,
            main_selector_key=keys["affine_main"],
            map_selector_key=keys["affine_map"],
            scan_selector_key=keys["affine_scan"],
            correction_selector_key=keys["affine_correction"],
            capturing=capturing,
        )

    cold = get_bundle(selector_keys)
    assert resolver_calls == [selector_keys[role] for role in roles]
    assert get_bundle(selector_keys) is cold
    assert get_bundle(selector_keys, capturing=True) is cold
    assert len(resolver_calls) == 4

    observed = []

    def observer(role, selector_key, metadata, module):
        observed.append((role, selector_key, metadata, module))

    entries = (cold.main, cold.map, cold.scan, cold.correction)
    with kda_prefill_api._observe_generated_affine_launches(observer):
        launch_observer = kda_prefill_api._generated_affine_launch_observer.get()
        for entry in entries:
            kda_prefill_api._generated_affine_module_for_launch(
                entry, launch_observer
            ).run()
        hot = get_bundle(selector_keys)
        for entry in (hot.main, hot.map, hot.scan, hot.correction):
            kda_prefill_api._generated_affine_module_for_launch(
                entry, launch_observer
            ).run()

    assert [row[0] for row in observed] == list(roles) * 2
    for row, entry in zip(observed[:4], entries, strict=True):
        assert row[1] is entry.selector_key
        assert row[2] is entry.metadata
        assert row[3] is entry.module
        assert entry.module.calls == [(), ()]
    assert len(resolver_calls) == 4

    changed_keys = dict(selector_keys)
    changed_keys["affine_main"] = {"role": "affine_main", "shape": 64}
    changed = get_bundle(changed_keys)
    assert changed is not cold
    assert workspace._generated_affine_module_bundle is changed
    assert len(resolver_calls) == 8
    assert [row["role"] for row in resolver_calls[4:]] == list(roles)

    cold_workspace = SimpleNamespace(_generated_affine_module_bundle=None)
    workspace = cold_workspace
    with pytest.raises(RuntimeError, match="not warmed for CUDA graph capture"):
        get_bundle(selector_keys, capturing=True)
    assert cold_workspace._generated_affine_module_bundle is None
    assert len(resolver_calls) == 8


def test_generated_affine_launch_plan_caches_only_workspace_views(monkeypatch):
    buffer_calls = []
    buffers = {}

    def workspace_buffer(*, name, shape, dtype, **_kwargs):
        buffer_calls.append((name, shape, dtype))
        tensor = torch.zeros(shape, dtype=dtype)
        buffers[name] = tensor
        return tensor

    modules = object()
    monkeypatch.setattr(kda_prefill_api, "_affine_workspace_buffer", workspace_buffer)
    monkeypatch.setattr(
        kda_prefill_api,
        "_cached_tensor",
        lambda key, _factory, **_kwargs: ("metadata", key),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_stream_cache_key",
        lambda _device: (0, 17),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_identity_seq_order",
        lambda **kwargs: ("seq_order", kwargs["num_sequences"]),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_generated_affine_module_bundle",
        lambda **_kwargs: modules,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_affine_descriptor_storage",
        lambda **kwargs: ("descriptor", kwargs["role"]),
    )
    workspace = SimpleNamespace(
        _generated_affine_launch_plan=None,
        _affine_map_identity_data_ptr=None,
    )

    def get_plan(token_offsets, *, capturing=False):
        return kda_prefill_api._generated_affine_launch_plan(
            workspace=workspace,
            target="sm100a",
            device=torch.device("cuda:0"),
            token_offsets=token_offsets,
            num_heads=4,
            state_dtype=torch.bfloat16,
            beta_layouts=("padded", "padded", "padded"),
            capturing=capturing,
        )

    cold = get_plan((0, 4096, 8192))
    assert len(buffer_calls) == 14
    assert get_plan((0, 4096, 8192)) is cold
    assert get_plan((0, 4096, 8192), capturing=True) is cold
    assert len(buffer_calls) == 14
    assert cold.modules is modules
    assert all(not isinstance(value, torch.Tensor) for value in vars(cold.key).values())
    assert set(buffers) == {
        "main_final_fp32",
        "map_identity_bfloat16",
        "map_state_bfloat16",
        "carry_float32",
        "correction_final_float32",
        "final_compact_float32",
        "zero_v",
        "map_out",
        "correction_out",
        "state_indices_i64",
        "final_external",
        "beta_tma_main",
        "beta_tma_map",
        "beta_tma_correction",
    }

    changed = get_plan((0, 4096, 8192, 12288))
    assert changed is not cold
    assert workspace._generated_affine_launch_plan is changed
    assert len(buffer_calls) == 28

    workspace._generated_affine_launch_plan = None
    with pytest.raises(RuntimeError, match="not warmed for CUDA graph capture"):
        get_plan((0, 4096, 8192), capturing=True)
    assert workspace._generated_affine_launch_plan is None
    assert len(buffer_calls) == 28


def test_affine_route_skips_general_metadata_and_dummy_materialization(monkeypatch):
    q = torch.empty((1, 256, 4, 128), dtype=torch.bfloat16)
    beta = torch.empty((1, 256, 4), dtype=torch.bfloat16)
    state = torch.empty((3, 4, 128, 128), dtype=torch.bfloat16)
    state_indices = torch.tensor([2], dtype=torch.int32)
    output = torch.empty_like(q)
    workspace = SimpleNamespace(_lock=threading.Lock())
    launches = []

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda _device: SimpleNamespace(cuda_stream=17),
    )
    monkeypatch.setattr(
        kda_prefill_api, "_select_flash_kda_prefill_target", lambda _device: "sm100a"
    )
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda _device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_flash_kda_device_sm_count", lambda _device: 148
    )
    monkeypatch.setattr(
        kda_prefill_api, "_get_stream_workspace", lambda _device: workspace
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_affine_token_offsets",
        lambda **_kwargs: (0, 128, 256),
    )
    monkeypatch.setattr(
        kda_prefill_api, "_bind_workspace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_run_generated_affine_route",
        lambda **kwargs: launches.append(kwargs),
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("affine fast path materialized general route metadata")

    for name in (
        "select_bf16_schedule_route",
        "_fixed_cu_seqlens",
        "_validate_prefill_seq_order",
        "_dummy_bf16",
        "_dummy_i32",
        "_dummy_i64",
    ):
        monkeypatch.setattr(kda_prefill_api, name, unexpected)

    result = kda_prefill_api._run_flash_kda_prefill(
        q=q,
        k=q,
        v=q,
        g=q,
        beta=beta,
        A_log=torch.empty(4, dtype=torch.float32),
        dt_bias=torch.empty((4, 128), dtype=torch.float32),
        scale=0.125,
        initial_state=state,
        output_final_state=True,
        lower_bound=-5.0,
        cu_seqlens=None,
        output=output,
        seq_order=None,
        prefill_workspace=None,
        state_indices=state_indices,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )

    assert result[0] is output
    assert result[1] is state
    assert len(launches) == 1
    assert launches[0]["state_indices"] is state_indices
    assert launches[0]["token_offsets"] == (0, 128, 256)


def test_generated_affine_launch_observer_scope_is_context_local_and_resets():
    def outer(*_args):
        return None

    def inner(*_args):
        return None

    observer_context = kda_prefill_api._generated_affine_launch_observer
    assert observer_context.get() is None

    child_values = []
    with kda_prefill_api._observe_generated_affine_launches(outer):
        assert observer_context.get() is outer
        thread = threading.Thread(
            target=lambda: child_values.append(observer_context.get())
        )
        thread.start()
        thread.join()
        with kda_prefill_api._observe_generated_affine_launches(inner):
            assert observer_context.get() is inner
        assert observer_context.get() is outer
    assert child_values == [None]
    assert observer_context.get() is None

    with (
        pytest.raises(RuntimeError, match="observer scope failure"),
        kda_prefill_api._observe_generated_affine_launches(outer),
    ):
        raise RuntimeError("observer scope failure")
    assert observer_context.get() is None

    module = _RecorderModule()
    resolved = kda_prefill_api._GeneratedAffineModule(
        "affine_main", {"role": "affine_main"}, object(), module
    )

    def failing_observer(*_args):
        raise RuntimeError("observer callback failure")

    with (
        pytest.raises(RuntimeError, match="observer callback failure"),
        kda_prefill_api._observe_generated_affine_launches(failing_observer),
    ):
        kda_prefill_api._generated_affine_module_for_launch(
            resolved, observer_context.get()
        ).run()
    assert module.calls == []
    assert observer_context.get() is None


def test_generated_affine_direct_role_uses_supplied_carriers(monkeypatch):
    def unexpected_lookup(*_args, **_kwargs):
        pytest.fail("affine role repeated a global dummy/empty carrier lookup")

    for helper in (
        "_dummy_bf16",
        "_dummy_i32",
        "_dummy_i64",
        "_dummy_f32",
        "_dummy_u32",
        "_empty_cuda_tensor",
    ):
        monkeypatch.setattr(kda_prefill_api, helper, unexpected_lookup)

    module = _RecorderModule()
    metadata = SimpleNamespace(variant_id="affine_map_test")
    selector_key = {"selector": "affine_map_test"}
    resolved_module = kda_prefill_api._GeneratedAffineModule(
        "affine_map", selector_key, metadata, module
    )
    descriptor_storage = object()
    monkeypatch.setattr(
        kda_prefill_api,
        "_affine_descriptor_storage",
        lambda **_kwargs: descriptor_storage,
    )

    carrier_values = {
        name: object()
        for name in (
            "dummy_bf16",
            "dummy_i32",
            "dummy_i64",
            "dummy_f32",
            "dummy_u32",
            "empty_bf16",
            "empty_f32",
            "empty_i64",
            "empty_u8",
        )
    }
    carriers = kda_prefill_api._GeneratedAffineCarriers(**carrier_values)
    workspace = SimpleNamespace(_descriptor_signatures={})
    q = torch.empty((1, 32, 4, 128), dtype=torch.bfloat16)
    beta = torch.empty((1, 32, 4), dtype=torch.bfloat16)
    state = torch.empty((1, 4, 128, 128), dtype=torch.bfloat16)
    dependency = torch.empty((1, 4, 128, 128), dtype=torch.float32)
    observed = []

    def run(role_q, *, capturing, role="affine_map", role_state=state):
        kda_prefill_api._run_generated_affine_direct_role(
            workspace=workspace,
            carriers=carriers,
            resolved_module=resolved_module,
            descriptor_storage=descriptor_storage,
            launch_observer=lambda *args: observed.append(args),
            role=role,
            q=role_q,
            k=role_q,
            v=role_q,
            g=role_q,
            beta=beta,
            beta_tma=beta,
            A_log=torch.empty(4, dtype=torch.float32),
            dt_bias=torch.empty((4, 128), dtype=torch.float32),
            cu_seqlens=torch.tensor([0, 32], dtype=torch.int64),
            seq_order=torch.tensor([0], dtype=torch.int32),
            state_indices=carriers.dummy_i32,
            initial_state=role_state,
            out=role_q,
            final_state=role_state,
            initial_state_f32_dependency=dependency,
            sequence_lengths=(32,),
            num_heads=4,
            use_state_indices=False,
            state_slot_stride=4 * 128 * 128,
            scale=0.125,
            lower_bound=-5.0,
            grid_x=4,
            stream_ptr=17,
            capturing=capturing,
        )

    run(q, capturing=False)

    assert observed == [("affine_map", selector_key, metadata, module)]
    (args,) = module.calls
    assert len(args) == 49
    assert args[14] is carriers.empty_bf16
    assert args[15] is carriers.empty_i64
    assert args[16] is carriers.dummy_i64
    assert args[17] is carriers.dummy_bf16
    assert args[18] is carriers.dummy_u32
    assert args[30] is carriers.dummy_u32
    assert args[31] is carriers.empty_u8
    assert args[32] is descriptor_storage

    run(q, capturing=False, role="affine_main")
    assert module.calls[1][14] is carriers.empty_bf16
    fp32_state = torch.empty((1, 4, 128, 128), dtype=torch.float32)
    run(q, capturing=False, role="affine_main", role_state=fp32_state)
    assert module.calls[2][14] is carriers.empty_f32

    with pytest.raises(RuntimeError, match="descriptors are not warmed"):
        run(torch.empty_like(q), capturing=True)
    assert observed == [("affine_map", selector_key, metadata, module)] * 3
    assert len(module.calls) == 3


def test_generated_prefill_runtime_specialization_helpers():
    assert not kda_prefill_api._flash_kda_generated_serving_native_abi(
        use_state_indices=False,
        checkpoint_every_n_tokens=0,
        beta_token_stride=64,
        num_heads=64,
        state_slot_stride=64 * 128 * 128,
    )
    assert kda_prefill_api._flash_kda_generated_serving_native_abi(
        use_state_indices=True,
        checkpoint_every_n_tokens=0,
        beta_token_stride=64,
        num_heads=64,
        state_slot_stride=64 * 128 * 128,
    )
    assert (
        kda_prefill_api._flash_kda_generated_bt16_stage_count(
            total_tasks=96, sm_count=148, use_beta_tma=False
        )
        == 7
    )
    assert (
        kda_prefill_api._flash_kda_generated_bt16_stage_count(
            total_tasks=8, sm_count=148, use_beta_tma=False
        )
        == 9
    )
    assert (
        kda_prefill_api._flash_kda_generated_bt16_stage_count(
            total_tasks=64, sm_count=148, use_beta_tma=False
        )
        == 8
    )
    assert kda_prefill_api._flash_kda_generated_full_n32_chunks((32, 64))
    assert not kda_prefill_api._flash_kda_generated_full_n32_chunks((32, 63))

    direct = kda_prefill_api._flash_kda_generated_direct_specialization(
        target="sm103a",
        route="direct_m128",
        num_heads=96,
        num_sequences=1,
        uniform_sequences=True,
        max_sequence_length=512,
        serving_native_abi=False,
        gate_kind="lower_bound",
        checkpoint_every_n_tokens=0,
        pair_packed_beta=False,
        state_dtype_is_fp32=False,
    )
    assert direct["chunk"] == 32
    assert direct["generic_register_inverse"]
    assert direct["n32_prediction_first"]
    assert direct["tensor_state_decay"]

    with pytest.raises(
        ValueError, match="affine indexed initial state requires FP32 state I/O"
    ):
        kda_prefill_api._flash_kda_generated_direct_specialization(
            target="sm103a",
            route="affine_split_m128",
            num_heads=96,
            num_sequences=1,
            uniform_sequences=True,
            max_sequence_length=512,
            serving_native_abi=False,
            gate_kind="lower_bound",
            checkpoint_every_n_tokens=0,
            pair_packed_beta=False,
            state_dtype_is_fp32=False,
            affine_main_indexed_initial=True,
        )

    vtile = kda_prefill_api._flash_kda_generated_vtile_specialization(
        sequence_lengths=(512,) * 8,
        num_heads=96,
        fixed_layout=False,
        use_initial_state=True,
        store_final_state=True,
        scale=0.08838834764831845,
        lower_bound=-5.0,
        state_dtype_is_fp32=False,
    )
    assert vtile["full_n32_chunks"]
    assert vtile["persistent_mode"]
    assert vtile["persistent_six_task_schedule"]
    assert not vtile["persistent_stride_head_aligned"]

    assert kda_prefill_api._flash_kda_generated_bt16_prepare_specialization() == {}
    assert kda_prefill_api._flash_kda_generated_bt16_chain_specialization(
        total_tasks=64,
        sm_count=148,
        use_beta_tma=False,
        state_dtype_is_fp32=False,
        serving_native_abi=False,
    ) == {
        "bt16_stage_count": 8,
        "state_dtype_is_fp32": False,
        "serving_native_abi": False,
    }
    assert kda_prefill_api._flash_kda_generated_m64_specialization(
        sequence_lengths=(512,),
        num_heads=64,
        use_initial_state=True,
        store_final_state=True,
        scale=0.08838834764831845,
        lower_bound=-5.0,
        state_dtype_is_fp32=False,
    )["full_n32_chunks"]
    assert kda_prefill_api._flash_kda_generated_scalar_lpt_specialization(
        num_heads=96,
        use_initial_state=True,
        store_final_state=True,
        scale=0.08838834764831845,
        lower_bound=-5.0,
        state_dtype_is_fp32=True,
    )["persistent_schedule"]
    assert kda_prefill_api._flash_kda_generated_taskized_persistent_specialization(
        piece_tasks=True, state_dtype_is_fp32=False
    ) == {"piece_tasks": True, "state_dtype_is_fp32": False}
    assert kda_prefill_api._flash_kda_generated_small_bh_specialization(
        serving_native_abi=True, state_dtype_is_fp32=True
    ) == {"serving_native_abi": True, "state_dtype_is_fp32": True}
    assert kda_prefill_api._flash_kda_generated_affine_scan_specialization() == {
        "use_pdl": True
    }


@pytest.mark.parametrize(
    (
        "compute_capability",
        "route",
        "uniform_sequences",
        "num_heads",
        "total_tasks",
        "max_sequence_length",
        "expected",
    ),
    [
        ((10, 3), "direct_m128", True, 96, 96, 8192, True),
        ((10, 3), "direct_m128", True, 64, 512, 1024, True),
        ((10, 0), "direct_m128", True, 96, 96, 8192, False),
        ((10, 3), "direct_m128", False, 96, 96, 8192, False),
        ((10, 3), "direct_m128", True, 96, 96, 8191, False),
        ((10, 3), "direct_m128", True, 64, 64, 1024, False),
        ((10, 3), "independent_dvsplit_m64", True, 96, 96, 8192, False),
    ],
)
def test_n32_tensor_state_decay_policy_matches_measured_region(
    compute_capability,
    route,
    uniform_sequences,
    num_heads,
    total_tasks,
    max_sequence_length,
    expected,
):
    assert (
        kda_prefill_api._should_use_n32_tensor_state_decay(
            compute_capability=compute_capability,
            route=route,
            uniform_sequences=uniform_sequences,
            num_heads=num_heads,
            total_tasks=total_tasks,
            max_sequence_length=max_sequence_length,
        )
        is expected
    )


def test_variant_selector_exposes_specialized_routes_only_when_requested():
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=2,
            num_heads=4,
            unbounded_softplus=True,
        )
        == "m128_unbounded_softplus"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=2,
            num_heads=4,
            unbounded_softplus=True,
            use_bt64_unbounded_softplus=True,
        )
        == "m128_bt64_unbounded_softplus"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=True,
            num_sequences=1,
            num_heads=8,
            use_persistent_m128=True,
            use_small_bh_m128=True,
        )
        == "small_bh_m128"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=8,
            num_heads=96,
            use_persistent_m128=True,
        )
        == "persistent_m128"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=128,
            num_heads=96,
            use_persistent_m128=True,
            use_exact_n16=True,
        )
        == "m128_n16"
    )
    assert (
        kda_prefill_api._select_flash_kda_prefill_variant(
            fixed_layout=False,
            num_sequences=8,
            num_heads=12,
            use_persistent_m128=True,
        )
        == "m128_n16"
    )


@pytest.mark.parametrize(
    (
        "compute_capability",
        "sm_count",
        "num_sequences",
        "num_heads",
        "sequence_length",
        "expected",
    ),
    [
        ((10, 0), 148, 1, 8, 2048, True),
        ((10, 3), 152, 2, 4, 65536, True),
        ((10, 3), 64, 8, 1, 131072, True),
        ((10, 0), 63, 8, 1, 2048, False),
        ((10, 0), 148, 1, 8, 2047, False),
        ((10, 0), 148, 3, 3, 2048, False),
        ((10, 0), 148, 1, 9, 2048, False),
    ],
)
def test_small_bh_owner_helper_policy_matches_residency_contract(
    compute_capability,
    sm_count,
    num_sequences,
    num_heads,
    sequence_length,
    expected,
):
    assert (
        kda_prefill_api._should_use_small_bh_owner_helper(
            compute_capability=compute_capability,
            sm_count=sm_count,
            num_sequences=num_sequences,
            num_heads=num_heads,
            sequence_length=sequence_length,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "fixed_layout",
        "num_sequences",
        "num_heads",
        "uniform_sequences",
        "max_sequence_length",
        "expected_route",
    ),
    [
        (True, 1, 64, True, 4096, "bt16_prepare_chain_m64"),
        (True, 1, 12, True, 512, "bt16_prepare_chain_m64"),
        (True, 8, 12, True, 1024, "direct_m128"),
        (False, 8, 12, False, 3072, "bt16_prepare_chain_m64"),
        (True, 1, 4, True, 65_536, "bt16_prepare_chain_m64"),
        (True, 1, 1, True, 512, "direct_m128"),
        (True, 1, 1, True, 65_535, "small_bh_owner_helper_m128"),
        (True, 1, 1, True, 65_536, "bt16_prepare_chain_m64"),
        (True, 1, 64, True, 512, "independent_dvsplit_m64"),
    ],
)
def test_bt16_route_policy_matches_measured_crossovers(
    fixed_layout,
    num_sequences,
    num_heads,
    uniform_sequences,
    max_sequence_length,
    expected_route,
):
    assert (
        kda_prefill_api._select_flash_kda_bf16_route(
            compute_capability=(10, 3),
            sm_count=152,
            fixed_layout=fixed_layout,
            num_sequences=num_sequences,
            num_heads=num_heads,
            uniform_sequences=uniform_sequences,
            max_sequence_length=max_sequence_length,
        )
        == expected_route
    )


def test_uniform_piece_bins_cover_tasks_and_match_h96_h64_bounds():
    for num_heads, expected_handoffs, expected_max in (
        (96, 32, 167),
        (64, 56, 112),
    ):
        (
            tasks,
            offsets,
            token_starts,
            token_counts,
            sources,
            destinations,
            handoff_count,
            loads,
        ) = kda_prefill_api._make_uniform_piece_task_bins(
            num_sequences=8,
            num_heads=num_heads,
            sequence_length=1024,
            worker_count=152,
        )
        assert len(offsets) == 153
        assert len(tasks) == len(token_starts) == len(token_counts)
        assert len(tasks) == len(sources) == len(destinations)
        assert handoff_count == expected_handoffs
        assert max(loads) == expected_max
        assert sum(loads) == 8 * num_heads * 32

        produced = sorted(value for value in destinations if value >= 0)
        consumed = sorted(value for value in sources if value >= 0)
        assert produced == consumed == list(range(handoff_count))
        coverage = {}
        for task, start, count in zip(tasks, token_starts, token_counts, strict=True):
            coverage.setdefault(task, []).append((start, start + count))
        assert set(coverage) == set(range(8 * num_heads))
        for intervals in coverage.values():
            ordered = sorted(intervals)
            assert ordered[0][0] == 0
            assert ordered[-1][1] == 1024
            assert all(
                left[1] == right[0]
                for left, right in zip(ordered, ordered[1:], strict=False)
            )


def test_uniform_piece_policy_uses_occupancy_and_dependency_dag():
    common = {
        "compute_capability": (10, 3),
        "sm_count": 152,
        "num_sequences": 8,
        "uniform_sequences": True,
        "max_sequence_length": 1024,
    }
    for num_heads in (60, 64, 96, 104):
        estimate = kda_prefill_api._persistent_m128_roofline(
            compute_capability=common["compute_capability"],
            sm_count=common["sm_count"],
            num_sequences=common["num_sequences"],
            num_heads=num_heads,
            sequence_length=common["max_sequence_length"],
            use_initial_state=True,
            store_final_state=True,
        )
        assert estimate is not None
        assert estimate.resident_ctas_per_sm == 1
        assert estimate.worker_count == common["sm_count"]
        assert estimate.handoff_count > 0
        assert estimate.piece_ns < estimate.direct_ns
        assert kda_prefill_api._should_use_uniform_piece_persistent(
            num_heads=num_heads,
            **common,
        )
        assert (
            kda_prefill_api._select_flash_kda_bf16_route(
                fixed_layout=False,
                num_heads=num_heads,
                **common,
            )
            == "piece_persistent_m128"
        )

    assert kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96,
        **(common | {"compute_capability": (10, 0), "sm_count": 148}),
    )
    assert not kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96, **(common | {"num_sequences": 4})
    )
    assert not kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96, **(common | {"sm_count": 149})
    )
    assert kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96, **(common | {"max_sequence_length": 992})
    )
    assert not kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96, **(common | {"max_sequence_length": 32})
    )
    assert not kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=48, **common
    )
    assert not kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96, **(common | {"uniform_sequences": False})
    )
    assert kda_prefill_api._should_use_uniform_piece_persistent(
        num_heads=96, **(common | {"num_sequences": 16})
    )
    for num_sequences in (32, 64):
        assert not kda_prefill_api._should_use_uniform_piece_persistent(
            num_heads=96, **(common | {"num_sequences": num_sequences})
        )


def test_bt16_prepare_walk_and_physical_variants_match_production_policy():
    assert (
        kda_prefill_api._direct_m128_route(num_heads=64, max_sequence_length=16)
        == "direct_m128_n16"
    )
    assert (
        kda_prefill_api._direct_m128_route(num_heads=64, max_sequence_length=17)
        == "direct_m128"
    )
    assert (
        kda_prefill_api._direct_m128_route(num_heads=96, max_sequence_length=16)
        == "direct_m128_n16"
    )
    assert (
        kda_prefill_api._direct_m128_route(num_heads=12, max_sequence_length=16)
        == "direct_m128_n16"
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=12, total_chunks=128)
        == 1
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=12, total_chunks=129)
        == 4
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=64, total_chunks=255)
        == 6
    )
    assert (
        kda_prefill_api._bt16_chunks_per_prepare_cta(num_heads=64, total_chunks=256)
        == 8
    )

    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 3),
        sm_count=152,
        fixed_layout=True,
        num_sequences=1,
        num_heads=64,
        max_sequence_length=4096,
    ) == ("bt16_prepare_beta_tma", "bt16_chain_m64_s9", True)
    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 3),
        sm_count=152,
        fixed_layout=True,
        num_sequences=1,
        num_heads=4,
        max_sequence_length=65_536,
    ) == ("bt16_prepare", "bt16_chain_m64_s9", False)
    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 3),
        sm_count=152,
        fixed_layout=True,
        num_sequences=1,
        num_heads=8,
        max_sequence_length=65_536,
    ) == ("bt16_prepare", "bt16_chain_m64_s9", False)
    assert kda_prefill_api._select_bt16_physical_variants(
        compute_capability=(10, 0),
        sm_count=148,
        fixed_layout=False,
        num_sequences=8,
        num_heads=12,
        max_sequence_length=3072,
    ) == ("bt16_prepare", "bt16_chain_m64_s7", False)


def test_bt16_two_stage_adapter_reuses_descriptors_across_state_rotations(monkeypatch):
    prepare_module = _RecorderModule()
    chain_module = _RecorderModule()
    modules = {
        "bt16_prepare_beta_tma": prepare_module,
        "bt16_chain_m64_s9": chain_module,
    }
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: modules[variant],
    )
    q = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16)
    factor = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16)
    kd = factor.clone()
    w = factor.clone()
    qk = torch.empty((1, 1, 1, 1, 1), dtype=torch.bfloat16)
    diag = torch.empty((1, 1, 1, 1), dtype=torch.float32)
    cu_chunks = torch.tensor([0, 256], dtype=torch.int32)
    chunk_to_seq = torch.zeros(256, dtype=torch.int32)
    monkeypatch.setattr(
        kda_prefill_api,
        "_bt16_workspace",
        lambda **kwargs: (
            cu_chunks,
            chunk_to_seq,
            factor,
            kd,
            w,
            qk,
            diag,
            256,
            760,
        ),
    )
    workspace = SimpleNamespace(
        _descriptor_signatures={},
        _descriptor_storages={
            variant: torch.empty(896, dtype=torch.uint8) for variant in modules
        },
    )
    cu_seqlens = torch.tensor([0, 4096], dtype=torch.int64)
    seq_order = torch.tensor([0], dtype=torch.int32)
    state = torch.empty((1, 1, 1, 1), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    beta = torch.empty((1, 1, 1), dtype=torch.bfloat16)
    a_log = torch.empty(64, dtype=torch.float32)
    dt_bias = torch.empty((64, 128), dtype=torch.float32)

    kda_prefill_api._run_bt16_prepare_chain(
        workspace=workspace,
        target="sm100f",
        q=q,
        k=q,
        v=q,
        g=q,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seq_order=seq_order,
        initial_state=state,
        out=output,
        final_state=state,
        offsets=(0, 4096),
        num_heads=64,
        sm_count=152,
        compute_capability=(10, 3),
        fixed_layout=True,
        max_sequence_length=4096,
        use_initial_state=True,
        store_final_state=True,
        scale=0.125,
        lower_bound=-5.0,
        stream_ptr=17,
        capturing=False,
    )

    (prepare_args,) = prepare_module.calls
    assert len(prepare_args) == 21
    assert prepare_args[6] is cu_seqlens
    assert prepare_args[7] is cu_chunks
    assert prepare_args[8] is chunk_to_seq
    assert prepare_args[15] == 1
    assert prepare_args[16:21] == (256, 64, -5.0, 760, 17)
    (chain_args,) = chain_module.calls
    assert len(chain_args) == 20
    assert chain_args[6] is cu_seqlens
    assert chain_args[7] is cu_chunks
    assert chain_args[8] is seq_order
    assert chain_args[12].dtype == torch.uint8
    assert chain_args[13:20] == (1, 64, 1, 1, 0.125, 128, 17)

    rotated_state = torch.empty_like(state)
    kda_prefill_api._run_bt16_prepare_chain(
        workspace=workspace,
        target="sm100f",
        q=q,
        k=q,
        v=q,
        g=q,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seq_order=seq_order,
        initial_state=rotated_state,
        out=output,
        final_state=rotated_state,
        offsets=(0, 4096),
        num_heads=64,
        sm_count=152,
        compute_capability=(10, 3),
        fixed_layout=True,
        max_sequence_length=4096,
        use_initial_state=True,
        store_final_state=True,
        scale=0.125,
        lower_bound=-5.0,
        stream_ptr=17,
        capturing=False,
    )

    second_prepare_args = prepare_module.calls[1]
    second_chain_args = chain_module.calls[1]
    assert second_prepare_args[15] == 0
    assert second_chain_args[9] is rotated_state
    assert second_chain_args[11] is rotated_state
    assert second_chain_args[13] == 0


def test_bt16_combined_adapter_reuses_both_descriptor_sets(monkeypatch):
    combined_module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: combined_module
        if variant == "bt16_prepare_chain_m64_s8"
        else (_ for _ in ()).throw(AssertionError(variant)),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_select_bt16_physical_variants",
        lambda **kwargs: ("bt16_prepare", "bt16_chain_m64_s8", False),
    )
    q = torch.empty((1, 1, 12, 128), dtype=torch.bfloat16)
    factor = torch.empty((1, 12, 16, 128), dtype=torch.bfloat16)
    kd = factor.clone()
    w = factor.clone()
    qk = torch.empty((1, 12, 1, 16, 16), dtype=torch.bfloat16)
    diag = torch.empty((1, 12, 1, 128), dtype=torch.float32)
    cu_chunks = torch.tensor([0, 1], dtype=torch.int32)
    chunk_to_seq = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(
        kda_prefill_api,
        "_bt16_workspace",
        lambda **kwargs: (
            cu_chunks,
            chunk_to_seq,
            factor,
            kd,
            w,
            qk,
            diag,
            1,
            12,
        ),
    )
    workspace = SimpleNamespace(
        _descriptor_signatures={},
        _descriptor_storages={
            variant: torch.empty(896, dtype=torch.uint8)
            for variant in ("bt16_prepare", "bt16_chain_m64_s8")
        },
    )
    cu_seqlens = torch.tensor([0, 512], dtype=torch.int64)
    seq_order = torch.tensor([0], dtype=torch.int32)
    state = torch.empty((1, 12, 128, 128), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    beta = torch.empty((1, 1, 12), dtype=torch.bfloat16)
    a_log = torch.empty(12, dtype=torch.float32)
    dt_bias = torch.empty((12, 128), dtype=torch.float32)

    def run(initial_state):
        kda_prefill_api._run_bt16_prepare_chain(
            workspace=workspace,
            target="sm100f",
            q=q,
            k=q,
            v=q,
            g=q,
            beta=beta,
            A_log=a_log,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            seq_order=seq_order,
            initial_state=initial_state,
            out=output,
            final_state=initial_state,
            offsets=(0, 512),
            num_heads=12,
            sm_count=152,
            compute_capability=(10, 3),
            fixed_layout=True,
            max_sequence_length=512,
            use_initial_state=True,
            store_final_state=True,
            scale=0.125,
            lower_bound=-5.0,
            stream_ptr=17,
            capturing=False,
        )

    run(state)
    rotated_state = torch.empty_like(state)
    run(rotated_state)

    first_args, second_args = combined_module.calls
    assert len(first_args) == 32
    assert first_args[6] is cu_seqlens
    assert first_args[7] is cu_chunks
    assert first_args[8] is chunk_to_seq
    assert first_args[15] is seq_order
    assert first_args[21:32] == (1, 1, 1, 12, -5.0, 12, 1, 1, 0.125, 24, 17)
    assert second_args[16] is rotated_state
    assert second_args[18] is rotated_state
    assert second_args[21:23] == (0, 0)


def test_h96_uniform_n128_keeps_n16_on_148_sm():
    for compute_capability in ((10, 0), (10, 3)):
        for sm_count in (148, 152):
            assert kda_prefill_api._requires_exact_n16_recurrence(
                compute_capability=compute_capability,
                sm_count=sm_count,
                fixed_layout=False,
                num_sequences=128,
                num_heads=96,
                uniform_sequences=True,
            ) is (sm_count == 148)


class _RecorderModule:
    def __init__(self, *, final_value=None):
        self.calls = []
        self.final_value = final_value

    def run(self, *args):
        self.calls.append(args)
        if self.final_value is not None:
            if len(args) == 21:
                store_final_state = args[17]
                final_state = args[12]
            elif len(args) == 23:
                store_final_state = args[19]
                final_state = args[14]
            elif len(args) == 25:
                store_final_state = args[21]
                final_state = args[12]
            else:
                store_final_state = args[23]
                final_state = args[13]
            if bool(store_final_state):
                final_state.fill_(self.final_value)


def test_decode_and_spec_stay_on_existing_backend(monkeypatch):
    sentinel = (object(), object())
    calls = []

    def old_backend(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", old_backend)
    monkeypatch.setattr(
        kda_decode_api,
        "recurrent_kda",
        lambda *args, **kwargs: pytest.fail("facade nested the decorated decode API"),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: pytest.fail(f"unexpected frozen route {variant}/{arch}"),
    )
    q = torch.empty((2, 1, 4, 128), dtype=torch.bfloat16)
    result = recurrent_kda(q, q, q, q, torch.empty((2, 1, 4)))
    assert result is sentinel
    result = recurrent_kda(
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        q.expand(2, 2, 4, 128),
        torch.empty((2, 2, 4)),
        num_spec_tokens=1,
    )
    assert result is sentinel
    assert len(calls) == 2


def test_multi_token_gqa_stays_on_existing_backend(cuda_device, monkeypatch):
    sentinel = (object(), object())
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_decode_api, "_run_recurrent_kda", lambda **kwargs: sentinel)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: pytest.fail(f"unexpected frozen route {variant}/{arch}"),
    )
    q = torch.randn((1, 2, 2, 128), dtype=torch.bfloat16, device=cuda_device)
    v = torch.randn((1, 2, 4, 128), dtype=torch.bfloat16, device=cuda_device)
    result = recurrent_kda(
        q,
        q.clone(),
        v,
        v.clone(),
        torch.randn((1, 2, 4), dtype=torch.bfloat16, device=cuda_device),
        A_log=torch.randn(2, device=cuda_device),
        dt_bias=torch.randn((2, 128), device=cuda_device),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
    )
    assert result is sentinel


@pytest.mark.parametrize(
    ("packed", "num_heads", "expected_variant"),
    [
        (False, 64, "m128_n16_short"),
        (True, 64, "m128_n16_short"),
        (True, 4, "m128_n16_short"),
        (True, 2, "m128_n16_short"),
        (False, 12, "m128_n16"),
    ],
)
@pytest.mark.parametrize(
    ("compute_capability", "expected_target"),
    [((10, 0), "sm100a"), ((10, 3), "sm103a")],
)
def test_frozen_route_and_ffi_abi(
    cuda_device,
    monkeypatch,
    packed,
    num_heads,
    expected_variant,
    compute_capability,
    expected_target,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    modules = {}
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        modules.setdefault(variant, _RecorderModule())
        return modules[variant]

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[1, 2] if packed else [2],
        num_heads=num_heads,
        packed=packed,
    )
    if packed and num_heads == 2:
        inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
    output = torch.zeros_like(inputs["q"])
    seq_order = (
        torch.tensor([1, 0], dtype=torch.int32, device="cuda") if packed else None
    )
    actual, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        seq_order=seq_order,
        backend="cake",
    )
    assert actual.data_ptr() == output.data_ptr()
    assert state is None
    assert set(modules) == {expected_variant}
    assert routes == [(expected_variant, expected_target)]
    (args,) = modules[expected_variant].calls
    assert len(args) == 28
    assert args[0].data_ptr() == inputs["q"].data_ptr()
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].shape == (
        max(
            inputs["q"].numel() // (num_heads * 128),
            16 if expected_variant in ("m128_n16", "m128_n16_short") else 32,
        ),
        (num_heads + 7) // 8 * 8,
    )
    assert args[8].dtype == torch.int64
    assert args[9].dtype == torch.int32
    if packed:
        assert args[9].data_ptr() == seq_order.data_ptr()
    if expected_variant == "m64":
        assert args[10].data_ptr() == args[12].data_ptr()
        assert args[13].dtype == torch.uint8
        assert args[13].shape == (768,)
        assert args[14] == 1
        assert args[15] == num_heads
        assert args[16] == 0
        assert args[17] == 0
        assert math.isclose(args[18], 128**-0.5)
        assert args[19] == -5.0
        assert args[20] == int(torch.cuda.current_stream(cuda_device).cuda_stream)
    else:
        assert args[11].data_ptr() == args[13].data_ptr()
        assert args[16].dtype == torch.uint8
        assert args[16].shape == (768,)
        assert args[17] == 1
        assert args[18] == num_heads
        assert args[19] == inputs["beta"].stride(-2)
        assert args[21] == 0
        assert args[22] == 0
        assert args[23] == 0
        assert args[24] == 0
        assert math.isclose(args[25], 128**-0.5)
        assert args[26] == -5.0
        assert args[27] == int(torch.cuda.current_stream(cuda_device).cuda_stream)
    if num_heads % 8 != 0:
        assert args[5].data_ptr() != inputs["beta"].data_ptr()


@pytest.mark.parametrize(
    ("seq_lens", "expected_variant", "expected_beta_tma_shape", "expect_alias"),
    [
        ([128] * 8, "m128_h12_short", (1024, 16), False),
        ([1024] * 8, "m128_h12_long", (4096, 24), True),
        ([513] * 7, "m128", (3591, 16), False),
    ],
)
def test_h12_n32_specializations_reach_ffi(
    cuda_device,
    monkeypatch,
    seq_lens,
    expected_variant,
    expected_beta_tma_shape,
    expect_alias,
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 3)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(
        kda_prefill_api, "_flash_kda_device_sm_count", lambda device: 152
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(seq_lens=seq_lens, num_heads=12, packed=True)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )

    assert routes == [(expected_variant, "sm103a")]
    (args,) = module.calls
    assert len(args) == 28
    assert tuple(args[5].shape) == expected_beta_tma_shape
    assert (args[5].data_ptr() == inputs["beta"].data_ptr()) is expect_alias


def test_sm103_uniform_n32_tensor_state_decay_reaches_m128_ffi(
    cuda_device,
    monkeypatch,
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 3)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(
        kda_prefill_api, "_flash_kda_device_sm_count", lambda device: 152
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(seq_lens=[256], num_heads=96, packed=False)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )

    assert routes == [("m128_tensor_state_decay", "sm103a")]
    assert len(module.calls[0]) == 28


def test_frozen_sm103_tensor_state_decay_matches_scalar_control(
    flash_kda_device,
    monkeypatch,
):
    if get_compute_capability(flash_kda_device) != (10, 3):
        pytest.skip("tensor-core state decay is selected only on CC 10.3")

    inputs = _make_inputs(
        seq_lens=[256],
        num_heads=96,
        packed=False,
        initial_state=True,
        seed=25696,
    )
    initial_state_seed = inputs["initial_state"].clone()
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append((variant, target))
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    actual_output = actual_output.clone()
    actual_state = actual_state.clone()

    inputs["initial_state"].copy_(initial_state_seed)
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_n32_tensor_state_decay",
        lambda **kwargs: False,
    )
    expected_output, expected_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    expected_target = kda_prefill_api._select_flash_kda_prefill_target(flash_kda_device)
    assert routes == [
        ("m128_tensor_state_decay", expected_target),
        ("m128", expected_target),
    ]
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_pair_packed_h12_beta_requires_an_even_dense_carrier(cuda_device):
    dense = torch.empty((1, 128, 12), dtype=torch.bfloat16, device=cuda_device)
    paired = kda_prefill_api._pair_packed_beta_tma_source(dense)
    assert paired is not None
    assert paired.shape == (64, 24)
    assert paired.data_ptr() == dense.data_ptr()
    odd = torch.empty((1, 129, 12), dtype=torch.bfloat16, device=cuda_device)
    assert kda_prefill_api._pair_packed_beta_tma_source(odd) is None


@pytest.mark.parametrize(
    ("seq_lens", "expected_variant"),
    [
        ([128] * 2, "m128_h12_short"),
        ([160] * 2, "m128_h12_long"),
    ],
)
def test_frozen_h12_n32_specializations_match_reference(
    flash_kda_device,
    monkeypatch,
    seq_lens,
    expected_variant,
):
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=sum(seq_lens),
    )
    expected_output, expected_state = _chunk16_debug_reference(inputs)
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append((variant, target))
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    expected_target = kda_prefill_api._select_flash_kda_prefill_target(flash_kda_device)
    assert routes == [(expected_variant, expected_target)]
    assert actual_state is inputs["initial_state"]
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_h12_long_cuda_graph_replay_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[160] * 2,
        num_heads=12,
        packed=True,
        initial_state=False,
        seed=321,
    )
    expected_output, expected_state = _chunk16_debug_reference(inputs)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "cake",
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        output.fill_(float("nan"))
        captured_state.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert workspace._captured
    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    torch.testing.assert_close(
        captured_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_h12_long_ffi_rejects_disjoint_pair_carrier(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[160] * 2,
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=322,
    )
    recorder = _RecorderModule()
    get_module = kda_prefill_api._get_flash_kda_prefill_module
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: recorder,
    )
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    (args,) = recorder.calls
    assert args[5].data_ptr() == inputs["beta"].data_ptr()

    target = kda_prefill_api._select_flash_kda_prefill_target(flash_kda_device)
    module = get_module("m128_h12_long", target)
    disjoint_args = list(args)
    disjoint_args[5] = args[5].clone()
    with pytest.raises(Exception, match="must exactly alias beta storage"):
        module.run(*disjoint_args)


@pytest.mark.parametrize("num_heads", [4, 8, 16, 32])
@pytest.mark.parametrize(
    ("compute_capability", "expected_target"),
    [((10, 0), "sm100a"), ((10, 3), "sm103a")],
)
def test_unbounded_softplus_prefill_routes_to_cake_runtime_head_module(
    cuda_device, monkeypatch, num_heads, compute_capability, expected_target
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_cake_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(
        kda_prefill_api, "_get_cake_kda_prefill_module", get_cake_module
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: pytest.fail(
            f"unbounded route reached provenanced FlashKDA module {variant}/{target}"
        ),
    )
    inputs = _make_inputs(
        seq_lens=[33, 65],
        num_heads=num_heads,
        packed=True,
    )
    output, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs, lower_bound=None),
        output=torch.empty_like(inputs["q"]),
    )

    assert output.shape == inputs["q"].shape
    assert state is None
    assert routes == [("m128_unbounded_softplus", expected_target)]
    (args,) = module.calls
    assert len(args) == 28
    assert args[18] == num_heads
    assert args[26] == 0.0


@pytest.mark.parametrize(
    ("num_heads", "checkpoint_every_n_tokens", "checkpoint_cu_starts", "expected"),
    [
        (4, 64, [0, 2, 5], "m128_bt64_unbounded_softplus"),
        (4, 32, [0, 3, 8], "m128_unbounded_softplus"),
        (8, 64, [0, 2, 5], "m128_unbounded_softplus"),
    ],
)
def test_unbounded_softplus_bt64_route_is_checkpoint_and_head_specific(
    cuda_device,
    monkeypatch,
    num_heads,
    checkpoint_every_n_tokens,
    checkpoint_cu_starts,
    expected,
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_cake_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(
        kda_prefill_api, "_get_cake_kda_prefill_module", get_cake_module
    )
    inputs = _make_inputs(
        seq_lens=[65, 131],
        num_heads=num_heads,
        packed=True,
    )
    checkpoint_cu_starts_tensor = torch.tensor(
        checkpoint_cu_starts, dtype=torch.int64, device=cuda_device
    )
    state_checkpoints = torch.empty(
        (checkpoint_cu_starts[-1], num_heads, 128, 128),
        dtype=torch.bfloat16,
        device=cuda_device,
    )

    output, state, returned_checkpoints = recurrent_kda(
        **_strict_prefill_kwargs(inputs, lower_bound=None),
        output=torch.empty_like(inputs["q"]),
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts_tensor,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        backend="cake",
    )

    assert output.shape == inputs["q"].shape
    assert state is None
    assert returned_checkpoints is state_checkpoints
    assert routes == [(expected, "sm100a")]
    (args,) = module.calls
    assert len(args) == 28
    assert args[14].data_ptr() == state_checkpoints.data_ptr()
    assert args[15].data_ptr() == checkpoint_cu_starts_tensor.data_ptr()
    assert args[18] == num_heads
    assert args[24] == checkpoint_every_n_tokens
    assert args[26] == 0.0


@pytest.mark.parametrize("sm_count", [148, 152])
def test_sm100_uniform_prefill_reaches_persistent_worker_abi(
    cuda_device,
    monkeypatch,
    sm_count,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 0),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: sm_count,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[32, 32],
        num_heads=96,
        packed=True,
        initial_state=True,
    )
    output, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    assert output.shape == inputs["q"].shape
    assert state is None
    assert routes == [("persistent_m128", "sm100a")]
    (args,) = module.calls
    assert len(args) == 23
    assert args[9].tolist() == [0, 1]
    assert sorted(args[10].tolist()) == list(range(2 * 96))
    assert args[11].numel() == sm_count + 1
    assert args[11][0].item() == 0
    assert args[11][-1].item() == 2 * 96
    assert args[15].dtype == torch.uint8
    assert args[15].shape == (768,)
    assert args[16] == 1
    assert args[17] == 96


def test_uniform_piece_prefill_reaches_extended_worker_abi(
    cuda_device,
    monkeypatch,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 3),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 152,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_uniform_piece_persistent",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_persistent_m128_roofline",
        lambda **kwargs: SimpleNamespace(
            worker_count=152,
            piece_ns=1.0,
            direct_ns=2.0,
        ),
    )
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[64] * 8,
        num_heads=96,
        packed=True,
        initial_state=True,
    )
    output, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )

    assert output.shape == inputs["q"].shape
    assert state is None
    assert routes == [("piece_persistent_m128", "sm103a")]
    (args,) = module.calls
    assert len(args) == 29
    assert args[9].tolist() == list(range(8))
    assert args[10].numel() > 8 * 96
    assert args[11].numel() == 153
    assert args[10].numel() == args[12].numel() == args[13].numel()
    assert args[10].numel() == args[14].numel() == args[15].numel()
    assert args[16].shape[1:] == (128, 128)
    assert args[17].dtype == torch.uint32
    assert args[16].shape[0] == args[17].numel()
    assert args[21].dtype == torch.uint8
    assert args[21].shape == (768,)
    assert args[22] == 1
    assert args[23] == 96


def test_explicit_workspace_keeps_uniform_piece_candidate_on_direct_abi(
    cuda_device,
    monkeypatch,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 3),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 152,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_uniform_piece_persistent",
        lambda **kwargs: True,
    )
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[64] * 8,
        num_heads=96,
        packed=True,
        initial_state=True,
    )
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        prefill_workspace=workspace,
        backend="cake",
    )

    assert routes == [("m128", "sm103a")]
    assert len(module.calls[0]) == 28


@pytest.mark.parametrize("num_heads", [40, 64, 96])
def test_frozen_uniform_piece_prefill_repeats_and_matches_direct_control(
    flash_kda_device,
    monkeypatch,
    num_heads,
):
    inputs = _make_inputs(
        seq_lens=[1024] * 8,
        num_heads=num_heads,
        packed=True,
        initial_state=True,
        seed=102400 + num_heads,
    )
    initial_state_seed = inputs["initial_state"].clone()
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append((variant, target))
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )

    piece_results = []
    for _ in range(2):
        inputs["initial_state"].copy_(initial_state_seed)
        piece_output, piece_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            output_final_state=True,
            backend="cake",
        )
        piece_results.append((piece_output.clone(), piece_state.clone()))

    stream_workspace = kda_prefill_api._get_stream_workspace(flash_kda_device)
    assert stream_workspace._piece_mid_state_ready is not None
    assert not torch.count_nonzero(stream_workspace._piece_mid_state_ready).item()

    inputs["initial_state"].copy_(initial_state_seed)
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_uniform_piece_persistent",
        lambda **kwargs: False,
    )
    direct_output, direct_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        seq_order=torch.arange(8, dtype=torch.int32, device=flash_kda_device),
        backend="cake",
    )

    expected_target = kda_prefill_api._select_flash_kda_prefill_target(flash_kda_device)
    direct_variant = (
        "m128_tensor_state_decay"
        if get_compute_capability(flash_kda_device) == (10, 3) and num_heads >= 64
        else "m128"
    )
    assert routes == [
        ("piece_persistent_m128", expected_target),
        ("piece_persistent_m128", expected_target),
        (direct_variant, expected_target),
    ]
    for piece_output, piece_state in piece_results:
        torch.testing.assert_close(
            piece_output.float(), direct_output.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            piece_state.float(), direct_state.float(), atol=1e-2, rtol=1e-2
        )
    assert torch.equal(piece_results[0][0], piece_results[1][0])
    assert torch.equal(piece_results[0][1], piece_results[1][1])


def test_explicit_seq_order_keeps_direct_worker_and_reaches_ffi(
    cuda_device,
    monkeypatch,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 0),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 148,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[32, 32],
        num_heads=96,
        packed=True,
        initial_state=True,
    )
    seq_order = torch.tensor([1, 0], dtype=torch.int32, device=cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        seq_order=seq_order,
        backend="cake",
    )

    assert routes == [("m128", "sm100a")]
    (args,) = module.calls
    assert len(args) == 28
    assert args[9].data_ptr() == seq_order.data_ptr()


def test_b200_prefill_without_initial_state_stays_direct(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 0),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 148,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[3, 1],
        num_heads=96,
        packed=True,
        initial_state=False,
    )
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    assert routes == [("m128_n16_short", "sm100a")]
    (args,) = module.calls
    assert args[9].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("compute_capability", "sm_count", "expected_target"),
    [((10, 0), 148, "sm100a"), ((10, 3), 152, "sm103a")],
)
def test_fixed_small_bh_prefill_reaches_owner_helper_abi(
    cuda_device,
    monkeypatch,
    compute_capability,
    sm_count,
    expected_target,
):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: compute_capability,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: sm_count,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule(final_value=0.5)
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[2048],
        num_heads=1,
        packed=False,
    )
    output, state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    assert output.shape == inputs["q"].shape
    assert state is not None
    assert torch.all(state == 0.5)
    assert routes == [("small_bh_m128", expected_target)]
    (args,) = module.calls
    assert len(args) == 25
    assert args[13].dtype == torch.uint8
    assert args[13].shape == (896,)
    assert args[14].dtype == torch.bfloat16
    assert args[14].shape == (35 * 123, 128)
    assert args[15].dtype == torch.uint32
    assert args[15].shape == (35,)
    assert args[16].dtype == torch.uint32
    assert args[16].shape == (35,)
    assert args[17].dtype == torch.uint32
    assert args[17].shape == (1,)
    assert args[18] == 1
    assert args[19] == 1
    assert args[21] == 1
    assert math.isclose(args[22], 128**-0.5)
    assert args[23] == -5.0
    assert args[24] == int(torch.cuda.current_stream(cuda_device).cuda_stream)


def test_b200_packed_metadata_is_cached_for_unchanged_offsets(cuda_device):
    workspace = kda_prefill_api._FlashKDAStreamWorkspace(cuda_device)
    offsets = torch.tensor([0, 3, 6], dtype=torch.int64, device=cuda_device)
    first = kda_prefill_api._cached_packed_task_metadata(
        workspace,
        offsets,
        total_tokens=6,
        num_heads=96,
        sm_count=148,
        build_persistent_plan=True,
    )
    second = kda_prefill_api._cached_packed_task_metadata(
        workspace,
        offsets,
        total_tokens=6,
        num_heads=96,
        sm_count=148,
        build_persistent_plan=True,
    )
    assert first[0] == (0, 1)
    assert first[1] is not None
    assert first[3] == (0, 3, 6)
    assert first[4] == (3, 3)
    assert first is second


def test_packed_metadata_is_self_contained_across_threads(cuda_device):
    workspace = kda_prefill_api._FlashKDAStreamWorkspace(cuda_device)
    device_index = torch.cuda.current_device()
    layouts = {
        "short_first": ((0, 1, 6), (1, 5)),
        "long_first": ((0, 4, 6), (4, 2)),
    }
    barrier = threading.Barrier(len(layouts), timeout=10)
    result_lock = threading.Lock()
    results = {}
    failures = []

    def build_metadata(name, expected):
        try:
            torch.cuda.set_device(device_index)
            offsets, _ = expected
            cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=cuda_device)
            metadata = kda_prefill_api._cached_packed_task_metadata(
                workspace,
                cu_seqlens,
                total_tokens=6,
                num_heads=96,
                sm_count=148,
                build_persistent_plan=False,
            )
            barrier.wait()
            with result_lock:
                results[name] = metadata
        except BaseException as error:
            barrier.abort()
            with result_lock:
                failures.append(error)

    threads = [
        threading.Thread(target=build_metadata, args=(name, expected))
        for name, expected in layouts.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert all(not thread.is_alive() for thread in threads)
    assert not failures
    for name, (expected_offsets, expected_lengths) in layouts.items():
        metadata = results[name]
        assert metadata[3] == expected_offsets
        assert metadata[4] == expected_lengths


def test_direct_packed_prefill_automatically_sorts_sequences(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api,
        "get_compute_capability",
        lambda device: (10, 3),
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_is_cuda_version_at_least",
        lambda version: True,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_flash_kda_device_sm_count",
        lambda device: 152,
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, target: module,
    )
    inputs = _make_inputs(
        seq_lens=[1, 3, 2],
        num_heads=96,
        packed=True,
    )
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        backend="cake",
    )
    (args,) = module.calls
    assert args[9].tolist() == [1, 2, 0]


def test_strided_beta_indexed_state_and_checkpoints_reach_native_ffi(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    module = _RecorderModule()
    routes = []

    def get_module(variant, target):
        routes.append((variant, target))
        return module

    monkeypatch.setattr(kda_prefill_api, "_get_flash_kda_prefill_module", get_module)
    inputs = _make_inputs(
        seq_lens=[65, 131],
        num_heads=12,
        packed=True,
        initial_state=True,
    )
    total_tokens = inputs["q"].shape[1]
    beta_carrier = torch.empty(
        (total_tokens, 32), dtype=torch.bfloat16, device=cuda_device
    )
    beta_carrier[:, 8:20].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8:20]
    assert not inputs["beta"].is_contiguous()

    state_slot_numel = 12 * 128 * 128
    state_storage = torch.zeros(
        (5, state_slot_numel + 64), dtype=torch.bfloat16, device=cuda_device
    )
    state_pool = state_storage.as_strided(
        (5, 12, 128, 128),
        (state_storage.stride(0), 128 * 128, 128, 1),
    )
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device=cuda_device)
    state_pool[state_indices.to(torch.int64)] = inputs["initial_state"]
    inputs["initial_state"] = state_pool

    checkpoint_cu_starts = torch.tensor(
        [0, 5, 14], dtype=torch.int64, device=cuda_device
    )
    state_checkpoints = torch.empty(
        (14, 12, 128, 128), dtype=torch.bfloat16, device=cuda_device
    )
    output, returned_state, returned_checkpoints = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        ssm_state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=16,
    )
    assert output.shape == inputs["q"].shape
    assert returned_state is state_pool
    assert returned_checkpoints is state_checkpoints
    assert routes == [("m128_n16_checkpoint", "sm100a")]
    (args,) = module.calls
    assert len(args) == 28
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].data_ptr() == inputs["beta"].data_ptr()
    assert args[10].data_ptr() == state_indices.data_ptr()
    assert args[11].data_ptr() == state_pool.data_ptr()
    assert args[13].data_ptr() == state_pool.data_ptr()
    assert args[14].data_ptr() == state_checkpoints.data_ptr()
    assert args[15].data_ptr() == checkpoint_cu_starts.data_ptr()
    assert args[19] == inputs["beta"].stride(-2)
    assert args[20] == state_pool.stride(0)
    assert args[21:25] == (1, 1, 1, 16)


def test_unaligned_strided_beta_uses_internal_tma_workspace(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant, target: module
    )
    inputs = _make_inputs(seq_lens=[32], num_heads=12, packed=True)
    beta_carrier = torch.empty((32, 32), dtype=torch.bfloat16, device=cuda_device)
    beta_carrier[:, 7:19].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 7:19]

    recurrent_kda(
        **_strict_prefill_kwargs(inputs), output=torch.empty_like(inputs["q"])
    )

    (args,) = module.calls
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].data_ptr() != inputs["beta"].data_ptr()
    assert args[5].shape == (32, 16)


def test_aligned_h6_strided_beta_uses_head_padded_tma_workspace(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api, "_is_cuda_version_at_least", lambda version: True
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api, "_get_flash_kda_prefill_module", lambda variant, target: module
    )
    inputs = _make_inputs(seq_lens=[128], num_heads=6, packed=True)
    beta_carrier = torch.empty((128, 32), dtype=torch.bfloat16, device=cuda_device)
    beta_carrier[:, 8:14].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8:14]
    assert inputs["beta"].data_ptr() % 16 == 0
    assert inputs["beta"].stride(-2) == 32

    recurrent_kda(
        **_strict_prefill_kwargs(inputs), output=torch.empty_like(inputs["q"])
    )

    (args,) = module.calls
    assert args[4].data_ptr() == inputs["beta"].data_ptr()
    assert args[5].data_ptr() != inputs["beta"].data_ptr()
    assert args[5].shape == (128, 8)


def test_frozen_route_passes_nondefault_stream(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    stream = torch.cuda.Stream(device=cuda_device)
    stream.wait_stream(torch.cuda.current_stream(cuda_device))
    with torch.cuda.stream(stream):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            backend="cake",
        )
    (args,) = module.calls
    assert args[27] == int(stream.cuda_stream)


def test_frozen_route_rejects_output_overlap(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(ValueError, match="output must not overlap q"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=inputs["q"].view_as(inputs["q"]),
        )
    assert module.calls == []


def test_initial_state_is_updated_in_place(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule(final_value=0.25)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False, initial_state=True)
    original_state = inputs["initial_state"]
    actual, returned_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    assert actual.shape == inputs["q"].shape
    assert returned_state is original_state
    (args,) = module.calls
    assert args[11].data_ptr() == original_state.data_ptr()
    assert args[13].data_ptr() == original_state.data_ptr()
    assert args[22] == 1
    assert args[23] == 1
    torch.testing.assert_close(
        original_state,
        torch.full_like(original_state, 0.25),
    )


def test_stream_workspace_does_not_allocate_state_scratch_for_inplace_update(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(kda_prefill_api, "_flash_kda_stream_workspaces", {})
    module = _RecorderModule(final_value=0.0)
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    cases = [
        _make_inputs(
            seq_lens=[2],
            num_heads=2,
            packed=False,
            initial_state=True,
        ),
        _make_inputs(
            seq_lens=[1, 1, 2],
            num_heads=2,
            packed=True,
            initial_state=True,
        ),
        _make_inputs(
            seq_lens=[2, 2],
            num_heads=2,
            packed=False,
            initial_state=True,
        ),
    ]
    for inputs in cases:
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
            backend="cake",
        )

    assert len(kda_prefill_api._flash_kda_stream_workspaces) == 1
    (workspace,) = kda_prefill_api._flash_kda_stream_workspaces.values()
    assert workspace._state_scratch is None
    assert workspace._beta_padding.numel() == 16 * 8


@pytest.mark.parametrize(
    ("dtype", "size_delta"),
    [(torch.int64, 0), (torch.int32, 1)],
)
def test_packed_seq_order_validation(cuda_device, monkeypatch, dtype, size_delta):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: _RecorderModule(),
    )
    inputs = _make_inputs(seq_lens=[1, 2], num_heads=2, packed=True)
    seq_order = torch.arange(2 + size_delta, dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="seq_order"):
        recurrent_kda(**_strict_prefill_kwargs(inputs), seq_order=seq_order)


def test_fixed_prefill_rejects_seq_order(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(ValueError, match="only supported for packed"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            seq_order=torch.zeros(1, dtype=torch.int32, device=cuda_device),
        )


def test_graph_capture_requires_packed_int64_offsets(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inputs = _make_inputs(seq_lens=[1, 2], num_heads=2, packed=True)
    inputs["cu_seqlens"] = inputs["cu_seqlens"].to(torch.int32)
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    with pytest.raises(RuntimeError, match="requires int64 cu_seqlens"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            prefill_workspace=workspace,
            backend="cake",
        )


def test_graph_capture_requires_explicit_workspace(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    with pytest.raises(
        RuntimeError, match="requires an explicit RecurrentKDAPrefillWorkspace"
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(inputs["q"]),
        )


def test_explicit_workspace_descriptor_prepare_and_reuse(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)

    for _ in range(2):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
            backend="cake",
        )
    assert [args[17] for args in module.calls] == [1, 0]
    assert (
        module.calls[0][16].data_ptr()
        == module.calls[1][16].data_ptr()
        == workspace._descriptor_storages["m128_n16_short"].data_ptr()
    )

    changed_output = torch.empty_like(output)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=changed_output,
        prefill_workspace=workspace,
        backend="cake",
    )
    assert module.calls[-1][17] == 1


def test_captured_workspace_rejects_eager_reuse_and_capture_mismatch(
    cuda_device, monkeypatch
):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
        backend="cake",
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="not warmed for the exact"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=torch.empty_like(output),
            prefill_workspace=workspace,
            backend="cake",
        )

    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
        backend="cake",
    )
    assert module.calls[-1][17] == 0
    assert workspace._captured

    with pytest.raises(RuntimeError, match="captured by another CUDA graph"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
            backend="cake",
        )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with pytest.raises(RuntimeError, match="cannot be reused eagerly"):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
            backend="cake",
        )


def test_workspace_rejects_a_different_stream(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_api, "get_compute_capability", lambda device: (10, 0)
    )
    module = _RecorderModule()
    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        lambda variant, arch: module,
    )
    inputs = _make_inputs(seq_lens=[2], num_heads=2, packed=False)
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(cuda_device)
    recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        prefill_workspace=workspace,
    )

    other_stream = torch.cuda.Stream(device=cuda_device)
    with (
        torch.cuda.stream(other_stream),
        pytest.raises(RuntimeError, match="different CUDA stream"),
    ):
        recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            prefill_workspace=workspace,
        )


def test_flash_kda_jit_getter_is_importable():
    import flashinfer
    from flashinfer.jit.flash_kda import get_flash_kda_prefill_module

    assert callable(get_flash_kda_prefill_module)
    assert flashinfer.RecurrentKDAPrefillWorkspace is RecurrentKDAPrefillWorkspace


@pytest.mark.parametrize("non_default_stream", [False, True])
def test_frozen_small_bh_prefill_matches_direct_control(
    flash_kda_device,
    monkeypatch,
    non_default_stream,
):
    inputs = _make_inputs(
        seq_lens=[2048],
        num_heads=1,
        packed=False,
        initial_state=True,
        seed=2048,
    )
    direct_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    small_output = torch.empty_like(inputs["q"])
    direct_output = torch.empty_like(inputs["q"])
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append((variant, target))
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )

    if non_default_stream:
        stream = torch.cuda.Stream(device=flash_kda_device)
        stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
        with torch.cuda.stream(stream):
            actual_output, actual_state = recurrent_kda(
                **_strict_prefill_kwargs(inputs),
                output=small_output,
                output_final_state=True,
                backend="cake",
            )
        stream.synchronize()
    else:
        actual_output, actual_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=small_output,
            output_final_state=True,
            backend="cake",
        )

    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_small_bh_owner_helper",
        lambda **kwargs: False,
    )
    expected_output, expected_state = recurrent_kda(
        **_strict_prefill_kwargs(direct_inputs),
        output=direct_output,
        output_final_state=True,
        backend="cake",
    )

    expected_target = kda_prefill_api._select_flash_kda_prefill_target(flash_kda_device)
    assert routes == [
        ("small_bh_m128", expected_target),
        ("m128", expected_target),
    ]
    assert actual_output.data_ptr() == small_output.data_ptr()
    assert actual_state is inputs["initial_state"]
    assert expected_output.data_ptr() == direct_output.data_ptr()
    assert expected_state is direct_inputs["initial_state"]
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


def test_frozen_small_bh_prefill_cuda_graph_replay_matches_direct_control(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[2048],
        num_heads=1,
        packed=False,
        initial_state=False,
        seed=2049,
    )
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "cake",
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["q"].mul_(0.875)
        inputs["beta"].add_(0.125)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replay_output = captured_output.clone()
    replay_state = captured_state.clone()

    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_small_bh_owner_helper",
        lambda **kwargs: False,
    )
    direct_output, direct_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(output),
        output_final_state=True,
        backend="cake",
    )

    assert workspace._captured
    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    torch.testing.assert_close(
        replay_output.float(),
        direct_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        replay_state.float(),
        direct_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_bt16_scalar_prepare_subgroup_heads_matches_direct_control(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=4,
        packed=False,
        initial_state=True,
        seed=2050,
    )
    initial_state_seed = inputs["initial_state"].clone()
    routes = []
    get_module = kda_prefill_api._get_flash_kda_prefill_module

    def recording_get_module(variant, target):
        routes.append(variant)
        return get_module(variant, target)

    monkeypatch.setattr(
        kda_prefill_api,
        "_get_flash_kda_prefill_module",
        recording_get_module,
    )
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: True,
    )
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    actual_output = actual_output.clone()
    actual_state = actual_state.clone()

    inputs["initial_state"].copy_(initial_state_seed)
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: False,
    )
    expected_output, expected_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )

    assert routes[:2] == ["bt16_prepare", "bt16_chain_m64_s9"]
    assert routes[-1] == "m128"
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_bt16_combined_h12_fixed512_matches_cute(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512],
        num_heads=12,
        packed=False,
        initial_state=True,
        seed=12002,
    )
    initial_state = inputs["initial_state"]
    assert initial_state is not None
    initial_state_seed = initial_state.clone()

    cake_output, cake_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    assert cake_state is initial_state
    cake_state = cake_state.clone()

    initial_state.copy_(initial_state_seed)
    cute_output, cute_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cute-dsl",
    )
    assert cute_state is initial_state
    torch.testing.assert_close(
        cake_output.float(), cute_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        cake_state.float(), cute_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_bt16_combined_h12_cuda_graph_replay(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512],
        num_heads=12,
        packed=False,
        initial_state=False,
        seed=12012,
    )
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "cake",
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["q"].mul_(0.875)
        inputs["beta"].add_(0.125)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replay_output = captured_output.clone()
    replay_state = captured_state.clone()

    direct_output, direct_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(output),
        output_final_state=True,
        backend="cake",
    )

    assert workspace._captured
    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    torch.testing.assert_close(
        replay_output.float(), direct_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        replay_state.float(), direct_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_bt16_scalar_prepare_subgroup_heads_cuda_graph_replay(
    flash_kda_device,
    monkeypatch,
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=4,
        packed=False,
        initial_state=False,
        seed=2051,
    )
    output = torch.empty_like(inputs["q"])
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: True,
    )
    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": True,
        "prefill_workspace": workspace,
        "backend": "cake",
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        inputs["q"].mul_(0.875)
        inputs["beta"].add_(0.125)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replay_output = captured_output.clone()
    replay_state = captured_state.clone()

    monkeypatch.setattr(
        kda_prefill_api,
        "_should_use_bt16_prepare_chain",
        lambda **kwargs: False,
    )
    direct_output, direct_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(output),
        output_final_state=True,
        backend="cake",
    )

    assert workspace._captured
    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    torch.testing.assert_close(
        replay_output.float(), direct_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        replay_state.float(), direct_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("non_default_stream", [False, True])
def test_frozen_prefill_matches_reference(flash_kda_device, packed, non_default_stream):
    inputs = _make_inputs(
        seq_lens=[3, 5] if packed else [4, 4],
        num_heads=2,
        packed=packed,
        initial_state=True,
        seed=2026,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]
    seq_order = (
        torch.tensor([1, 0], dtype=torch.int32, device=flash_kda_device)
        if packed
        else None
    )

    if non_default_stream:
        stream = torch.cuda.Stream(device=flash_kda_device)
        stream.wait_stream(torch.cuda.current_stream(flash_kda_device))
        with torch.cuda.stream(stream):
            actual_output, actual_state = recurrent_kda(
                **_strict_prefill_kwargs(inputs),
                output=output,
                output_final_state=True,
                seq_order=seq_order,
            )
        stream.synchronize()
    else:
        actual_output, actual_state = recurrent_kda(
            **_strict_prefill_kwargs(inputs),
            output=output,
            output_final_state=True,
            seq_order=seq_order,
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


@pytest.mark.parametrize(("packed", "seed"), ((False, 11018), (True, 11019)))
def test_frozen_prefill_h96_short_beta_workspace_matches_reference(
    flash_kda_device, packed, seed
):
    """Cover token-padded beta TMA storage when H is already eight-aligned."""

    inputs = _make_inputs(
        seq_lens=[16],
        num_heads=96,
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

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_prefill_without_initial_or_final_state(flash_kda_device):
    inputs = _make_inputs(seq_lens=[3], num_heads=2, packed=False, initial_state=False)
    expected_output, _ = _reference(inputs)
    output = torch.empty_like(inputs["q"])
    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=False,
    )
    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is None
    torch.testing.assert_close(
        actual_output.float(),
        expected_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )


def test_frozen_prefill_h6_full_tma_chunk_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=6,
        packed=True,
        initial_state=True,
        seed=2032,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
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


@pytest.mark.parametrize("seq_len", [32, 33])
def test_frozen_prefill_h12_tma_chunks_match_reference(flash_kda_device, seq_len):
    inputs = _make_inputs(
        seq_lens=[seq_len],
        num_heads=12,
        packed=False,
        initial_state=True,
        seed=2012 + seq_len,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _chunk16_debug_reference(reference_inputs)
    output = torch.empty_like(inputs["q"])

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
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


def test_frozen_prefill_h12_packed_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[32, 3],
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=2047,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _chunk16_debug_reference(reference_inputs)
    output = torch.empty_like(inputs["q"])

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_state is inputs["initial_state"]
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


def test_frozen_prefill_h12_strided_beta_indexed_state_and_checkpoints_match_reference(
    flash_kda_device,
):
    checkpoint_interval = 16
    inputs = _make_inputs(
        seq_lens=[65, 131],
        num_heads=12,
        packed=True,
        initial_state=True,
        seed=2064,
    )
    compact_initial_state = inputs["initial_state"].clone()
    beta_carrier = torch.empty(
        (inputs["q"].shape[1], 32),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )
    beta_carrier[:, 8:20].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8:20]
    expected_output, expected_state, expected_checkpoints = _chunk16_debug_reference(
        {**inputs, "initial_state": compact_initial_state},
        checkpoint_every_n_tokens=checkpoint_interval,
    )

    state_slot_numel = 12 * 128 * 128
    state_storage = torch.zeros(
        (5, state_slot_numel + 64),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )
    state_pool = state_storage.as_strided(
        (5, 12, 128, 128),
        (state_storage.stride(0), 128 * 128, 128, 1),
    )
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device=flash_kda_device)
    state_indices_i64 = state_indices.to(torch.int64)
    state_pool[state_indices_i64] = compact_initial_state
    untouched_before = state_pool[[0, 2, 4]].clone()
    inputs["initial_state"] = state_pool
    checkpoint_cu_starts = torch.tensor(
        [0, 5, 14], dtype=torch.int64, device=flash_kda_device
    )
    state_checkpoints = torch.empty(
        (14, 12, 128, 128), dtype=torch.bfloat16, device=flash_kda_device
    )

    actual_output, actual_state, actual_checkpoints = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        ssm_state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_interval,
        backend="cake",
    )

    assert actual_state is state_pool
    assert actual_checkpoints is state_checkpoints
    assert inputs["beta"].data_ptr() == beta_carrier[:, 8:20].data_ptr()
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_pool[state_indices_i64].float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_checkpoints.float(),
        expected_checkpoints.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(state_pool[[0, 2, 4]], untouched_before)


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "packed", "has_initial_state"),
    [
        ((33,), 96, False, False),
        ((33, 65), 12, True, True),
    ],
)
def test_cute_dsl_checkpoints_match_cake(
    flash_kda_device,
    seq_lens,
    num_heads,
    packed,
    has_initial_state,
):
    interval = 32
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=has_initial_state,
        seed=2091 + num_heads,
    )
    counts = [(length + interval - 1) // interval for length in seq_lens]
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    checkpoint_cu_starts = torch.tensor(
        starts, dtype=torch.int64, device=flash_kda_device
    )

    results = {}
    for backend in ("cake", "cute-dsl"):
        backend_inputs = {
            **inputs,
            "initial_state": (
                inputs["initial_state"].clone()
                if inputs["initial_state"] is not None
                else None
            ),
        }
        checkpoints = torch.empty(
            starts[-1],
            num_heads,
            128,
            128,
            dtype=torch.bfloat16,
            device=flash_kda_device,
        )
        run_kwargs = {
            **_strict_prefill_kwargs(backend_inputs),
            "output": torch.empty_like(inputs["q"]),
            "output_final_state": True,
            "state_checkpoints": checkpoints,
            "checkpoint_cu_starts": checkpoint_cu_starts,
            "checkpoint_every_n_tokens": interval,
        }
        if backend == "cute-dsl" and packed:
            wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
            wrapper.plan(run_kwargs.pop("cu_seqlens"))
            results[backend] = wrapper.run(**run_kwargs)
        else:
            results[backend] = recurrent_kda(**run_kwargs, backend=backend)

    for cute_value, cake_value in zip(
        results["cute-dsl"], results["cake"], strict=True
    ):
        torch.testing.assert_close(
            cute_value.float(), cake_value.float(), atol=1e-2, rtol=1e-2
        )


def test_prefill_without_cute_dsl_experimental_falls_back_to_cake(
    flash_kda_device, monkeypatch
):
    if not kda_prefill_cute_api._is_cute_dsl_kda_runtime_available():
        pytest.skip("reference output requires nvidia-cutlass-dsl>=4.7.0")
    run_kwargs = _strict_prefill_kwargs(
        _make_inputs(seq_lens=(33,), num_heads=12, packed=False, seed=4711)
    )
    expected = recurrent_kda(**run_kwargs, backend="cute-dsl")

    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_runtime_available",
        lambda: False,
    )
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_run_cute_dsl_kda_prefill",
        lambda **kwargs: pytest.fail("auto must not reach the CuTe DSL kernel"),
    )

    fallback = recurrent_kda(**run_kwargs)
    torch.testing.assert_close(
        fallback[0].float(), expected[0].float(), atol=1e-2, rtol=1e-2
    )
    with pytest.raises(ImportError, match=r"nvidia-cutlass-dsl>=4\.7\.0"):
        recurrent_kda(**run_kwargs, backend="cute-dsl")


def test_dsl_version_guard_is_scoped_to_the_sm100_family(cuda_device, monkeypatch):
    monkeypatch.setattr(
        kda_prefill_cute_api,
        "_is_cute_dsl_kda_runtime_available",
        lambda: False,
    )
    q = torch.empty(1, 1, 1, 1, device=cuda_device)
    for capability, blocked in (((10, 0), True), ((10, 3), True), ((12, 0), False)):
        monkeypatch.setattr(
            kda_prefill_cute_api,
            "get_compute_capability",
            lambda _device, capability=capability: capability,
        )
        assert (
            kda_prefill_cute_api._is_cute_dsl_kda_prefill_dsl_too_old(q) is blocked
        ), capability


@pytest.mark.parametrize(
    ("seq_lens", "num_heads", "packed"),
    [((17,), 96, False), ((17, 33), 12, True)],
)
def test_cute_dsl_padded_indexed_state_matches_cake(
    flash_kda_device, seq_lens, num_heads, packed
):
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=True,
        seed=2117 + num_heads,
    )
    # Exercise Cake's minimum int32 alignment contract, including a +4-byte
    # contiguous view that is not 8-byte aligned.
    state_indices = torch.tensor(
        [0, 3, 1][: len(seq_lens) + 1],
        dtype=torch.int32,
        device=flash_kda_device,
    )[1:]

    def make_state_pool():
        slot_numel = num_heads * 128 * 128
        storage = torch.zeros(
            (4, slot_numel + 64),
            dtype=torch.bfloat16,
            device=flash_kda_device,
        )
        pool = storage.as_strided(
            (4, num_heads, 128, 128),
            (storage.stride(0), 128 * 128, 128, 1),
        )
        pool[state_indices.to(torch.int64)] = inputs["initial_state"]
        return pool

    results = {}
    for backend in ("cake", "cute-dsl"):
        backend_inputs = {**inputs, "initial_state": make_state_pool()}
        run_kwargs = {
            **_strict_prefill_kwargs(backend_inputs),
            "output": torch.empty_like(inputs["q"]),
            "output_final_state": True,
            "ssm_state_indices": state_indices,
        }
        if backend == "cute-dsl" and packed:
            wrapper = RecurrentKDAPrefillWrapper(flash_kda_device)
            wrapper.plan(run_kwargs.pop("cu_seqlens"))
            results[backend] = wrapper.run(**run_kwargs)
        else:
            results[backend] = recurrent_kda(**run_kwargs, backend=backend)

    for cute_value, cake_value in zip(
        results["cute-dsl"], results["cake"], strict=True
    ):
        torch.testing.assert_close(
            cute_value.float(), cake_value.float(), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize(
    ("num_heads", "checkpoint_every_n_tokens"),
    [
        (1, 64),
        (2, 64),
        (4, 32),
        (4, 64),
        (8, 64),
        (16, 64),
        (32, 64),
    ],
)
def test_frozen_unbounded_softplus_tp_shapes_strided_beta_state_and_checkpoints(
    flash_kda_device, num_heads, checkpoint_every_n_tokens
):
    inputs = _make_inputs(
        seq_lens=[65, 131],
        num_heads=num_heads,
        packed=True,
        initial_state=True,
        seed=2088 + num_heads,
    )
    compact_initial_state = inputs["initial_state"].clone()
    beta_carrier = torch.empty(
        (inputs["q"].shape[1], num_heads + 16),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )
    beta_carrier[:, 8 : 8 + num_heads].copy_(inputs["beta"][0])
    inputs["beta"] = beta_carrier[None, :, 8 : 8 + num_heads]
    expected_output, expected_state, expected_checkpoints = _reference(
        {**inputs, "initial_state": compact_initial_state},
        lower_bound=None,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )

    state_slot_numel = num_heads * 128 * 128
    state_storage = torch.zeros(
        (5, state_slot_numel + 64),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )
    state_pool = state_storage.as_strided(
        (5, num_heads, 128, 128),
        (state_storage.stride(0), 128 * 128, 128, 1),
    )
    state_indices = torch.tensor([1, 3], dtype=torch.int32, device=flash_kda_device)
    state_indices_i64 = state_indices.to(torch.int64)
    state_pool[state_indices_i64] = compact_initial_state
    untouched_before = state_pool[[0, 2, 4]].clone()
    inputs["initial_state"] = state_pool
    checkpoint_counts = [
        (seq_len + checkpoint_every_n_tokens - 1) // checkpoint_every_n_tokens
        for seq_len in (65, 131)
    ]
    checkpoint_cu_starts = torch.tensor(
        [0, checkpoint_counts[0], sum(checkpoint_counts)],
        dtype=torch.int64,
        device=flash_kda_device,
    )
    state_checkpoints = torch.empty(
        (sum(checkpoint_counts), num_heads, 128, 128),
        dtype=torch.bfloat16,
        device=flash_kda_device,
    )

    actual_output, actual_state, actual_checkpoints = recurrent_kda(
        **_strict_prefill_kwargs(inputs, lower_bound=None),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        ssm_state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )

    assert actual_state is state_pool
    assert actual_checkpoints is state_checkpoints
    assert inputs["beta"].data_ptr() == beta_carrier[:, 8 : 8 + num_heads].data_ptr()
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        state_pool[state_indices_i64].float(),
        expected_state.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_checkpoints.float(),
        expected_checkpoints.float(),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(state_pool[[0, 2, 4]], untouched_before)


def test_frozen_unbounded_softplus_h32_prefix_resume_matches_uninterrupted(
    flash_kda_device,
):
    """A BF16 radix checkpoint must reproduce uninterrupted prefill exactly."""

    inputs = _make_inputs(
        seq_lens=[321],
        num_heads=32,
        packed=False,
        initial_state=True,
        seed=2091,
    )
    initial_state = inputs["initial_state"].clone()

    def run_slice(start, end, state):
        sliced = {
            **inputs,
            "q": inputs["q"][:, start:end],
            "k": inputs["k"][:, start:end],
            "v": inputs["v"][:, start:end],
            "g": inputs["g"][:, start:end],
            "beta": inputs["beta"][:, start:end],
            "initial_state": state,
        }
        return recurrent_kda(
            **_strict_prefill_kwargs(sliced, lower_bound=None),
            output_final_state=True,
        )

    full_output, full_state = run_slice(0, 321, initial_state.clone())
    resumed_state = initial_state.clone()
    prefix_output, returned_prefix_state = run_slice(0, 256, resumed_state)
    assert returned_prefix_state is resumed_state
    suffix_output, returned_suffix_state = run_slice(256, 321, resumed_state)
    assert returned_suffix_state is resumed_state

    resumed_output = torch.cat((prefix_output, suffix_output), dim=1)
    torch.testing.assert_close(resumed_output, full_output, atol=0, rtol=0)
    torch.testing.assert_close(returned_suffix_state, full_state, atol=0, rtol=0)


@pytest.mark.parametrize("num_sequences", [171, 256])
def test_cute_dsl_packed_tensor_map_stride_above_int32_matches_cake(
    flash_kda_device,
    num_sequences,
):
    # The natural stride of the unused packed-batch mode is
    # 128 * (171 * 1024) * 96 = 2,151,677,952 BF16 elements, just above
    # signed Int32.  The singleton mode must not make the TensorMap invalid.
    inputs = _make_inputs(
        seq_lens=(1024,) * num_sequences,
        num_heads=96,
        packed=True,
        initial_state=True,
        seed=2197,
    )
    initial_state = inputs["initial_state"]
    assert initial_state is not None
    initial_state_seed = initial_state.clone()

    cake_output, cake_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cake",
    )
    assert cake_state is initial_state
    cake_state = cake_state.clone()

    initial_state.copy_(initial_state_seed)
    cute_output, cute_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=torch.empty_like(inputs["q"]),
        output_final_state=True,
        backend="cute-dsl",
    )
    assert cute_state is initial_state
    torch.testing.assert_close(
        cute_output.float(), cake_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        cute_state.float(), cake_state.float(), atol=1e-2, rtol=1e-2
    )


def test_frozen_prefill_m64_matches_reference(flash_kda_device):
    inputs = _make_inputs(
        seq_lens=[512],
        num_heads=64,
        packed=False,
        initial_state=True,
        seed=2027,
    )
    reference_inputs = {
        **inputs,
        "initial_state": inputs["initial_state"].clone(),
    }
    expected_output, expected_state = _reference(reference_inputs)
    output = torch.empty_like(inputs["q"])
    state_identity = inputs["initial_state"]

    actual_output, actual_state = recurrent_kda(
        **_strict_prefill_kwargs(inputs),
        output=output,
        output_final_state=True,
        backend="cake",
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


@pytest.mark.parametrize(
    (
        "packed",
        "num_heads",
        "has_initial_state",
        "seq_lens",
        "output_final_state",
        "seed",
        "compare_eager_control",
    ),
    [
        (False, 64, True, (2,), True, 2028, False),
        (True, 2, False, (1, 2), True, 2028, False),
        (True, 96, True, (16,), False, 11018, True),
    ],
)
def test_frozen_prefill_cuda_graph_capture_and_replay(
    flash_kda_device,
    packed,
    num_heads,
    has_initial_state,
    seq_lens,
    output_final_state,
    seed,
    compare_eager_control,
):
    inputs = _make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        packed=packed,
        initial_state=has_initial_state,
        seed=seed,
    )
    initial_state_seed = (
        inputs["initial_state"].clone() if inputs["initial_state"] is not None else None
    )
    expected_output, expected_state = _reference(
        {
            **inputs,
            "initial_state": (
                initial_state_seed.clone() if initial_state_seed is not None else None
            ),
        }
    )
    output = torch.empty_like(inputs["q"])
    seq_order = (
        torch.arange(
            len(seq_lens) - 1,
            -1,
            -1,
            dtype=torch.int32,
            device=flash_kda_device,
        )
        if packed
        else None
    )
    workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    capture_stream.wait_stream(torch.cuda.current_stream(flash_kda_device))

    call_kwargs = {
        **_strict_prefill_kwargs(inputs),
        "output": output,
        "output_final_state": output_final_state,
        "seq_order": seq_order,
        "prefill_workspace": workspace,
        "backend": "cake",
    }
    with torch.cuda.stream(capture_stream):
        warm_output, warm_state = recurrent_kda(**call_kwargs)
    capture_stream.synchronize()
    observed_warm_state = (
        inputs["initial_state"] if inputs["initial_state"] is not None else warm_state
    )
    assert observed_warm_state is not None
    with torch.cuda.stream(capture_stream):
        warm_output_control = warm_output.clone()
        warm_state_control = observed_warm_state.clone()
        if initial_state_seed is not None:
            inputs["initial_state"].copy_(initial_state_seed)
        output.zero_()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    with torch.cuda.stream(capture_stream):
        if initial_state_seed is not None:
            inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    if not output_final_state:
        assert captured_state is None
    elif inputs["initial_state"] is None:
        assert captured_state is not None
        assert captured_state.data_ptr() == workspace._state_scratch.data_ptr()
    else:
        assert captured_state is inputs["initial_state"]
    assert workspace._captured
    observed_captured_state = (
        inputs["initial_state"]
        if inputs["initial_state"] is not None
        else captured_state
    )
    assert observed_captured_state is not None
    if compare_eager_control:
        torch.testing.assert_close(
            warm_output_control.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            warm_state_control.float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        assert torch.equal(captured_output, warm_output_control)
        assert torch.equal(observed_captured_state, warm_state_control)
    else:
        torch.testing.assert_close(
            captured_output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            observed_captured_state.float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
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
    assert wrapper._workspace._captured
    torch.testing.assert_close(
        captured_output.float(), expected_output.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        captured_state.float(), expected_state.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("num_heads", [6, 12])
def test_frozen_prefill_non_aligned_heads_graph_refreshes_beta(
    flash_kda_device, num_heads
):
    inputs = _make_inputs(
        seq_lens=[32],
        num_heads=num_heads,
        packed=False,
        initial_state=True,
        seed=2033 + num_heads,
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
    }

    with torch.cuda.stream(capture_stream):
        recurrent_kda(**call_kwargs)
        inputs["initial_state"].copy_(initial_state_seed)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output, captured_state = recurrent_kda(**call_kwargs)

    # Establish an original-beta replay result before mutating graph inputs.
    with torch.cuda.stream(capture_stream):
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    original_output = captured_output.clone()
    original_state = captured_state.clone()

    # The captured public call must repack the changed beta values on replay.
    with torch.cuda.stream(capture_stream):
        inputs["beta"].fill_(2.0)
        inputs["initial_state"].copy_(initial_state_seed)
        output.fill_(float("nan"))
    capture_stream.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    # Compare against a separate eager launch with distinct tensors/workspace.
    eager_inputs = {
        name: value.clone() if value is not None else None
        for name, value in inputs.items()
    }
    eager_inputs["initial_state"] = initial_state_seed.clone()
    eager_output_storage = torch.empty_like(output)
    eager_workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
    eager_output, eager_state = recurrent_kda(
        **_strict_prefill_kwargs(eager_inputs),
        output=eager_output_storage,
        output_final_state=True,
        prefill_workspace=eager_workspace,
    )
    torch.cuda.synchronize()

    assert captured_output.data_ptr() == output.data_ptr()
    assert captured_state is inputs["initial_state"]
    assert eager_output.data_ptr() == eager_output_storage.data_ptr()
    assert eager_state is eager_inputs["initial_state"]
    assert torch.equal(captured_output, eager_output)
    assert torch.equal(captured_state, eager_state)
    assert not torch.equal(captured_output, original_output)
    assert not torch.equal(captured_state, original_state)


def test_frozen_prefill_cuda_graph_workspaces_are_isolated(flash_kda_device):
    capture_stream = torch.cuda.Stream(device=flash_kda_device)
    launch_stream = torch.cuda.Stream(device=flash_kda_device)
    bundles = []

    for seed in (2030, 2031):
        inputs = _make_inputs(
            seq_lens=[2],
            num_heads=2,
            packed=False,
            initial_state=True,
            seed=seed,
        )
        state_seed = inputs["initial_state"].clone()
        expected_output, expected_state = _reference(
            {
                **inputs,
                "initial_state": state_seed.clone(),
            }
        )
        output = torch.empty_like(inputs["q"])
        workspace = RecurrentKDAPrefillWorkspace(flash_kda_device)
        call_kwargs = {
            **_strict_prefill_kwargs(inputs),
            "output": output,
            "output_final_state": True,
            "prefill_workspace": workspace,
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
        bundles.append(
            (
                graph,
                workspace,
                inputs,
                state_seed,
                output,
                expected_output,
                expected_state,
            )
        )

    assert bundles[0][1]._state_scratch is None
    assert bundles[1][1]._state_scratch is None
    assert (
        bundles[0][1]._descriptor_storages["m128"].data_ptr()
        != bundles[1][1]._descriptor_storages["m128"].data_ptr()
    )

    for bundle_index in (0, 1, 0, 1):
        (
            graph,
            _workspace,
            inputs,
            state_seed,
            output,
            expected_output,
            expected_state,
        ) = bundles[bundle_index]
        with torch.cuda.stream(launch_stream):
            inputs["initial_state"].copy_(state_seed)
            output.fill_(float("nan"))
        launch_stream.synchronize()
        with torch.cuda.stream(launch_stream):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(),
            expected_output.float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            inputs["initial_state"].float(),
            expected_state.float(),
            atol=1e-2,
            rtol=1e-2,
        )
