"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Numerical accuracy tests for b12x Fused MoE on SM120/SM121 GPUs.

These are SM120-only APIs that take bf16 input directly (no x_sf needed).
The kernel fuses quantization + routing + FC1 + activation + FC2 + scatter.

This test file covers both APIs:
1. Functional API: `b12x_fused_moe`
2. Wrapper API: `B12xMoEWrapper`

Tests include:
- Numerical accuracy against reference implementation (SiLU and ReLU2)
- CUDA graph capture and replay
- API consistency between functional and wrapper APIs
- Micro kernel path for small decode batches
- ReLU2 (non-gated) activation for Nemotron-Super
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from .utils import (
    check_accuracy,
    compute_reference_moe_fp4,
    compute_reference_moe_relu2,
    create_b12x_moe_tensors as create_moe_tensors,
    create_relu2_moe_tensors,
)


def is_sm120_family():
    """Check for SM120 family (SM120, SM121)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 12


def _is_sm12x_supported():
    """Check SM120/SM121 support using repo-standard utility checks."""
    from flashinfer.utils import is_sm120a_supported, is_sm121a_supported

    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _cuda_13_or_newer():
    """b12x fused MoE kernels require the CUDA 13 toolkit."""
    try:
        from flashinfer.jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False


# Skip decorators
cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuteDSL not available"
)
sm120_required = pytest.mark.skipif(
    not _is_sm12x_supported(),
    reason="Requires SM120/SM121 GPU with CUDA 12.8+",
)
cuda_13_required = pytest.mark.skipif(
    not _cuda_13_or_newer(),
    reason="b12x fused MoE requires CUDA 13 or later",
)

_STATIC_CUTOVER_ENV_VARS = (
    "FLASHINFER_B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS",
    "B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS",
    "FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS",
    "B12X_STATIC_COMPACT_CUTOVER_PAIRS",
)


def _clear_static_cutover_env(monkeypatch):
    for name in _STATIC_CUTOVER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# =============================================================================
# Unit regressions for SM120 dispatch decisions
# =============================================================================


@cute_dsl_available
def test_w4a16_static_tiler_uses_64_when_intermediate_not_128_aligned():
    """The W4A16 backend must not use a 128-wide N tile for n=64 mod 128."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
        _select_tile_config,
    )

    _, tile_n, _, _ = _select_tile_config(
        problem_m=32,
        problem_n=192,
        problem_k=256,
        top_k=2,
        moe_block_size=64,
        sms=1,
        max_shared_mem=101_376,
    )

    assert tile_n == 64


@cute_dsl_available
def test_w4a16_quant_mode_selects_internal_workspace(monkeypatch):
    """Callers provide quant_mode; dispatch owns the concrete workspace type."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    captured = {}

    def fake_allocate(**kwargs):
        captured.update(kwargs)
        return "w4a16-workspace"

    monkeypatch.setattr(moe_dispatch, "_allocate_sm120_w4a16_workspace", fake_allocate)

    workspace = moe_dispatch.allocate_sm120_moe_workspace(
        state_E=1,
        weight_E=1,
        routed_rows=64,
        k=256,
        n=192,
        num_topk=2,
        device=torch.device("cuda"),
        quant_mode="w4a16",
    )

    assert workspace == "w4a16-workspace"
    assert captured["routed_rows"] == 64
    assert captured["k"] == 256
    assert captured["n"] == 192


@cute_dsl_available
def test_sm120_backend_cutovers_are_precision_specific(monkeypatch):
    """W4A16 bypasses NVFP4 static/dynamic cutovers via quant_mode."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    _clear_static_cutover_env(monkeypatch)
    moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    try:
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=80,
                num_topk=8,
                activation_precision="fp4",
            )
            == "static"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=81,
                num_topk=8,
                activation_precision="fp4",
            )
            == "dynamic"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=16,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=1024,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
    finally:
        moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()


@cute_dsl_available
def test_w4a16_static_cutover_env_override_is_precision_scoped(monkeypatch):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    _clear_static_cutover_env(monkeypatch)
    moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    monkeypatch.setenv("FLASHINFER_B12X_W4A16_STATIC_COMPACT_CUTOVER_PAIRS", "256")
    try:
        assert moe_dispatch._get_static_compact_cutover_pairs("fp4") == 640
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=32,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=33,
                num_topk=8,
                quant_mode="w4a16",
            )
            == "w4a16"
        )
    finally:
        moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()

    moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()
    monkeypatch.setenv("FLASHINFER_B12X_STATIC_COMPACT_CUTOVER_PAIRS", "256")
    try:
        assert moe_dispatch._get_static_compact_cutover_pairs("fp4") == 256
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=32,
                num_topk=8,
                quant_mode="nvfp4",
            )
            == "static"
        )
        assert (
            moe_dispatch.select_sm120_moe_backend(
                num_tokens=33,
                num_topk=8,
                quant_mode="nvfp4",
            )
            == "dynamic"
        )
    finally:
        moe_dispatch._STATIC_COMPACT_CUTOVER_PAIRS_CACHE.clear()


@cute_dsl_available
def test_w4a16_direct_micro_rejects_cutlass45_wide_multi_token_shape(monkeypatch):
    """The former direct-micro rejected shape now uses the W4A16 backend."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    captured = {}

    def fake_allocate(**kwargs):
        captured.update(kwargs)
        return "w4a16-workspace"

    monkeypatch.setattr(moe_dispatch, "_allocate_sm120_w4a16_workspace", fake_allocate)

    workspace = moe_dispatch.allocate_sm120_moe_workspace(
        state_E=32,
        weight_E=32,
        routed_rows=4 * 8,
        k=4096,
        n=4096,
        num_topk=8,
        device=torch.device("cuda"),
        quant_mode="w4a16",
    )

    assert workspace == "w4a16-workspace"
    assert captured["k"] == 4096
    assert captured["n"] == 4096
    assert captured["routed_rows"] == 32


@cute_dsl_available
def test_w4a16_direct_micro_shape_guard_rejects_cached_wide_shape(monkeypatch):
    """Wide-shape graph coverage is now expressed as W4A16 workspace capacity."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        max_w4a16_route_capacity,
    )

    routed_rows = 4 * 8
    route_slots, route_blocks = max_w4a16_route_capacity(routed_rows, 32)
    monkeypatch.setattr(moe_dispatch, "get_num_sm", lambda device: 120)
    (
        workspace_slots,
        workspace_blocks,
        fc1_scratch,
        fc2_scratch,
        fc1_cols,
    ) = moe_dispatch._w4a16_workspace_geometry(
        routed_rows=routed_rows,
        route_num_experts=32,
        k=4096,
        n=4096,
        is_gated=True,
        device=torch.device("cuda"),
    )

    assert workspace_slots >= route_slots
    assert workspace_blocks >= route_blocks
    assert fc1_cols == 8192
    assert fc1_scratch > 0
    assert fc2_scratch > 0


@cute_dsl_available
def test_legacy_static_dynamic_allocators_reject_w4a16():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    kwargs = dict(
        state_E=1,
        weight_E=1,
        k=256,
        n=512,
        num_topk=2,
        device=torch.device("cuda"),
        activation_precision="bf16",
    )
    with pytest.raises(ValueError, match="allocate_sm120_moe_workspace"):
        moe_dispatch.allocate_sm120_static_workspace(max_rows=64, **kwargs)
    with pytest.raises(ValueError, match="allocate_sm120_moe_workspace"):
        moe_dispatch.allocate_sm120_dynamic_workspace(routed_rows=64, **kwargs)


def _fake_cuda_13_version():
    class CudaVersion:
        major = 13

        def __str__(self):
            return "13.0"

    return CudaVersion()


def test_functional_cuda_graph_capture_requires_output(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    monkeypatch.setattr(b12x_moe_mod, "_is_cuda_graph_capturing", lambda: True)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 1, 1), dtype=torch.uint8)
    scale = torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="pre-allocated output"):
        b12x_moe_mod.b12x_fused_moe(
            x=x,
            w1_weight=weight,
            w1_weight_sf=scale,
            w1_alpha=alpha,
            fc2_input_scale=alpha,
            w2_weight=weight,
            w2_weight_sf=scale,
            w2_alpha=alpha,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            num_experts=1,
            top_k=1,
            quant_mode="w4a16",
        )


@cute_dsl_available
def test_functional_api_passes_source_format_to_dispatch(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return kwargs["scatter_output"]

    monkeypatch.setattr(moe_dispatch, "launch_sm120_moe", fake_launch)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    output = b12x_moe_mod.b12x_fused_moe(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        num_experts=1,
        top_k=1,
        quant_mode="w4a16",
        source_format="compressed_tensors",
    )

    assert output is captured["scatter_output"]
    assert captured["source_format"] == "compressed_tensors"


@cute_dsl_available
def test_wrapper_stores_and_passes_source_format(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)
        return kwargs["scatter_output"]

    monkeypatch.setattr(moe_dispatch, "launch_sm120_moe", fake_launch)
    moe = b12x_moe_mod.B12xMoEWrapper(
        num_experts=1,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        use_cuda_graph=False,
        quant_mode="w4a16",
        source_format="compressed_tensors",
    )

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    output = moe.run(
        x=x,
        w1_weight=weight,
        w1_weight_sf=scale,
        w1_alpha=alpha,
        w2_weight=weight,
        w2_weight_sf=scale,
        w2_alpha=alpha,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
    )

    assert moe.source_format == "compressed_tensors"
    assert output is captured["scatter_output"]
    assert captured["source_format"] == "compressed_tensors"


@cute_dsl_available
def test_dispatch_rejects_nvfp4_compressed_tensors_source_format():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 32, 8), dtype=torch.uint8)
    scale = torch.empty((1, 32, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    output = torch.empty((1, 16), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"compressed_tensors.*w4a16"):
        moe_dispatch.launch_sm120_moe(
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=weight,
            w1_weight_sf=scale,
            w1_alpha=alpha,
            fc2_input_scale=alpha,
            w2_weight=weight,
            w2_weight_sf=scale,
            w2_alpha=alpha,
            num_experts=1,
            top_k=1,
            num_local_experts=1,
            scatter_output=output,
            quant_mode="nvfp4",
            source_format="compressed_tensors",
        )


@cute_dsl_available
def test_w4a16_workspace_validation_rejects_activation_mismatch():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    workspace = moe_dispatch.Sm120W4A16MoEWorkspace(
        state_E=1,
        weight_E=1,
        max_rows=1,
        k=16,
        n=16,
        num_topk=1,
        device=torch.device("cpu"),
        activation="relu2",
        activation_precision="bf16",
        quant_mode="w4a16",
        routed_rows_capacity=1,
        route_num_experts=1,
        intermediate_cache13=torch.empty((16,), dtype=torch.bfloat16),
        intermediate_cache2=torch.empty((1, 16), dtype=torch.bfloat16),
        fc1_c_tmp=torch.empty((1,), dtype=torch.float32),
        fc2_c_tmp=torch.empty((1,), dtype=torch.float32),
        packed_route_indices=torch.empty((1,), dtype=torch.int32),
        block_expert_ids=torch.empty((1,), dtype=torch.int32),
        packed_route_count=torch.empty((1,), dtype=torch.int32),
        expert_offsets=torch.empty((2,), dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="activation mismatch"):
        moe_dispatch._validate_w4a16_workspace(
            workspace,
            state_E=1,
            weight_E=1,
            routed_rows=1,
            k=16,
            n=16,
            num_topk=1,
            device=torch.device("cpu"),
            activation="silu",
        )


@cute_dsl_available
def test_wrapper_cuda_graph_capture_requires_preallocated_buffers(monkeypatch):
    from flashinfer.fused_moe.cute_dsl import b12x_moe as b12x_moe_mod
    from flashinfer.jit import cpp_ext

    monkeypatch.setattr(cpp_ext, "get_cuda_version", _fake_cuda_13_version)
    moe = b12x_moe_mod.B12xMoEWrapper(
        num_experts=1,
        top_k=1,
        hidden_size=16,
        intermediate_size=16,
        use_cuda_graph=False,
    )
    monkeypatch.setattr(b12x_moe_mod, "_is_cuda_graph_capturing", lambda: True)

    x = torch.empty((1, 16), dtype=torch.bfloat16)
    weight = torch.empty((1, 1, 1), dtype=torch.uint8)
    scale = torch.empty((1, 1, 1), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((1,), dtype=torch.float32)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match=r"use_cuda_graph=True"):
        moe.run(
            x=x,
            w1_weight=weight,
            w1_weight_sf=scale,
            w1_alpha=alpha,
            fc2_input_scale=alpha,
            w2_weight=weight,
            w2_weight_sf=scale,
            w2_alpha=alpha,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
        )


@cute_dsl_available
def test_preallocated_dynamic_workspace_rejects_remapped_experts():
    """Dynamic workspaces index expert buffers directly with global topk ids."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    workspace = object.__new__(moe_dispatch.Sm120DynamicMoEWorkspace)
    workspace.activation_precision = "fp4"

    x = torch.empty((1, 256), dtype=torch.bfloat16)
    topk_ids = torch.zeros((1, 1), dtype=torch.int32)
    topk_weights = torch.ones((1, 1), dtype=torch.float32)
    w1 = torch.empty((2, 384, 128), dtype=torch.uint8)
    w1_sf = torch.empty((2, 384, 16), dtype=torch.float8_e4m3fn)
    w2 = torch.empty((2, 256, 96), dtype=torch.uint8)
    w2_sf = torch.empty((2, 256, 12), dtype=torch.float8_e4m3fn)
    alpha = torch.ones((2,), dtype=torch.float32)
    output = torch.empty((1, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match=r"dynamic.*num_local_experts.*num_experts"):
        moe_dispatch.launch_sm120_moe(
            a=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            w1_weight=w1,
            w1_weight_sf=w1_sf,
            w1_alpha=alpha,
            fc2_input_scale=torch.ones((1,), dtype=torch.float32),
            w2_weight=w2,
            w2_weight_sf=w2_sf,
            w2_alpha=alpha,
            num_experts=4,
            top_k=1,
            num_local_experts=2,
            scatter_output=output,
            activation_precision="fp4",
            _workspace=workspace,
            _weight_views=object(),
        )


# =============================================================================
# Test Class: Functional API (b12x_fused_moe)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xFunctional:
    """Tests for the functional API: b12x_fused_moe."""

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size", [(256, 512), (1024, 2048)]
    )
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    @pytest.mark.parametrize("num_tokens", [128, 515, 1024])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_numerical_accuracy(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        """Accuracy test for b12x functional API across configurations."""
        from flashinfer import b12x_fused_moe

        num_local_experts = num_experts

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_local_experts,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_local_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    @pytest.mark.parametrize(
        "activation", ["silu", "gelu_tanh", "swigluoai_uninterleave"]
    )
    @pytest.mark.parametrize("num_tokens", [8, 128, 515])
    def test_activation_accuracy(self, activation: str, num_tokens: int):
        """Accuracy of each gated activation: SwiGLU, GeGLU and SwiGLU-OAI.

        Num tokens chosen to trigger the micro, static and dynamic backends to ensure
        that all three backends are tested.
        """
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 1536, 768
        num_experts, top_k = 8, 2
        swiglu_limit = (
            7.0 if activation == "swigluoai_uninterleave" else None
        )  # Minimax-M3 clamp limit
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        assert not torch.isnan(result).any() and not torch.isinf(result).any()
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, f"Only {percent_within * 100:.2f}% within tol (atol={atol:.4f})"

    @pytest.mark.parametrize(
        "activation", ["silu", "gelu_tanh", "swigluoai_uninterleave"]
    )
    @pytest.mark.parametrize("num_tokens", [8, 128])
    def test_intermediate_not_128_aligned(self, activation: str, num_tokens: int):
        """NVFP4 transparently pads non-128-aligned intermediate sizes (e.g.
        Gemma-4's 704) up to a tile multiple; result matches the unpadded ref."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 512, 704  # 704 = 128 * 5.5
        num_experts, top_k = 8, 2
        swiglu_limit = 7.0 if activation == "swigluoai_uninterleave" else None
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any() and not torch.isinf(result).any()
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, f"Only {percent_within * 100:.2f}% within tol (atol={atol:.4f})"

    def test_activation_precision_api_validation(self):
        """W4A4 requires fc2_input_scale; W4A16 tolerates it."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 4, 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        with pytest.raises(ValueError, match="fc2_input_scale is required"):
            b12x_fused_moe(**kwargs)

        result = b12x_fused_moe(**kwargs, activation_precision="bf16")
        assert result.shape == (num_tokens, hidden_size)

        result = b12x_fused_moe(
            **kwargs,
            fc2_input_scale=tensors["fc2_input_scale"],
            quant_mode="w4a16",
        )
        assert result.shape == (num_tokens, hidden_size)

    @pytest.mark.parametrize("num_tokens", [64, 384])
    def test_w4a16_functional_accuracy(self, num_tokens: int):
        """Accuracy test for the W4A16 activation path."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    def test_w4a16_static_accuracy_intermediate_not_128_aligned(self):
        """W4A16 must handle n=64 mod 128 without crossing gate/up tiles."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 32, 256, 192
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=789,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 n=192: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    def test_w4a16_dynamic_accuracy_with_wide_scale_tile(self):
        """Exercise W4A16 scale loads when a row spans two scale words."""
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        moe_dispatch._DYNAMIC_KERNEL_CACHE.clear()
        moe_dispatch._WORKSPACE_CACHE.clear()

        num_tokens, hidden_size, intermediate_size = 384, 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        try:
            result = b12x_fused_moe(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
                num_experts=num_experts,
                top_k=top_k,
                quant_mode="w4a16",
            )
        finally:
            moe_dispatch._DYNAMIC_KERNEL_CACHE.clear()
            moe_dispatch._WORKSPACE_CACHE.clear()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 wide scale tile: {percent_within * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )

    @pytest.mark.parametrize("num_tokens", [64, 384])
    def test_w4a16_is_more_accurate_than_w4a4(self, num_tokens: int):
        """W4A16 should be closer than W4A4 to the BF16-activation reference."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 2
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )
        kwargs = dict(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        a4_result = b12x_fused_moe(**kwargs, quant_mode="nvfp4")
        a16_result = b12x_fused_moe(**kwargs, quant_mode="w4a16")
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        a4_error = (a4_result.float() - ref_output).abs()
        a16_error = (a16_result.float() - ref_output).abs()
        a4_mae = a4_error.mean()
        a16_mae = a16_error.mean()
        a4_mse = a4_error.square().mean()
        a16_mse = a16_error.square().mean()

        assert a16_mae < 0.75 * a4_mae, (
            f"W4A16 MAE should be lower than W4A4 for tokens={num_tokens}: "
            f"a16={a16_mae.item():.6f}, a4={a4_mae.item():.6f}"
        )
        assert a16_mse < 0.5 * a4_mse, (
            f"W4A16 MSE should be lower than W4A4 for tokens={num_tokens}: "
            f"a16={a16_mse.item():.6f}, a4={a4_mse.item():.6f}"
        )


# =============================================================================
# Test Class: Wrapper API (B12xMoEWrapper)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xWrapper:
    """Tests for the wrapper API: B12xMoEWrapper."""

    @pytest.mark.parametrize(
        "activation", ["silu", "gelu_tanh", "swigluoai_uninterleave"]
    )
    def test_wrapper_intermediate_not_128_aligned(self, activation: str):
        """Wrapper transparently pads a non-128-aligned intermediate size
        (Gemma-4's 704), caching the padded weights across calls."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 128, 512, 704
        num_experts, top_k = 8, 2
        swiglu_limit = 7.0 if activation == "swigluoai_uninterleave" else None
        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )
        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any() and not torch.isinf(result).any()
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
            activation=activation,
            swiglu_limit=swiglu_limit,
        )
        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, f"Only {percent_within * 100:.2f}% within tol (atol={atol:.4f})"

    @pytest.mark.parametrize("num_tokens", [128, 256, 512])
    @pytest.mark.parametrize("top_k", [2, 8])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_wrapper_accuracy(self, num_tokens: int, top_k: int, num_experts: int):
        """Accuracy test for B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Create wrapper WITHOUT CUDA graph
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Only {percent_within * 100:.2f}% within tolerance (atol={atol:.4f})"
        )

    def test_w4a16_wrapper_accuracy(self):
        """Accuracy test for B12xMoEWrapper with BF16 intermediates."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 64, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=321,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_mode="w4a16",
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 wrapper: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    @pytest.mark.parametrize("num_tokens", [64, 128, 256])
    @pytest.mark.parametrize("num_experts", [256, 384])
    def test_wrapper_cuda_graph(self, num_tokens: int, num_experts: int):
        """Test B12xMoEWrapper with CUDA graph capture and replay."""
        from flashinfer import B12xMoEWrapper

        hidden_size, intermediate_size = 256, 512
        top_k = 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Create wrapper WITH CUDA graph
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )

        # Warmup
        for _ in range(3):
            moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        # Capture CUDA graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        # Note: CUDA graph capture doesn't execute - output may be zeros here
        # Actual execution happens during replay
        assert output.shape == (num_tokens, hidden_size)

        # First replay to get actual output
        g.replay()
        torch.cuda.synchronize()

        # Verify output is valid after first replay
        assert not torch.isnan(output).any(), "NaN after first replay"
        assert not (output == 0).all(), "All zeros after first replay"

        # Test replay consistency (allow small numerical differences due to FP4 atomics)
        results = []
        for _ in range(3):
            g.replay()
            torch.cuda.synchronize()
            results.append(output.clone())

        # All replays should produce very similar results (small FP4 tolerance)
        for i in range(1, len(results)):
            max_diff = (results[0] - results[i]).abs().max().item()
            # FP4 atomics can have small non-determinism
            assert max_diff < 0.5, f"Replay {i} differs too much: max_diff={max_diff}"

        # Verify accuracy
        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(results[0], ref_output)
        assert passed, (
            f"CUDA graph accuracy: {percent_within * 100:.2f}% (atol={atol:.4f})"
        )


# =============================================================================
# Test Class: API Consistency
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestB12xApiConsistency:
    """Tests verifying consistency between b12x functional and wrapper APIs."""

    def test_functional_vs_wrapper_output(self):
        """Verify b12x_fused_moe and B12xMoEWrapper produce the same output."""
        from flashinfer import B12xMoEWrapper, b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        # Functional API
        result_functional = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        # Wrapper API
        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result_wrapper = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        # Both should produce valid outputs
        assert result_functional.shape == result_wrapper.shape
        assert not torch.isnan(result_functional).any()
        assert not torch.isnan(result_wrapper).any()

        # Outputs should be very close (may not be exactly equal due to different
        # code paths, but should be within FP4 tolerance)
        diff = (result_functional - result_wrapper).abs()
        max_diff = diff.max().item()
        # Allow small differences from code path differences
        assert max_diff < 1e-3, f"Max diff between APIs: {max_diff}"


# =============================================================================
# Test Class: Micro Kernel (SM120-only, small decode batches)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestMicroKernel:
    """Tests for the micro kernel path (routed_rows <= 20-40).

    The micro kernel is selected automatically when routed_rows is small.
    These tests use num_tokens=1-4 to exercise the micro dispatch path,
    including the all_rows_unique fast path (num_tokens=1).
    """

    @pytest.mark.parametrize("num_tokens", [1, 2, 4])
    @pytest.mark.parametrize("top_k", [2, 8])
    @pytest.mark.parametrize("num_experts", [256])
    def test_micro_functional_accuracy(
        self, num_tokens: int, top_k: int, num_experts: int
    ):
        """Accuracy test for micro kernel via b12x functional API."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro kernel: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens}, top_k={top_k})"
        )

    def test_micro_wrapper_accuracy(self):
        """Accuracy test for micro kernel via B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 2, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro wrapper: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    @pytest.mark.parametrize("num_tokens", [1, 2, 4])
    def test_w4a16_direct_micro_functional_accuracy(self, num_tokens: int):
        """Accuracy test for the W4A16 small-batch route-packing path."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 512, 256
        num_experts, top_k = 64, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=777,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 direct micro: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    def test_w4a16_direct_micro_wrapper_accuracy(self):
        """Accuracy test for W4A16 small-batch route-packing via B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=778,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_mode="w4a16",
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"W4A16 direct micro wrapper: {percent_within * 100:.2f}% "
            f"within tolerance (atol={atol:.4f})"
        )

    def test_w4a16_direct_micro_cuda_graph(self):
        """CUDA graph replay test for the W4A16 route-packing wrapper path."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=780,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_mode="w4a16",
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )

        for _ in range(3):
            moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        graph.replay()
        torch.cuda.synchronize()

        assert output.shape == (num_tokens, hidden_size)
        assert torch.isfinite(output).all()
        assert not (output == 0).all()

    def test_w4a16_direct_micro_workspace_capacity(self):
        """The unified allocator reserves W4A16 route and GEMM scratch space."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            allocate_sm120_moe_workspace,
        )

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2
        routed_rows = num_tokens * top_k
        workspace = allocate_sm120_moe_workspace(
            state_E=num_experts,
            weight_E=num_experts,
            routed_rows=routed_rows,
            k=hidden_size,
            n=intermediate_size,
            num_topk=top_k,
            device=torch.device("cuda"),
            quant_mode="w4a16",
        )

        assert workspace.quant_mode == "w4a16"
        assert workspace.routed_rows_capacity >= routed_rows
        assert workspace.intermediate_cache13.numel() >= routed_rows * max(
            hidden_size, 2 * intermediate_size
        )
        assert workspace.intermediate_cache2.shape == (
            workspace.routed_rows_capacity,
            intermediate_size,
        )
        assert workspace.fc1_c_tmp.numel() > 0
        assert workspace.fc2_c_tmp.numel() > 0
        assert workspace.packed_route_indices.numel() >= routed_rows
        assert workspace.block_expert_ids.numel() > 0
        assert workspace.expert_offsets.numel() == workspace.route_num_experts + 1

    def test_micro_single_token_unique_path(self):
        """Test the all_rows_unique fast path (num_tokens=1, top_k=8).

        With 1 token and 8 experts, every expert has exactly 1 row.
        The micro kernel detects this and uses O(1) work tile assignment.
        """
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 1, 256, 512
        num_experts, top_k = 256, 8

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (1, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro unique path: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    @pytest.mark.parametrize(
        "num_tokens,top_k,num_experts",
        [
            (2, 8, 8),  # total_pairs=16 > num_local_experts=8
            (4, 8, 16),  # total_pairs=32 > num_local_experts=16
            (4, 4, 8),  # total_pairs=16 > num_local_experts=8
        ],
    )
    def test_micro_pairs_exceed_local_experts(
        self, num_tokens: int, top_k: int, num_experts: int
    ):
        """Regression test: micro kernel when num_tokens * top_k > num_local_experts.

        The workspace compact_topk_ids buffer was previously sized state_E
        (num_local_experts), but the micro kernel fills it with total_pairs =
        num_tokens * top_k.  When total_pairs > num_local_experts the assertion
        'flat_ids.numel() <= workspace.compact_topk_ids.numel()' fired.

        Fixed by sizing compact_topk_ids as max(state_E, max_rows).
        """
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512

        tensors = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_fp4(
            hidden_states=tensors["x_bf16"].float().cuda(),
            gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
            gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"Micro pairs>experts: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens}, top_k={top_k}, experts={num_experts})"
        )


# =============================================================================
# Test Class: ReLU2 Activation (SM120-only, non-gated)
# =============================================================================


@cute_dsl_available
@sm120_required
@cuda_13_required
class TestRelu2Activation:
    """Tests for ReLU2 activation (non-gated, Nemotron-Super)."""

    @pytest.mark.parametrize(
        "hidden_size,intermediate_size", [(256, 512), (1024, 2048)]
    )
    @pytest.mark.parametrize("top_k", [1, 2, 8])
    @pytest.mark.parametrize("num_tokens", [1, 2, 128, 512])
    @pytest.mark.parametrize("num_experts", [256])
    def test_relu2_functional_accuracy(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        """Accuracy test for ReLU2 via b12x functional API."""
        from flashinfer import b12x_fused_moe

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    @pytest.mark.parametrize("num_tokens", [64, 384])
    def test_relu2_w4a16_functional_accuracy(self, num_tokens: int):
        """Accuracy test for ReLU2 with the W4A16 activation path."""
        from flashinfer import b12x_fused_moe

        hidden_size, intermediate_size = 256, 512
        num_experts, top_k = 256, 2
        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=123,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 W4A16: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}, tokens={num_tokens})"
        )

    def test_relu2_wrapper_accuracy(self):
        """Accuracy test for ReLU2 via B12xMoEWrapper."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=False,
            activation="relu2",
        )

        result = moe.run(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 wrapper: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    def test_relu2_micro_accuracy(self):
        """Accuracy test for ReLU2 with micro kernel (small decode batch)."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 2, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
        )

        assert result.shape == (num_tokens, hidden_size)
        assert not torch.isnan(result).any()

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=tensors["fc2_input_scale"],
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 micro: {percent_within * 100:.2f}% within tolerance "
            f"(atol={atol:.4f})"
        )

    def test_relu2_w4a16_direct_micro_accuracy(self):
        """Accuracy test for ReLU2 with the W4A16 route-packing small-batch path."""
        from flashinfer import b12x_fused_moe

        num_tokens, hidden_size, intermediate_size = 2, 512, 256
        num_experts, top_k = 64, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
            seed=779,
        )

        result = b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            activation="relu2",
            quant_mode="w4a16",
        )

        ref_output = compute_reference_moe_relu2(
            hidden_states=tensors["x_bf16"].float().cuda(),
            fc1_weights=tensors["w1_weight_bf16"].float().cuda(),
            fc2_weights=tensors["w2_weight_bf16"].float().cuda(),
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            fc2_input_scale=None,
        )

        passed, percent_within, atol = check_accuracy(result, ref_output)
        assert passed, (
            f"ReLU2 W4A16 direct micro: {percent_within * 100:.2f}% "
            f"within tolerance (atol={atol:.4f})"
        )

    def test_relu2_cuda_graph(self):
        """Test ReLU2 with CUDA graph capture and replay."""
        from flashinfer import B12xMoEWrapper

        num_tokens, hidden_size, intermediate_size = 128, 256, 512
        num_experts, top_k = 256, 2

        tensors = create_relu2_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_local_experts=num_experts,
            top_k=top_k,
        )

        moe = B12xMoEWrapper(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
            activation="relu2",
        )

        # Warmup
        for _ in range(3):
            moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            output = moe.run(
                x=tensors["x_bf16"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                fc2_input_scale=tensors["fc2_input_scale"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
            )
        torch.cuda.synchronize()

        assert output.shape == (num_tokens, hidden_size)

        # Replay and verify
        g.replay()
        torch.cuda.synchronize()
        assert not torch.isnan(output).any(), "NaN after ReLU2 CUDA graph replay"
        assert not (output == 0).all(), "All zeros after ReLU2 CUDA graph replay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
