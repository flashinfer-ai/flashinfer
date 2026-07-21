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

from dataclasses import dataclass
import csv
import importlib
import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import flashinfer.fused_moe as fused_moe_api
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fp4_quantization import block_scale_interleave, fp4_quantize
from flashinfer.fused_moe import (
    ActivationType,
    RoutingMethodType,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
)
from flashinfer.fused_moe import core as moe_core
from flashinfer.fused_moe.dist_aware import (
    da_capture,
    da_core,
    da_profile,
    da_single_graph,
    da_state,
)
from flashinfer.fused_moe.core import (
    DtypeTrtllmGen,
    Fp8QuantizationType,
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.fused_moe.dist_aware.da_utils import (
    generate_da_distribution_assignments,
    get_da_distribution_specs,
)
from flashinfer.utils import get_compute_capability
from tests.moe import test_trtllm_gen_fused_moe as gen_moe_tests
from tests.moe import trtllm_gen_fused_moe_utils as gen_moe_utils
from tests.moe.test_trtllm_gen_fused_moe import (
    BF16Moe,
    FP4Moe,
    FP8BlockScaleMoe,
    FP8PerTensorMoe,
    MxInt4BlockScaleMoe,
    QuantMode,
    run_moe_test,
)

NUM_TOKENS = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 1024
NUM_EXPERTS = 16
TOP_K = 1
N_GROUP = 4
TOPK_GROUP = 4
ROUTED_SCALING_FACTOR = 1.0
SF_VEC_SIZE = 16
TUNE_MAX_NUM_TOKENS = NUM_TOKENS
DA_DISTRIBUTIONS = get_da_distribution_specs("uniform,exp:2,single")
DA_DISTRIBUTION_NAMES = ("uniform", "exp:2", "single")
DA_ROUTING_METHODS = (RoutingMethodType.DeepSeekV3, RoutingMethodType.Renormalize)


@pytest.mark.parametrize("pack_topk_ids", [False, True])
def test_future_default_profile_uses_valid_representative_routing(pack_topk_ids):
    invocation = SimpleNamespace(top_k=8, num_experts=128)
    execution = SimpleNamespace(invocation=invocation)
    generator = da_core.DADistributionTensorGenerator(
        execution, pack_topk_ids=pack_topk_ids
    )
    original = torch.zeros(64, 8, dtype=torch.int32)
    profiled = torch.zeros(512, 8, dtype=torch.int32)

    generated = generator(
        da_core.DEFAULT_PROFILE_VALUE_BUCKET,
        profiled,
        original,
        [],
    )
    expert_ids = generated >> 16 if pack_topk_ids else generated

    assert generated.shape == profiled.shape
    assert int(expert_ids.min()) >= 0
    assert int(expert_ids.max()) < invocation.num_experts
    assert torch.all(expert_ids.sort(dim=1).values.diff(dim=1) != 0)


def test_da_distribution_samples_are_stable_across_tactic_profiles():
    distribution = get_da_distribution_specs("ddist:2")[0]
    invocation = SimpleNamespace(
        top_k=8,
        num_experts=128,
        num_local_experts=128,
        local_expert_offset=0,
    )
    execution = SimpleNamespace(
        invocation=invocation,
        config=da_profile.DAConfig(distributions=(distribution,)),
    )
    generator = da_core.DADistributionTensorGenerator(execution, pack_topk_ids=False)
    profiled = torch.zeros(512, 8, dtype=torch.int32)

    torch.manual_seed(9284)
    expected = generate_da_distribution_assignments(
        distribution,
        profiled,
        invocation.num_local_experts,
        invocation.num_experts,
        invocation.top_k,
        invocation.local_expert_offset,
    )
    torch.manual_seed(9284)
    first = generator(0, profiled, profiled, [], sample_index=3)
    torch.rand(4096)
    repeated = generator(0, profiled, profiled, [], sample_index=3)
    next_sample = generator(0, profiled, profiled, [], sample_index=4)

    assert torch.equal(first, expected)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, next_sample)


def test_da_single_graph_has_no_fp4_launcher_code():
    """The graph injector remains independent of precision-owned launchers."""
    source = inspect.getsource(da_single_graph).lower()
    assert "capture_safe_trtllm_fp4_moe" not in source
    assert "trtllm_fp4_block_scale_moe" not in source


@dataclass
class _FakeCUDAStream:
    cuda_stream: int


def _clear_da_stream_caches() -> None:
    da_single_graph._DA_INLINE_POOL_HANDLES.clear()
    da_single_graph._DA_INLINE_SIDE_STREAMS.clear()
    da_single_graph._DA_INLINE_ROUTING_STREAMS.clear()


def test_da_capture_stream_pool_wrap_is_repaired(monkeypatch):
    """A pooled handle that wraps onto the outer stream is reacquired."""
    _clear_da_stream_caches()
    replacements = iter((_FakeCUDAStream(101),))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: next(replacements))

    side, routing = da_single_graph._validated_capture_streams(
        device=torch.device("cuda", 0),
        outer_stream=_FakeCUDAStream(32),
        side_stream=_FakeCUDAStream(32),
        routing_stream=_FakeCUDAStream(64),
    )

    assert (side.cuda_stream, routing.cuda_stream) == (101, 64)


def test_da_capture_stream_reacquires_internal_counterpart(monkeypatch):
    """A supplied stream is preserved while its internal alias is repaired."""
    _clear_da_stream_caches()
    supplied_side = _FakeCUDAStream(41)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: _FakeCUDAStream(99))

    side, routing = da_single_graph._validated_capture_streams(
        device=torch.device("cuda", 0),
        outer_stream=_FakeCUDAStream(7),
        side_stream=supplied_side,
        routing_stream=_FakeCUDAStream(41),
        side_stream_supplied=True,
    )

    assert side is supplied_side
    assert routing.cuda_stream == 99


def test_da_capture_stream_replacement_is_cached(monkeypatch):
    """A repaired internal stream becomes the stable per-device primitive."""
    _clear_da_stream_caches()
    replacement = _FakeCUDAStream(83)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: replacement)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: "pool")

    repaired_side, repaired_routing = da_single_graph._validated_capture_streams(
        device=torch.device("cuda", 0),
        outer_stream=_FakeCUDAStream(11),
        side_stream=_FakeCUDAStream(11),
        routing_stream=_FakeCUDAStream(72),
    )
    side, routing, pool = da_single_graph.capture_primitives(torch.device("cuda", 0))

    assert side is repaired_side
    assert routing is repaired_routing
    assert pool == "pool"


@pytest.mark.parametrize(
    ("side", "routing", "match"),
    [
        (17, 31, "side_stream aliases the outer"),
        (23, 17, "routing_stream aliases the outer"),
        (23, 23, "auxiliary streams alias each other"),
    ],
)
def test_da_capture_rejects_supplied_stream_aliases(side, routing, match):
    """Framework-owned aliases fail rather than being silently replaced."""
    with pytest.raises(RuntimeError, match=match):
        da_single_graph._validated_capture_streams(
            device=torch.device("cuda", 0),
            outer_stream=_FakeCUDAStream(17),
            side_stream=_FakeCUDAStream(side),
            routing_stream=_FakeCUDAStream(routing),
            side_stream_supplied=True,
            routing_stream_supplied=True,
        )


def test_da_capture_stream_reacquisition_exhaustion_is_diagnostic(monkeypatch):
    """Ten failed repairs report every handle and the retry count."""
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: _FakeCUDAStream(5))
    with pytest.raises(RuntimeError, match="outer=5, side=5, routing=9") as error:
        da_single_graph._validated_capture_streams(
            device=torch.device("cuda", 0),
            outer_stream=_FakeCUDAStream(5),
            side_stream=_FakeCUDAStream(5),
            routing_stream=_FakeCUDAStream(9),
        )
    assert "reacquire_attempts=10" in str(error.value)


def test_da_capture_stream_validation_precedes_ffi_mutation(monkeypatch):
    """An invalid supplied stream cannot partially mutate the CUDA graph."""

    class _FakeTensor:
        is_cuda = True
        dtype = torch.int32
        device = torch.device("cuda", 0)

    class _RecordingFFI:
        mutated = False

        def da_inline_switch_begin_with_handle(self, *_args):
            self.mutated = True
            return 1

    ffi = _RecordingFFI()
    injector = da_single_graph.DAInlineGraphInjector(ffi)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda _device: _FakeCUDAStream(55)
    )

    with (
        pytest.raises(RuntimeError, match="side_stream aliases the outer"),
        injector.inject(
            topk_ids=_FakeTensor(),
            routing_input_mode=2,
            tile_sizes=(16, 32),
            num_tokens_bucket=16,
            num_local_experts=8,
            local_expert_offset=0,
            top_k=1,
            side_stream=_FakeCUDAStream(55),
            routing_stream=_FakeCUDAStream(77),
            pool_handle="pool",
        ),
    ):
        pass
    assert not ffi.mutated


def test_da_missing_metadata_skips_before_backend_graph_mutation(monkeypatch):
    """An unsupported metadata path exits before constructing the FFI backend."""
    backend_accessed = False

    def backend():
        nonlocal backend_accessed
        backend_accessed = True
        raise AssertionError("backend graph mutation must not be reached")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        da_capture,
        "lookup_capture_resources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("metadata unsupported")
        ),
    )
    context = _da_test_context(
        "flashinfer::test_metadata_skip",
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.E2m1,
        Fp8QuantizationType.NoneFp8,
        device=torch.device("cuda", 0),
        top_k=1,
    )

    result = da_capture.try_trtllm_capture_aware_da(
        backend=backend,
        upload_bucket=lambda tokens, _maximum: tokens,
        debug_log=lambda _message: None,
        da_context=context,
        run_from_routing_metadata=lambda _metadata, _tactic: None,
        routing_input_mode=int(moe_core.RoutingInputMode.FromLogits),
        internal_routing_mode=int(moe_core.RoutingInputMode.PackedPrecomputed),
        hidden_states=None,
        hidden_states_scale=None,
        routing_logits=None,
        topk_ids=None,
        expert_weights=None,
        routing_bias=None,
        gemm1_weights=None,
        gemm1_weights_scale=None,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=None,
        gemm2_weights_scale=None,
        gemm2_bias=None,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        output=None,
        num_experts=8,
        top_k=1,
        n_group=None,
        topk_group=None,
        intermediate_size=128,
        local_expert_offset=0,
        num_local_experts=8,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.TopK),
        enable_pdl=False,
        activation_type=int(ActivationType.Swiglu),
        num_tokens=8,
        tune_max_num_tokens=8,
        dtype_act=DtypeTrtllmGen.E2m1,
        norm_topk_prob=True,
    )

    assert result is None
    assert not backend_accessed


@dataclass(frozen=True)
class _DAPrecision:
    name: str
    moe_impl: object
    op_name: str
    dtype_act: moe_core.DtypeTrtllmGen
    dtype_weights: moe_core.DtypeTrtllmGen
    quantization_type: moe_core.Fp8QuantizationType
    match_ratio: float


DA_PRECISION_CONTRACTS = (
    _DAPrecision(
        "bf16",
        BF16Moe(),
        "flashinfer::trtllm_bf16_moe",
        moe_core.DtypeTrtllmGen.Bfloat16,
        moe_core.DtypeTrtllmGen.Bfloat16,
        moe_core.Fp8QuantizationType.NoneFp8,
        0.925,
    ),
    _DAPrecision(
        "fp8_per_tensor",
        FP8PerTensorMoe(),
        "flashinfer::trtllm_fp8_per_tensor_scale_moe",
        moe_core.DtypeTrtllmGen.E4m3,
        moe_core.DtypeTrtllmGen.E4m3,
        moe_core.Fp8QuantizationType.NoneFp8,
        0.92,
    ),
    _DAPrecision(
        "fp8_block",
        FP8BlockScaleMoe(QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
        "flashinfer::trtllm_fp8_block_scale_moe",
        moe_core.DtypeTrtllmGen.E4m3,
        moe_core.DtypeTrtllmGen.E4m3,
        moe_core.Fp8QuantizationType.DeepSeekFp8,
        0.79,
    ),
    _DAPrecision(
        "mxfp8",
        FP8BlockScaleMoe(QuantMode.FP8_BLOCK_SCALE_MXFP8),
        "flashinfer::trtllm_fp8_block_scale_moe",
        moe_core.DtypeTrtllmGen.MxE4m3,
        moe_core.DtypeTrtllmGen.MxE4m3,
        moe_core.Fp8QuantizationType.MxFp8,
        0.79,
    ),
    _DAPrecision(
        "nvfp4",
        FP4Moe(QuantMode.FP4_NVFP4_NVFP4),
        "flashinfer::trtllm_fp4_block_scale_moe",
        moe_core.DtypeTrtllmGen.E2m1,
        moe_core.DtypeTrtllmGen.E2m1,
        moe_core.Fp8QuantizationType.NoneFp8,
        0.92,
    ),
    _DAPrecision(
        "mxfp4_mxfp8",
        FP4Moe(QuantMode.FP4_MXFP4_MXFP8),
        "flashinfer::trtllm_fp4_block_scale_moe",
        moe_core.DtypeTrtllmGen.MxE4m3,
        moe_core.DtypeTrtllmGen.MxE2m1,
        moe_core.Fp8QuantizationType.NoneFp8,
        0.92,
    ),
    _DAPrecision(
        "mxfp4_bf16",
        FP4Moe(QuantMode.FP4_MXFP4_Bf16),
        "flashinfer::trtllm_fp4_block_scale_moe",
        moe_core.DtypeTrtllmGen.Bfloat16,
        moe_core.DtypeTrtllmGen.MxE2m1,
        moe_core.Fp8QuantizationType.NoneFp8,
        0.92,
    ),
    _DAPrecision(
        "mxint4",
        MxInt4BlockScaleMoe(),
        "flashinfer::trtllm_mxint4_block_scale_moe",
        moe_core.DtypeTrtllmGen.Bfloat16,
        moe_core.DtypeTrtllmGen.MxInt4,
        moe_core.Fp8QuantizationType.NoneFp8,
        0.925,
    ),
)

NON_FP4_DA_PRECISION_CONTRACTS = tuple(
    precision
    for precision in DA_PRECISION_CONTRACTS
    if precision.name in {"bf16", "fp8_per_tensor", "fp8_block", "mxfp8", "mxint4"}
)
DA_RUNTIME_PLAN_POLICIES = {
    "da_switch",
    "da_singleton",
}


def _assert_bit_exact(expected: torch.Tensor, actual: torch.Tensor) -> None:
    max_abs_err = float((expected.float() - actual.float()).abs().max().item())
    assert torch.equal(expected, actual), f"max_abs_err={max_abs_err}"


@dataclass
class _MoEState:
    hidden_states: torch.Tensor
    hidden_states_scale: torch.Tensor
    gemm1_weights: torch.Tensor
    gemm1_weights_scale: torch.Tensor
    gemm2_weights: torch.Tensor
    gemm2_weights_scale: torch.Tensor
    scales: torch.Tensor
    routing_bias: torch.Tensor


def _require_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] != 10:
        pytest.skip("requires SM100 family GPU")


def _load_moe_ffi_op():
    return moe_core.get_trtllm_moe_sm100_module().ffi_moe_op


@pytest.mark.parametrize(
    "precision", DA_PRECISION_CONTRACTS, ids=lambda item: item.name
)
def test_factorized_moe_config_query_matches_combined_query(precision):
    """The additive rows preserve combined ordering and FC1×FC2 metadata."""
    _require_sm100()
    module = _load_moe_ffi_op()
    weight_layout = (
        moe_core.WeightLayout.BlockMajorK
        if precision.name == "bf16"
        else moe_core.WeightLayout.MajorK
    )
    query_args = (
        precision.dtype_act,
        precision.dtype_weights,
        precision.quantization_type,
        2,  # top_k
        1024,  # hidden_size
        1024,  # intermediate_size
        8,  # num_local_experts
        int(moe_core.ActivationType.Swiglu),
        True,  # use_shuffled_weight
        int(weight_layout),
        False,  # use_per_token_scaling
        128,  # num_tokens
        False,  # has_gemm1_lora_delta
    )

    combined = [
        tuple(int(value) for value in row)
        for row in module.trtllm_get_valid_moe_configs(*query_args)
    ]
    factorized = [
        tuple(int(value) for value in row)
        for row in module.trtllm_get_valid_moe_factorized_configs(*query_args)
    ]

    assert combined
    assert [row[:2] for row in factorized] == combined
    assert all(len(row) == 5 for row in factorized)
    for tile in {row[0] for row in factorized}:
        tile_rows = [row for row in factorized if row[0] == tile]
        assert sum(row[4] for row in tile_rows) == 1
        fc1_configs = {row[2] for row in tile_rows}
        fc2_configs = {row[3] for row in tile_rows}
        assert {(row[2], row[3]) for row in tile_rows} == {
            (fc1, fc2) for fc1 in fc1_configs for fc2 in fc2_configs
        }


def _da_test_context(
    op_name,
    dtype_act,
    dtype_weights,
    quantization_type=moe_core.Fp8QuantizationType.NoneFp8,
    *,
    device="cuda:0",
    top_k=2,
    hidden_size=1024,
):
    """Build a selector context for the captured-context lifetime test."""
    return da_state.make_context(
        op_name,
        device=device,
        dtype_act=dtype_act,
        dtype_weights=dtype_weights,
        quantization_type=quantization_type,
        top_k=top_k,
        num_experts=8,
        num_local_experts=8,
        local_expert_offset=0,
        hidden_size=hidden_size,
        intermediate_size=1024,
        activation_type=int(moe_core.ActivationType.Swiglu),
        weight_layout=moe_core.WeightLayout.MajorK,
        use_shuffled_weight=True,
    )


def test_da_captured_selector_survives_conflicting_context_upload():
    """Captured selectors retain their own context after a conflicting upload."""
    if os.getenv("_FLASHINFER_DA_SELECTOR_LIFETIME_CHILD") != "1":
        # This low-level test creates CUDA conditional body graphs directly.
        # Keep that process-global capture state out of the public-wrapper
        # matrix, which intentionally creates dozens of unrelated graphs.
        env = os.environ.copy()
        env["_FLASHINFER_DA_SELECTOR_LIFETIME_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_da_captured_selector_survives_conflicting_context_upload",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    _require_sm100()
    ffi = _load_moe_ffi_op()
    injector = object.__new__(da_single_graph.DAInlineGraphInjector)
    injector._ffi = ffi

    # Selector isolation depends on the context handle, not the precision body.
    # One pair with different op/dtype/shape fields proves the lifetime contract
    # without accumulating sixteen conditional graph bodies in one process.
    precision = DA_PRECISION_CONTRACTS[0]
    conflicting_precision = next(
        item for item in DA_PRECISION_CONTRACTS if item.name == "nvfp4"
    )

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    context_a = _da_test_context(
        precision.op_name,
        precision.dtype_act,
        precision.dtype_weights,
        precision.quantization_type,
        device=device,
        top_k=1,
    )
    context_b = _da_test_context(
        conflicting_precision.op_name,
        conflicting_precision.dtype_act,
        conflicting_precision.dtype_weights,
        conflicting_precision.quantization_type,
        device=device,
        top_k=1,
        hidden_size=2048,
    )
    handle_a = da_state.selector_handle(context_a)
    handle_b = da_state.selector_handle(context_b)
    assert handle_a != handle_b

    num_tokens = 128
    num_experts = 8
    tile_sizes = [16, 32]
    single_expert = torch.zeros(num_experts, device=device, dtype=torch.float32)
    single_expert[0] = 1.0
    uniform = torch.full(
        (num_experts,), num_experts**-0.5, device=device, dtype=torch.float32
    )
    exemplars = torch.stack((single_expert, uniform)).flatten()

    def upload(context, body_indices):
        da_profile.upload_exemplars_for_context(
            ffi,
            context,
            exemplars,
            body_indices,
            tile_sizes,
            [0, 1],
            tile_sizes,
            num_experts,
            0,
            1,
            num_tokens,
        )

    topk_ids = torch.zeros((num_tokens, 1), device=device, dtype=torch.int32)

    def capture_selector(selector_handle):
        output = torch.zeros((), device=device, dtype=torch.int32)
        side_stream = torch.cuda.Stream(device=device)
        routing_stream = torch.cuda.Stream(device=device)
        pool_handle = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.graph(graph),
            injector.inject(
                selector_handle=selector_handle,
                topk_ids=topk_ids,
                routing_input_mode=int(moe_core.RoutingInputMode.UnpackedPrecomputed),
                tile_sizes=tile_sizes,
                num_tokens_bucket=num_tokens,
                num_local_experts=num_experts,
                local_expert_offset=0,
                top_k=1,
                side_stream=side_stream,
                routing_stream=routing_stream,
                pool_handle=pool_handle,
            ) as switch,
        ):
            assert switch.num_bodies == 2
            with switch.body(0):
                output.fill_(1)
            with switch.body(1):
                output.fill_(2)
        return graph, output, side_stream, routing_stream

    graph_a = graph_b = None
    try:
        upload(context_a, [0, 1])
        graph_a, output_a, side_a, routing_a = capture_selector(handle_a)
        graph_a.replay()
        torch.cuda.synchronize()
        assert output_a.item() == 1

        upload(context_b, [1, 0])
        graph_b, output_b, side_b, routing_b = capture_selector(handle_b)
        graph_b.replay()
        torch.cuda.synchronize()
        assert output_b.item() == 2

        for _ in range(2):
            output_a.zero_()
            graph_a.replay()
            torch.cuda.synchronize()
            assert output_a.item() == 1
    finally:
        torch.cuda.synchronize()
        # Explicitly release graph executables before selector device state and
        # their capture streams.  CUDA graph nodes retain the selector pointers.
        for graph in (graph_a, graph_b):
            if graph is not None:
                graph.reset()
        del graph_a, graph_b
        ffi.da_destroy_knn_selector(handle_a)
        ffi.da_destroy_knn_selector(handle_b)


def test_da_sparse_register_sort_preserves_zero_tail():
    """Sparse register sorting must not leak sentinels into selector similarity."""
    if os.getenv("_FLASHINFER_DA_SPARSE_REGISTER_SORT_CHILD") != "1":
        # Direct conditional-graph construction leaves process-global CUDA
        # capture state. Isolate it from the public-wrapper numerical matrix.
        env = os.environ.copy()
        env["_FLASHINFER_DA_SPARSE_REGISTER_SORT_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_da_sparse_register_sort_preserves_zero_tail",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    _require_sm100()
    ffi = _load_moe_ffi_op()
    injector = object.__new__(da_single_graph.DAInlineGraphInjector)
    injector._ffi = ffi

    precision = DA_PRECISION_CONTRACTS[0]
    device = torch.device("cuda", torch.cuda.current_device())
    num_local_experts = 128
    num_tokens_bucket = 64
    context = da_state.make_context(
        precision.op_name,
        device=device,
        dtype_act=precision.dtype_act,
        dtype_weights=precision.dtype_weights,
        quantization_type=precision.quantization_type,
        top_k=1,
        num_experts=num_local_experts,
        num_local_experts=num_local_experts,
        local_expert_offset=0,
        hidden_size=1024,
        intermediate_size=1024,
        activation_type=int(moe_core.ActivationType.Swiglu),
        weight_layout=moe_core.WeightLayout.MajorK,
        use_shuffled_weight=True,
    )
    selector_handle = da_state.selector_handle(context)
    tile_sizes = [16, 32]

    # Forty active experts force sort_len=64 and the register-bitonic path.
    # Exemplar 0 is the correct CPU dot-product winner but has a small dense
    # tail; leaked INT_MIN sentinels make exemplar 1 win on the broken kernel.
    dense_tail = torch.cat(
        (
            torch.full((20,), 2.0, device=device),
            torch.ones(20, device=device),
            torch.full((88,), 0.01, device=device),
        )
    )
    sparse = torch.cat((torch.ones(40, device=device), torch.zeros(88, device=device)))
    exemplars = torch.stack(
        (dense_tail / dense_tail.norm(), sparse / sparse.norm())
    ).flatten()
    da_profile.upload_exemplars_for_context(
        ffi,
        context,
        exemplars,
        [0, 1],
        tile_sizes,
        [0, 1],
        tile_sizes,
        num_local_experts,
        0,
        1,
        num_tokens_bucket,
    )

    expert_ids = torch.cat(
        (
            torch.arange(20, device=device, dtype=torch.int32).repeat(2),
            torch.arange(20, 40, device=device, dtype=torch.int32),
        )
    ).reshape(-1, 1)
    expected_counts = torch.cat(
        (
            torch.full((20,), 2.0, device=device),
            torch.ones(20, device=device),
            torch.zeros(88, device=device),
        )
    )
    expected_sims = torch.mv(exemplars.reshape(2, -1), expected_counts)
    assert int(expected_sims.argmax()) == 0

    output = torch.zeros((), device=device, dtype=torch.int32)
    side_stream = torch.cuda.Stream(device=device)
    routing_stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    try:
        with (
            torch.cuda.graph(graph),
            injector.inject(
                selector_handle=selector_handle,
                topk_ids=expert_ids,
                routing_input_mode=int(moe_core.RoutingInputMode.UnpackedPrecomputed),
                tile_sizes=tile_sizes,
                num_tokens_bucket=num_tokens_bucket,
                num_local_experts=num_local_experts,
                local_expert_offset=0,
                top_k=1,
                side_stream=side_stream,
                routing_stream=routing_stream,
                pool_handle=torch.cuda.graph_pool_handle(),
            ) as switch,
        ):
            with switch.body(0):
                output.fill_(1)
            with switch.body(1):
                output.fill_(2)

        graph.replay()
        torch.cuda.synchronize()
        assert output.item() == 1
    finally:
        torch.cuda.synchronize()
        graph.reset()
        del graph
        ffi.da_destroy_knn_selector(selector_handle)


def _fp4_global_scale(tensor: torch.Tensor) -> torch.Tensor:
    return torch.finfo(torch.float8_e4m3fn).max * 6.0 / tensor.float().abs().max()


def _prepare_weight_batch(
    weights: torch.Tensor,
    m_dim: int,
    k_dim: int,
    permute_fn,
    cache: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_rows = []
    sf_rows = []
    for expert_idx in range(weights.shape[0]):
        q, sf = fp4_quantize(
            weights[expert_idx],
            _fp4_global_scale(weights[expert_idx]),
            SF_VEC_SIZE,
            False,
            False,
        )
        q = q.view(torch.uint8).reshape(m_dim, k_dim // 2)
        sf = sf.view(torch.float8_e4m3fn).reshape(m_dim, k_dim // SF_VEC_SIZE)

        perm = permute_fn(cache, q, 128)
        q_rows.append(q[perm.to(q.device)].contiguous())

        sf_u8 = sf.view(torch.uint8)
        sf_perm = permute_fn(cache, sf_u8, 128, SF_VEC_SIZE)
        sf_rows.append(
            block_scale_interleave(sf_u8[sf_perm.to(sf_u8.device)].contiguous())
        )

    q_batch = torch.stack(q_rows)
    sf_batch = (
        torch.stack(sf_rows)
        .view(torch.float8_e4m3fn)
        .reshape(weights.shape[0], m_dim, k_dim // SF_VEC_SIZE)
    )
    return q_batch, sf_batch


def _build_moe_state() -> _MoEState:
    torch.manual_seed(7)
    device = torch.device("cuda")

    hidden_bf16 = (
        torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) / 10
    )
    hidden_fp4, hidden_sf = fp4_quantize(
        hidden_bf16,
        _fp4_global_scale(hidden_bf16),
        SF_VEC_SIZE,
        False,
        True,
    )
    hidden_fp4 = hidden_fp4.view(torch.uint8).reshape(NUM_TOKENS, HIDDEN_SIZE // 2)
    hidden_sf = (
        hidden_sf.view(torch.float8_e4m3fn)
        .flatten()[: NUM_TOKENS * HIDDEN_SIZE // SF_VEC_SIZE]
        .reshape(NUM_TOKENS, HIDDEN_SIZE // SF_VEC_SIZE)
    )

    w1_bf16 = (
        torch.randn(
            NUM_EXPERTS,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    w2_bf16 = (
        torch.randn(
            NUM_EXPERTS,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / 10
    )
    cache = {}
    w1_fp4, w1_sf = _prepare_weight_batch(
        w1_bf16,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        _maybe_get_cached_w3_w1_permute_indices,
        cache,
    )
    w2_fp4, w2_sf = _prepare_weight_batch(
        w2_bf16,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        get_w2_permute_indices_with_cache,
        cache,
    )

    return _MoEState(
        hidden_states=hidden_fp4,
        hidden_states_scale=hidden_sf,
        gemm1_weights=w1_fp4,
        gemm1_weights_scale=w1_sf,
        gemm2_weights=w2_fp4,
        gemm2_weights_scale=w2_sf,
        scales=torch.ones(NUM_EXPERTS, device=device, dtype=torch.float32),
        routing_bias=torch.zeros(NUM_EXPERTS, device=device, dtype=torch.bfloat16),
    )


def _assignments_for_distribution(distribution) -> torch.Tensor:
    return generate_da_distribution_assignments(
        distribution,
        torch.zeros(NUM_TOKENS, TOP_K, dtype=torch.int32, device="cuda"),
        NUM_EXPERTS,
        NUM_EXPERTS,
        TOP_K,
        0,
    )


def _routing_logits_for_distribution(distribution) -> torch.Tensor:
    """Construct deterministic logits whose top-k follows a DA distribution."""
    topk_ids = _assignments_for_distribution(distribution).to(torch.long)
    logits = torch.full(
        (NUM_TOKENS, NUM_EXPERTS),
        -20.0,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    values = torch.linspace(20.0, 19.0, steps=TOP_K, device=topk_ids.device).reshape(
        1, TOP_K
    )
    logits.scatter_(1, topk_ids, values.expand(NUM_TOKENS, -1))
    return logits


def _matrix_moe_impl(name: str):
    return {
        "bf16": BF16Moe,
        "fp8_per_tensor": FP8PerTensorMoe,
        "fp8_block": lambda: FP8BlockScaleMoe(QuantMode.FP8_BLOCK_SCALE_DEEPSEEK),
        "mxfp8": lambda: FP8BlockScaleMoe(QuantMode.FP8_BLOCK_SCALE_MXFP8),
        "nvfp4": lambda: FP4Moe(QuantMode.FP4_NVFP4_NVFP4),
        "mxfp4_mxfp8": lambda: FP4Moe(QuantMode.FP4_MXFP4_MXFP8),
        "mxfp4_bf16": lambda: FP4Moe(QuantMode.FP4_MXFP4_Bf16),
        "mxint4": MxInt4BlockScaleMoe,
    }[name]()


def _matrix_logits_transform(distribution_name: str, top_k: int):
    distribution = next(
        item for item in DA_DISTRIBUTIONS if item[0] == distribution_name
    )

    def transform(logits: torch.Tensor) -> torch.Tensor:
        assignments = generate_da_distribution_assignments(
            distribution,
            torch.zeros(
                logits.shape[0], top_k, dtype=torch.int32, device=logits.device
            ),
            logits.shape[1],
            logits.shape[1],
            top_k,
            0,
        ).to(torch.long)
        routed = torch.full_like(logits, -4)
        values = torch.linspace(
            4, 3, steps=top_k, dtype=logits.dtype, device=logits.device
        ).reshape(1, top_k)
        return routed.scatter_(1, assignments, values.expand(logits.shape[0], -1))

    return transform


def _near_tie_logits(logits: torch.Tensor) -> torch.Tensor:
    """Create deterministic logits separated by only a few BF16 ULPs."""
    expert_offsets = torch.linspace(
        -0.01,
        0.01,
        logits.shape[1],
        dtype=logits.dtype,
        device=logits.device,
    )
    token_offsets = (torch.arange(logits.shape[0], device=logits.device) % 3).to(
        logits.dtype
    )
    return expert_offsets.unsqueeze(0) + token_offsets.unsqueeze(1) * 1.0e-4


def _warmup_matrix_fp4_da(execution) -> None:
    moe_impl, static_data, hidden_states_orig, hidden_scale_global, kwargs = execution
    inputs = moe_impl.quantize_inputs(
        hidden_states_orig, hidden_scale_global, is_swizzling=False
    )
    with autotune(True):
        trtllm_fp4_block_scale_moe(
            routing_logits=kwargs["expert_logits"],
            routing_bias=kwargs["routing_bias"],
            hidden_states=inputs["hidden_states"],
            hidden_states_scale=inputs["hidden_states_scale"],
            gemm1_weights=static_data["gemm1_weights_fp4_shuffled"],
            gemm1_weights_scale=static_data["gemm1_scales_fp4_shuffled"],
            gemm1_bias=kwargs["gemm1_bias"],
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=static_data["gemm2_weights_fp4_shuffled"],
            gemm2_weights_scale=static_data["gemm2_scales_fp4_shuffled"],
            gemm2_bias=kwargs["gemm2_bias"],
            output1_scale_scalar=static_data["scale_c_fc1"],
            output1_scale_gate_scalar=static_data["scale_gate_fc1"],
            output2_scale_scalar=static_data["scale_c_fc2"],
            num_experts=kwargs["num_experts"],
            top_k=kwargs["top_k"],
            n_group=kwargs["n_groups"],
            topk_group=kwargs["top_k_groups"],
            intermediate_size=kwargs["intermediate_size"],
            local_expert_offset=0,
            local_num_experts=kwargs["num_experts"],
            routed_scaling_factor=kwargs["routed_scaling"],
            routing_method_type=kwargs["routing_method_type"],
            activation_type=kwargs["activation_type"],
            tune_max_num_tokens=gen_moe_utils.TUNE_MAX_NUM_TOKENS,
            norm_topk_prob=kwargs["norm_topk_prob"],
        )


def _run_matrix_fp4_logits(
    execution, output: torch.Tensor | None = None
) -> torch.Tensor:
    moe_impl, static_data, hidden_states_orig, hidden_scale_global, kwargs = execution
    inputs = moe_impl.quantize_inputs(
        hidden_states_orig, hidden_scale_global, is_swizzling=False
    )
    return trtllm_fp4_block_scale_moe(
        routing_logits=kwargs["expert_logits"],
        routing_bias=kwargs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=static_data["gemm1_weights_fp4_shuffled"],
        gemm1_weights_scale=static_data["gemm1_scales_fp4_shuffled"],
        gemm1_bias=kwargs["gemm1_bias"],
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=static_data["gemm2_weights_fp4_shuffled"],
        gemm2_weights_scale=static_data["gemm2_scales_fp4_shuffled"],
        gemm2_bias=kwargs["gemm2_bias"],
        output1_scale_scalar=static_data["scale_c_fc1"],
        output1_scale_gate_scalar=static_data["scale_gate_fc1"],
        output2_scale_scalar=static_data["scale_c_fc2"],
        num_experts=kwargs["num_experts"],
        top_k=kwargs["top_k"],
        n_group=kwargs["n_groups"],
        topk_group=kwargs["top_k_groups"],
        intermediate_size=kwargs["intermediate_size"],
        local_expert_offset=0,
        local_num_experts=kwargs["num_experts"],
        routed_scaling_factor=kwargs["routed_scaling"],
        routing_method_type=kwargs["routing_method_type"],
        activation_type=kwargs["activation_type"],
        output=output,
        tune_max_num_tokens=64,
        norm_topk_prob=kwargs["norm_topk_prob"],
    )[0]


def _run_matrix_fp4_routed(
    execution,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    moe_impl, static_data, hidden_states_orig, hidden_scale_global, kwargs = execution
    inputs = moe_impl.quantize_inputs(
        hidden_states_orig, hidden_scale_global, is_swizzling=False
    )
    return trtllm_fp4_block_scale_routed_moe(
        (expert_ids, expert_weights),
        None,
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        static_data["gemm1_weights_fp4_shuffled"],
        static_data["gemm1_scales_fp4_shuffled"],
        kwargs["gemm1_bias"],
        None,
        None,
        None,
        static_data["gemm2_weights_fp4_shuffled"],
        static_data["gemm2_scales_fp4_shuffled"],
        kwargs["gemm2_bias"],
        static_data["scale_c_fc1"],
        static_data["scale_gate_fc1"],
        static_data["scale_c_fc2"],
        kwargs["num_experts"],
        kwargs["top_k"],
        None,
        None,
        kwargs["intermediate_size"],
        0,
        kwargs["num_experts"],
        None,
        int(RoutingMethodType.TopK),
        activation_type=int(kwargs["activation_type"]),
        output=output,
        tune_max_num_tokens=64,
    )[0]


def _run_matrix_same_tactic(precision, execution, replay: bool = False) -> None:
    moe_impl, static_data, hidden_states_orig, hidden_scale_global, kwargs = execution
    module = _load_moe_ffi_op()

    if isinstance(moe_impl, FP4Moe):
        inputs = moe_impl.quantize_inputs(
            hidden_states_orig, hidden_scale_global, is_swizzling=False
        )
    else:
        inputs = moe_impl.quantize_inputs(hidden_states_orig, hidden_scale_global)
    hidden_states = inputs["hidden_states"]
    hidden_states_scale = inputs["hidden_states_scale"]
    if isinstance(moe_impl, FP8BlockScaleMoe):
        hidden_states = kwargs["hidden_states_quant"]
        hidden_states_scale = kwargs["hidden_states_scale"]

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states_orig.shape[1]
    num_experts = kwargs["num_experts"]
    top_k = kwargs["top_k"]
    intermediate_size = kwargs["intermediate_size"]
    activation_type = int(kwargs["activation_type"])
    use_shuffled_weight = static_data.get("use_shuffled_weight", True)
    weight_layout = int(static_data.get("weight_layout", moe_core.WeightLayout.MajorK))
    tactics = list(
        module.trtllm_get_valid_moe_configs(
            precision.dtype_act,
            precision.dtype_weights,
            precision.quantization_type,
            top_k,
            hidden_size,
            intermediate_size,
            num_experts,
            activation_type,
            use_shuffled_weight,
            weight_layout,
            False,
            num_tokens,
            False,
        )
    )
    assert tactics
    selected_tactic = next((item for item in tactics if int(item[0]) == 32), tactics[0])
    tactic = [int(value) for value in selected_tactic]

    packed_topk_ids = torch.empty(
        num_tokens, top_k, dtype=torch.int32, device=hidden_states.device
    )
    metadata = [
        torch.from_dlpack(tensor)
        for tensor in module.trtllm_moe_allocate_routing_metadata_from_logits(
            kwargs["expert_logits"],
            kwargs["routing_bias"],
            num_experts,
            top_k,
            kwargs["n_groups"],
            kwargs["top_k_groups"],
            0,
            num_experts,
            kwargs["routed_scaling"],
            int(kwargs["routing_method_type"]),
            tactic[0],
            True,
            False,
            packed_topk_ids,
        )
    ]
    monolithic = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=hidden_states.device
    )
    split = torch.empty_like(monolithic)
    expert_weights = metadata[3]
    no_expert_weights = torch.empty(
        0, dtype=torch.bfloat16, device=hidden_states.device
    )
    no_topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
    extracted_weights = torch.empty_like(expert_weights)
    common = (
        num_experts,
        top_k,
        kwargs["n_groups"],
        kwargs["top_k_groups"],
        intermediate_size,
        0,
        num_experts,
        kwargs["routed_scaling"],
        int(kwargs["routing_method_type"]),
    )

    if precision.name == "bf16":
        module.trtllm_bf16_moe(
            kwargs["expert_logits"],
            kwargs["routing_bias"],
            no_topk_ids,
            no_expert_weights,
            hidden_states,
            static_data["gemm1_weights"],
            static_data["gemm2_weights"],
            None,
            None,
            None,
            None,
            monolithic,
            *common,
            use_shuffled_weight,
            weight_layout,
            True,
            False,
            tactic,
            activation_type,
            True,
            None,
        )

        def run_split():
            module.trtllm_bf16_moe_run_from_routing_metadata(
                metadata,
                hidden_states,
                static_data["gemm1_weights"],
                static_data["gemm2_weights"],
                num_experts,
                top_k,
                intermediate_size,
                0,
                num_experts,
                use_shuffled_weight,
                weight_layout,
                True,
                False,
                split,
                tactic,
                activation_type,
            )
    elif precision.name == "fp8_per_tensor":
        module.trtllm_fp8_per_tensor_scale_moe(
            kwargs["expert_logits"],
            kwargs["routing_bias"],
            hidden_states,
            static_data["gemm1_weights"],
            static_data["scale_c_fc1"],
            static_data["scale_gate_fc1"],
            static_data["gemm2_weights"],
            static_data["scale_c_fc2"],
            monolithic,
            *common[:-1],
            False,
            common[-1],
            True,
            False,
            tactic,
            activation_type,
            True,
            None,
        )

        def run_split():
            module.trtllm_fp8_per_tensor_scale_moe_run_from_routing_metadata(
                metadata,
                hidden_states,
                static_data["gemm1_weights"],
                static_data["scale_c_fc1"],
                static_data["scale_gate_fc1"],
                static_data["gemm2_weights"],
                static_data["scale_c_fc2"],
                num_experts,
                top_k,
                intermediate_size,
                0,
                num_experts,
                False,
                True,
                False,
                split,
                tactic,
                activation_type,
            )
    elif precision.name in {"fp8_block", "mxfp8"}:
        module.trtllm_fp8_block_scale_moe(
            kwargs["expert_logits"],
            no_topk_ids,
            no_expert_weights,
            kwargs["routing_bias"],
            hidden_states,
            hidden_states_scale,
            static_data["gemm1_weights"],
            static_data["gemm1_scales"],
            None,
            None,
            None,
            None,
            static_data["gemm2_weights"],
            static_data["gemm2_scales"],
            monolithic,
            *common[:2],
            0,
            *common[2:],
            use_shuffled_weight,
            weight_layout,
            True,
            False,
            tactic,
            precision.quantization_type,
            activation_type,
            True,
            None,
        )

        def run_split():
            module.trtllm_fp8_block_scale_moe_run_from_routing_metadata(
                metadata,
                hidden_states,
                hidden_states_scale,
                static_data["gemm1_weights"],
                static_data["gemm1_scales"],
                static_data["gemm2_weights"],
                static_data["gemm2_scales"],
                num_experts,
                top_k,
                intermediate_size,
                0,
                num_experts,
                use_shuffled_weight,
                weight_layout,
                True,
                False,
                split,
                tactic,
                precision.quantization_type,
                activation_type,
            )
    elif precision.name in {"nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"}:
        module.trtllm_fp4_block_scale_moe(
            int(moe_core.RoutingInputMode.FromLogits),
            kwargs["expert_logits"],
            packed_topk_ids,
            extracted_weights,
            kwargs["routing_bias"],
            hidden_states,
            hidden_states_scale,
            static_data["gemm1_weights_fp4_shuffled"],
            static_data["gemm1_scales_fp4_shuffled"],
            None,
            None,
            None,
            None,
            static_data["gemm2_weights_fp4_shuffled"],
            static_data["gemm2_scales_fp4_shuffled"],
            None,
            static_data["scale_c_fc1"],
            static_data["scale_gate_fc1"],
            static_data["scale_c_fc2"],
            None,
            *common,
            True,
            False,
            activation_type,
            monolithic,
            tactic,
            True,
            None,
        )

        def run_split():
            module.trtllm_fp4_block_scale_moe_run_from_routing_metadata(
                metadata,
                hidden_states,
                hidden_states_scale,
                static_data["gemm1_weights_fp4_shuffled"],
                static_data["gemm1_scales_fp4_shuffled"],
                None,
                None,
                None,
                None,
                static_data["gemm2_weights_fp4_shuffled"],
                static_data["gemm2_scales_fp4_shuffled"],
                None,
                static_data["scale_c_fc1"],
                static_data["scale_gate_fc1"],
                static_data["scale_c_fc2"],
                num_experts,
                top_k,
                intermediate_size,
                0,
                num_experts,
                True,
                False,
                activation_type,
                split,
                tactic,
            )
    else:
        module.trtllm_mxint4_block_scale_moe(
            kwargs["expert_logits"],
            kwargs["routing_bias"],
            no_topk_ids,
            no_expert_weights,
            hidden_states,
            static_data["gemm1_weights"],
            static_data["gemm1_scales"],
            None,
            None,
            None,
            None,
            static_data["gemm2_weights"],
            static_data["gemm2_scales"],
            *common,
            True,
            False,
            monolithic,
            tactic,
            True,
            None,
        )

        def run_split():
            module.trtllm_mxint4_block_scale_moe_run_from_routing_metadata(
                metadata,
                hidden_states,
                static_data["gemm1_weights"],
                static_data["gemm1_scales"],
                None,
                None,
                None,
                static_data["gemm2_weights"],
                static_data["gemm2_scales"],
                num_experts,
                top_k,
                intermediate_size,
                0,
                num_experts,
                True,
                False,
                split,
                tactic,
            )

    run_split()
    torch.cuda.synchronize()
    _assert_bit_exact(monolithic, split)

    if replay:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_split()
        for _ in range(2):
            split.zero_()
            graph.replay()
            torch.cuda.synchronize()
            _assert_bit_exact(monolithic, split)
        # This helper runs dozens of graph cases before the public DA capture
        # tests.  Explicit reset prevents completed graph executables from
        # retaining capture-stream state until cyclic Python GC runs.
        graph.reset()
        del graph
        torch.cuda.synchronize()


@pytest.mark.parametrize(
    ("precision", "distribution_name", "routing_method"),
    [
        (precision, distribution_name, routing_method)
        for precision in DA_PRECISION_CONTRACTS
        for distribution_name in DA_DISTRIBUTION_NAMES
        for routing_method in DA_ROUTING_METHODS
    ],
    ids=lambda item: item.name if hasattr(item, "name") else str(item),
)
def test_da_all_precision_production_reference_matrix(
    monkeypatch, precision, distribution_name, routing_method
):
    _require_sm100()
    monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", ",".join(DA_DISTRIBUTION_NAMES))

    moe_impl = _matrix_moe_impl(precision.name)
    top_k = 2
    deepseek = routing_method == RoutingMethodType.DeepSeekV3
    block_major = precision.name in {"bf16", "fp8_block", "mxint4"}
    execution = []
    output_reference, output_actual, _ = run_moe_test(
        num_tokens=64,
        hidden_size=1024,
        intermediate_size=1024,
        moe_impl=moe_impl,
        routing_config={
            "num_experts": 16,
            "top_k": top_k,
            "padding": 8,
            "n_groups": 4 if deepseek else None,
            "top_k_groups": 2 if deepseek else None,
            "routed_scaling": 1.0 if deepseek else None,
            "has_routing_bias": deepseek,
            "routing_method_type": routing_method,
            "compatible_moe_impls": [type(moe_impl)],
            "compatible_intermediate_size": [1024],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": (
                moe_core.WeightLayout.BlockMajorK
                if block_major
                else moe_core.WeightLayout.MajorK
            ),
            "compatible_moe_impls": [type(moe_impl)],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices={},
        routing_logits_dtype=torch.bfloat16,
        zero_hidden_states=False,
        expert_logits_transform=_matrix_logits_transform(distribution_name, top_k),
        execution_observer=lambda *args: execution.append(args),
    )

    assert torch.isfinite(output_reference).all()
    assert torch.isfinite(output_actual).all()
    assert len(execution) == 1
    replay = (
        distribution_name == "uniform"
        and routing_method == RoutingMethodType.DeepSeekV3
    )
    _run_matrix_same_tactic(precision, execution[0], replay=replay)


@pytest.mark.parametrize(
    "precision",
    [
        precision
        for precision in DA_PRECISION_CONTRACTS
        if precision.name in {"nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"}
    ],
    ids=lambda item: item.name,
)
@pytest.mark.parametrize(
    "weight_dtype",
    [torch.bfloat16, torch.float32],
    ids=["bf16_weights", "fp32_weights"],
)
def test_fp4_unpacked_public_wrapper_da_graph_is_numerically_stable(
    monkeypatch, precision, weight_dtype
):
    """Every supported FP4 activation mode captures raw routing numerically."""
    _require_sm100()
    monkeypatch.setattr(gen_moe_utils, "TUNE_MAX_NUM_TOKENS", 64)
    tuner = _reset_tuner_and_da(monkeypatch)
    graph = None
    try:
        monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", "uniform,single")
        monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTION_SAMPLES", "1")
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
        execution = []
        moe_impl = _matrix_moe_impl(precision.name)
        run_moe_test(
            num_tokens=64,
            hidden_size=1024,
            intermediate_size=1024,
            moe_impl=moe_impl,
            routing_config={
                "num_experts": 16,
                "top_k": 2,
                "padding": 8,
                "n_groups": None,
                "top_k_groups": None,
                "routed_scaling": None,
                "has_routing_bias": False,
                "routing_method_type": RoutingMethodType.TopK,
                "compatible_moe_impls": [type(moe_impl)],
                "compatible_intermediate_size": [1024],
                "compatible_activation_types": [ActivationType.Swiglu],
                "enable_autotune": False,
            },
            weight_processing={
                "use_shuffled_weight": True,
                "layout": moe_core.WeightLayout.MajorK,
                "compatible_moe_impls": [type(moe_impl)],
            },
            activation_type=ActivationType.Swiglu,
            cache_permute_indices={},
            routing_logits_dtype=torch.bfloat16,
            zero_hidden_states=False,
            execution_observer=lambda *args: execution.append(args),
        )
        assert len(execution) == 1
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
        kwargs = execution[0][-1]
        expert_ids = (
            kwargs["expert_logits"].topk(kwargs["top_k"], dim=1).indices.to(torch.int32)
        )
        expert_weights = torch.linspace(
            0.125,
            0.875,
            expert_ids.numel(),
            dtype=weight_dtype,
            device=expert_ids.device,
        ).reshape_as(expert_ids)

        with autotune(True):
            tuned = _run_matrix_fp4_routed(execution[0], expert_ids, expert_weights)
        torch.cuda.synchronize()
        assert torch.isfinite(tuned).all()
        da_context = da_state.make_context(
            precision.op_name,
            device=expert_ids.device,
            dtype_act=precision.dtype_act,
            dtype_weights=precision.dtype_weights,
            quantization_type=precision.quantization_type,
            top_k=2,
            num_experts=16,
            num_local_experts=16,
            local_expert_offset=0,
            hidden_size=1024,
            intermediate_size=1024,
            activation_type=int(ActivationType.Swiglu),
            weight_layout=moe_core.WeightLayout.MajorK,
            use_shuffled_weight=True,
        )
        body_tactics = da_state.PER_BODY_TACTICS[da_state.cache_key(da_context, 64)]
        assert body_tactics

        graph_output = torch.empty_like(tuned)
        _run_matrix_fp4_routed(execution[0], expert_ids, expert_weights, graph_output)
        torch.cuda.synchronize()
        before = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_matrix_fp4_routed(
                execution[0], expert_ids, expert_weights, graph_output
            )
        after = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
        assert after == before + 1
        resources = next(
            item
            for item in da_capture.CAPTURE_RESOURCES.values()
            if item.da_context.dtype_act == int(precision.dtype_act)
            and item.da_context.dtype_weights == int(precision.dtype_weights)
            and item.input_routing_mode
            == int(moe_core.RoutingInputMode.UnpackedPrecomputed)
        )
        assert resources.internal_routing_mode == int(
            moe_core.RoutingInputMode.UnpackedPrecomputed
        )
        assert resources.topk_ids.data_ptr() == expert_ids.data_ptr()
        assert resources.topk_weights.data_ptr() == expert_weights.data_ptr()
        for _, bundle in resources.routing_metadata_by_tile:
            assert bundle.tensors[3].data_ptr() == expert_weights.data_ptr()

        expert_weights.copy_(torch.flip(expert_weights, dims=(0,)))
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
        reference = _run_matrix_fp4_routed(execution[0], expert_ids, expert_weights)
        torch.cuda.synchronize()
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, reference, rtol=0, atol=0)
    finally:
        if graph is not None:
            graph.reset()
            del graph
        _clear_da_test_state(tuner)


@pytest.mark.parametrize(
    "precision",
    [
        precision
        for precision in DA_PRECISION_CONTRACTS
        if precision.name in {"nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"}
    ],
    ids=lambda item: item.name,
)
def test_fp4_monolithic_from_logits_matches_independent_reference(precision):
    """NoDA owns independent coverage of the monolithic FromLogits launcher."""
    _require_sm100()
    execution = []
    moe_impl = _matrix_moe_impl(precision.name)
    reference, actual, _ = run_moe_test(
        num_tokens=16,
        hidden_size=1024,
        intermediate_size=1024,
        moe_impl=moe_impl,
        routing_config={
            "num_experts": 16,
            "top_k": 2,
            "padding": 8,
            "n_groups": 4,
            "top_k_groups": 2,
            "routed_scaling": 1.0,
            "has_routing_bias": True,
            "routing_method_type": RoutingMethodType.DeepSeekV3,
            "compatible_moe_impls": [type(moe_impl)],
            "compatible_intermediate_size": [1024],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": False,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": moe_core.WeightLayout.MajorK,
            "compatible_moe_impls": [type(moe_impl)],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices={},
        routing_logits_dtype=torch.bfloat16,
        zero_hidden_states=False,
        expert_logits_transform=_matrix_logits_transform("uniform", 2),
        execution_observer=lambda *args: execution.append(args),
    )
    assert len(execution) == 1
    monolithic = _run_matrix_fp4_logits(execution[0])
    torch.cuda.synchronize()
    _assert_precision_reference(precision, reference, actual)
    _assert_precision_reference(precision, reference, monolithic.float())


@pytest.mark.parametrize(
    "precision",
    [
        precision
        for precision in DA_PRECISION_CONTRACTS
        if precision.name in {"nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"}
    ],
    ids=lambda item: item.name,
)
def test_fp4_logits_da_graph_replays_packed_metadata(monkeypatch, precision):
    """DA routes near-tie logits once and replays every FP4 body from metadata."""
    _require_sm100()
    monkeypatch.setattr(gen_moe_utils, "TUNE_MAX_NUM_TOKENS", 64)
    tuner = _reset_tuner_and_da(monkeypatch)
    graph = None
    try:
        monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", "ddist:0.1,ddist:10")
        monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTION_SAMPLES", "1")
        monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "0")
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
        execution = []
        moe_impl = _matrix_moe_impl(precision.name)
        run_moe_test(
            num_tokens=64,
            hidden_size=1024,
            intermediate_size=1024,
            moe_impl=moe_impl,
            routing_config={
                "num_experts": 16,
                "top_k": 2,
                "padding": 8,
                "n_groups": 4,
                "top_k_groups": 2,
                "routed_scaling": 1.0,
                "has_routing_bias": True,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "compatible_moe_impls": [type(moe_impl)],
                "compatible_intermediate_size": [1024],
                "compatible_activation_types": [ActivationType.Swiglu],
                "enable_autotune": False,
            },
            weight_processing={
                "use_shuffled_weight": True,
                "layout": moe_core.WeightLayout.MajorK,
                "compatible_moe_impls": [type(moe_impl)],
            },
            activation_type=ActivationType.Swiglu,
            cache_permute_indices={},
            routing_logits_dtype=torch.bfloat16,
            zero_hidden_states=False,
            expert_logits_transform=_near_tie_logits,
            execution_observer=lambda *args: execution.append(args),
        )
        assert len(execution) == 1
        monolithic_reference = _run_matrix_fp4_logits(execution[0]).detach().clone()
        torch.cuda.synchronize()
        monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
        with autotune(True):
            tuned = _run_matrix_fp4_logits(execution[0])
        torch.cuda.synchronize()
        assert torch.isfinite(tuned).all()

        # Autotuning may legitimately deduplicate to one body on a fast test
        # shape. Publish two already-tuned tactics against deterministic
        # exemplars so this acceptance case always exercises the real SWITCH.
        config_key = next(
            key
            for key in da_state.PER_TILE_TACTICS
            if key[0] == 64
            and key[1].dtype_act == int(precision.dtype_act)
            and key[1].dtype_weights == int(precision.dtype_weights)
        )
        da_context = config_key[1]
        tactics_by_tile = {}
        for cache_key, (_, tactic, _) in tuner.profiling_cache.items():
            op_name = (
                cache_key.custom_op if hasattr(cache_key, "custom_op") else cache_key[0]
            )
            if (
                op_name == precision.op_name
                and hasattr(tactic, "__getitem__")
                and len(tactic) >= 2
            ):
                tactics_by_tile.setdefault(
                    int(tactic[0]), (int(tactic[0]), int(tactic[1]))
                )
        assert len(tactics_by_tile) >= 2
        body_tactics = list(tactics_by_tile.values())[:2]
        uniform = torch.full((16,), 16**-0.5, dtype=torch.float32, device=tuned.device)
        concentrated = torch.zeros_like(uniform)
        concentrated[0] = 1.0
        da_profile.upload_and_publish_selector_tactics(
            moe_core.get_trtllm_moe_sm100_module().ffi_moe_op,
            da_context,
            torch.stack((uniform, concentrated)).flatten(),
            [0, 1],
            [int(tactic[0]) for tactic in body_tactics],
            [int(tactic[1]) for tactic in body_tactics],
            sorted({int(tactic[0]) for tactic in body_tactics}),
            16,
            0,
            2,
            64,
            per_tile_tactics={int(tactic[0]): tactic for tactic in body_tactics},
            per_body_tactics=body_tactics,
        )
        da_capture.CAPTURE_RESOURCES.clear()

        output = torch.empty_like(monolithic_reference)
        _run_matrix_fp4_logits(execution[0], output)
        torch.cuda.synchronize()
        resources = next(
            item
            for item in da_capture.CAPTURE_RESOURCES.values()
            if item.da_context.dtype_act == int(precision.dtype_act)
            and item.input_routing_mode == int(moe_core.RoutingInputMode.FromLogits)
        )
        assert resources.internal_routing_mode == int(
            moe_core.RoutingInputMode.PackedPrecomputed
        )
        assert resources.topk_ids.dtype == torch.int32
        decoded_topk_ids = torch.bitwise_right_shift(resources.topk_ids, 16)
        assert int(decoded_topk_ids.min()) >= 0
        assert int(decoded_topk_ids.max()) < 16
        assert resources.topk_weights.dtype == torch.bfloat16
        assert all(
            bundle.routing_input_mode
            == int(moe_core.RoutingInputMode.PackedPrecomputed)
            for _, bundle in resources.routing_metadata_by_tile
        )
        assert len(resources.per_body_tactics) == 2
        assert len(set(resources.candidate_tile_sizes)) == 2

        before = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run_matrix_fp4_logits(execution[0], output)
        after = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
        assert after == before + 1
        profile_replay = os.environ.get("FLASHINFER_DA_TEST_PROFILE_REPLAY") == "1"
        if profile_replay:
            torch.cuda.profiler.start()
        try:
            graph.replay()
            torch.cuda.synchronize()
        finally:
            if profile_replay:
                torch.cuda.profiler.stop()
        torch.testing.assert_close(output, monolithic_reference, rtol=0, atol=0)
    finally:
        if graph is not None:
            graph.reset()
            del graph
        _clear_da_test_state(tuner)


def _run_public_moe(state: _MoEState, routing_logits: torch.Tensor) -> torch.Tensor:
    """Run the public FP4 logits wrapper used by graph acceptance tests."""
    return trtllm_fp4_block_scale_moe(
        routing_logits=routing_logits,
        routing_bias=state.routing_bias,
        hidden_states=state.hidden_states,
        hidden_states_scale=state.hidden_states_scale,
        gemm1_weights=state.gemm1_weights,
        gemm1_weights_scale=state.gemm1_weights_scale,
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=state.gemm2_weights,
        gemm2_weights_scale=state.gemm2_weights_scale,
        gemm2_bias=None,
        output1_scale_scalar=state.scales,
        output1_scale_gate_scalar=state.scales,
        output2_scale_scalar=state.scales,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=INTERMEDIATE_SIZE,
        local_expert_offset=0,
        local_num_experts=NUM_EXPERTS,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        routing_method_type=int(RoutingMethodType.DeepSeekV3),
        do_finalize=True,
        enable_pdl=False,
        activation_type=int(ActivationType.Swiglu.value),
        tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
    )[0]


def _run_public_routed_moe(
    state: _MoEState,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    return trtllm_fp4_block_scale_routed_moe(
        (expert_ids, expert_weights),
        None,
        state.hidden_states,
        state.hidden_states_scale,
        state.gemm1_weights,
        state.gemm1_weights_scale,
        None,
        None,
        None,
        None,
        state.gemm2_weights,
        state.gemm2_weights_scale,
        None,
        state.scales,
        state.scales,
        state.scales,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        INTERMEDIATE_SIZE,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        output=output,
        tune_max_num_tokens=TUNE_MAX_NUM_TOKENS,
    )[0]


def test_nvfp4_per_token_metadata_path_is_same_tactic_bit_exact():
    _require_sm100()
    state = _build_moe_state()
    expert_ids = _assignments_for_distribution(DA_DISTRIBUTIONS[1])
    expert_weights = torch.rand(
        NUM_TOKENS, TOP_K, dtype=torch.bfloat16, device=expert_ids.device
    )
    packed_topk_ids = (expert_ids << 16) | (
        expert_weights.view(torch.int16).to(torch.int32) & 0xFFFF
    )
    per_token_scale = torch.linspace(
        0.75, 1.25, NUM_TOKENS, dtype=torch.float32, device=expert_ids.device
    )
    module = _load_moe_ffi_op()

    tactics = list(
        module.trtllm_get_valid_moe_configs(
            DtypeTrtllmGen.E2m1,
            DtypeTrtllmGen.E2m1,
            Fp8QuantizationType.NoneFp8,
            TOP_K,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            NUM_EXPERTS,
            int(ActivationType.Swiglu),
            True,
            int(moe_core.WeightLayout.MajorK),
            True,
            NUM_TOKENS,
            False,
        )
    )
    if not tactics:
        pytest.skip("no per-token FP4 tactics advertised")
    tactic = list(tactics[0])
    metadata = fused_moe_api.trtllm_moe_allocate_routing_metadata(
        packed_topk_ids,
        None,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        tactic[0],
    )
    multi_tile_metadata = fused_moe_api.trtllm_moe_allocate_routing_metadata_multi_tile(
        packed_topk_ids,
        None,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        [tactic[0]],
    )
    assert len(multi_tile_metadata) == 1
    monolithic_output = torch.empty(
        NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device=expert_ids.device
    )
    split_output = torch.empty_like(monolithic_output)
    multi_tile_split_output = torch.empty_like(monolithic_output)
    extracted_weights = torch.empty_like(expert_weights)

    module.trtllm_fp4_block_scale_moe(
        int(moe_core.RoutingInputMode.PackedPrecomputed),
        None,
        packed_topk_ids,
        extracted_weights,
        None,
        state.hidden_states,
        state.hidden_states_scale,
        state.gemm1_weights,
        state.gemm1_weights_scale,
        None,
        None,
        None,
        None,
        state.gemm2_weights,
        state.gemm2_weights_scale,
        None,
        state.scales,
        state.scales,
        state.scales,
        per_token_scale,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        INTERMEDIATE_SIZE,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        True,
        False,
        int(ActivationType.Swiglu),
        monolithic_output,
        tactic,
        True,
        None,
    )
    module.trtllm_fp4_block_scale_moe_run_from_routing_metadata_with_per_token_scale(
        metadata,
        state.hidden_states,
        state.hidden_states_scale,
        state.gemm1_weights,
        state.gemm1_weights_scale,
        None,
        None,
        None,
        None,
        state.gemm2_weights,
        state.gemm2_weights_scale,
        None,
        state.scales,
        state.scales,
        state.scales,
        per_token_scale,
        NUM_EXPERTS,
        TOP_K,
        INTERMEDIATE_SIZE,
        0,
        NUM_EXPERTS,
        True,
        False,
        int(ActivationType.Swiglu),
        split_output,
        tactic,
    )
    module.trtllm_fp4_block_scale_moe_run_from_routing_metadata_with_per_token_scale(
        multi_tile_metadata[0],
        state.hidden_states,
        state.hidden_states_scale,
        state.gemm1_weights,
        state.gemm1_weights_scale,
        None,
        None,
        None,
        None,
        state.gemm2_weights,
        state.gemm2_weights_scale,
        None,
        state.scales,
        state.scales,
        state.scales,
        per_token_scale,
        NUM_EXPERTS,
        TOP_K,
        INTERMEDIATE_SIZE,
        0,
        NUM_EXPERTS,
        True,
        False,
        int(ActivationType.Swiglu),
        multi_tile_split_output,
        tactic,
    )
    torch.cuda.synchronize()

    assert torch.equal(monolithic_output, split_output)
    assert torch.equal(monolithic_output, multi_tile_split_output)
    assert (monolithic_output - split_output).abs().max().item() == 0


def test_precomputed_all_expert_route_prepares_multi_tile_metadata():
    """Precomputed routing may select every expert; logits routing may not."""
    _require_sm100()
    num_tokens = 16
    num_experts = top_k = 8
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device="cuda").repeat(
        num_tokens, 1
    )
    expert_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        dtype=torch.bfloat16,
        device=expert_ids.device,
    )

    metadata_by_tile = fused_moe_api.trtllm_moe_allocate_routing_metadata_multi_tile(
        expert_ids,
        None,
        num_experts,
        top_k,
        8,
        4,
        0,
        num_experts,
        None,
        int(RoutingMethodType.DeepSeekV3),
        [16, 32],
        int(moe_core.RoutingInputMode.UnpackedPrecomputed),
        expert_weights,
    )
    torch.cuda.synchronize()

    assert len(metadata_by_tile) == 2
    for tile_tokens, metadata in zip((16, 32), metadata_by_tile, strict=True):
        assert metadata[3].data_ptr() == expert_weights.data_ptr()
        assert metadata[0].item() == num_experts * tile_tokens
        assert torch.all(metadata[1] >= 0)


def test_nvfp4_unpacked_public_wrapper_metadata_replay_is_bit_exact():
    """Raw FP4 routing IDs and weights survive metadata preparation unchanged."""
    _require_sm100()
    state = _build_moe_state()
    module = _load_moe_ffi_op()
    expert_ids = _assignments_for_distribution(DA_DISTRIBUTIONS[1])
    expert_weights = torch.linspace(
        0.125,
        0.875,
        NUM_TOKENS * TOP_K,
        dtype=torch.bfloat16,
        device=expert_ids.device,
    ).reshape(NUM_TOKENS, TOP_K)
    per_token_scale = torch.linspace(
        0.75, 1.25, NUM_TOKENS, dtype=torch.float32, device=expert_ids.device
    )
    tactics = list(
        module.trtllm_get_valid_moe_configs(
            DtypeTrtllmGen.E2m1,
            DtypeTrtllmGen.E2m1,
            Fp8QuantizationType.NoneFp8,
            TOP_K,
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            NUM_EXPERTS,
            int(ActivationType.Swiglu),
            True,
            int(moe_core.WeightLayout.MajorK),
            False,
            NUM_TOKENS,
            False,
        )
    )
    assert tactics
    tactics_by_tile = {}
    for raw_tactic in tactics:
        tactics_by_tile.setdefault(int(raw_tactic[0]), list(raw_tactic))
    selected = list(tactics_by_tile.values())[:2]
    tile_sizes = [int(tactic[0]) for tactic in selected]

    def run_routed(output):
        return trtllm_fp4_block_scale_routed_moe(
            (expert_ids, expert_weights),
            None,
            state.hidden_states,
            state.hidden_states_scale,
            state.gemm1_weights,
            state.gemm1_weights_scale,
            None,
            None,
            None,
            None,
            state.gemm2_weights,
            state.gemm2_weights_scale,
            None,
            state.scales,
            state.scales,
            state.scales,
            NUM_EXPERTS,
            TOP_K,
            None,
            None,
            INTERMEDIATE_SIZE,
            0,
            NUM_EXPERTS,
            None,
            int(RoutingMethodType.TopK),
            per_token_scale=per_token_scale,
            output=output,
            tune_max_num_tokens=NUM_TOKENS,
        )[0]

    reference = torch.empty(
        NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device=expert_ids.device
    )
    run_routed(reference)

    metadata = fused_moe_api.trtllm_moe_allocate_routing_metadata(
        expert_ids,
        None,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        tile_sizes[0],
        int(moe_core.RoutingInputMode.UnpackedPrecomputed),
        expert_weights,
    )
    metadata_by_tile = fused_moe_api.trtllm_moe_allocate_routing_metadata_multi_tile(
        expert_ids,
        None,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        tile_sizes,
        int(moe_core.RoutingInputMode.UnpackedPrecomputed),
        expert_weights,
    )
    assert len(metadata_by_tile) == len(tile_sizes)
    assert metadata[3].dtype == expert_weights.dtype
    assert torch.equal(metadata[3], expert_weights)
    for prepared in metadata_by_tile:
        assert prepared[3].data_ptr() == expert_weights.data_ptr()

    for tactic, prepared in zip(selected, metadata_by_tile, strict=True):
        replay = torch.empty_like(reference)
        module.trtllm_fp4_block_scale_moe_run_from_routing_metadata_with_per_token_scale(
            prepared,
            state.hidden_states,
            state.hidden_states_scale,
            state.gemm1_weights,
            state.gemm1_weights_scale,
            None,
            None,
            None,
            None,
            state.gemm2_weights,
            state.gemm2_weights_scale,
            None,
            state.scales,
            state.scales,
            state.scales,
            per_token_scale,
            NUM_EXPERTS,
            TOP_K,
            INTERMEDIATE_SIZE,
            0,
            NUM_EXPERTS,
            True,
            False,
            int(ActivationType.Swiglu),
            replay,
            tactic,
        )
        torch.cuda.synchronize()
        _assert_bit_exact(reference, replay)

    graph_output = torch.empty_like(reference)
    run_routed(graph_output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_routed(graph_output)
    expert_weights.copy_(torch.flip(expert_weights, dims=(0,)))
    updated_reference = torch.empty_like(reference)
    run_routed(updated_reference)
    graph.replay()
    torch.cuda.synchronize()
    _assert_bit_exact(updated_reference, graph_output)
    graph.reset()
    del graph
    torch.cuda.synchronize()


def _reset_tuner_and_da(monkeypatch: pytest.MonkeyPatch) -> AutoTuner:
    monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
    # These tests exercise DA body publication and graph topology. Baseline
    # guard rejection is covered separately by the monolithic-NoDA contracts.
    monkeypatch.setenv("FLASHINFER_DA_BASELINE_GUARD", "0")
    monkeypatch.setenv("FLASHINFER_DA_KNN_TIE_EPS", "0.05")
    monkeypatch.setenv(
        "FLASHINFER_DA_DISTRIBUTIONS",
        ",".join(str(distribution[0]) for distribution in DA_DISTRIBUTIONS),
    )
    monkeypatch.delenv("FLASHINFER_DA_BUNDLE", raising=False)
    monkeypatch.setattr(da_profile, "_bundle_loaded", False)
    monkeypatch.setattr(da_profile, "_bundle_has_tactics", False)

    da_state.PER_TILE_TACTICS.clear()
    da_state.PER_BODY_TACTICS.clear()
    da_capture.CAPTURE_RESOURCES.clear()
    da_state.CAPTURE_KEEPALIVE.clear()
    da_single_graph._DA_INLINE_POOL_HANDLES.clear()
    da_single_graph._DA_INLINE_SIDE_STREAMS.clear()
    da_single_graph._DA_INLINE_ROUTING_STREAMS.clear()
    da_state.STATIC_FALLBACK_TACTICS.clear()
    da_state.BUNDLE_EAGER_TACTICS.clear()
    da_state.BASELINE_GUARD_DECISIONS.clear()
    fused_moe_api.reset_da_fast_path_stats()

    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    tuner.is_tuning_mode = False
    tuner.warmup = 1
    tuner.repeat = 1
    tuner._post_autotune_callbacks.clear()
    return tuner


def _clear_da_test_state(tuner: AutoTuner) -> None:
    # Capture resources own tensors referenced by graph and side-stream work.
    # Ensure that work is complete before dropping the keepalive references.
    torch.cuda.synchronize()
    da_state.PER_TILE_TACTICS.clear()
    da_state.PER_BODY_TACTICS.clear()
    da_capture.CAPTURE_RESOURCES.clear()
    da_state.CAPTURE_KEEPALIVE.clear()
    # Tests destroy every graph executable before reaching this point. Drop
    # its per-device pool and stream caches as well so the next test cannot
    # inherit CUDA capture state or allocations from an unrelated graph.
    da_single_graph._DA_INLINE_POOL_HANDLES.clear()
    da_single_graph._DA_INLINE_SIDE_STREAMS.clear()
    da_single_graph._DA_INLINE_ROUTING_STREAMS.clear()
    da_state.STATIC_FALLBACK_TACTICS.clear()
    da_state.BUNDLE_EAGER_TACTICS.clear()
    da_state.BASELINE_GUARD_DECISIONS.clear()
    fused_moe_api.reset_da_fast_path_stats()
    tuner.clear_cache()
    tuner.reset_statistics()
    tuner.is_tuning_mode = False
    tuner._post_autotune_callbacks.clear()


def _collect_profile_latencies(tuner: AutoTuner) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for cache_key, (_runner_id, tactic, _profile) in tuner.profiling_cache.items():
        if hasattr(cache_key, "custom_op"):
            op_name = cache_key.custom_op
            runner_name = cache_key.runner_class_name
            profile_key = cache_key.nearest_profile
        else:
            op_name, runner_name, _runner_hash, profile_key, *_extras = cache_key
        if op_name != "flashinfer::trtllm_fp4_block_scale_moe":
            continue
        if runner_name != "MoERunner":
            continue
        if not isinstance(profile_key, tuple) or len(profile_key) != 2:
            continue
        shapes, value_buckets = profile_key
        if not value_buckets or len(value_buckets) < 2:
            continue
        if int(shapes[0][0]) != NUM_TOKENS:
            continue
        tile = int(tactic[0])
        dist_idx = int(value_buckets[1])
        ms = float(tuner.profiling_time_cache[cache_key])
        out.setdefault(dist_idx, {})[tile] = ms
    return out


def _da_context_for_state(state: _MoEState):
    return da_state.make_context(
        "flashinfer::trtllm_fp4_block_scale_moe",
        device=state.hidden_states.device,
        dtype_act=moe_core.deduce_trtllm_gen_tensor_dtype(
            state.hidden_states, state.hidden_states_scale
        ),
        dtype_weights=moe_core.deduce_trtllm_gen_tensor_dtype(
            state.gemm1_weights, state.gemm1_weights_scale
        ),
        quantization_type=Fp8QuantizationType.NoneFp8,
        top_k=TOP_K,
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_EXPERTS,
        local_expert_offset=0,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        activation_type=int(ActivationType.Swiglu.value),
        weight_layout=moe_core.WeightLayout.MajorK,
        use_shuffled_weight=True,
    )


def _expected_body_tactics_from_profiles(
    tuner: AutoTuner,
    state: _MoEState,
) -> list[tuple[int, int]]:
    latencies = _collect_profile_latencies(tuner)
    assert set(latencies) == set(range(len(DA_DISTRIBUTIONS)))

    da_context = _da_context_for_state(state)
    config_key = da_state.cache_key(da_context, NUM_TOKENS)
    tile_map = da_state.PER_TILE_TACTICS[config_key]
    switch_tiles = sorted(
        moe_core._da_knn_switch_tile_sizes(
            NUM_TOKENS,
            top_k=TOP_K,
            num_local_experts=NUM_EXPERTS,
            local_expert_offset=0,
            dtype_act=DtypeTrtllmGen.E2m1,
            quantization_type=Fp8QuantizationType.NoneFp8,
            da_context=da_context,
        )
    )
    tie_eps = float(os.environ.get("FLASHINFER_DA_KNN_TIE_EPS", "0.05"))

    expected = []
    for dist_idx in range(len(DA_DISTRIBUTIONS)):
        tile_latencies = {
            tile: latency
            for tile, latency in latencies[dist_idx].items()
            if tile in switch_tiles
        }
        assert tile_latencies
        min_latency = min(tile_latencies.values())
        near_best = {
            tile: latency
            for tile, latency in tile_latencies.items()
            if (latency - min_latency) / min_latency <= tie_eps
        }
        best_tile = max(near_best)
        expected.append(tuple(int(v) for v in tile_map[best_tile]))

    return da_profile.deduplicate_body_tactics(expected)[0]


def test_da_baseline_guard_cost_model_keeps_switch_when_all_bodies_win():
    """A deduplicated multi-body plan stays a switch when every body wins."""
    config = da_profile.DAConfig(
        baseline_guard=True,
        control_overhead_us=100.0,
        baseline_guard_margin=0.0,
    )
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=((64, 9), 1.0),
        switch_tiles=[16, 32],
        best_idxs=[0, 1],
        per_distribution_latencies=[{16: 0.7, 32: 1.2}, {16: 1.1, 32: 0.8}],
        per_distribution_tactic_latencies=[
            {(16, 1): 0.7, (32, 2): 1.2, (64, 9): 1.0},
            {(16, 1): 1.1, (32, 2): 0.8, (64, 9): 1.0},
        ],
        tile_to_tactic={16: (16, 1), 32: (32, 2)},
        config=config,
    )
    assert policy == "da_switch"
    assert forced_singleton is None
    assert decision.dynamic_worst_ms == pytest.approx(0.9)


def test_da_baseline_guard_uses_pre_recorded_control_overhead_once():
    """The calibrated control-plane estimate is added once per assignment."""
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=((64, 9), 1.0),
        switch_tiles=[16, 32],
        best_idxs=[0, 1],
        per_distribution_latencies=[{16: 0.8, 32: 1.2}, {16: 1.2, 32: 0.8}],
        per_distribution_tactic_latencies=[
            {(16, 1): 0.8, (32, 2): 1.2, (64, 9): 1.0},
            {(16, 1): 1.2, (32, 2): 0.8, (64, 9): 1.0},
        ],
        tile_to_tactic={16: (16, 1), 32: (32, 2)},
        config=da_profile.DAConfig(
            baseline_guard=True,
            control_overhead_us=10.0,
        ),
    )

    assert policy == "da_switch"
    assert forced_singleton is None
    assert decision.overhead_ms == pytest.approx(0.01)
    assert decision.dynamic_worst_ms == pytest.approx(0.81)


def test_da_baseline_guard_default_control_overhead_is_twelve_us():
    assert da_profile.DAConfig().control_overhead_us == pytest.approx(12.0)


def test_da_config_reads_bundle_path_from_environment(monkeypatch):
    monkeypatch.setenv("FLASHINFER_DA_BUNDLE", "/tmp/da-bundle.pkl")

    assert da_profile.DAConfig().bundle_path == "/tmp/da-bundle.pkl"


@pytest.mark.parametrize(
    ("baseline", "latencies", "limitation"),
    [
        (None, [{16: 0.8}], "missing_noda_baseline"),
        (((16, 1), 1.0), [], "missing_fixed_candidate_timing"),
    ],
)
def test_da_baseline_guard_missing_timing_skips_admission(
    baseline, latencies, limitation
):
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=baseline,
        switch_tiles=[16],
        best_idxs=[0],
        per_distribution_latencies=latencies,
        per_distribution_tactic_latencies=([{(16, 1): 0.8}] if latencies else []),
        tile_to_tactic={16: (16, 1)},
        config=da_profile.DAConfig(baseline_guard=True),
    )

    assert policy == "da_singleton"
    assert forced_singleton is None
    assert not decision.admission_applied
    assert decision.limitation == limitation


def test_da_baseline_guard_source_has_no_cuda_graph_profiling():
    source = inspect.getsource(da_profile.auto_profile_knn_exemplars)
    assert "CUDAGraph" not in source
    assert "graph.replay" not in source
    assert "runtime_cuda_graph" not in source


@pytest.mark.parametrize(
    "precision",
    [
        "nvfp4",
        "mxfp4_mxfp8",
        "mxfp4_bf16",
        "bf16",
        "fp8_per_tensor",
        "fp8_block",
        "mxfp8",
        "mxint4",
    ],
)
def test_guard_preserves_unconstrained_candidate_tactics_for_all_precisions(
    precision,
):
    """Every precision applies the guard after fixing the candidate bodies."""
    switch_tiles = [8, 16, 32]
    tile_to_tactic = {8: (8, 65), 16: (16, 90), 32: (32, 121)}
    best_idxs = [2, 0]
    latencies = [
        {8: 0.90, 16: 0.82, 32: 0.70},
        {8: 0.72, 16: 0.84, 32: 0.91},
    ]
    tactic_latencies = [
        {(8, 65): 0.90, (16, 90): 0.82, (32, 121): 0.70, (32, 43): 0.85},
        {(8, 65): 0.72, (16, 90): 0.84, (32, 121): 0.91, (32, 43): 0.85},
    ]
    candidate_policy, candidate_tactics = da_profile.unconstrained_candidate_plan(
        switch_tiles, best_idxs, tile_to_tactic
    )
    assert candidate_policy == "da_switch", precision
    assert candidate_tactics == [(32, 121), (8, 65)]

    off_policy, off_singleton, off_decision = da_profile.choose_baseline_guard_policy(
        baseline=((32, 43), 0.85),
        switch_tiles=switch_tiles,
        best_idxs=best_idxs,
        per_distribution_latencies=latencies,
        per_distribution_tactic_latencies=tactic_latencies,
        tile_to_tactic=tile_to_tactic,
        config=da_profile.DAConfig(baseline_guard=False),
    )
    on_policy, on_singleton, _ = da_profile.choose_baseline_guard_policy(
        baseline=((32, 43), 0.85),
        switch_tiles=switch_tiles,
        best_idxs=best_idxs,
        per_distribution_latencies=latencies,
        per_distribution_tactic_latencies=tactic_latencies,
        tile_to_tactic=tile_to_tactic,
        config=da_profile.DAConfig(
            baseline_guard=True,
            control_overhead_us=200.0,
        ),
    )

    assert off_policy == "da_switch"
    assert off_singleton is None
    assert off_decision.limitation is None
    assert on_policy == "noda_baseline_guard"
    assert on_singleton == (32, 43)
    assert da_profile.unconstrained_candidate_plan(
        switch_tiles, best_idxs, tile_to_tactic
    ) == (candidate_policy, candidate_tactics)


def test_bundle_guard_signature_rejects_incompatible_final_policy():
    """Guard-off final plans cannot silently load into a guarded process."""
    guard_off = da_profile.DAConfig(baseline_guard=False)
    guard_on = da_profile.DAConfig(baseline_guard=True)
    meta = {"baseline_guard_signature": da_profile.baseline_guard_signature(guard_off)}

    assert da_profile.bundle_guard_signature_matches(meta, guard_off)
    assert not da_profile.bundle_guard_signature_matches(meta, guard_on)
    assert not da_profile.bundle_guard_signature_matches({}, guard_on)
    assert da_profile.bundle_default_profile_contract_matches(
        {"default_profile_contract": da_profile.DEFAULT_PROFILE_CONTRACT}
    )
    assert not da_profile.bundle_default_profile_contract_matches({})


def test_da_baseline_guard_classifies_natural_singleton_after_deduplication():
    """Repeated distribution assignments deduplicate to a natural singleton."""
    config = da_profile.DAConfig(
        baseline_guard=True,
        control_overhead_us=300.0,
        baseline_guard_margin=0.0,
    )
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=((64, 3), 1.0),
        switch_tiles=[16, 32],
        best_idxs=[0, 0],
        per_distribution_latencies=[{16: 0.8, 32: 1.2}, {16: 0.9, 32: 1.1}],
        per_distribution_tactic_latencies=[
            {(16, 1): 0.8, (32, 2): 1.2, (64, 3): 1.0},
            {(16, 1): 0.9, (32, 2): 1.1, (64, 3): 1.0},
        ],
        tile_to_tactic={16: (16, 1), 32: (32, 2)},
        config=config,
    )
    assert policy == "da_singleton"
    assert forced_singleton is None
    assert decision.singleton_tactic == (16, 1)
    assert decision.singleton_source == "profiled"
    assert decision.singleton_worst_ms == pytest.approx(0.9)


def test_da_baseline_guard_matches_fixed_tactics_on_each_distribution():
    """One noisy pairwise crossing cannot reject a better robust singleton."""
    candidate = (128, 42)
    baseline = (32, 13)
    candidate_times = [6, 4, 5]
    baseline_times = [5, 10, 9]
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        # This is the unrelated balanced default-profile time.  It remains a
        # useful eager diagnostic but is not an admission threshold.
        baseline=(baseline, 3),
        switch_tiles=[32, 128],
        best_idxs=[1] * len(candidate_times),
        per_distribution_latencies=[
            {32: baseline_ms, 128: candidate_ms}
            for candidate_ms, baseline_ms in zip(
                candidate_times, baseline_times, strict=True
            )
        ],
        per_distribution_tactic_latencies=[
            {baseline: baseline_ms, candidate: candidate_ms}
            for candidate_ms, baseline_ms in zip(
                candidate_times, baseline_times, strict=True
            )
        ],
        tile_to_tactic={32: baseline, 128: candidate},
        config=da_profile.DAConfig(baseline_guard=True),
    )

    assert policy == "da_singleton"
    assert forced_singleton is None
    assert decision.baseline_ms == pytest.approx(3)
    assert decision.baseline_worst_ms == pytest.approx(max(baseline_times))
    assert decision.singleton_tactic == candidate


def test_da_baseline_guard_rejects_uncompetitive_natural_singleton():
    """A natural singleton that loses to NoDA selects the monolithic fallback."""
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=((64, 3), 0.7),
        switch_tiles=[16, 32],
        best_idxs=[0, 0],
        per_distribution_latencies=[{16: 0.8, 32: 1.2}, {16: 0.9, 32: 1.1}],
        per_distribution_tactic_latencies=[
            {(16, 1): 0.8, (32, 2): 1.2, (64, 3): 0.7},
            {(16, 1): 0.9, (32, 2): 1.1, (64, 3): 0.7},
        ],
        tile_to_tactic={16: (16, 1), 32: (32, 2)},
        config=da_profile.DAConfig(
            baseline_guard=True,
            control_overhead_us=0.0,
        ),
    )

    assert policy == "noda_baseline_guard"
    assert forced_singleton == (64, 3)
    assert decision.singleton_source == "noda_baseline"


def test_da_baseline_guard_rejects_switch_to_monolithic_noda():
    """A rejected switch selects NoDA rather than a metadata-replay singleton."""
    config = da_profile.DAConfig(
        baseline_guard=True,
        control_overhead_us=300.0,
        baseline_guard_margin=0.0,
    )
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=((64, 3), 1.0),
        switch_tiles=[16, 32],
        best_idxs=[0, 1],
        per_distribution_latencies=[{16: 0.8, 32: 0.95}, {16: 0.9, 32: 0.7}],
        per_distribution_tactic_latencies=[
            {(16, 1): 0.8, (32, 2): 0.95, (64, 3): 1.0},
            {(16, 1): 0.9, (32, 2): 0.7, (64, 3): 1.0},
        ],
        tile_to_tactic={16: (16, 1), 32: (32, 2)},
        config=config,
    )
    assert policy == "noda_baseline_guard"
    assert forced_singleton == (64, 3)
    assert decision.singleton_tactic == (64, 3)
    assert decision.singleton_source == "noda_baseline"
    assert decision.singleton_worst_ms == pytest.approx(1.0)
    assert decision.dynamic_worst_ms == pytest.approx(1.1)


def test_da_baseline_guard_reports_baseline_tactic_for_noda_fallback():
    """An uncompetitive profile reports the monolithic NoDA tactic."""
    config = da_profile.DAConfig(
        baseline_guard=True,
        control_overhead_us=300.0,
        baseline_guard_margin=0.0,
    )
    policy, forced_singleton, decision = da_profile.choose_baseline_guard_policy(
        baseline=((64, 3), 1.0),
        switch_tiles=[16, 32],
        best_idxs=[0, 1],
        per_distribution_latencies=[{16: 1.1, 32: 1.4}, {16: 1.2, 32: 1.05}],
        per_distribution_tactic_latencies=[
            {(16, 1): 1.1, (32, 2): 1.4, (64, 3): 1.0},
            {(16, 1): 1.2, (32, 2): 1.05, (64, 3): 1.0},
        ],
        tile_to_tactic={16: (16, 1), 32: (32, 2)},
        config=config,
    )
    assert policy == "noda_baseline_guard"
    assert forced_singleton == (64, 3)
    assert decision.baseline_tactic == (64, 3)
    assert decision.singleton_tactic == (64, 3)
    assert decision.singleton_source == "noda_baseline"
    assert decision.singleton_worst_ms == pytest.approx(1.0)


def test_guarded_noda_skips_before_capture_backend_construction():
    """A rejected plan must not reach any graph-mutating DA backend."""
    context = _da_test_context(
        "flashinfer::trtllm_fp4_block_scale_moe",
        moe_core.DtypeTrtllmGen.E2m1,
        moe_core.DtypeTrtllmGen.E2m1,
    )
    key = da_state.cache_key(context, 512)
    da_state.BASELINE_GUARD_DECISIONS[key] = {
        "policy": "noda_baseline_guard",
        "final_policy": "noda_baseline_guard",
        "baseline_tactic": (32, 86),
    }
    execution = SimpleNamespace(
        invocation=SimpleNamespace(
            da_context=context,
            num_tokens=512,
            tune_max_num_tokens=512,
        ),
        backend=SimpleNamespace(
            capture_backend=lambda: pytest.fail(
                "guarded NoDA constructed a graph-mutating capture backend"
            )
        ),
        config=da_profile.DAConfig(),
    )
    try:
        assert da_core.try_capture_dispatch(execution, lambda *_args: None) is None
    finally:
        da_state.BASELINE_GUARD_DECISIONS.pop(key, None)


def test_guarded_noda_bundle_restores_policy_without_da_bodies():
    """Bundle reuse preserves the monolithic fallback and eager tactic."""
    config = da_profile.DAConfig(baseline_guard=True)
    context = _da_test_context(
        "flashinfer::trtllm_fp4_block_scale_moe",
        moe_core.DtypeTrtllmGen.E2m1,
        moe_core.DtypeTrtllmGen.E2m1,
    )
    key = da_state.cache_key(context, 512)
    bundle = {
        "meta": {
            "schema_version": context.schema_version,
            "device_type": context.device_type,
            "device_index": context.device_index,
            "profile_signature": list(config.profile_signature),
            "baseline_guard_signature": da_profile.baseline_guard_signature(config),
            "default_profile_contract": da_profile.DEFAULT_PROFILE_CONTRACT,
            "num_local": context.num_local_experts,
            "num_global_experts": context.num_experts,
            "local_offset": context.local_expert_offset,
            "top_k": context.top_k,
        },
        "plans": {
            "512": {
                "policy": "noda_baseline_guard",
                "candidate_policy": "da_switch",
                "candidate_tactics": [(32, 202), (128, 14)],
                "final_policy": "noda_baseline_guard",
                "baseline_tactic": (128, 14),
            }
        },
        "exemplars": {},
        "eager_tactics": {"512": {"tactic": (128, 14), "time_ms": 0.51}},
    }
    da_state.BASELINE_GUARD_DECISIONS.pop(key, None)
    da_state.PER_TILE_TACTICS[key] = {32: (32, 202)}
    da_state.PER_BODY_TACTICS[key] = [(32, 202)]
    da_state.BUNDLE_EAGER_TACTICS.pop(key, None)
    try:
        uploaded = da_profile.load_knn_v2_bundle(
            bundle,
            "unused.pkl",
            config=config,
            backend=SimpleNamespace(
                get_ffi_moe_op=lambda: pytest.fail(
                    "guarded NoDA bundle uploaded selector metadata"
                )
            ),
            da_context=context,
        )
        assert uploaded == 1
        decision = da_state.BASELINE_GUARD_DECISIONS[key]
        assert decision["candidate_tactics"] == [(32, 202), (128, 14)]
        assert decision["final_policy"] == "noda_baseline_guard"
        assert decision["final_tactics"] == [(128, 14)]
        assert key not in da_state.PER_TILE_TACTICS
        assert key not in da_state.PER_BODY_TACTICS
        assert da_state.BUNDLE_EAGER_TACTICS[key] == ((128, 14), 0.51)
    finally:
        da_state.BASELINE_GUARD_DECISIONS.pop(key, None)
        da_state.PER_TILE_TACTICS.pop(key, None)
        da_state.PER_BODY_TACTICS.pop(key, None)
        da_state.BUNDLE_EAGER_TACTICS.pop(key, None)


@dataclass(frozen=True)
class _BaselineLookupContext:
    hidden_size: int = 7168
    top_k: int = 8


def _baseline_lookup_tuner(entries):
    class _Tuner:
        profiling_cache = {key: (0, tactic, None) for key, tactic, _ in entries}
        profiling_time_cache = {key: latency for key, _, latency in entries}

    return _Tuner()


def test_da_baseline_lookup_accepts_flat_shape_only_key_and_prefers_exact_hash():
    """NoDA keys are flat shapes; exact runner identity wins when available."""
    op = "flashinfer::trtllm_fp4_block_scale_moe"
    shapes = ((64, 7168), (64, 64), (64, 8))
    exact = (op, "MoERunner", 7, shapes, ())
    compatible = (op, "MoERunner", 8, shapes, ())
    value_aware = (op, "MoERunner", 7, (shapes, (0, 1)), ())
    tuner = _baseline_lookup_tuner(
        [
            (exact, (16, 1), 1.0),
            (compatible, (32, 2), 0.5),
            (value_aware, (64, 3), 0.1),
        ]
    )
    da_state.STATIC_FALLBACK_TACTICS.clear()
    assert da_profile.best_static_tactic_from_profiles(
        tuner,
        op,
        "MoERunner",
        7,
        64,
        da_context=_BaselineLookupContext(),
    ) == ((16, 1), 1.0)


def test_da_baseline_lookup_uses_only_shape_compatible_hash_fallback():
    """A policy-only hash mismatch may fall back, but another shape may not."""
    op = "flashinfer::trtllm_fp4_block_scale_moe"
    compatible_shapes = ((64, 7168), (64, 64), (64, 8))
    wrong_shapes = ((64, 4096), (64, 64), (64, 8))
    compatible = (op, "MoERunner", 8, compatible_shapes, ())
    wrong = (op, "MoERunner", 9, wrong_shapes, ())
    tuner = _baseline_lookup_tuner([(compatible, (32, 2), 0.8), (wrong, (64, 3), 0.1)])
    da_state.STATIC_FALLBACK_TACTICS.clear()
    assert da_profile.best_static_tactic_from_profiles(
        tuner,
        op,
        "MoERunner",
        7,
        64,
        da_context=_BaselineLookupContext(),
    ) == ((32, 2), 0.8)


def test_da_baseline_lookup_restores_bundle_default_profile_tactic():
    context = _BaselineLookupContext()
    da_state.STATIC_FALLBACK_TACTICS.clear()
    da_state.BUNDLE_EAGER_TACTICS.clear()
    da_state.BUNDLE_EAGER_TACTICS[da_state.cache_key(context, 64)] = (
        (32, 9),
        0.75,
    )

    assert da_profile.best_static_tactic_from_profiles(
        _baseline_lookup_tuner([]),
        "flashinfer::trtllm_fp4_block_scale_moe",
        "MoERunner",
        7,
        64,
        da_context=context,
    ) == ((32, 9), 0.75)
    da_state.BUNDLE_EAGER_TACTICS.clear()


def _capture_and_replay_da_graph(
    state: _MoEState,
    routing_logits: torch.Tensor,
    warmup_tactic: tuple[int, int],
) -> torch.Tensor:
    """Prepare, capture, replay, and explicitly release a public FP4 graph."""
    module = moe_core.get_trtllm_moe_sm100_module()
    output = torch.empty(
        NUM_TOKENS,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=state.hidden_states.device,
    )
    topk_ids = torch.empty(
        NUM_TOKENS, TOP_K, dtype=torch.int32, device=state.hidden_states.device
    )
    expert_weights = torch.empty(
        NUM_TOKENS, TOP_K, dtype=torch.bfloat16, device=state.hidden_states.device
    )
    tactic_arg = [int(warmup_tactic[0]), int(warmup_tactic[1])]

    def warmup_fn() -> None:
        module.ffi_moe_op.trtllm_fp4_block_scale_moe(
            int(moe_core.RoutingInputMode.FromLogits),
            routing_logits,
            topk_ids,
            expert_weights,
            state.routing_bias,
            state.hidden_states,
            state.hidden_states_scale,
            state.gemm1_weights,
            state.gemm1_weights_scale,
            None,
            None,
            None,
            None,
            state.gemm2_weights,
            state.gemm2_weights_scale,
            None,
            state.scales,
            state.scales,
            state.scales,
            None,
            NUM_EXPERTS,
            TOP_K,
            N_GROUP,
            TOPK_GROUP,
            INTERMEDIATE_SIZE,
            0,
            NUM_EXPERTS,
            ROUTED_SCALING_FACTOR,
            int(RoutingMethodType.DeepSeekV3),
            True,
            False,
            int(ActivationType.Swiglu.value),
            output,
            tactic_arg,
            True,
            None,
        )

    def capture_fn() -> None:
        module.trtllm_fp4_block_scale_moe(
            int(moe_core.RoutingInputMode.FromLogits),
            routing_logits,
            topk_ids,
            expert_weights,
            state.routing_bias,
            state.hidden_states,
            state.hidden_states_scale,
            state.gemm1_weights,
            state.gemm1_weights_scale,
            None,
            None,
            None,
            None,
            state.gemm2_weights,
            state.gemm2_weights_scale,
            None,
            state.scales,
            state.scales,
            state.scales,
            None,
            NUM_EXPERTS,
            TOP_K,
            N_GROUP,
            TOPK_GROUP,
            INTERMEDIATE_SIZE,
            0,
            NUM_EXPERTS,
            ROUTED_SCALING_FACTOR,
            int(RoutingMethodType.DeepSeekV3),
            True,
            False,
            int(ActivationType.Swiglu.value),
            output,
            TUNE_MAX_NUM_TOKENS,
            True,
            None,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        warmup_fn()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        capture_fn()
    torch.cuda.synchronize()

    before = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        capture_fn()
        graph.capture_end()
    torch.cuda.synchronize()
    after = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
    assert after == before + 1

    graph.replay()
    torch.cuda.synchronize()
    result = output.clone()
    graph.reset()
    del graph
    torch.cuda.synchronize()
    return result


def _capture_unpacked_da_graph(
    state: _MoEState,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    warmup_tactic: tuple[int, int],
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    """Capture the public unpacked FP4 path with graph-stable caller tensors."""
    module = moe_core.get_trtllm_moe_sm100_module()
    output = torch.empty(
        NUM_TOKENS,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device=state.hidden_states.device,
    )
    tactic_arg = [int(warmup_tactic[0]), int(warmup_tactic[1])]

    module.ffi_moe_op.trtllm_fp4_block_scale_moe(
        int(moe_core.RoutingInputMode.UnpackedPrecomputed),
        None,
        expert_ids,
        expert_weights,
        None,
        state.hidden_states,
        state.hidden_states_scale,
        state.gemm1_weights,
        state.gemm1_weights_scale,
        None,
        None,
        None,
        None,
        state.gemm2_weights,
        state.gemm2_weights_scale,
        None,
        state.scales,
        state.scales,
        state.scales,
        None,
        NUM_EXPERTS,
        TOP_K,
        None,
        None,
        INTERMEDIATE_SIZE,
        0,
        NUM_EXPERTS,
        None,
        int(RoutingMethodType.TopK),
        True,
        False,
        int(ActivationType.Swiglu.value),
        output,
        tactic_arg,
        True,
        None,
    )
    torch.cuda.synchronize()

    # The eager public call prepares stable raw-ID/weight buffers and one
    # routing-metadata bundle per candidate tile for active capture.
    _run_public_routed_moe(state, expert_ids, expert_weights, output)
    torch.cuda.synchronize()

    before = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_public_routed_moe(state, expert_ids, expert_weights, output)
    after = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
    assert after == before + 1
    return graph, output


def test_nvfp4_unpacked_da_graph_tracks_live_caller_weights(monkeypatch):
    """DA graph replay consumes the caller's live raw routing pair."""
    _require_sm100()
    tuner = _reset_tuner_and_da(monkeypatch)
    graph = None
    try:
        monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", "uniform")
        monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTION_SAMPLES", "1")
        state = _build_moe_state()
        expert_ids = _assignments_for_distribution(DA_DISTRIBUTIONS[0])
        expert_weights = torch.linspace(
            0.125,
            0.875,
            NUM_TOKENS * TOP_K,
            dtype=torch.bfloat16,
            device=expert_ids.device,
        ).reshape(NUM_TOKENS, TOP_K)

        with autotune(True):
            tuned = _run_public_routed_moe(state, expert_ids, expert_weights)
        torch.cuda.synchronize()
        assert torch.isfinite(tuned).all()

        da_context = _da_context_for_state(state)
        body_tactics = da_state.PER_BODY_TACTICS[
            da_state.cache_key(da_context, NUM_TOKENS)
        ]
        assert len(body_tactics) == 1
        graph, graph_output = _capture_unpacked_da_graph(
            state, expert_ids, expert_weights, body_tactics[0]
        )

        for new_weights in (
            torch.flip(expert_weights, dims=(0,)),
            torch.linspace(
                0.25,
                1.0,
                NUM_TOKENS * TOP_K,
                dtype=torch.bfloat16,
                device=expert_ids.device,
            ).reshape(NUM_TOKENS, TOP_K),
        ):
            expert_weights.copy_(new_weights)
            monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "0")
            reference = _run_public_routed_moe(state, expert_ids, expert_weights)
            torch.cuda.synchronize()
            monkeypatch.setenv("FLASHINFER_DIST_AWARE_AUTOTUNE", "1")
            graph.replay()
            torch.cuda.synchronize()
            _assert_bit_exact(reference, graph_output)
    finally:
        if graph is not None:
            graph.reset()
            del graph
        _clear_da_test_state(tuner)


def test_trtllm_da_autotune_graph_outputs_are_finite(monkeypatch):
    """A real DA-autotuned public wrapper produces usable graph outputs."""
    if os.getenv("_FLASHINFER_DA_FINITE_GRAPH_CHILD") != "1":
        # This legacy test intentionally captures three independent public FP4
        # graphs. Run it in its own CUDA context so its graph-pool lifetime does
        # not consume the budget of the end-to-end precision graph matrix.
        env = os.environ.copy()
        env["_FLASHINFER_DA_FINITE_GRAPH_CHILD"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_trtllm_da_autotune_graph_outputs_are_finite",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    _require_sm100()
    tuner = _reset_tuner_and_da(monkeypatch)
    try:
        state = _build_moe_state()
        with autotune(True):
            output = _run_public_moe(
                state, _routing_logits_for_distribution(DA_DISTRIBUTIONS[0])
            )
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()

        da_context = _da_context_for_state(state)
        body_tactics = da_state.PER_BODY_TACTICS.get(
            da_state.cache_key(da_context, NUM_TOKENS), ()
        )
        assert body_tactics
        warmup_tactic = body_tactics[0]
        for distribution in DA_DISTRIBUTIONS:
            graph_output = _capture_and_replay_da_graph(
                state,
                _routing_logits_for_distribution(distribution),
                warmup_tactic,
            )
            assert graph_output.shape == (NUM_TOKENS, HIDDEN_SIZE)
            assert graph_output.dtype == torch.bfloat16
            assert torch.isfinite(graph_output).all()
            assert not torch.all(graph_output == 0)
    finally:
        _clear_da_test_state(tuner)


def _run_factorized_public_reference_case(monkeypatch, tmp_path, precision):
    """Tune one public precision and return its independent reference and call."""
    _require_sm100()
    monkeypatch.delenv("FLASHINFER_DA_FACTORIZED_AUTOTUNE", raising=False)
    monkeypatch.setattr(gen_moe_utils, "TUNE_MAX_NUM_TOKENS", 64)
    autotuner_module = importlib.import_module("flashinfer.autotuner.autotuner")
    profile_dump = tmp_path / f"factorized-{precision.name}.csv"
    monkeypatch.setattr(autotuner_module, "_AUTOTUNE_DUMP_PATH", str(profile_dump))

    monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTIONS", "uniform")
    monkeypatch.setenv("FLASHINFER_DA_DISTRIBUTION_SAMPLES", "1")
    moe_impl = _matrix_moe_impl(precision.name)
    block_major = precision.name in {"bf16", "fp8_block", "mxint4"}
    execution = []

    reference, output, _ = run_moe_test(
        num_tokens=64,
        hidden_size=1024,
        intermediate_size=1024,
        moe_impl=moe_impl,
        routing_config={
            "num_experts": 16,
            "top_k": 2,
            "padding": 8,
            "n_groups": 4,
            "top_k_groups": 2,
            "routed_scaling": 1.0,
            "has_routing_bias": True,
            "routing_method_type": RoutingMethodType.DeepSeekV3,
            "compatible_moe_impls": [type(moe_impl)],
            "compatible_intermediate_size": [1024],
            "compatible_activation_types": [ActivationType.Swiglu],
            "enable_autotune": True,
        },
        weight_processing={
            "use_shuffled_weight": True,
            "layout": (
                moe_core.WeightLayout.BlockMajorK
                if block_major
                else moe_core.WeightLayout.MajorK
            ),
            "compatible_moe_impls": [type(moe_impl)],
        },
        activation_type=ActivationType.Swiglu,
        cache_permute_indices={},
        routing_logits_dtype=torch.bfloat16,
        zero_hidden_states=False,
        expert_logits_transform=_matrix_logits_transform("uniform", 2),
        execution_observer=lambda *args: execution.append(args),
    )
    assert torch.isfinite(reference).all()
    assert torch.isfinite(output).all()

    with profile_dump.open(newline="") as profile_file:
        records = list(csv.DictReader(profile_file))
    assert any(record.get("selection_stage") == "composed" for record in records)
    assert len(execution) == 1
    return reference, output, execution[0]


def _call_recorded_public_moe(execution):
    """Call the public wrapper recorded by run_moe_test without retuning it."""
    moe_impl, static_data, hidden_states, hidden_scale, kwargs = execution
    replay_kwargs = dict(kwargs)
    replay_kwargs["enable_autotune"] = False
    if isinstance(moe_impl, FP8BlockScaleMoe):
        # FP8BlockScaleMoe.call_moe contains an eager-only ``isnan().any()``
        # test assertion. Keep that assertion in run_moe_test, then exercise
        # the real public wrapper directly for graph preparation and capture.
        quantization_type = (
            moe_core.Fp8QuantizationType.MxFp8
            if moe_impl.fp8_quantization_type == QuantMode.FP8_BLOCK_SCALE_MXFP8
            else moe_core.Fp8QuantizationType.DeepSeekFp8
        )
        with autotune(False):
            output = trtllm_fp8_block_scale_moe(
                replay_kwargs["expert_logits"],
                replay_kwargs["routing_bias"],
                replay_kwargs["hidden_states_quant"].to(torch.float8_e4m3fn),
                replay_kwargs["hidden_states_scale"],
                static_data["gemm1_weights"],
                static_data["gemm1_scales"],
                static_data["gemm2_weights"],
                static_data["gemm2_scales"],
                replay_kwargs["num_experts"],
                replay_kwargs["top_k"],
                replay_kwargs["n_groups"],
                replay_kwargs["top_k_groups"],
                replay_kwargs["intermediate_size"],
                0,
                replay_kwargs["num_experts"],
                replay_kwargs["routed_scaling"],
                replay_kwargs["routing_method_type"],
                use_shuffled_weight=static_data["use_shuffled_weight"],
                weight_layout=static_data["weight_layout"],
                enable_pdl=replay_kwargs.get("enable_pdl"),
                tune_max_num_tokens=64,
                fp8_quantization_type=quantization_type,
                activation_type=replay_kwargs["activation_type"],
                norm_topk_prob=replay_kwargs.get("norm_topk_prob", True),
            )
        return output[0].to(torch.float) if isinstance(output, list) else output.float()
    return moe_impl.call_moe(
        static_data,
        hidden_states,
        hidden_scale,
        **replay_kwargs,
    )


def _assert_published_runtime_plan(precision):
    """Require a DA runtime plan for the exact tuned precision context."""
    matches = [
        (key, bodies)
        for key, bodies in da_state.PER_BODY_TACTICS.items()
        if key[0] == 64
        and key[1].op_name == precision.op_name
        and key[1].dtype_act == int(precision.dtype_act)
        and key[1].dtype_weights == int(precision.dtype_weights)
        and key[1].quantization_type == int(precision.quantization_type)
    ]
    assert len(matches) == 1, matches
    key, bodies = matches[0]
    assert bodies
    decision = da_state.BASELINE_GUARD_DECISIONS.get(key)
    assert decision is not None
    assert decision.get("policy") in DA_RUNTIME_PLAN_POLICIES


def _assert_precision_reference(precision, reference, output):
    """Apply the precision implementation's independent-reference tolerance."""
    tolerances = precision.moe_impl.get_tolerances()
    gen_moe_tests.check_accuracy(
        reference,
        output,
        atol=tolerances["atol"],
        rtol=tolerances["rtol"],
        percent=tolerances["percent"],
    )


@pytest.mark.parametrize(
    "precision", NON_FP4_DA_PRECISION_CONTRACTS, ids=lambda item: item.name
)
def test_factorized_da_public_wrapper_eager_matches_independent_reference(
    monkeypatch, tmp_path, precision
):
    """The tuned public eager call must match run_moe_test's dequant reference."""
    tuner = _reset_tuner_and_da(monkeypatch)
    try:
        reference, output, _ = _run_factorized_public_reference_case(
            monkeypatch, tmp_path, precision
        )
        _assert_precision_reference(precision, reference, output)
        _assert_published_runtime_plan(precision)
        assert fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"] == 0
    finally:
        _clear_da_test_state(tuner)


@pytest.mark.parametrize(
    "precision", NON_FP4_DA_PRECISION_CONTRACTS, ids=lambda item: item.name
)
def test_factorized_da_public_wrapper_graph_matches_independent_reference(
    monkeypatch, tmp_path, precision
):
    """The real public CUDA graph must dispatch DA and match the dequant reference."""
    child_precision = os.getenv("_FLASHINFER_DA_FACTOR_GRAPH_CHILD")
    if child_precision != precision.name:
        # Earlier conditional-graph tests can leave CUDA state that is only
        # reported by a later allocation.  Keep every precision contract real,
        # but give it an independent CUDA context so failures are attributable
        # to that public-wrapper capture rather than inherited graph lifetime.
        env = os.environ.copy()
        env["_FLASHINFER_DA_FACTOR_GRAPH_CHILD"] = precision.name
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::{test_factorized_da_public_wrapper_graph_matches_independent_reference.__name__}[{precision.name}]",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return

    tuner = _reset_tuner_and_da(monkeypatch)
    graph = None
    try:
        reference, _, execution = _run_factorized_public_reference_case(
            monkeypatch, tmp_path, precision
        )
        prepared_output = _call_recorded_public_moe(execution)
        torch.cuda.synchronize()
        _assert_precision_reference(precision, reference, prepared_output)
        _assert_published_runtime_plan(precision)

        fused_moe_api.reset_da_fast_path_stats()
        before = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = _call_recorded_public_moe(execution)
        after = fused_moe_api.get_da_fast_path_stats()["capture_dispatch_count"]
        assert after == before + 1

        graph.replay()
        torch.cuda.synchronize()
        _assert_precision_reference(precision, reference, graph_output)
        _assert_published_runtime_plan(precision)
    finally:
        if graph is not None:
            graph.reset()
            del graph
        _clear_da_test_state(tuner)
