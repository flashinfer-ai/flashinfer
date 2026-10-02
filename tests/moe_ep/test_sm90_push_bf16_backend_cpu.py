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

CPU-only contracts for the SM90 push BF16 mega-MoE backend.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib import resources
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


_PACKAGE_NAME = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "flashinfer"
    / "moe_ep"
    / "kernel_src"
    / "sm90"
    / "push_style_megamoe"
)


def _package_text(*parts: str) -> str:
    source_tree = _PACKAGE_ROOT.joinpath(*parts)
    if source_tree.is_file():
        return source_tree.read_text(encoding="utf-8")

    resource = resources.files(_PACKAGE_NAME)
    for part in parts:
        resource = resource / part
    return resource.read_text(encoding="utf-8")


def _module_text(module_name: str) -> str:
    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        raise AssertionError(f"module source is unavailable: {module_name}")
    return Path(spec.origin).read_text(encoding="utf-8")


def test_sm90_push_bf16_import_defers_kernel_package() -> None:
    code = textwrap.dedent(
        """
        import importlib
        import sys
        import typing

        kernel_name = "flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe"
        backend_package = importlib.import_module(
            "flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda"
        )
        weights_module = importlib.import_module(
            "flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.weights"
        )
        assert kernel_name not in sys.modules
        hints = typing.get_type_hints(weights_module.preprocess_mega_weights)
        assert hints["return"] is typing.Any
        assert kernel_name not in sys.modules
        transformed_type = backend_package.TransformedMegaWeights
        kernel_package = importlib.import_module(kernel_name)
        assert transformed_type is kernel_package.Sm90PushBf16Weights
        """
    )
    env = os.environ.copy()
    env["FLASHINFER_DISABLE_JIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_sm90_push_bf16_timeout_requires_supported_setter() -> None:
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.backend import (
        _set_process_group_timeout,
    )

    with (
        mock.patch.object(dist, "set_timeout", None, create=True),
        mock.patch.object(dist, "distributed_c10d", SimpleNamespace(), create=True),
        pytest.raises(RuntimeError, match="exposes neither set_timeout"),
    ):
        _set_process_group_timeout(object(), 1.0)


def test_sm90_push_bf16_config_defaults() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.config import (
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    config = Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(intermediate_size=256, top_k=2)

    assert config.kernel_name == "sm90_bf16_bf16_bf16_push_cuda"
    assert config.capacity_factor == 1.0
    assert config.dedup_dispatch is True
    assert config.grouped_combine is False
    assert config.fuse_fc1_epilogue is False
    assert config.allow_unverified_p2p is False
    assert config.wave_schedule == "mono"


def test_sm90_push_bf16_fused_fc1_rebuilds_the_fc2_schedule() -> None:
    source = _package_text("shim", "bf16_runner.py")

    assert "create_sm90_push_bf16_fused_fc1_runner" in source
    assert "prepare_schedule=self.pipe.config.fuse_fc1_epilogue" in source
    assert (
        'return None if self.pipe.config.fuse_fc1_epilogue else "activation"' in source
    )


def test_sm90_push_bf16_fused_fc1_has_a_valid_protocol_config() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.protocol import (
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushPayload,
        _validate_fusion_contract,
    )

    _validate_fusion_contract(
        Sm90PushConfig(
            payload_dtype=Sm90PushPayload.BF16,
            combine_dtype=Sm90PushCombine.BF16,
            fuse_act=False,
            fuse_fc1_epilogue=True,
        )
    )

    with pytest.raises(ValueError, match="FP8.*fused activation path"):
        _validate_fusion_contract(
            Sm90PushConfig(
                payload_dtype=Sm90PushPayload.FP8,
                fuse_act=False,
                fuse_fc1_epilogue=True,
            )
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("top_k", 3, "top_k"),
        ("capacity_factor", 0.0, "capacity_factor"),
        ("capacity_factor", float("nan"), "capacity_factor"),
        ("init_timeout_s", 0.0, "init_timeout_s"),
        ("wave_schedule", "bad", "wave_schedule"),
    ],
)
def test_sm90_push_bf16_rejects_invalid_config(
    field: str, value: object, error: str
) -> None:
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda import (
        backend as backend_module,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.backend import (
        Sm90PushBf16MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.config import (
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    kwargs = {"intermediate_size": 256, "top_k": 2, field: value}
    backend = Sm90PushBf16MegaKernelBackend(
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(**kwargs)
    )
    bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fleet = FleetParams(
        num_experts=4,
        max_tokens_per_rank=16,
        token_hidden_size=256,
    )

    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        pytest.raises(MoEEpConfigError, match=error),
    ):
        backend.validate_init(bootstrap, fleet)


def test_sm90_push_bf16_rejects_multinode_sized_ep_group() -> None:
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda import (
        backend as backend_module,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.backend import (
        Sm90PushBf16MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.config import (
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushBf16MegaKernelBackend(
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(intermediate_size=256, top_k=2)
    )
    bootstrap = BootstrapConfig(world_size=33, rank=0, auto_bootstrap=False)
    fleet = FleetParams(
        num_experts=66,
        max_tokens_per_rank=16,
        token_hidden_size=256,
    )

    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        pytest.raises(MoEEpConfigError, match="at most 32"),
    ):
        backend.validate_init(bootstrap, fleet)


def test_sm90_push_bf16_two_wave_requires_full_capacity() -> None:
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda import (
        backend as backend_module,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.backend import (
        Sm90PushBf16MegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.config import (
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushBf16MegaKernelBackend(
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            capacity_factor=0.75,
            wave_schedule="pipe2",
        )
    )
    bootstrap = BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    fleet = FleetParams(
        num_experts=4,
        max_tokens_per_rank=16,
        token_hidden_size=256,
    )

    with (
        mock.patch.object(backend_module, "_validate_sm90_arch"),
        pytest.raises(MoEEpConfigError, match="capacity_factor=1.0"),
    ):
        backend.validate_init(bootstrap, fleet)


def test_sm90_push_bf16_two_wave_event_order_is_fail_closed() -> None:
    source = _package_text("shim", "bf16_overlap.py")

    assert source.index("self._dispatch0_done.record") < source.index(
        "self._runner0.compute"
    )
    pair_body = source[source.index("def compute(") :]
    serial_wait = pair_body.index('if self._schedule == "serial2"')
    stage1 = pair_body.index("self._runner1.stage_inputs")
    overlap_wait = pair_body.index('if self._schedule == "pipe2"')
    compute1 = pair_body.index("self._runner1.compute")
    assert serial_wait < stage1 < overlap_wait < compute1
    assert "two-wave scheduling does not support CUDA graph capture" in source
    assert "runner.abort()" in source


def test_sm90_push_bf16_staging_validates_context_before_runner() -> None:
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.backend import (
        Sm90PushBf16MegaKernelBackend,
        _Sm90PushBf16Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.config import (
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushBf16MegaKernelBackend(
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(intermediate_size=256, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    runner = mock.Mock()
    workspace = _Sm90PushBf16Workspace(
        pipe=object(),
        runner=runner,
        transformed_weights=transformed,
    )
    inputs = SimpleNamespace(
        hidden_states=object(),
        topk_ids=object(),
        topk_weights=object(),
        num_tokens=3,
    )
    backend._transformed_weights = object()
    with pytest.raises(RuntimeError, match="workspace bundle"):
        backend.stage_inputs(inputs, workspace, quantize_input=True)
    runner.stage_inputs.assert_not_called()

    backend._transformed_weights = transformed
    with pytest.raises(MoEEpConfigError, match="quantize_input=True"):
        backend.stage_inputs(inputs, workspace, quantize_input=False)
    runner.stage_inputs.assert_not_called()

    backend.stage_inputs(inputs, workspace, quantize_input=True)
    runner.stage_inputs.assert_called_once_with(
        inputs.hidden_states,
        inputs.topk_ids,
        inputs.topk_weights,
    )
    assert workspace.staged_tokens == 3


def test_sm90_push_bf16_compute_finishes_round_before_weight_rejection() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.backend import (
        Sm90PushBf16MegaKernelBackend,
        _Sm90PushBf16Workspace,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.config import (
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig,
    )

    backend = Sm90PushBf16MegaKernelBackend(
        Sm90_Bf16_Bf16_Bf16_PushCuda_MegaMoeConfig(intermediate_size=256, top_k=2)
    )
    transformed = object()
    backend._transformed_weights = transformed
    output = object()
    runner = mock.Mock(state="idle")
    runner.compute.return_value = output
    workspace = _Sm90PushBf16Workspace(
        pipe=object(),
        runner=runner,
        transformed_weights=transformed,
        staged_tokens=3,
    )

    with pytest.raises(RuntimeError, match="different weight bundle"):
        backend.compute(workspace, object(), output=output)

    runner.compute.assert_called_once_with(output=output)
    assert workspace.staged_tokens is None
    assert workspace.poisoned is False


def test_sm90_push_bf16_preprocess_rejects_scaled_weights() -> None:
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cuda.weights import (
        preprocess_mega_weights,
    )

    weights = MoEWeightPack(
        w13=torch.empty(2, 512, 256, dtype=torch.uint8),
        w2=torch.empty(2, 256, 256, dtype=torch.uint8),
        w13_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )

    with pytest.raises(MoEEpConfigError, match="canonical BF16 weights only"):
        preprocess_mega_weights(
            weights,
            intermediate_size=256,
            hidden_size=256,
            num_local_experts=2,
        )


def test_sm90_push_bf16_weights_reject_cpu_tensors() -> None:
    import torch

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe import (
        Sm90PushBf16Weights,
    )

    with pytest.raises(ValueError, match="CUDA"):
        Sm90PushBf16Weights(
            w13=torch.empty(2, 512, 256, dtype=torch.bfloat16),
            w2=torch.empty(2, 256, 256, dtype=torch.bfloat16),
        )


def test_sm90_push_bf16_gemm_requires_cuda_12_0(tmp_path, monkeypatch) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    monkeypatch.setattr(bf16_gemm, "is_cuda_version_at_least", lambda _version: False)
    monkeypatch.setattr(bf16_gemm.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)

    with pytest.raises(RuntimeError, match=r"requires CUDA 12\.0"):
        bf16_gemm.gen_sm90_push_bf16_gemm_module()

    assert not any(tmp_path.iterdir())


def test_sm90_push_bf16_gemm_digest_covers_sources_and_dependencies() -> None:
    from dataclasses import replace

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    snapshot = bf16_gemm._capture_source_snapshot()
    baseline = bf16_gemm._source_digest(snapshot)
    source_name, source = snapshot.sources[0]
    changed_source = replace(
        snapshot,
        sources=((source_name, source + b"\nchanged"), *snapshot.sources[1:]),
    )
    dependency_name, dependency = snapshot.dependencies[0]
    changed_dependency = replace(
        snapshot,
        dependencies=(
            (dependency_name, dependency + b"\nchanged"),
            *snapshot.dependencies[1:],
        ),
    )
    changed_tactic_generator = replace(
        snapshot,
        tactic_generator=snapshot.tactic_generator + b"\nchanged",
    )

    assert bf16_gemm._source_digest(changed_source) != baseline
    assert bf16_gemm._source_digest(changed_dependency) != baseline
    assert bf16_gemm._source_digest(changed_tactic_generator) != baseline


def test_sm90_push_bf16_gemm_workspace_query_does_not_bind_null_storage() -> None:
    source = _package_text("src", "bf16_gemm", "bf16_grouped_gemm_binding.cu")

    assert "WorkspaceView const workspace{}" in source
    assert "bind_workspace(nullptr" not in source


def test_sm90_push_bf16_gemm_serializes_shared_workspace_across_streams() -> None:
    source = _package_text("shim", "bf16_gemm.py")

    assert "completion_event.query()" in source
    assert "active_stream == stream" in source
    assert "cannot overlap calls on" in source
    assert "if self._graph_owned:" in source
    assert "reserved for CUDA graph replay" in source


def test_sm90_push_bf16_capture_handoff_avoids_event_apis() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    current_stream = mock.Mock()
    completion_event = mock.Mock()
    bf16_gemm._guard_stream_handoff(
        current_stream=current_stream,
        stream=2,
        active_stream=1,
        completion_event=completion_event,
        capturing=True,
        error_message="overlap",
    )

    current_stream.wait_event.assert_not_called()
    completion_event.query.assert_not_called()


def test_sm90_push_bf16_schedule_lease_is_fail_closed() -> None:
    import torch

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    workspace = bf16_gemm._Bf16ScheduleWorkspace(
        tensor=object(), signature=(512, 4, torch.device("cuda", 0))
    )
    prepared = bf16_gemm._PreparedSchedule(
        offsets_address=1024,
        offsets_shape=(5,),
        row_capacity=512,
        device=torch.device("cuda", 0),
        stream=7,
    )
    lease = workspace.begin_prepare(prepared)
    workspace.commit_prepare(lease)
    consumed = workspace.consume_prepared(prepared)

    assert consumed.epoch == 1
    assert workspace.state is bf16_gemm._ScheduleState.CONSUMED
    with pytest.raises(RuntimeError, match="unavailable"):
        workspace.consume_prepared(prepared)

    next_lease = workspace.begin_prepare(prepared)
    workspace.abort_prepare(next_lease)
    assert workspace.state is bf16_gemm._ScheduleState.INVALID


def test_sm90_push_bf16_schedule_rejects_identity_mismatch() -> None:
    import torch

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    workspace = bf16_gemm._Bf16ScheduleWorkspace(
        tensor=object(), signature=(512, 4, torch.device("cuda", 0))
    )
    prepared = bf16_gemm._PreparedSchedule(
        offsets_address=1024,
        offsets_shape=(5,),
        row_capacity=512,
        device=torch.device("cuda", 0),
        stream=7,
    )
    lease = workspace.begin_prepare(prepared)
    workspace.commit_prepare(lease)
    mismatched = bf16_gemm._PreparedSchedule(
        offsets_address=2048,
        offsets_shape=(5,),
        row_capacity=512,
        device=torch.device("cuda", 0),
        stream=7,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        workspace.consume_prepared(mismatched)
    assert workspace.state is bf16_gemm._ScheduleState.INVALID


def test_sm90_push_bf16_failed_prepare_invalidates_lease() -> None:
    import torch

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    device = torch.device("cuda", 0)
    runner = bf16_gemm.Sm90PushBf16GroupedGemm.__new__(
        bf16_gemm.Sm90PushBf16GroupedGemm
    )
    runner.device = device
    runner.ffi_runner = mock.Mock()
    runner.ffi_runner.grouped_run.side_effect = RuntimeError("injected")
    runner.schedule_workspace = bf16_gemm._Bf16ScheduleWorkspace(
        tensor=object(), signature=(8, 2, device)
    )
    runner._active_stream = None
    runner._completion_event = None
    runner._graph_owned = False
    activation = SimpleNamespace(shape=(8, 128), device=device)
    offsets = SimpleNamespace(shape=(3,), data_ptr=lambda: 4096)

    with (
        mock.patch.object(
            torch.cuda,
            "current_stream",
            return_value=SimpleNamespace(cuda_stream=7),
        ),
        mock.patch.object(runner, "_is_capturing", return_value=False),
        pytest.raises(RuntimeError, match="injected"),
    ):
        runner.run(object(), activation, object(), offsets, prepare_schedule=True)

    assert runner.schedule_workspace.state is bf16_gemm._ScheduleState.INVALID
    assert runner.schedule_workspace.lease is None


def test_sm90_push_bf16_gemm_has_device_schedule_and_finite_tactics() -> None:
    kernel = _package_text("src", "bf16_gemm", "bf16_grouped_gemm.cuh")
    binding = _package_text("src", "bf16_gemm", "bf16_grouped_gemm_binding.cu")

    assert "prepare_schedule_and_arguments_kernel<<<" in kernel
    assert "prepare_bf16_schedule_kernel<<<" not in kernel
    assert "validate_bf16_schedule_kernel<<<" not in kernel
    assert "using M64Traits = GemmTraits<64" in kernel
    assert "using M128Traits = GemmTraits<128" in kernel
    assert "SM90_PUSH_BF16_FAMILY_MASK" in kernel
    assert "static_assert(!kSwapAB || kNumMTileFamilies == 1)" in kernel
    assert "select_row_families" in kernel
    assert "select_row_families(0, 0).m64_rows == 0" in kernel
    assert "grouped_run_prepared" in binding
    assert "kernel_resource_usage" in binding


def test_sm90_push_bf16_gemm_tactic_matrix_is_finite_and_roundtrips() -> None:
    from itertools import product

    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        CORE_BF16_GEMM_TACTICS,
        SUPPORTED_BF16_GEMM_FAMILY_TACTICS,
        SUPPORTED_BF16_GEMM_TACTICS,
        normalize_bf16_gemm_tactic,
    )

    assert SUPPORTED_BF16_GEMM_FAMILY_TACTICS
    assert 8 <= len(CORE_BF16_GEMM_TACTICS) <= 16
    assert SUPPORTED_BF16_GEMM_TACTICS
    tags = [tactic.tag for tactic in SUPPORTED_BF16_GEMM_TACTICS]
    assert len(tags) == len(set(tags))
    assert {tactic.family_mode for tactic in SUPPORTED_BF16_GEMM_TACTICS} == {
        "m64",
        "m128",
        "dual",
    }
    assert {family.block_n for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        64,
        128,
    }
    assert {family.block_k for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        64,
        128,
    }
    assert {family.stages for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        2,
        3,
        4,
    }
    assert {family.cluster_m for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {1, 2}
    assert {family.schedule for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS} == {
        "pingpong",
        "cooperative",
    }
    assert {tactic.swap_ab for tactic in SUPPORTED_BF16_GEMM_TACTICS} == {
        False,
        True,
    }
    for tactic in SUPPORTED_BF16_GEMM_TACTICS:
        assert normalize_bf16_gemm_tactic(tactic) is tactic
        assert normalize_bf16_gemm_tactic(tactic.tag) == tactic
        if tactic.family_mode == "dual":
            assert tactic.swap_ab is False
        assert tactic.families
    core_families = [
        family for tactic in CORE_BF16_GEMM_TACTICS for family in tactic.families
    ]
    assert {tactic.family_mode for tactic in CORE_BF16_GEMM_TACTICS} == {
        "m64",
        "m128",
        "dual",
    }
    assert {family.block_n for family in core_families} == {64, 128}
    assert {family.block_k for family in core_families} == {64, 128}
    assert {family.stages for family in core_families} == {2, 3, 4}
    assert {family.cluster_m for family in core_families} == {1, 2}
    assert {tactic.swap_ab for tactic in CORE_BF16_GEMM_TACTICS} == {False, True}

    expected_families = set()
    for block_m in (64, 128):
        schedules = ("pingpong",) if block_m == 64 else ("pingpong", "cooperative")
        for block_n, block_k, stages, cluster_m, schedule in product(
            (64, 128), (64, 128), (2, 3, 4), (1, 2), schedules
        ):
            operand_bytes = stages * (block_m + block_n) * block_k * 2
            if operand_bytes <= 220 * 1024:
                expected_families.add(
                    (block_m, block_n, block_k, stages, cluster_m, schedule)
                )
    actual_families = {
        (
            family.block_m,
            family.block_n,
            family.block_k,
            family.stages,
            family.cluster_m,
            family.schedule,
        )
        for family in SUPPORTED_BF16_GEMM_FAMILY_TACTICS
    }
    assert actual_families == expected_families
    assert len(actual_families) == 68

    expected_plans = set()
    m64_families = {family for family in expected_families if family[0] == 64}
    m128_families = {family for family in expected_families if family[0] == 128}
    for family in expected_families:
        mode = "m64" if family[0] == 64 else "m128"
        expected_plans.add(
            (
                mode,
                family if mode == "m64" else None,
                family if mode == "m128" else None,
                False,
            )
        )
        if family[5] != "cooperative" or family[1] == 128:
            expected_plans.add(
                (
                    mode,
                    family if mode == "m64" else None,
                    family if mode == "m128" else None,
                    True,
                )
            )
    for m64 in m64_families:
        for m128 in m128_families:
            if m64[1:4] == m128[1:4]:
                expected_plans.add(("dual", m64, m128, False))
    actual_plans = {
        (
            tactic.family_mode,
            None
            if tactic.m64 is None
            else (
                tactic.m64.block_m,
                tactic.m64.block_n,
                tactic.m64.block_k,
                tactic.m64.stages,
                tactic.m64.cluster_m,
                tactic.m64.schedule,
            ),
            None
            if tactic.m128 is None
            else (
                tactic.m128.block_m,
                tactic.m128.block_n,
                tactic.m128.block_k,
                tactic.m128.stages,
                tactic.m128.cluster_m,
                tactic.m128.schedule,
            ),
            tactic.swap_ab,
        )
        for tactic in SUPPORTED_BF16_GEMM_TACTICS
    }
    assert actual_plans == expected_plans
    assert len(actual_plans) == 212


@pytest.mark.parametrize(
    (
        "expected_m",
        "n",
        "k",
        "sm_count",
        "family_mode",
        "block_n",
        "block_k",
        "stages",
        "cluster_m",
        "swap_ab",
    ),
    [
        (0.0, 64, 64, 132, "m64", 64, 64, 2, 1, False),
        (32.0, 4096, 2048, 132, "m64", 128, 64, 2, 1, True),
        (64.0, 7168, 2048, 132, "m64", 128, 64, 3, 1, False),
        (64.01, 7168, 2048, 132, "m128", 128, 64, 3, 1, False),
        (96.0, 2048, 2048, 132, "m128", 64, 64, 3, 1, False),
        (96.0, 2049, 2049, 78, "m128", 64, 64, 3, 1, False),
        (96.0, 2880, 4096, 78, "m128", 64, 128, 3, 1, False),
        (96.0, 4096, 2880, 78, "m128", 128, 64, 3, 1, False),
        (128.0, 7168, 3072, 114, "m128", 128, 128, 3, 1, False),
        (128.01, 7168, 2048, 132, "dual", 128, 64, 3, 1, False),
    ],
)
def test_sm90_push_bf16_auto_selector_is_deterministic_at_boundaries(
    expected_m: float,
    n: int,
    k: int,
    sm_count: int,
    family_mode: str,
    block_n: int,
    block_k: int,
    stages: int,
    cluster_m: int,
    swap_ab: bool,
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        SUPPORTED_BF16_GEMM_TACTICS,
        select_sm90_push_bf16_gemm_tactic,
    )

    first = select_sm90_push_bf16_gemm_tactic(
        expected_m=expected_m, n=n, k=k, sm_count=sm_count
    )
    second = select_sm90_push_bf16_gemm_tactic(
        expected_m=expected_m, n=n, k=k, sm_count=sm_count
    )

    assert first == second
    tactic, reason = first
    assert tactic in SUPPORTED_BF16_GEMM_TACTICS
    assert reason
    assert tactic.family_mode == family_mode
    assert tactic.swap_ab is swap_ab
    assert {family.block_n for family in tactic.families} == {block_n}
    assert {family.block_k for family in tactic.families} == {block_k}
    assert {family.stages for family in tactic.families} == {stages}
    if tactic.m128 is not None:
        assert tactic.m128.cluster_m == cluster_m
        assert tactic.m128.schedule == ("cooperative" if cluster_m == 2 else "pingpong")


def test_sm90_push_bf16_forced_tactics_change_uri_and_digest() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        CORE_BF16_GEMM_TACTICS,
        _capture_source_snapshot,
        _source_digest,
        sm90_push_bf16_gemm_uri,
    )

    snapshot = _capture_source_snapshot()
    uris = [sm90_push_bf16_gemm_uri(tactic) for tactic in CORE_BF16_GEMM_TACTICS]
    digests = [
        _source_digest(snapshot, tactic=tactic) for tactic in CORE_BF16_GEMM_TACTICS
    ]

    assert len(uris) == len(set(uris))
    assert len(digests) == len(set(digests))
    for tactic, uri in zip(CORE_BF16_GEMM_TACTICS, uris, strict=True):
        assert tactic.tag in uri


def test_sm90_push_bf16_auto_selection_is_independent_of_jit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim import bf16_gemm

    ffi_runner = SimpleNamespace(
        get_workspace_size=lambda *_: 0,
        get_schedule_workspace_size=lambda: 0,
        configure_workspace=lambda *_: None,
    )
    module = SimpleNamespace(init=lambda: ffi_runner)
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")
    monkeypatch.setattr(
        bf16_gemm, "_load_sm90_push_bf16_gemm_module", lambda _tactic: module
    )
    monkeypatch.setattr(bf16_gemm.torch, "empty", lambda *_args, **_kwargs: object())

    production = bf16_gemm.Sm90PushBf16GroupedGemm(
        max_rows=128,
        num_experts=8,
        n=7168,
        k=2048,
        device="cuda:0",
        expected_m=16,
        sm_count=132,
    )
    automatic = bf16_gemm.Sm90PushBf16GroupedGemm(
        max_rows=128,
        num_experts=8,
        n=7168,
        k=2048,
        device="cuda:0",
        tactic="auto",
        expected_m=16,
        sm_count=132,
    )
    forced = bf16_gemm.Sm90PushBf16GroupedGemm(
        max_rows=128,
        num_experts=8,
        n=7168,
        k=2048,
        device="cuda:0",
        tactic=bf16_gemm.CORE_BF16_GEMM_TACTICS[0],
        expected_m=16,
        sm_count=132,
    )

    assert production.tactic is bf16_gemm.DEFAULT_BF16_GEMM_TACTIC
    assert production.selector_kind == "production_default"
    expected, reason = bf16_gemm.select_sm90_push_bf16_gemm_tactic(
        expected_m=16,
        n=7168,
        k=2048,
        sm_count=132,
    )
    assert automatic.tactic == expected
    assert automatic.selector_kind == "shape_load_sm_count_v2"
    assert automatic.selection_reason == reason
    assert forced.tactic == bf16_gemm.CORE_BF16_GEMM_TACTICS[0]
    assert forced.selector_kind == "forced"
    assert forced.selection_reason == "forced internal BF16 GEMM tactic"


def test_sm90_push_bf16_stays_out_of_global_aot_registration() -> None:
    source = _module_text("flashinfer.aot")

    assert "gen_sm90_push_bf16_gemm_module" not in source
    assert "gen_sm90_push_bf16_persistent_gemm_module" not in source
    assert "gen_sm90_push_bf16_fused_fc1_module" not in source


def test_sm90_push_bf16_gemm_rejects_invalid_tactic_combinations() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        Bf16GemmFamilyTactic,
        Bf16GemmTactic,
        normalize_bf16_gemm_tactic,
    )

    m64_c1 = Bf16GemmFamilyTactic(64, 128, 128, 3, 1, "pingpong")
    m128_c1 = Bf16GemmFamilyTactic(128, 128, 128, 3, 1, "pingpong")
    m128_c2 = Bf16GemmFamilyTactic(128, 128, 128, 3, 2, "cooperative")
    with pytest.raises(ValueError, match="single M-tile"):
        Bf16GemmTactic("dual", m64_c1, m128_c2, swap_ab=True)
    with pytest.raises(ValueError, match="inconsistent"):
        Bf16GemmTactic("m64", None, m128_c1)
    with pytest.raises(ValueError, match="inconsistent"):
        Bf16GemmTactic("m128", m64_c1, None)
    with pytest.raises(ValueError, match="inconsistent"):
        Bf16GemmTactic("dual", m64_c1, None)

    unsupported = Bf16GemmTactic(
        "m128",
        m128=Bf16GemmFamilyTactic(128, 128, 128, 4, 2, "cooperative"),
    )
    with pytest.raises(ValueError, match="unknown|canonical"):
        normalize_bf16_gemm_tactic(unsupported)

    with pytest.raises(ValueError, match="BlockN=128"):
        Bf16GemmTactic(
            "m128",
            m128=Bf16GemmFamilyTactic(128, 64, 64, 3, 2, "cooperative"),
            swap_ab=True,
        )


def test_sm90_push_bf16_gemm_fuses_schedule_and_descriptor_preparation() -> None:
    kernel = _package_text("src", "bf16_gemm", "bf16_grouped_gemm.cuh")
    binding = _package_text("src", "bf16_gemm", "bf16_grouped_gemm_binding.cu")

    assert "prepare_schedule_and_arguments_kernel" in kernel
    assert "bool write_schedule" in kernel
    assert "prepare_family_arguments<M64Traits>" in kernel
    assert "prepare_family_arguments<M128Traits>" in kernel
    assert "SM90_PUSH_BF16_FAMILY_MASK" in kernel
    assert "SM90_PUSH_BF16_SWAP_AB" in kernel
    assert "bool prepare_schedule" in binding
    assert "std::conditional_t<SwapAB" in kernel
    assert "UnderlyingProblemShape(n, rows, k)" in kernel
    assert "view.activation_ptrs[expert] = weights" in kernel
    assert "view.weight_ptrs[expert] =" in kernel
    assert "make_cute_packed_stride(StrideD{}, {n, rows, 1})" in kernel


def test_sm90_push_bf16_expected_expert_m_is_independent_of_ep_size() -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        estimate_sm90_push_bf16_expected_m,
    )

    token_capacity, top_k, local_experts = 512, 8, 32
    expected = token_capacity * top_k / local_experts
    ep1 = estimate_sm90_push_bf16_expected_m(
        token_capacity=token_capacity,
        top_k=top_k,
        num_local_experts=local_experts,
    )
    ep8 = estimate_sm90_push_bf16_expected_m(
        token_capacity=token_capacity,
        top_k=top_k,
        num_local_experts=local_experts,
    )

    assert ep1 == expected
    assert ep8 == expected
    assert (1 * token_capacity * top_k) / (1 * local_experts) == expected
    assert (8 * token_capacity * top_k) / (8 * local_experts) == expected


@pytest.mark.parametrize(
    ("token_capacity", "top_k", "local_experts", "error"),
    [
        (-1, 8, 32, "token_capacity"),
        (512, 0, 32, "top_k"),
        (512, 8, 0, "num_local_experts"),
    ],
)
def test_sm90_push_bf16_expected_expert_m_rejects_invalid_envelope(
    token_capacity: int, top_k: int, local_experts: int, error: str
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.push_style_megamoe.shim.bf16_gemm import (
        estimate_sm90_push_bf16_expected_m,
    )

    with pytest.raises(ValueError, match=error):
        estimate_sm90_push_bf16_expected_m(
            token_capacity=token_capacity,
            top_k=top_k,
            num_local_experts=local_experts,
        )


def test_sm90_push_bf16_runtime_uses_the_expected_m_helper() -> None:
    runner = _package_text("shim", "bf16_runner.py")

    assert "expected_m = estimate_sm90_push_bf16_expected_m(" in runner
    assert "token_capacity=pipe.token_capacity" in runner
    assert "top_k=pipe.K" in runner
    assert "num_local_experts=pipe.E" in runner
    assert runner.count("expected_m=expected_m") >= 2
    assert 'fc1_gemm_tactic: Bf16GemmTactic | str = "production"' in runner
    assert 'fc2_gemm_tactic: Bf16GemmTactic | str = "production"' in runner
