"""SVDQuant tuning must respect the provisioned scratch workspace."""

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from flashinfer.gemm import svdquant_sm120_cutlass


@pytest.mark.parametrize(
    ("config", "workspace_index"),
    (
        (svdquant_sm120_cutlass._NVFP4_SVDQUANT_GEMM_TUNING_CONFIG_EXACT, 9),
        (svdquant_sm120_cutlass._SM120_LINEAR_TUNING_CONFIG_COLD, 13),
    ),
    ids=("gemm", "linear"),
)
def test_scratch_synthesis_does_not_randomize_workspace(
    monkeypatch: pytest.MonkeyPatch,
    config: TuningConfig,
    workspace_index: int,
) -> None:
    """Exercise the actual profiles while rejecting random scratch synthesis."""
    tuner_module = importlib.import_module("flashinfer.autotuner.autotuner")
    initialize = tuner_module.autotuner_initializer_rand_scaled
    workspace_shape = (4096,)

    def reject_scratch_randomization(
        shapes: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        assert shapes != workspace_shape, (
            "scratch workspace used random FP32 temporaries"
        )
        return initialize(shapes, dtype, device)

    monkeypatch.setattr(
        tuner_module, "autotuner_initializer_rand_scaled", reject_scratch_randomization
    )
    inputs = [torch.ones((2, 256), dtype=torch.uint8) for _ in range(workspace_index)]
    inputs.append(torch.empty(workspace_shape, dtype=torch.uint8))
    tuner = AutoTuner.get()
    profiles = tuner._generate_optimization_profiles(config, inputs)
    assert len(profiles) == 1
    prepared = tuner._prepare_input_tensors(profiles[0], inputs)
    workspace = prepared[workspace_index]
    assert workspace is not None
    assert workspace.shape == workspace_shape
    assert workspace.dtype == torch.uint8
    assert workspace.data_ptr() != inputs[workspace_index].data_ptr()


@dataclass(frozen=True)
class _WorkspaceCase:
    runner: TunableRunner
    inputs: list[torch.Tensor]
    config: TuningConfig
    op_name: str
    large_tactic: int
    module: SimpleNamespace
    launches: list[int]


@pytest.fixture(
    params=(
        (route, rank) for route in ("gemm", "fused", "unfused") for rank in (32, 64)
    )
)
def workspace_case(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> _WorkspaceCase:
    backend = svdquant_sm120_cutlass
    route, rank = request.param
    m, n, k = 64, 128, 3072
    launches: list[int] = []

    def check_launch(workspace: torch.Tensor, tactic: int) -> None:
        row = (
            module.nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank)
            if tactic == -1
            else tactic
        )
        required = (16, 64)[row]
        assert required <= workspace.numel() * workspace.element_size(), (
            f"tactic {row} requires {required} bytes, workspace buffer too small"
        )
        launches.append(row)

    module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 2,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, r, t: t in (0, 1),
        nvfp4_svdquant_gemm_workspace_size=lambda m, n, k, t: (16, 64)[t],
        nvfp4_svdquant_gemm_fallback_tactic=lambda m, n, k, r: 0,
        nvfp4_quantize_smooth=lambda *args: None,
        nvfp4_svdquant_gemm=lambda *args: check_launch(args[9], args[10]),
        nvfp4_svdquant_linear_sm120=lambda *args: check_launch(
            args[13], backend._sm120_unpack_tactic(args[14])[0]
        ),
    )
    monkeypatch.setattr(
        backend, "get_nvfp4_svdquant_sm120_module", lambda rank=32: module
    )
    monkeypatch.setattr(
        backend, "_get_nvfp4_svdquant_module_for_device", lambda *args: module
    )
    monkeypatch.setattr(backend, "_sm120_workspace_budget_bytes", lambda *args: 16)
    monkeypatch.setattr(backend, "_SM120_WORKSPACE_BUDGET_CACHE", {})
    monkeypatch.setattr(backend, "_SM120_WORKSPACE_BYTES_CACHE", {})
    monkeypatch.setattr(backend, "_SM120_LINEAR_DISPATCH_CACHE", {})
    x = torch.zeros((m, k), dtype=torch.bfloat16)
    weight = torch.empty((n, k // 2), dtype=torch.uint8)
    sf = torch.empty(128 * (k // 16), dtype=torch.uint8)
    scale = torch.ones(1)
    l2t = torch.zeros((k, rank), dtype=torch.bfloat16)
    l1 = torch.zeros((n, rank), dtype=torch.bfloat16)
    bias = torch.zeros(n, dtype=torch.bfloat16)
    xq = torch.empty((m, k // 2), dtype=torch.uint8)
    down = torch.empty((m, rank), dtype=torch.bfloat16)
    out = torch.empty((m, n), dtype=torch.bfloat16)
    workspace = torch.empty(64, dtype=torch.uint8)
    if route == "gemm":
        runner = backend._nvfp4_svdquant_gemm_runner(False, x.device, rank)
        inputs = [xq, weight, sf, sf, scale, down, l1, bias, out, workspace]
        config = backend._NVFP4_SVDQUANT_GEMM_TUNING_CONFIG_EXACT
        op_name = backend._svdquant_op_name("sm120", rank)
        large_tactic = 1
    else:
        runner = (
            backend._sm120_fused_linear_runner(False, x.device, rank)
            if route == "fused"
            else backend._sm120_cutlass_linear_runner(False, x.device, rank)
        )
        inputs = [
            x,
            weight,
            sf,
            scale,
            torch.ones(k, dtype=torch.bfloat16),
            l2t,
            l1,
            scale,
            bias,
            xq,
            sf,
            down,
            out,
            workspace,
        ]
        config = backend._SM120_LINEAR_TUNING_CONFIG_COLD
        op_name = backend._sm120_linear_op_name(m, k, rank, enable_pdl=False)
        # Nonzero producer bits must be stripped before querying GEMM workspace.
        large_tactic = backend._sm120_pack_tactic(1, 1) if route == "fused" else 1
    return _WorkspaceCase(
        runner, inputs, config, op_name, large_tactic, module, launches
    )


@pytest.mark.parametrize("source", ("memory", "file"))
@pytest.mark.parametrize("workspace_bytes", (16, 64))
def test_cached_tactic_must_fit_actual_workspace(
    workspace_case: _WorkspaceCase,
    source: Literal["memory", "file"],
    workspace_bytes: int,
) -> None:
    case = workspace_case
    tuner = AutoTuner()
    shapes = tuple(t.shape for t in case.inputs)
    key = tuner._get_cache_key(case.op_name, case.runner, shapes, case.config, ())
    if source == "memory":
        tuner.profiling_cache[key] = (case.large_tactic, None)
    else:
        tuner._file_configs[key.file_key] = (
            type(case.runner).__name__,
            case.large_tactic,
        )
    # The storage remains large; only the supplied view is available to the kernel.
    case.inputs[-1] = case.inputs[-1][:workspace_bytes]
    assert key == tuner._get_cache_key(
        case.op_name, case.runner, tuple(t.shape for t in case.inputs), case.config, ()
    )
    runner, tactic = tuner.choose_one(
        case.op_name, [case.runner], case.config, case.inputs
    )
    assert (tactic == case.large_tactic) == (workspace_bytes >= 64)
    runner(case.inputs, tactic=tactic)


def test_linear_dispatch_cache_rechecks_smaller_workspace(
    workspace_case: _WorkspaceCase,
) -> None:
    case = workspace_case
    if len(case.inputs) != 14:
        pytest.skip("linear-only dispatch cache")
    tuner = AutoTuner()
    key = tuner._get_cache_key(
        case.op_name, case.runner, tuple(t.shape for t in case.inputs), case.config, ()
    )
    tuner.profiling_cache[key] = (case.large_tactic, None)

    def choose() -> tuple[TunableRunner, int]:
        return svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            [case.runner],
            case.config,
            case.inputs,
            device=case.inputs[0].device,
            m=64,
            n=128,
            k=3072,
            rank=case.inputs[5].shape[1],
            enable_pdl=False,
            has_bias=True,
        )

    runner, tactic = choose()
    assert tactic == case.large_tactic
    runner(case.inputs, tactic=tactic)
    case.inputs[-1] = torch.empty(16, dtype=torch.uint8)
    runner, tactic = choose()
    assert tactic != case.large_tactic
    runner(case.inputs, tactic=tactic)


@pytest.mark.parametrize("source", ("bundled", "fallback"))
@pytest.mark.parametrize("workspace_bytes", (8, 16))
def test_public_dispatch_checks_tactics_that_bypass_cache_validation(
    monkeypatch: pytest.MonkeyPatch,
    workspace_case: _WorkspaceCase,
    source: Literal["bundled", "fallback"],
    workspace_bytes: int,
) -> None:
    case = workspace_case
    backend = svdquant_sm120_cutlass
    tuner_module = importlib.import_module("flashinfer.autotuner.autotuner")
    monkeypatch.setattr(AutoTuner, "_instance", AutoTuner())
    monkeypatch.setenv(
        "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1" if source == "bundled" else "0"
    )
    monkeypatch.setattr(
        tuner_module, "load_from_file", lambda key: (True, 0, case.large_tactic, None)
    )
    monkeypatch.setattr(
        case.module, "nvfp4_svdquant_gemm_fallback_tactic", lambda *args: 1
    )
    monkeypatch.setattr(backend, "DEFAULT_WORKSPACE_SIZE", 16)
    case.inputs[-1] = case.inputs[-1][:workspace_bytes]
    monkeypatch.setattr(backend, "_get_cache_buf", lambda *args: case.inputs[-1])
    monkeypatch.setattr(
        backend, "_cached_sm120_linear_runners", lambda *args: [case.runner]
    )
    monkeypatch.setattr(
        backend, "_nvfp4_svdquant_gemm_runners", lambda *args: [case.runner]
    )

    def dispatch() -> torch.Tensor:
        if len(case.inputs) == 10:
            a, b, a_sf, b_sf, alpha, d, l1, bias, out = case.inputs[:9]
            return backend.mm_nvfp4_svdquant(
                a, b, a_sf, b_sf, alpha, d, l1, bias, out, enable_pdl=False
            )
        x, weight, weight_sf, alpha, pqs, l2t, l1, scale, bias = case.inputs[:9]
        return backend.svdquant_linear(
            x, weight, weight_sf, alpha, pqs, l2t, l1, scale, bias, enable_pdl=False
        )

    if workspace_bytes < 16:
        with pytest.raises(RuntimeError, match="No feasible SVDQuant tactic fits"):
            dispatch()
        assert not case.launches
    else:
        assert dispatch().shape == (64, 128)
        assert case.launches == [0]


def test_sm100_without_workspace_metadata_keeps_cached_tactic() -> None:
    assert svdquant_sm120_cutlass._svdquant_tactic_fits_workspace(
        SimpleNamespace(), (64, 128, 3072, 32), torch.empty(16, dtype=torch.uint8), 1
    )
