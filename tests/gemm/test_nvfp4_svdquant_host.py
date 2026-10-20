"""NVFP4 SVDQuant dispatch, routes, workspace, JIT and native host guards.

The C++ harnesses compile production guards and arithmetic under UBSan while
stubbing CUDA/CUTLASS calls. Operator and GPU correctness tests live in
``test_nvfp4_svdquant_gemm.py``."""

import ast
import importlib
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Final, Literal

import pytest
import torch
from packaging.version import Version

from flashinfer import svdquant_linear
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from flashinfer.gemm import gemm_svdquant, svdquant_sm120_cutlass
from flashinfer.gemm import svdquant_sm120_cutlass as backend
from flashinfer.gemm import svdquant_sm120_routes as _sm120_routes
from flashinfer.gemm import svdquant_sm120_routes as routes
from flashinfer.gemm.svdquant_sm120_routes import (
    SM120_FAMILY_M537,
    SM120_FAMILY_SMALL_M_PACKED,
    SM120_FUSED_LINEAR_MK,
    SM120_LINEAR_ROUTE_ABI_OVERRIDES,
    SM120_LINEAR_ROUTE_V6_MKR,
    SM120_PRODUCER_GEOMETRY_LADDER,
    SM120_RANK64_PRODUCER_GEOMETRY_LADDER,
    sm120_linear_route_abi_version,
    sm120_producer_geometry_is_valid,
    sm120_producer_variants,
    sm120_variant_packs_l2t,
)
from flashinfer.jit import core
from flashinfer.jit.gemm import svdquant_sm120
from flashinfer.jit.gemm.svdquant_sm120_configs import SVDQUANT_SM120_CONFIGS
from flashinfer.testing.svdq_model_shapes import (
    MODEL_SHAPE_CASES,
    MODEL_SHAPE_COUNT,
    ShapeCase,
)

# Backend dispatch, rank handling and linear runner selection.


@pytest.fixture
def auto_backend_case(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Run real public dispatch and tuning with CPU kernels and deterministic timings."""
    import flashinfer.cute_dsl
    import flashinfer.utils
    from flashinfer.gemm import svdquant_sm120_auto

    svdquant_sm120_auto._auto_runners.cache_clear()
    request.addfinalizer(svdquant_sm120_auto._auto_runners.cache_clear)
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    monkeypatch.setattr(flashinfer.utils, "get_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(gemm_svdquant, "get_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(gemm_svdquant, "get_cuda_version", lambda: Version("12.9"))
    monkeypatch.setattr(flashinfer.cute_dsl, "is_cute_dsl_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    profiles: list[tuple[str, int]] = []
    builds: list[str] = []

    class Runner(TunableRunner):
        def __init__(self, label: str, output_index: int) -> None:
            self.label = label
            self.output_index = output_index

        def get_cache_key_extras(self, inputs):
            return (self.label,)

        def get_valid_tactics(self, inputs, profile):
            return [0]

        def validate_tactic(self, inputs, tactic):
            return tactic in (-1, 0)

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            value = 2 if self.label.startswith("cutlass") else 1
            inputs[self.output_index].fill_(value)
            return inputs[self.output_index]

    def native(label: str, output_index: int):
        builds.append(label)
        return Runner(label, output_index)

    monkeypatch.setattr(
        gemm_svdquant, "_sm120_nvfp4_svdquant_runner", lambda _: Runner("cute", 8)
    )
    monkeypatch.setattr(
        gemm_svdquant,
        "_sm120_nvfp4_svdquant_unfused_runner",
        lambda _: Runner("cute-unfused", 8),
    )
    monkeypatch.setattr(
        backend, "_nvfp4_svdquant_gemm_runner", lambda *args: native("cutlass", 8)
    )
    monkeypatch.setattr(
        backend, "_sm120_cutlass_linear_runner", lambda *args: native("cutlass", 12)
    )
    monkeypatch.setattr(
        backend, "_sm120_fused_linear_runner", lambda *args: native("cutlass-fused", 12)
    )
    monkeypatch.setattr(backend, "_sm120_max_workspace_bytes", lambda *args: 16)
    monkeypatch.setattr(backend, "DEFAULT_WORKSPACE_SIZE", 16)
    monkeypatch.setattr(
        backend,
        "_get_cache_buf",
        lambda _, size, device: torch.empty(size, dtype=torch.uint8),
    )

    def quantize(x, scale, global_scale, **kwargs):
        m, k = x.shape
        return (
            torch.zeros((m, k // 2), dtype=torch.uint8),
            torch.zeros(gemm_svdquant._swizzled_sf_size(m, k // 16), dtype=torch.uint8),
        )

    monkeypatch.setattr(gemm_svdquant, "nvfp4_quantize_smooth", quantize)

    def profile(runner, inputs, tactic, tuning_config, **kwargs):
        result = runner(inputs, tactic=tactic)
        value = int(result[0, 0])
        profiles.append((type(runner).__name__, len(inputs)))
        return 1.0 if value == 2 else 10.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    m, n, k, rank = getattr(request, "param", (64, 128, 256, 32))
    x = torch.ones((m, k), dtype=torch.bfloat16)
    weight = torch.zeros((n, k // 2), dtype=torch.uint8)
    sf = torch.zeros(gemm_svdquant._swizzled_sf_size(n, k // 16), dtype=torch.uint8)
    scale = torch.ones(1)
    l2 = torch.ones((k, rank), dtype=torch.bfloat16)
    l1 = torch.ones((n, rank), dtype=torch.bfloat16)
    operands = [x, weight, sf, scale, x[0], l2, l1, scale, None]

    def call(operation: str, enable_pdl: bool = False):
        x, weight, sf, scale, pqs, l2, l1, global_scale, bias = operands
        if operation == "gemm":
            xq, x_sf = quantize(x, pqs, global_scale)
            return gemm_svdquant.mm_nvfp4_svdquant(
                xq,
                weight,
                x_sf,
                sf,
                scale,
                torch.mm(x, l2),
                l1,
                bias=bias,
                backend="auto",
                enable_pdl=enable_pdl,
            )
        return gemm_svdquant.svdquant_linear(
            *operands,
            backend="auto",
            enable_pdl=enable_pdl,
        )

    return tuner, call, profiles, builds, operands


@pytest.mark.parametrize("operation", ["gemm", "linear"])
@pytest.mark.parametrize("source", ["memory", "file"])
def test_auto_tunes_cutlass_and_replays_winner(
    auto_backend_case, operation, source, tmp_path
):
    from flashinfer import autotune

    tuner, call, profiles, builds, _ = auto_backend_case
    with autotune(True):
        result = call(operation)
    assert torch.all(result == 2), "auto never offered the faster CUTLASS backend"
    assert profiles and builds
    assert tuner.profiling_cache
    if source == "file":
        path = str(tmp_path / "auto.json")
        tuner.save_configs(path)
        tuner.clear_cache()
        assert tuner.load_configs(path)
    count = len(profiles)
    replay = call(operation)
    assert torch.all(replay == 2)
    assert len(profiles) == count
    if operation == "linear":
        assert all(size == 10 for _, size in profiles), "tune the full linear chain"


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_untuned_keeps_cute_without_cutlass_build(auto_backend_case, operation):
    _, call, profiles, builds, _ = auto_backend_case
    assert torch.all(call(operation) == 1)
    assert not profiles
    assert not builds


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_can_choose_cute_over_cutlass(auto_backend_case, operation, monkeypatch):
    from flashinfer import autotune

    tuner, call, profiles, builds, _ = auto_backend_case
    profile = tuner._profile_single_kernel
    monkeypatch.setattr(
        tuner,
        "_profile_single_kernel",
        lambda *args, **kwargs: -profile(*args, **kwargs),
    )
    with autotune(True):
        assert torch.all(call(operation) == 1)
    assert builds and profiles
    assert torch.all(call(operation) == 1)


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_uses_cutlass_without_cute_dsl(auto_backend_case, operation, monkeypatch):
    import flashinfer.cute_dsl

    _, call, profiles, builds, _ = auto_backend_case
    monkeypatch.setattr(flashinfer.cute_dsl, "is_cute_dsl_available", lambda: False)
    assert torch.all(call(operation) == 2)
    assert builds and not profiles


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_drops_cached_cutlass_for_unaligned_lora(auto_backend_case, operation):
    from flashinfer import autotune

    _, call, profiles, builds, operands = auto_backend_case
    with autotune(True):
        assert torch.all(call(operation) == 2)
    aligned = operands[6]
    operands[6] = torch.ones(aligned.numel() + 1, dtype=aligned.dtype)[1:].reshape(
        aligned.shape
    )
    assert operands[6].data_ptr() % 16 != 0
    profiles.clear()
    builds.clear()
    assert torch.all(call(operation) == 1)
    assert not builds and not profiles


@pytest.mark.parametrize("operation", ["gemm", "linear"])
@pytest.mark.parametrize("change", ["pdl", "abi"])
def test_auto_does_not_reuse_winner_across_pdl_or_abi(
    auto_backend_case, operation, change, monkeypatch
):
    from flashinfer import autotune

    _, call, _, _, _ = auto_backend_case
    with autotune(True):
        assert torch.all(call(operation) == 2)
    if change == "abi":
        monkeypatch.setattr(
            backend, "_SM120_TACTIC_ABI_VERSION", backend._SM120_TACTIC_ABI_VERSION + 1
        )
        monkeypatch.setattr(
            backend,
            "_sm120_linear_op_name",
            lambda *args, **kwargs: "future_linear_abi",
        )
    assert torch.all(call(operation, enable_pdl=change == "pdl") == 1)


@pytest.mark.parametrize(
    "flag", ["FLASHINFER_NVFP4_4OVER6", "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"]
)
def test_auto_linear_quantization_policy_excludes_cached_native(
    auto_backend_case, monkeypatch, flag
):
    from flashinfer import autotune

    _, call, profiles, builds, _ = auto_backend_case
    with autotune(True):
        assert torch.all(call("linear") == 2)
    monkeypatch.setenv(flag, "1")
    profiles.clear()
    builds.clear()
    assert torch.all(call("linear") == 1)
    assert not profiles and not builds


@pytest.mark.parametrize(
    "auto_backend_case", [(64, 128, 3072, 32), (64, 128, 3072, 64)], indirect=True
)
def test_auto_linear_measures_both_native_prefixes(auto_backend_case):
    from flashinfer import autotune

    _, call, profiles, builds, _ = auto_backend_case
    with autotune(True):
        assert torch.all(call("linear") == 2)
    assert {"cutlass", "cutlass-fused"} <= set(builds)
    assert all(size == 10 for _, size in profiles)


@pytest.mark.parametrize("index", [1, 2, 3, 5, 6, 8])
def test_auto_linear_validates_before_tuning(auto_backend_case, index):
    _, call, profiles, builds, operands = auto_backend_case
    operands[index] = torch.zeros((1, 32) if index == 5 else (1,), dtype=torch.int64)
    with pytest.raises(ValueError):
        call("linear")
    assert not profiles and not builds


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_keeps_cute_when_native_build_is_unavailable(
    auto_backend_case, operation, monkeypatch
):
    from flashinfer import autotune

    _, call, profiles, _, _ = auto_backend_case

    def unavailable(*args):
        raise RuntimeError("native compiler unavailable")

    monkeypatch.setattr(backend, "_nvfp4_svdquant_gemm_runner", unavailable)
    monkeypatch.setattr(backend, "_sm120_cutlass_linear_runner", unavailable)
    monkeypatch.setattr(backend, "_sm120_fused_linear_runner", unavailable)
    with autotune(True):
        assert torch.all(call(operation) == 1)
    assert profiles


@pytest.mark.parametrize("operation", ["gemm", "linear"])
@pytest.mark.parametrize("capability", [(10, 0), (12, 1)])
def test_auto_keeps_other_architectures_on_existing_backends(
    auto_backend_case, operation, capability, monkeypatch
):
    import flashinfer.utils
    from flashinfer import autotune

    _, call, _, builds, _ = auto_backend_case
    monkeypatch.setattr(
        flashinfer.utils, "get_compute_capability", lambda _: capability
    )
    monkeypatch.setattr(gemm_svdquant, "get_compute_capability", lambda _: capability)
    monkeypatch.setattr(
        gemm_svdquant,
        "_nvfp4_svdquant_gemm_runner",
        gemm_svdquant._sm120_nvfp4_svdquant_runner,
    )
    monkeypatch.setattr(
        gemm_svdquant,
        "_get_cache_buf",
        lambda *args: torch.empty(16, dtype=torch.uint8),
    )
    with autotune(True):
        assert torch.all(call(operation) == 1)
    assert not builds


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_empty_batch_skips_tuning_and_native_build(auto_backend_case, operation):
    from flashinfer import autotune

    _, call, profiles, builds, operands = auto_backend_case
    operands[0] = operands[0][:0]
    with autotune(True):
        result = call(operation)
    assert result.shape == (0, operands[1].shape[0])
    assert not profiles and not builds


def test_auto_linear_prefix_and_pooled_alpha_match_composition(
    auto_backend_case, monkeypatch
):
    _, call, _, _, operands = auto_backend_case
    operands[3] = torch.tensor([0.5, 999.0])
    operands[8] = torch.ones(operands[1].shape[0], dtype=torch.bfloat16)

    class CpuLoraRunner(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, **kwargs):
            _, _, _, _, alpha, down, l1, bias, out = inputs
            assert alpha.shape == (1,)
            out.copy_((down.float() @ l1.float().T) * alpha + bias)
            return out

    monkeypatch.setattr(
        gemm_svdquant, "_sm120_nvfp4_svdquant_runner", lambda _: CpuLoraRunner()
    )
    expected = (
        torch.mm(operands[0], operands[5]).float() @ operands[6].float().T
    ) * 0.5 + operands[8]
    torch.testing.assert_close(call("linear"), expected.bfloat16(), rtol=0, atol=0)


def test_auto_native_adapter_rechecks_actual_workspace(workspace_case, monkeypatch):
    from flashinfer.gemm.svdquant_sm120_auto import _CutlassAutoRunner

    case = workspace_case
    linear = len(case.inputs) == 14
    fused = type(case.runner).__name__ == "Sm120FusedLinearRunner"
    adapter = _CutlassAutoRunner(False, linear=linear, fused=fused)
    for name in (
        "_nvfp4_svdquant_gemm_runner",
        "_sm120_cutlass_linear_runner",
        "_sm120_fused_linear_runner",
    ):
        monkeypatch.setattr(backend, name, lambda *args: case.runner)
    monkeypatch.setattr(backend, "_sm120_max_workspace_bytes", lambda *args: 64)
    monkeypatch.setattr(backend, "_get_cache_buf", lambda *args: case.inputs[-1])
    values = case.inputs[:9] + ([case.inputs[12]] if linear else [])
    assert adapter.validate_tactic(values, case.large_tactic)
    case.inputs[-1] = case.inputs[-1][:16]
    assert not adapter.validate_tactic(values, case.large_tactic)
    adapter(values, tactic=case.large_tactic)
    assert case.launches == [0]


def test_backend_for_capability_routing():
    assert svdquant_sm120_cutlass._svdquant_backend_for_capability(10, 0) == "sm100"
    assert svdquant_sm120_cutlass._svdquant_backend_for_capability(10, 3) == "sm100"
    assert svdquant_sm120_cutlass._svdquant_backend_for_capability(12, 0) == "sm120"


@pytest.mark.parametrize(
    "major,minor", [(9, 0), (8, 9), (10, 1), (10, 2), (11, 0), (12, 1)]
)
def test_backend_for_capability_rejects_unsupported(major, minor):
    with pytest.raises(ValueError, match="not supported"):
        svdquant_sm120_cutlass._svdquant_backend_for_capability(major, minor)


class _FakeTensor:
    """Minimal stand-in for the tensor surface the public entries actually read.

    These tests grew SimpleNamespace fakes and then chased AttributeErrors one
    attribute at a time as the entry points came to read more of the tensor
    surface -- dtype, then numel, then size. A small class states the surface
    once, so a new read fails loudly here instead of somewhere downstream.
    """

    def __init__(self, shape, dtype=None, device="cuda:0", contiguous=True, **kwargs):
        # torch.empty takes either a size tuple or bare ints.
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype if dtype is not None else torch.float32
        self.device = device
        self.is_cuda = str(device).startswith("cuda")
        self.kwargs = kwargs
        self._contiguous = contiguous

    def numel(self):
        return math.prod(self.shape) if self.shape else 0

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def is_contiguous(self):
        return self._contiguous

    def __repr__(self):
        return f"_FakeTensor(shape={self.shape}, dtype={self.dtype})"


def _fake_tensor(shape, dtype, is_cuda=True, contiguous=True):
    """Duck-typed stand-in for what _check_mm_nvfp4_svdquant_problem reads."""
    return _FakeTensor(
        shape, dtype=dtype, device="cuda:0" if is_cuda else "cpu", contiguous=contiguous
    )


def _problem(m=256, n=256, k=256, rank=32):
    a = _fake_tensor((m, k // 2), torch.uint8)
    b = _fake_tensor((n, k // 2), torch.uint8)
    a_sf = _fake_tensor((m * k // 16,), torch.uint8)
    b_sf = _fake_tensor((n * k // 16,), torch.uint8)
    alpha = _fake_tensor((1,), torch.float32)
    d = _fake_tensor((m, rank), torch.bfloat16)
    l1 = _fake_tensor((n, rank), torch.bfloat16)
    return a, b, a_sf, b_sf, alpha, d, l1


@pytest.mark.parametrize("rank", [64, 96, 128])
def test_ranks_above_32_clear_the_shared_problem_check(monkeypatch, rank):
    """The shared check admits any positive multiple of 32; the rank gate is elsewhere.

    This used to assert that _check_mm_nvfp4_svdquant_problem refuses rank > 32
    on SM120, matching "rank 32 only". It never did -- that message comes from
    svdquant_linear in this backend, while the shared check only enforces the
    granularity -- and the premise has since stopped being true anyway: the
    collective takes its LoRA rank as a build parameter, so a module compiled
    for rank 64 serves it. What decides a rank now is which module you built
    and which entry point you call, not the architecture.
    """
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank)) is True


@pytest.mark.parametrize("rank", [8, 48, 33])
def test_ranks_off_the_granularity_are_refused(monkeypatch, rank):
    """The granularity itself is still a hard boundary, on every architecture."""
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    with pytest.raises(ValueError, match="multiple of 32"):
        gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank))


def test_fused_linear_admission_supports_ranks_32_and_64():
    """The fused prefix has separate rank-32 and rank-64 specializations."""
    for rank in (96, 128):
        assert not svdquant_sm120_cutlass._sm120_fused_linear_supported(64, 3072, rank)
    for rank in (32, 64):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(64, 3072, rank)


def test_sm120_accepts_rank_32(monkeypatch):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=32)) is True


def test_fused_prefix_rank_limit_does_not_restrict_shared_gemm_validation():
    """The shared GEMM check also admits ranks with no SM120 fused prefix."""
    m, k = 64, 3072
    for rank in (96, 128):
        assert not svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
        assert (
            gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank)) is True
        )


@pytest.mark.parametrize("rank", [32, 64, 128])
def test_sm100_keeps_multiple_of_32_ranks(monkeypatch, rank):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (10, 0)
    )
    assert gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=rank)) is True


def test_rank_not_multiple_of_32_rejected_everywhere(monkeypatch):
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (10, 0)
    )
    with pytest.raises(ValueError, match="positive multiple"):
        gemm_svdquant._check_mm_nvfp4_svdquant_problem(*_problem(rank=48))


def test_module_getter_routes_by_capability(monkeypatch):
    calls = []
    # Patch where the name is looked up, not where it is defined:
    # svdquant_sm120_cutlass imported get_nvfp4_svdquant_module into its own
    # namespace, so patching gemm_svdquant leaves the bound name untouched and
    # the real getter runs -- which then tries to build an SM100 module on
    # whatever card the tests are on.
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_module",
        lambda: calls.append("sm100") or "M100",
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_sm120_module",
        lambda lora_rank=32: calls.append("sm120") or "M120",
    )

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (10, 3)
    )
    assert svdquant_sm120_cutlass._get_nvfp4_svdquant_module_for_device("dev") == "M100"

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    assert svdquant_sm120_cutlass._get_nvfp4_svdquant_module_for_device("dev") == "M120"

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (9, 0)
    )
    with pytest.raises(ValueError, match="not supported"):
        svdquant_sm120_cutlass._get_nvfp4_svdquant_module_for_device("dev")

    assert calls == ["sm100", "sm120"]


class _FakeRunner:
    def get_valid_tactics(self, inputs, profile):
        return [0, 1]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return None


def _fake_runner():
    from flashinfer.autotuner import TunableRunner

    class FakeRunner(_FakeRunner, TunableRunner):
        pass

    return FakeRunner()


# Ten shapes matching the real input list (a, b, a_sf, b_sf, alpha, d, l1, bias,
# out, workspace): the tuning config's dynamic/constraint specs index into them.
_CACHE_TEST_SHAPES = (
    torch.Size([128, 32]),
    torch.Size([64, 32]),
    torch.Size([1024]),
    torch.Size([1024]),
    torch.Size([1]),
    torch.Size([128, 32]),
    torch.Size([64, 32]),
    torch.Size([64]),
    torch.Size([128, 64]),
    torch.Size([1024]),
)


def _seed_cache(tuner, op_name, runner, shapes, config, tactic):
    from flashinfer.autotuner import AutoTuner

    _, _, _, profile = tuner.search_cache(op_name, [runner], shapes, config)
    key = AutoTuner._get_cache_key(op_name, runner, shapes, config, ())
    # (tactic, profile). The runner id used to lead this tuple; AutoTuner reads
    # it as a pair now, so seeding the old shape raises on lookup rather than
    # missing, which is why this read as a cache-crossing failure.
    tuner.profiling_cache[key] = (tactic, profile)
    return key


def test_autotune_cache_namespaces_do_not_cross():
    """A tactic cached under the SM100 op name must not satisfy an SM120 lookup."""
    from flashinfer.autotuner import AutoTuner

    tuner = AutoTuner.get()
    runner = _fake_runner()
    config_sm100 = svdquant_sm120_cutlass._svdquant_tuning_config("sm100")
    config_sm120 = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")
    key_sm100 = _seed_cache(
        tuner,
        svdquant_sm120_cutlass._svdquant_op_name("sm100"),
        runner,
        _CACHE_TEST_SHAPES,
        config_sm100,
        7,
    )
    try:
        hit_sm100, _, tactic_sm100, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm100"),
            [runner],
            _CACHE_TEST_SHAPES,
            config_sm100,
        )
        assert hit_sm100 and tactic_sm100 == 7

        hit_sm120, _, tactic_sm120, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm120"),
            [runner],
            _CACHE_TEST_SHAPES,
            config_sm120,
        )
        assert not hit_sm120 or tactic_sm120 != 7, (
            "an SM100-cached tactic leaked into the SM120 lookup"
        )
    finally:
        tuner.profiling_cache.pop(key_sm100, None)


def test_sm120_op_name_is_tactic_abi_versioned():
    """The SM120 namespace embeds the tactic-table ABI version."""
    name = svdquant_sm120_cutlass._svdquant_op_name("sm120")
    assert name == (
        f"nvfp4_svdquant_gemm_sm120_v{svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION}"
    )
    assert name != "nvfp4_svdquant_gemm_sm120", (
        "v1 (pre-table) namespace must not be reused"
    )


def test_stale_v1_namespace_cache_never_replays():
    """Tactic ids cached under the pre-table v1 namespace must miss under v2."""
    tuner = _get_tuner()
    runner = _fake_runner()
    config = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")
    key_v1 = _seed_cache(
        tuner, "nvfp4_svdquant_gemm_sm120", runner, _CACHE_TEST_SHAPES, config, 9
    )
    try:
        hit, _, tactic, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm120"),
            [runner],
            _CACHE_TEST_SHAPES,
            config,
        )
        assert not hit or tactic != 9, "a v1-cached tactic id replayed under v2"
    finally:
        tuner.profiling_cache.pop(key_v1, None)


def test_same_process_stale_cache_invalidated_by_version_bump(monkeypatch):
    """An in-memory entry from the current version misses after an ABI bump."""
    tuner = _get_tuner()
    runner = _fake_runner()
    config = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")
    key_cur = _seed_cache(
        tuner,
        svdquant_sm120_cutlass._svdquant_op_name("sm120"),
        runner,
        _CACHE_TEST_SHAPES,
        config,
        5,
    )
    try:
        monkeypatch.setattr(
            svdquant_sm120_cutlass,
            "_SM120_TACTIC_ABI_VERSION",
            svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION + 1,
        )
        hit, _, tactic, _ = tuner.search_cache(
            svdquant_sm120_cutlass._svdquant_op_name("sm120"),
            [runner],
            _CACHE_TEST_SHAPES,
            config,
        )
        assert not hit or tactic != 5, (
            "a stale same-process tactic id survived the ABI version bump"
        )
    finally:
        tuner.profiling_cache.pop(key_cur, None)


def _get_tuner():
    from flashinfer.autotuner import AutoTuner

    return AutoTuner.get()


def test_sm120_exact_shape_keys_distinguish_m():
    """Two m values that share a hybrid bucket must get distinct exact keys."""
    from flashinfer.autotuner import AutoTuner

    runner = _fake_runner()
    config = svdquant_sm120_cutlass._svdquant_tuning_config("sm120")

    def shapes_for(m):
        return (
            torch.Size([m, 32]),
            torch.Size([64, 32]),
            torch.Size([1024]),
            torch.Size([1024]),
            torch.Size([1]),
            torch.Size([m, 32]),
            torch.Size([64, 32]),
            torch.Size([64]),
            torch.Size([m, 64]),
            torch.Size([1024]),
        )

    op = svdquant_sm120_cutlass._svdquant_op_name("sm120")
    key_a = AutoTuner._get_cache_key(op, runner, shapes_for(6889), config, ())
    key_b = AutoTuner._get_cache_key(op, runner, shapes_for(6890), config, ())
    assert key_a != key_b, "exact-shape tuning keys collided across different m"


def test_get_valid_tactics_filters_by_can_implement(monkeypatch):
    """The runner only offers tactics the backend says are implementable."""
    fake_module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 6,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: t % 2 == 0,
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_nvfp4_svdquant_module_for_device",
        lambda device, lora_rank=32: fake_module,
    )
    # The two-argument runner is this backend's; gemm_svdquant keeps upstream's
    # one-argument version, and the module split left these pointing at it.
    runner = svdquant_sm120_cutlass._nvfp4_svdquant_gemm_runner(False, "dev")
    inputs = [
        SimpleNamespace(shape=(64, 1536)),  # a [m, k/2]
        SimpleNamespace(shape=(3072, 1536)),  # b [n, k/2]
        None,
        None,
        None,
        SimpleNamespace(shape=(64, 32)),  # d [m, r]
    ]
    assert runner.get_valid_tactics(inputs, profile=None) == [0, 2, 4]


def test_get_valid_tactics_unfiltered_without_query(monkeypatch):
    """Backends without the feasibility query keep the historical behavior."""
    fake_module = SimpleNamespace(nvfp4_svdquant_gemm_tactic_num=lambda: 4)
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_nvfp4_svdquant_module_for_device",
        lambda device, lora_rank=32: fake_module,
    )
    # The two-argument runner is this backend's; gemm_svdquant keeps upstream's
    # one-argument version, and the module split left these pointing at it.
    runner = svdquant_sm120_cutlass._nvfp4_svdquant_gemm_runner(False, "dev")
    assert runner.get_valid_tactics([None] * 6, profile=None) == [0, 1, 2, 3]


def test_sm120_candidate_set_carries_no_shape_fitted_pruning(monkeypatch):
    """Beyond can_implement, the candidate set must not vary with the shape.

    What stood here pinned a filter that pruned tactics by fitted thresholds --
    `m <= 512` for the swap and 64-row driver rows, `k >= 8192` plus an output
    tile count for Split-K, `m <= 8192` for raster/swizzle, a one-wave envelope
    for the static-scheduler rows. Every one of those numbers was measured on a
    single SM120 card and then applied to every SM120 card, and the filter's own
    docstring priced its benefit as mixed in sign and inside the measurement
    floor. It bought scan time, and scan time is not a cost we
    are paying for.

    The bound that replaced it reads the card, not the shape. So on a card that
    can back every request, the candidate set is exactly the can_implement set
    -- on the very shapes the old thresholds singled out.
    """
    fake_module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 8,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: True,
        nvfp4_svdquant_gemm_workspace_size=lambda m, n, k, t: 1 << 20,
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_workspace_budget_bytes",
        lambda device_index: 1 << 40,
    )
    every = list(range(8))
    # The shapes the deleted thresholds treated differently from one another.
    for m, n, k in [
        (64, 3072, 3072),
        (512, 3072, 3072),
        (1024, 3072, 3072),
        (7800, 3072, 3072),
        (64, 3072, 12288),
        (4096, 3072, 12288),
        (27280, 3072, 14336),
        (82752, 28672, 5376),
    ]:
        assert (
            svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
                fake_module, m, n, k, 32, SimpleNamespace(index=0)
            )
            == every
        ), (m, n, k)


# No _pack_row helper here any more: the candidate bound reads a tactic's
# workspace size, never its packed kernel id, raster or swizzle fields.


def test_sm120_tuned_linear_route_bypasses_repeated_autotuner_lookup():
    """A tuned exact-shape winner should dispatch directly after the first lookup."""

    class FakeTuner:
        is_tuning_mode = True

        def __init__(self):
            self.calls = 0

        def choose_one(self, op_name, runners, tuning_config, inputs):
            self.calls += 1
            return runners[1], 43

    tuner = FakeTuner()
    runners = [object(), object()]
    inputs = [torch.empty(16, dtype=torch.uint8)]
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()
    try:
        first = svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            inputs,
            device="cuda:0",
            m=256,
            n=3072,
            k=12288,
            rank=32,
            enable_pdl=False,
            has_bias=True,
        )
        tuner.is_tuning_mode = False
        second = svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            inputs,
            device="cuda:0",
            m=256,
            n=3072,
            k=12288,
            rank=32,
            enable_pdl=False,
            has_bias=True,
        )
        third = svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            inputs,
            device="cuda:0",
            m=256,
            n=3072,
            k=12288,
            rank=32,
            enable_pdl=False,
            has_bias=False,
        )

        assert first == second == third == (runners[1], 43)
        assert tuner.calls == 2, "bias/no-bias profiles must not share a dispatch entry"
    finally:
        svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()


def _sm120_linear_op_name_in_use(m, k, rank, n=5376, enable_pdl=False):
    """Drive the production chooser and return the op name it actually used.

    Reads the name off both seams at once: what `choose_one` is keyed by (the
    persistent autotune cache) and what the in-process dispatch cache is keyed
    by. They must be one computed value, or a shape could profile under one name
    and replay under another.
    """

    class RecordingTuner:
        is_tuning_mode = True

        def __init__(self):
            self.op_names = []

        def choose_one(self, op_name, runners, tuning_config, inputs):
            self.op_names.append(op_name)
            return runners[0], 9

    tuner = RecordingTuner()
    runners = [object()]
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()
    try:
        svdquant_sm120_cutlass._choose_sm120_linear_runner(
            tuner,
            runners,
            object(),
            [torch.empty(16, dtype=torch.uint8)],
            device="cuda:0",
            m=m,
            n=n,
            k=k,
            rank=rank,
            enable_pdl=enable_pdl,
            has_bias=False,
        )
        (op_name,) = tuner.op_names
        (cache_key,) = svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE
        assert cache_key[0] == op_name, (
            "the dispatch cache and the tuner key must share one computed name"
        )
        return op_name
    finally:
        svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()


# Representative row-52 route used by the PDL key tests below.
_SM120_ROUTE_V6_MKR = (73984, 14336, 32)


@pytest.mark.parametrize(
    "m,k,rank,route_version",
    [
        (*_SM120_ROUTE_V6_MKR, 6),
        (73984, 7168, 32, 10),
        (82752, 5376, 32, 10),
        (64, 3072, 32, 10),
        (537, 5376, 32, 10),
        (17, 3072, 32, 10),
        (512, 3072, 64, 11),
    ],
)
def test_sm120_linear_op_name_uses_current_route_abi(m, k, rank, route_version):
    """Every admitted runner-set change invalidates its older route record."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
    assert _sm120_linear_op_name_in_use(m, k, rank).endswith(
        f"routes_v{route_version}_tactics_v3"
    )


@pytest.mark.parametrize(
    "m,k,rank",
    [
        (256, 12288, 32),  # K outside the packed-family admission
        (64, 1536, 32),  # native producers with no new packed family
        (27280, 14336, 32),  # fused route on the same K as row 52
        (73984, 14336, 16),  # right (m, k), rank the fused route never admitted
    ],
)
def test_sm120_linear_op_name_keeps_v5_for_every_other_shape(m, k, rank):
    """Shapes whose runner set did not change must reach their retained winners.

    A global bump made every unrelated shape miss its v5 record and re-profile;
    one such re-tune picked a slower tactic and regressed the shape.
    """
    assert _sm120_linear_op_name_in_use(m, k, rank).endswith("routes_v5_tactics_v3")


@pytest.mark.parametrize(
    "m,k,rank,expected_without_pdl",
    [
        (256, 12288, 32, "svdquant_linear_sm120_routes_v5_tactics_v3"),
        (*_SM120_ROUTE_V6_MKR, "svdquant_linear_sm120_routes_v6_tactics_v3"),
        (64, 3072, 32, "svdquant_linear_sm120_routes_v10_tactics_v3"),
        (17, 3072, 32, "svdquant_linear_sm120_routes_v10_tactics_v3"),
        (64, 3072, 64, "svdquant_linear_sm120_routes_v11_tactics_v3"),
    ],
)
def test_sm120_linear_op_name_separates_pdl_in_the_persistent_key(
    m, k, rank, expected_without_pdl
):
    """PDL must key its own autotune record, not replay the non-PDL winner.

    PDL changes what every runner costs, so a profile taken without it is not a
    valid answer for a call that has it. The in-process dispatch key always
    separated the two; the persistent name did not, so a PDL=False winner could
    be replayed for PDL=True. The names below are asserted exactly, not by
    suffix: each route version has a separate PDL=False name and a PDL=True
    name with one stable `_pdl` suffix. Changing the producer candidate set
    advances the route version in both modes; toggling PDL alone does not.

    The helper drives the real chooser and asserts, on each call, that the name
    `choose_one` was keyed by is the same computed value the in-process dispatch
    cache was keyed by -- so this covers both seams in both PDL modes.
    """
    without_pdl = _sm120_linear_op_name_in_use(m, k, rank, enable_pdl=False)
    with_pdl = _sm120_linear_op_name_in_use(m, k, rank, enable_pdl=True)

    assert without_pdl == expected_without_pdl
    assert with_pdl == f"{expected_without_pdl}_pdl"
    assert with_pdl != without_pdl, (
        "otherwise-identical PDL modes must not share one persistent record"
    )


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1935, 5376, 7168),  # was pinned to tactic 82
        (537, 5376, 14336),  # was pinned to tactic 81, route ABI v9
    ],
)
def test_sm120_linear_always_reaches_the_tuner(monkeypatch, m, n, k):
    """No shape short-circuits selection any more.

    Both of these used to be dispatched straight to nvfp4_svdquant_linear_sm120
    with a tactic read out of a source table, so the tuner never saw them and
    nothing ever re-checked the value. One such value measured off the best
    candidate in its own field. The
    contract now is that every SM120 linear shape is selected, and the only
    thing that may produce a tactic is the tuner.
    """
    allocated = []

    def fake_empty(shape, **kwargs):
        # The public entry validates dtype and sizes before choosing a route,
        # so a stand-in has to answer those too.
        # torch.empty accepts either a size tuple or bare ints, and the code
        # under test uses both spellings; tuple() on a bare int raises.
        tensor = _FakeTensor(shape, **kwargs)
        allocated.append(tensor)
        return tensor

    module = SimpleNamespace(
        nvfp4_svdquant_gemm_workspace_size=lambda *args: 0,
        nvfp4_svdquant_linear_sm120=lambda *args: pytest.fail(
            "a shape reached the fused FFI without going through the tuner"
        ),
    )

    launched = []

    class Recorder:
        def __call__(self, inputs, tactic):
            launched.append((inputs, tactic))

    recorder = Recorder()

    class Tuner:
        is_tuning_mode = False
        seen = []

        def choose_one(self, op_name, runners, tuning_config, inputs):
            Tuner.seen.append(op_name)
            return runners[0], 7

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_sm120_module",
        lambda lora_rank=32: module,
    )
    monkeypatch.setattr(gemm_svdquant.torch, "empty", fake_empty)
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_cache_buf",
        lambda *args, **kwargs: torch.zeros(16, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_sm120_max_workspace_bytes", lambda *args, **kwargs: 0
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_cached_sm120_linear_runners",
        lambda *args, **kwargs: [recorder],
    )
    monkeypatch.setattr(
        gemm_svdquant.AutoTuner, "get", classmethod(lambda cls: Tuner())
    )
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()

    x = torch.zeros((m, k), dtype=torch.bfloat16)
    weight = SimpleNamespace(ndim=2, shape=(n, k // 2))
    l2t = SimpleNamespace(ndim=2, shape=(k, 32))
    out = gemm_svdquant.svdquant_linear(
        x,
        weight,
        "weight_sf",
        "alpha",
        torch.ones(k, dtype=torch.bfloat16),
        l2t,
        "l1_scaled",
        "global_scale",
        bias="bias",
        enable_pdl=False,
        backend="cutlass-sm120",
    )

    # Identify the output by its shape, not by being the last allocation: the
    # workspace buffer is allocated too, and which of them comes last is not
    # part of the contract this test is about.
    outputs = [t for t in allocated if t.shape == (m, n)]
    assert outputs and out is outputs[-1]
    assert launched and launched[0][1] == 7, "the tactic must be the tuner's answer"
    assert Tuner.seen, "the tuner was never consulted"
    svdquant_sm120_cutlass._SM120_LINEAR_DISPATCH_CACHE.clear()


def test_sm120_linear_route_pool(monkeypatch):
    """Which routes a shape offers the tuner, now that there are only two.

    A shape the fused prefix admits offers it alongside CUTLASS and the tuner
    picks; every other shape offers CUTLASS alone. Nothing else may appear --
    a third route would mean the CuTeDSL fallback came back.
    """
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_fused_linear_runner",
        lambda enable_pdl, device, rank: "fused",
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_cutlass_linear_runner",
        lambda enable_pdl, device, rank: "cutlass",
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_fused_linear_supported",
        lambda m, k, rank: m == 4096,
    )
    pool = svdquant_sm120_cutlass._sm120_linear_runners(
        False, "dev", 4096, 3072, 3072, 32
    )
    assert pool == ["fused", "cutlass"]
    pool = svdquant_sm120_cutlass._sm120_linear_runners(
        False, "dev", 4097, 3072, 3072, 32
    )
    assert pool == ["cutlass"]


@pytest.mark.parametrize("device_name", ("cpu", "cuda"))
@pytest.mark.parametrize("rank", (32, 64))
@pytest.mark.parametrize("k", (3072, 14336))
@pytest.mark.parametrize("enable_pdl", (None, False, True))
def test_empty_linear_returns_before_gpu_preparation(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
    rank: int,
    k: int,
    enable_pdl: bool | None,
) -> None:
    if device_name == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    ):
        pytest.skip("requires SM120")

    def unexpected_work(*args: torch.Tensor | int | torch.device | None) -> None:
        pytest.fail("empty batch reached GPU preparation or tactic selection")

    for name in (
        "device_support_pdl",
        "_normalize_nvfp4_quantize_smooth_inputs",
        "_sm120_max_workspace_bytes",
        "_get_cache_buf",
        "_cached_sm120_linear_runners",
        "_choose_sm120_linear_runner",
    ):
        monkeypatch.setattr(svdquant_sm120_cutlass, name, unexpected_work)
    device = torch.device(device_name)
    n = 128
    out = gemm_svdquant.svdquant_linear(
        torch.empty((0, k), dtype=torch.bfloat16, device=device),
        torch.empty((n, k // 2), dtype=torch.uint8, device=device),
        torch.empty(n * k // 16, dtype=torch.uint8, device=device),
        torch.ones(1, device=device),
        torch.empty(k + 1, dtype=torch.bfloat16, device=device)[1:],
        torch.empty((k, rank), dtype=torch.bfloat16, device=device),
        torch.empty((n, rank), dtype=torch.bfloat16, device=device),
        torch.ones(1, device=device),
        bias=torch.empty(n, dtype=torch.bfloat16, device=device),
        enable_pdl=enable_pdl,
        backend="cutlass-sm120",
    )
    assert out.shape == (0, n)
    assert out.dtype == torch.bfloat16
    assert out.device.type == device_name


def _packing_tactic(m: int, k: int) -> int:
    """A tactic whose producer variant is the one that indexes L2T prepacked.

    A tactic used to be a consumer row and nothing else, so this test could name
    one as a bare integer. It now carries a producer variant too, and whether the
    L2T matrix is prepacked follows that variant rather than the shape -- so the
    integer has to be derived from the same ladder the runner decodes, not
    written down. Hardcoding it is how this test came to pass a variant that does
    not pack at all.
    """
    variants = range(len(_sm120_routes.sm120_producer_variants(m, k)))
    packing = [v for v in variants if _sm120_routes.sm120_variant_packs_l2t(m, k, v)]
    assert packing, f"({m}, {k}) has no L2T-packing producer variant"
    return _sm120_routes.sm120_pack_tactic(0, packing[0])


def test_table_miss_packs_m537_l2t_only_inside_fused_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared tuner input stays row-major until the fused route is selected."""
    allocated = []
    native_calls = []
    chooser_inputs = []
    # The packed L2T is handed back to the entry point, which reads its dtype
    # before deciding a route, so it needs the tensor surface too.
    packed_l2t = _FakeTensor((5376, 32), dtype=torch.bfloat16)

    def fake_empty(shape, **kwargs):
        # dtype and numel are read by the public entry's own validation before
        # the route is ever chosen, so a stand-in has to answer both.
        tensor = _FakeTensor(shape, **kwargs)
        allocated.append(tensor)
        return tensor

    module = SimpleNamespace(
        nvfp4_svdquant_linear_sm120=lambda *args: native_calls.append(args),
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_compute_capability", lambda device: (12, 0)
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_sm120_module",
        lambda lora_rank=32: module,
    )
    monkeypatch.setattr(gemm_svdquant.torch, "empty", fake_empty)
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_cache_buf",
        # Not the string "workspace": the entry point reads this buffer's dtype
        # on its way to a route.
        lambda *args, **kwargs: _FakeTensor((1 << 20,), dtype=torch.uint8),
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_sm120_max_workspace_bytes", lambda *args: 0
    )
    # No _sm120_cutedsl_fallback_device_info stand-in: the CuTeDSL runner it
    # described is deliberately not part of this backend, so the symbol is gone
    # and patching it would only assert that it still exists.
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_cached_sm120_m537_l2t",
        lambda l2t: packed_l2t,
    )
    monkeypatch.setattr(gemm_svdquant.AutoTuner, "get", staticmethod(lambda: "tuner"))

    runner = svdquant_sm120_cutlass._sm120_fused_linear_runner(False, "cuda:0")
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_cached_sm120_linear_runners", lambda *args: [runner]
    )

    def choose_runner(tuner, runners, tuning_config, inputs, **kwargs):
        chooser_inputs.append(inputs)
        return runner, _packing_tactic(537, 5376)

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_choose_sm120_linear_runner", choose_runner
    )

    x = torch.zeros((537, 5376), dtype=torch.bfloat16)
    weight = _FakeTensor((5376, 2688), dtype=torch.uint8)
    raw_l2t = _FakeTensor((5376, 32), dtype=torch.bfloat16)

    out = gemm_svdquant.svdquant_linear(
        x,
        weight,
        "weight_sf",
        "alpha",
        torch.ones(5376, dtype=torch.bfloat16),
        raw_l2t,
        "l1_scaled",
        "global_scale",
        bias="bias",
        enable_pdl=False,
        # The fused SM120 route is reached by name, not by capability:
        # svdquant_linear only enters it on backend == "cutlass-sm120".
        # Without this the call falls through to the generic path and dies in
        # nvfp4_quantize_smooth, which is not what this test is about.
        backend="cutlass-sm120",
    )

    assert out is allocated[-1]
    assert chooser_inputs[0][5] is raw_l2t
    assert native_calls[0][3] is packed_l2t


@pytest.mark.parametrize("rank", (32, 64))
@pytest.mark.parametrize("fused", (False, True))
@pytest.mark.parametrize(
    "x_offset,pqs_offset,x_stride,pqs_stride,pqs_matrix",
    [
        (0, 0, 1, 1, False),
        (1, 0, 1, 1, False),
        (0, 1, 1, 1, False),
        (0, 0, 2, 2, False),
        (0, 0, 1, 1, True),
    ],
    ids=("aligned", "offset-x", "offset-scale", "strided", "scale-matrix"),
)
def test_linear_normalizes_inputs_before_tuning_and_launch(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    fused: bool,
    x_offset: int,
    pqs_offset: int,
    x_stride: int,
    pqs_stride: int,
    pqs_matrix: bool,
) -> None:
    # Real CPU tensor views preserve the storage-offset/alignment behavior;
    # only tactic selection and the native CUDA calls are replaced.
    m, n, k = 64, 128, 3072
    x = (
        torch.arange(m * k * x_stride + x_offset, dtype=torch.float32)
        .to(torch.bfloat16)[x_offset::x_stride]
        .reshape(m, k)
    )
    pqs = torch.arange(k * pqs_stride + pqs_offset, dtype=torch.float32).to(
        torch.bfloat16
    )[pqs_offset::pqs_stride]
    if pqs_matrix:
        pqs = pqs.reshape(1, k)
    expected_x, expected_pqs = x.clone(), pqs.reshape(k).clone()
    observed: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_prefix(
        activation: torch.Tensor,
        scale: torch.Tensor,
        *_args: torch.Tensor | int | bool | None,
    ) -> None:
        observed.append((activation, scale))

    module = SimpleNamespace(
        nvfp4_svdquant_linear_sm120=record_prefix,
        nvfp4_quantize_smooth=record_prefix,
        nvfp4_svdquant_gemm=lambda *args: None,
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "get_nvfp4_svdquant_sm120_module",
        lambda rank=32: module,
    )
    runner = (
        svdquant_sm120_cutlass._sm120_fused_linear_runner(False, x.device, rank)
        if fused
        else svdquant_sm120_cutlass._sm120_cutlass_linear_runner(False, x.device, rank)
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_cached_sm120_linear_runners", lambda *args: [runner]
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_sm120_max_workspace_bytes", lambda *args: 0
    )
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_cache_buf",
        lambda *args: torch.empty(16, dtype=torch.uint8),
    )

    def choose(
        tuner: AutoTuner,
        runners: list[TunableRunner],
        config: TuningConfig,
        inputs: list[torch.Tensor],
        **kwargs: int | bool | torch.device,
    ) -> tuple[TunableRunner, int]:
        observed.append((inputs[0], inputs[4]))
        return runners[0], 0

    monkeypatch.setattr(svdquant_sm120_cutlass, "_choose_sm120_linear_runner", choose)
    gemm_svdquant.svdquant_linear(
        x,
        torch.empty((n, k // 2), dtype=torch.uint8),
        torch.empty(n * k // 16, dtype=torch.uint8),
        torch.ones(1),
        pqs,
        torch.zeros((k, rank), dtype=torch.bfloat16),
        torch.zeros((n, rank), dtype=torch.bfloat16),
        torch.ones(1),
        enable_pdl=False,
        backend="cutlass-sm120",
    )

    assert len(observed) == 2
    for actual_x, actual_pqs in observed:
        assert actual_x.is_contiguous() and actual_x.data_ptr() % 16 == 0
        assert actual_pqs.is_contiguous() and actual_pqs.data_ptr() % 16 == 0
        torch.testing.assert_close(actual_x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(actual_pqs, expected_pqs, rtol=0, atol=0)
        if x.is_contiguous() and x.data_ptr() % 16 == 0:
            assert actual_x.data_ptr() == x.data_ptr()
        if pqs.is_contiguous() and pqs.data_ptr() % 16 == 0:
            assert actual_pqs.data_ptr() == pqs.data_ptr()
    assert observed[0][0] is observed[1][0]
    assert observed[0][1] is observed[1][1]


@pytest.mark.parametrize("rank", [32, 64])
def test_linear_runners_load_the_requested_rank(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    requested_ranks: list[int] = []

    def get_module(lora_rank: int = 32) -> SimpleNamespace:
        requested_ranks.append(lora_rank)
        return SimpleNamespace()

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "get_nvfp4_svdquant_sm120_module", get_module
    )

    runners = svdquant_sm120_cutlass._sm120_linear_runners(
        False, torch.device("cuda:0"), 129, 256, 256, rank
    )

    assert requested_ranks
    assert set(requested_ranks) == {rank}
    expected_fallback = (
        "Sm120CutlassLinearRunner" if rank == 64 else "Sm120FusedLinearRunner"
    )
    assert type(runners[0]).__name__ == expected_fallback


@pytest.mark.parametrize("rank", [0, 48, 96, 128])
def test_linear_rejects_unsupported_ranks_before_loading_kernels(rank: int) -> None:
    with pytest.raises(ValueError, match="rank"):
        svdquant_linear(
            torch.empty((1, 32), dtype=torch.bfloat16),
            torch.empty((32, 16), dtype=torch.uint8),
            torch.empty(128, dtype=torch.uint8),
            torch.ones(1),
            torch.ones(32, dtype=torch.bfloat16),
            torch.empty((32, rank), dtype=torch.bfloat16),
            torch.empty((32, rank), dtype=torch.bfloat16),
            torch.ones(1),
            enable_pdl=False,
            backend="cutlass-sm120",
        )


# Producer geometry, route identity and cache namespaces.


@pytest.mark.parametrize("m", (0, -1))
@pytest.mark.parametrize("k", (3072, 14336))
@pytest.mark.parametrize("rank", (32, 64))
def test_nonpositive_batches_have_no_fused_producer(m: int, k: int, rank: int) -> None:
    geometries = (
        SM120_RANK64_PRODUCER_GEOMETRY_LADDER
        if rank == 64
        else SM120_PRODUCER_GEOMETRY_LADDER
    )
    assert not any(
        sm120_producer_geometry_is_valid(m, k, *geometry, rank)
        for geometry in geometries
    )
    assert sm120_producer_variants(m, k, rank) == ()


def test_route_manifest_containers_are_immutable() -> None:
    # Given: routing metadata is shared by dispatch, tests, and tuning tools.
    # When: its public container types are inspected.
    # Then: no consumer can mutate process-wide route decisions in place.
    assert isinstance(SM120_LINEAR_ROUTE_ABI_OVERRIDES, MappingProxyType)
    assert isinstance(SM120_FUSED_LINEAR_MK, frozenset)
    assert isinstance(SM120_LINEAR_ROUTE_V6_MKR, frozenset)


@pytest.mark.parametrize(
    "m,k,rank,expected_version",
    [
        (256, 12288, 32, 5),
        (73984, 14336, 32, 6),
        (1935, 14336, 32, 7),
        (537, 14336, 32, 9),
        (64, 3072, 32, 10),
        (64, 5120, 32, 10),
        (64, 5376, 32, 10),
        (64, 7168, 32, 10),
        (73984, 7168, 32, 10),
        (537, 5376, 32, 10),
        (82752, 5376, 32, 10),
        (17, 3072, 32, 10),
        (0, 3072, 32, 5),
        (64, 1536, 32, 5),
        (64, 3072, 16, 5),
        (64, 3072, 64, 11),
    ],
)
def test_route_abi_version_matches_manifest(
    m: int,
    k: int,
    rank: int,
    expected_version: int,
) -> None:
    # Given: affected and unaffected shapes, including an unlisted M and ranks.
    # When: its current cache namespace is selected.
    actual_version = sm120_linear_route_abi_version(m, k, rank)
    # Then: only runner-set changes advance beyond the retained v5 base.
    assert actual_version == expected_version


def test_route_abi_override_manifest_has_no_base_version_entries() -> None:
    # Given: the compact mapping contains only exceptions to the v5 base.
    # When: every override version is inspected.
    override_versions = set(SM120_LINEAR_ROUTE_ABI_OVERRIDES.values())
    # Then: redundant v5 entries cannot drift from the default independently.
    assert override_versions == {6, 7, 8, 9}


def test_the_two_formerly_pinned_problems_keep_their_fused_admission() -> None:
    # Given: 32760x1536x1536 and 32760x8960x1536 reached the fused runner only
    # because they carried a pinned tactic, and pinned prefixes were unioned into
    # admission separately from SM120_FUSED_LINEAR_MK.
    # When: the pins are deleted.
    # Then: their prefix has to survive in the admission set on its own, or the
    # tuner sees only Sm120CutlassLinearRunner for both -- measured slower on
    # 32760x1536x1536.
    assert (32760, 1536) in SM120_FUSED_LINEAR_MK


def _shape_keyed_tactic_tables(source: str) -> list[str]:
    """Names bound to a module-level mapping keyed by a four-int (m,n,k,rank)."""
    found: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        value = node.value
        if value is None:
            continue
        # MappingProxyType({...}) and frozenset({...}) both wrap the literal.
        while isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if not isinstance(key, ast.Tuple) or len(key.elts) != 4:
                break
            if not all(
                isinstance(e, ast.Constant) and isinstance(e.value, int)
                for e in key.elts
            ):
                break
        else:
            if value.keys:
                found.extend(targets)
    return found


def test_no_module_ships_a_shape_keyed_tactic_table() -> None:
    # Given: a tactic is a measurement of one kernel on one device, and the only
    # thing entitled to produce one is the tuner. A cache is fine; a source
    # literal is not, because nothing re-checks it and nothing dates it. One went
    # stale and was found only by measuring the whole candidate field.
    # When: every module of the gemm package is parsed.
    # Then: no module-level mapping is keyed by an exact (M, N, K, rank) problem,
    # which is the shape such a table has to have to be looked up at dispatch.
    root = Path(__file__).resolve().parents[2]
    package = root / "flashinfer" / "gemm"
    # Every SVDQuant module under flashinfer/gemm. The CuTeDSL kernels this
    # guard used to also scan are gone with that route; what replaced them as
    # the tempting place to write such a table is the SM120 CUTLASS backend,
    # which owns the route manifest and the runner set.
    offenders: dict[str, list[str]] = {}
    scanned = sorted(package.rglob("*svdquant*.py"))
    covered = {path.name for path in scanned}
    for expected in (
        "svdquant_sm120_routes.py",
        "svdquant_sm120_cutlass.py",
        "gemm_svdquant.py",
    ):
        assert expected in covered, (
            f"{expected} is no longer scanned; the guard passes vacuously for it"
        )
    for path in scanned:
        names = _shape_keyed_tactic_tables(path.read_text())
        if names:
            offenders[str(path.relative_to(root))] = names
    assert offenders == {}, (
        f"a shape-keyed table is back in source: {offenders}. Tactics come from "
        "the tuner and persist through AutoTuner.save_configs / load_configs."
    )


def _has_native_producer(m: int, k: int) -> bool:
    """Whether the shape has a legal producer geometry."""
    return bool(sm120_producer_variants(m, k))


def _specialized_producer_shapes(header: str) -> set[tuple[int, int]]:
    """Read the remaining fixed-M/K instantiations from the family dispatcher."""
    start = header.index(
        "inline cudaError_t nvfp4_smooth_quantize_lora_down_family_sm120("
    )
    end = header.index(
        "inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120(", start
    )
    return {
        (537, int(match.group(1)))
        for match in re.finditer(
            r"launch_m537_mixed_kernel<\s*(\d+)", header[start:end]
        )
    }


def _producer_header() -> str:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "include"
        / "flashinfer"
        / "gemm"
        / "nvfp4_smooth_quantize_lora_down_sm120.cuh"
    ).read_text()


def test_every_fused_admission_has_a_prefix_that_can_run() -> None:
    # Given: admitting (M, K) to SM120_FUSED_LINEAR_MK offers the fused runner to
    # the tuner, and the fused route reaches its prefix in one of two ways -- a
    # compiled native producer, or the cuBLASLt prefix the ladder can select.
    # When: the admission set is checked against both.
    # Then: nothing is admitted that has neither, which would make the C++
    # dispatcher return cudaErrorInvalidValue for a route the tuner can pick.
    # Reachable = the FFI lets it in AND the shape computes a geometry to launch.
    # The second term used to be the ladder's cuBLASLt column, for shapes that
    # had the prefix but no native producer. The prefix is a producer family now
    # and is offered exactly where the native families are, so a shape reachable
    # by one is reachable by the other and the ladder term has nothing to add.
    reachable = {s for s in SM120_FUSED_LINEAR_MK if _has_native_producer(*s)}
    orphans = sorted(set(SM120_FUSED_LINEAR_MK) - reachable)
    assert orphans == [], (
        f"{orphans} are admitted to the fused route but have no native producer "
        "and no cuBLASLt prefix rung"
    )


def test_every_specialized_producer_is_admitted() -> None:
    specialized = _specialized_producer_shapes(_producer_header())
    assert specialized, "the source scan must cover the fixed-M/K M537 family"
    assert specialized <= set(SM120_FUSED_LINEAR_MK)
    for m, k in specialized:
        assert any(
            family == SM120_FAMILY_M537
            for family, _, _ in sm120_producer_variants(m, k)
        ), f"the compiled M537 producer for {(m, k)} cannot be selected"


@pytest.mark.parametrize(
    "m,k",
    [(512, 1536), (7800, 1536), (256, 3072), (6889, 3072), (9216, 3072), (16384, 3072)],
)
def test_swept_producers_are_admitted_and_route_native(m: int, k: int) -> None:
    # Given: six prefixes gained a producer because a full sweep of the legal
    # launch geometries measured the fused producer beating the unfused prefix
    # by a measured margin.
    # When: each is resolved through the manifest.
    # Then: it has native candidates and no fixed-M537 candidate.
    assert (m, k) in SM120_FUSED_LINEAR_MK
    assert _has_native_producer(m, k)
    # Was `(m, k) in _ffi_guard_shapes()`. The FFI no longer carries a shape
    # list; being runnable is being able to compute a producer geometry.
    assert sm120_producer_variants(m, k)
    # Packed small-M candidates may coexist with row-major native candidates;
    # the selected family determines the L2T layout.
    variants = sm120_producer_variants(m, k)
    assert variants, f"({m}, {k}) computes no producer geometry"
    assert all(family != SM120_FAMILY_M537 for family, _, _ in variants)
    for variant, (family, _, _) in enumerate(variants):
        assert sm120_variant_packs_l2t(m, k, variant) == (
            family == SM120_FAMILY_SMALL_M_PACKED
        )


@pytest.mark.parametrize("m,k", sorted(SM120_FUSED_LINEAR_MK | {(17, 3072)}))
def test_packed_family_is_appended_after_all_legacy_variants(m: int, k: int) -> None:
    # Given: old producer IDs are persisted as part of each packed tactic.
    # When: the producer list admits the new layout family.
    variants = sm120_producer_variants(m, k)
    packed_indices = [
        index
        for index, (family, _, _) in enumerate(variants)
        if family == SM120_FAMILY_SMALL_M_PACKED
    ]
    # Then: new entries form a suffix, preserving all previous indices.
    if packed_indices:
        assert packed_indices == list(range(packed_indices[0], len(variants)))


@pytest.mark.parametrize(
    "m,k,variant,expected",
    [
        (64, 3072, 17, (1, (768, 8, 8), 0)),
        (64, 3072, 23, (3, (0, 0, 0), 0)),
        (537, 5376, 19, (2, (1024, 24, 16), 0)),
        (537, 5376, 20, (3, (0, 0, 0), 0)),
    ],
)
def test_legacy_producer_ids_keep_their_meaning(
    m: int, k: int, variant: int, expected: tuple[int, tuple[int, int, int], int]
) -> None:
    # Given: representative persisted IDs from every affected legacy family.
    # When: the current producer list resolves each index.
    actual = sm120_producer_variants(m, k)[variant]
    # Then: appending candidates changes no existing tactic interpretation.
    assert actual == expected


@pytest.mark.parametrize(
    "m,k",
    [
        (4096, 3072),
        (1024, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
        (512, 5120),
        (537, 14336),
        (1935, 5376),
    ],
)
def test_prefixes_where_fusing_measured_slower_are_offered_anyway(
    m: int, k: int
) -> None:
    # Given: the sweep measured these prefixes too, and on every one the fused
    # producer was SLOWER than the prefix the shape pays today, against both
    # the unfused prefix and the cuBLASLt one.
    # When: the manifest is checked.
    # Then: they are present regardless. Whether a candidate is worth running is
    # a measurement, and the only thing entitled to make it is the tuner; a
    # prefix left out of this table has no fused candidate for the tuner to
    # reject, and the source-side judgement that put it there never re-checks
    # itself. The measured costs live beside each dispatcher entry.
    assert (m, k) in SM120_FUSED_LINEAR_MK
    assert _has_native_producer(m, k)
    # Was `(m, k) in _ffi_guard_shapes()`. The FFI no longer carries a shape
    # list; being runnable is being able to compute a producer geometry.
    assert sm120_producer_variants(m, k)


def test_every_production_prefix_has_a_fused_producer() -> None:
    # Given: a shape whose (M, K) has no producer cannot offer the fused route
    # to the tuner at all -- not as a losing candidate, not as anything. The
    # coverage question is therefore separate from the performance question.
    # When: every registered production shape's prefix is looked up.
    # Then: all of them have one. This is the invariant that says the fused
    # implementation exists everywhere; which route each shape actually runs is
    # decided by measurement, per shape, per card.
    from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES

    prefixes = {(case[0], case[2]) for case in MODEL_SHAPE_CASES}
    missing = sorted(p for p in prefixes if not _has_native_producer(*p))
    assert missing == [], f"{missing} have no fused producer"
    assert prefixes <= set(SM120_FUSED_LINEAR_MK)
    # Same: reachability is computed, not listed.
    assert all(sm120_producer_variants(m, k) for m, k in prefixes)


# --- the producer axis of a packed tactic ------------------------------------
#
# A tactic is now (consumer row, producer variant) packed into one integer. The
# tests below hold the two halves of that grammar together: the ordering
# contract the packing rests on, its inertness where the axis is degenerate, and
# the one failure mode that is silent rather than loud.


def test_fused_linear_mk_is_the_benchmark_projection() -> None:
    """The named shape list must stay the benchmark matrix, not drift into a gate.

    SM120_FUSED_LINEAR_MK used to decide fused admission. It does not any more --
    _sm120_fused_linear_supported computes that from the launch geometry -- so
    what is left is a name for the shapes this backend was measured on. If the
    two ever diverge, the list has started encoding a decision again.
    """
    from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES

    assert set(SM120_FUSED_LINEAR_MK) == {(m, k) for m, _, k in MODEL_SHAPE_CASES}


@pytest.mark.parametrize("m,k", [(64, 3072), (537, 7168), (2048, 3072), (4096, 12288)])
def test_rank64_offers_fused_preprocessing(m: int, k: int) -> None:
    assert backend._sm120_fused_linear_supported(m, k, 64)
    variants = routes.sm120_producer_variants(m, k, 64)
    assert variants
    assert {family for family, _, _ in variants} == {
        routes.SM120_FAMILY_LARGE_M,
        routes.SM120_FAMILY_SMALL_M,
    }
    for variant, entry in enumerate(variants):
        assert routes.sm120_decode_producer_variant(m, k, variant, 64) == entry
        assert not routes.sm120_variant_packs_l2t(m, k, variant, 64)


def test_rank64_fusion_invalidates_unfused_route_cache() -> None:
    assert routes.sm120_linear_route_abi_version(64, 3072, 64) > 10
    assert routes.sm120_linear_route_abi_version(64, 3072, 32) == 10


# Workspace provisioning and cached tactic feasibility.


def _pack_row(kernel_id, splits=1, raster=0, swizzle=1, streamk=0):
    return (
        kernel_id | (splits << 8) | (raster << 16) | (swizzle << 24) | (streamk << 32)
    )


# MiniMax-H3 row 69, the largest shape in the frozen benchmark matrix and one of
# the three that fell back to tactic=-1 after the route autotuner OOMed.
_H3_LARGEST = (82752, 28672, 5376, 32)

# The one MiniMax-H3 shape that does keep a Split-K row exposed: k >= 8192 and
# only 5*42 = 210 output tiles, so it clears both halves of the Split-K rule.
# Narrowing the sizing set must be a no-op here.
_H3_SPLITK_SHAPE = (537, 5376, 14336, 32)

# MiniMax-H3 row 52, newly admitted to the fused K12 producer. No combined
# tactic is pinned for it, so the buffer has to cover every K3 tactic the
# autotuner may select -- which is the same invariant as everywhere else, just
# on a shape that now has one more runner offering it.
_H3_ROW52 = (73984, 5376, 14336, 32)

# The 27 frozen MiniMax-H3 benchmark cases, mirroring the registry's own
# construction (see tests/gemm/test_nvfp4_svdquant_host.py). Declared locally so
# this file stays readable, then cross-checked against the registry below.
_H3_TEXT_M = (537, 1935, 6913)
_H3_OMNI_M = (61056, 73984, 82752)
_H3_BLOCK_NK = ((21504, 5376), (5376, 7168), (28672, 5376), (5376, 14336))
_H3_SHAPES = frozenset(
    [(m, n, k) for m in (*_H3_TEXT_M, *_H3_OMNI_M) for n, k in _H3_BLOCK_NK]
    + [(m, 5376, 5120) for m in _H3_TEXT_M]
)

# Split-K on a legacy Stream-K kernel: implementable anywhere, exposed only on
# long-K problems with an underfilled grid, and by far the largest workspace.
_SPLITK_TACTIC = 1
_SPLITK_WORKSPACE_BYTES = 38 * 1024**3
_LEGACY_WORKSPACE_BYTES = 4 * 1024**2
# No list of Stream-K kernel ids here on purpose: the budget reads a tactic's
# workspace size and never its identity, so which kernels can do Split-K is no
# longer something this layer has to know.


def _fake_workspace_module(with_tactic_row=True):
    """Backend where only the Split-K row carries a multi-GB workspace.

    with_tactic_row=False models a backend compiled before the runtime tactic
    table was exported, where the exposure filter cannot run.
    """
    rows = {
        _SPLITK_TACTIC: _pack_row(2, splits=2, streamk=1),
    }
    module = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=lambda: 4,
        nvfp4_svdquant_gemm_can_implement=lambda m, n, k, rank, t: True,
        nvfp4_svdquant_gemm_workspace_size=lambda m, n, k, t: (
            _SPLITK_WORKSPACE_BYTES if t == _SPLITK_TACTIC else _LEGACY_WORKSPACE_BYTES
        ),
    )
    if with_tactic_row:
        module.nvfp4_svdquant_gemm_tactic_row = lambda t: rows.get(t, _pack_row(8))
    return module


def _patch_workspace_module(monkeypatch, module):
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_get_nvfp4_svdquant_module_for_device",
        lambda device, lora_rank=32: module,
    )
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BYTES_CACHE", {})
    # The budget is decided once per shape and held, so it has to be cleared
    # alongside the sizes or one test's card size answers for the next.
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BUDGET_CACHE", {})


# A card big enough to back the Split-K row, and one that is not. Neither
# number is a threshold the code knows: the budget is read off the device.
_CARD_BACKS_SPLITK = int((_SPLITK_WORKSPACE_BYTES + 1) / 0.25)
_CARD_TOO_SMALL_FOR_SPLITK = int((_SPLITK_WORKSPACE_BYTES - 1) / 0.25)


def _patch_budget(monkeypatch, total_memory_bytes):
    """Stand in for the card, so these stay CPU-only tests."""
    monkeypatch.setattr(
        svdquant_sm120_cutlass,
        "_sm120_workspace_budget_bytes",
        lambda device_index: int(
            total_memory_bytes * svdquant_sm120_cutlass._SM120_WORKSPACE_BUDGET_FRACTION
        ),
    )


def test_sm120_workspace_skips_tactics_the_card_cannot_back(monkeypatch):
    """Sizing ranges over what the card can back, not the wider can_implement set.

    Sizing over can_implement reserved the O(splits*m*n*4B) Split-K reduction
    buffer unconditionally, which OOMed the route autotuner on the largest H3
    shapes. On a card too small to back that row, it is neither offered nor
    sized for.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    _patch_budget(monkeypatch, _CARD_TOO_SMALL_FOR_SPLITK)
    m, n, k, rank = _H3_LARGEST
    device = SimpleNamespace(index=0)
    assert _SPLITK_TACTIC not in svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
        _fake_workspace_module(), m, n, k, rank, device
    )
    assert (
        svdquant_sm120_cutlass._sm120_max_workspace_bytes(device, m, n, k, rank)
        == _LEGACY_WORKSPACE_BYTES
    )


def test_sm120_workspace_covers_every_selectable_tactic(monkeypatch):
    """The buffer must be a superset of what the autotuner can return.

    Under-sizing is caught fail-closed by the TVM_FFI_ICHECK_GE in
    csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu, so this asserts the Python side
    never gets there. Both card sizes, because the selectable set differs.
    """
    for total in (_CARD_TOO_SMALL_FOR_SPLITK, _CARD_BACKS_SPLITK):
        module = _fake_workspace_module()
        _patch_workspace_module(monkeypatch, module)
        _patch_budget(monkeypatch, total)
        device = SimpleNamespace(index=0)
        for m, n, k in [
            _H3_LARGEST[:3],
            _H3_SPLITK_SHAPE[:3],
            (512, 3072, 3072),
            _H3_ROW52[:3],
        ]:
            rank = 32
            allotted = svdquant_sm120_cutlass._sm120_max_workspace_bytes(
                device, m, n, k, rank
            )
            selectable = svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
                module, m, n, k, rank, device
            )
            assert selectable, f"no tactic selectable for {(m, n, k)}"
            for t in selectable:
                assert allotted >= module.nvfp4_svdquant_gemm_workspace_size(
                    m, n, k, t
                ), f"tactic {t} under-runs the buffer at {(m, n, k)}"


def test_sm120_workspace_sizes_splitk_when_the_card_can_back_it(monkeypatch):
    """The bound is the card, not the shape.

    The old filter pruned Split-K by shape-fitted thresholds (`k >= 8192` and
    an output-tile count), so the same shape was pruned on every card. Here the
    identical shape that was refused above is served once the card is big
    enough -- which is the whole point of pricing legality instead of guessing
    profitability.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    _patch_budget(monkeypatch, _CARD_BACKS_SPLITK)
    m, n, k, rank = _H3_LARGEST
    device = SimpleNamespace(index=0)
    assert _SPLITK_TACTIC in svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
        _fake_workspace_module(), m, n, k, rank, device
    )
    assert (
        svdquant_sm120_cutlass._sm120_max_workspace_bytes(device, m, n, k, rank)
        == _SPLITK_WORKSPACE_BYTES
    )


def test_sm120_valid_tactics_are_unbounded_without_a_device():
    """No device means no budget: the caller gets the can_implement set.

    The CPU-only tests rely on this, and so does any backend too old to export
    a per-tactic workspace query.
    """
    module = _fake_workspace_module()
    m, n, k, rank = _H3_LARGEST
    every = list(range(module.nvfp4_svdquant_gemm_tactic_num()))
    assert (
        svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(module, m, n, k, rank)
        == every
    )
    no_query = SimpleNamespace(
        nvfp4_svdquant_gemm_tactic_num=module.nvfp4_svdquant_gemm_tactic_num,
        nvfp4_svdquant_gemm_can_implement=module.nvfp4_svdquant_gemm_can_implement,
    )
    assert (
        svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
            no_query, m, n, k, rank, SimpleNamespace(index=0)
        )
        == every
    )


def test_sm120_budget_keeps_one_tactic_when_the_card_backs_none(monkeypatch):
    """A shape whose every candidate overruns the budget still has to run.

    Refusing to return a tactic would fail the shape outright; returning the
    cheapest one lets it try, and the fail-closed ICHECK on the C++ side is
    what catches a buffer that genuinely cannot be allocated.
    """
    module = _fake_workspace_module()
    _patch_workspace_module(monkeypatch, module)
    _patch_budget(monkeypatch, 4096)
    m, n, k, rank = _H3_LARGEST
    selectable = svdquant_sm120_cutlass._nvfp4_svdquant_valid_tactics(
        module, m, n, k, rank, SimpleNamespace(index=0)
    )
    assert len(selectable) == 1
    assert module.nvfp4_svdquant_gemm_workspace_size(m, n, k, selectable[0]) == (
        _LEGACY_WORKSPACE_BYTES
    )


def test_sm120_h3_shapes_match_the_frozen_registry():
    """The inline H3 cases must be the registry's, not a drifted copy.

    This file names its 27 shapes locally so it stays readable; that is only
    safe while they still agree with the shared registry.
    """
    assert len(_H3_SHAPES) == 27
    assert set(MODEL_SHAPE_CASES) >= _H3_SHAPES
    for shape in [(73984, 28672, 5376), (82752, 21504, 5376), (82752, 28672, 5376)]:
        assert shape in _H3_SHAPES, "the three route-autotuner OOM rows"


def test_sm120_budget_is_priced_against_free_memory_not_total():
    """The quantity that decides is what is left, not what the card has.

    Measured on 82752x28672x5376: the widest per-tactic workspace is 8.84 GiB,
    under a quarter of a 71 GiB card, so a total-memory budget admits it. By
    the time the buffer is provisioned the problem's own operands hold 65 GiB
    and 5.74 GiB remains, and the allocation fails -- the OOM the shape-keyed
    exposure filter used to prevent, reintroduced by pricing total memory.
    """
    import inspect

    source = inspect.getsource(svdquant_sm120_cutlass._sm120_workspace_budget_bytes)
    assert "mem_get_info" in source, "the budget must read free memory"
    assert "get_device_properties" not in source, "total memory is the wrong quantity"


def test_sm120_budget_is_decided_once_per_shape(monkeypatch):
    """Sizing and selection must agree even though free memory moves.

    They are separate calls. If selection re-read free memory it could admit a
    tactic the already-provisioned buffer cannot hold, which the C++ ICHECK_GE
    refuses at launch.
    """
    monkeypatch.setattr(svdquant_sm120_cutlass, "_SM120_WORKSPACE_BUDGET_CACHE", {})
    seen = []

    def moving_target(device_index):
        seen.append(len(seen))
        return (1 << 40) if not seen[:-1] else 4096

    monkeypatch.setattr(
        svdquant_sm120_cutlass, "_sm120_workspace_budget_bytes", moving_target
    )
    device = SimpleNamespace(index=0)
    key = (0, 64, 64, 64, 32)
    first = svdquant_sm120_cutlass._sm120_shape_workspace_budget(device, key)
    second = svdquant_sm120_cutlass._sm120_shape_workspace_budget(device, key)
    assert first == second, "the budget moved between sizing and selection"
    assert len(seen) == 1, "free memory was read more than once for one shape"


def test_sm120_row52_admission_does_not_widen_its_workspace(monkeypatch):
    """Admitting row 52 to the fused route cannot re-OOM the route autotuner.

    The three shapes that OOMed did so because Split-K sizing was reserved
    where the card could not back it. On a card that cannot, row 52's admission
    adds a runner without adding a byte of workspace.
    """
    _patch_workspace_module(monkeypatch, _fake_workspace_module())
    _patch_budget(monkeypatch, _CARD_TOO_SMALL_FOR_SPLITK)
    m, n, k, rank = _H3_ROW52
    assert (m, n, k) in _H3_SHAPES
    assert (
        svdquant_sm120_cutlass._sm120_max_workspace_bytes(
            SimpleNamespace(index=0), m, n, k, rank
        )
        == _LEGACY_WORKSPACE_BYTES
    )
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)


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


# JIT specialization and compilation policy.


@pytest.fixture
def isolated_jit_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(svdquant_sm120.jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path)
    monkeypatch.setattr(
        svdquant_sm120.jit_env,
        "FLASHINFER_CSRC_DIR",
        Path(__file__).resolve().parents[2] / "csrc",
    )
    monkeypatch.setattr(core, "check_cuda_arch", lambda: None)
    monkeypatch.setattr(core, "jit_spec_registry", core.JitSpecRegistry())
    monkeypatch.setattr(
        svdquant_sm120.current_compilation_context,
        "get_nvcc_flags_list",
        lambda supported_major_versions: ["-gencode=arch=compute_120f,code=sm_120f"],
    )


@pytest.mark.usefixtures("isolated_jit_sources")
@pytest.mark.parametrize(
    "rank,name,rank_flags",
    [
        (32, "nvfp4_svdquant_gemm_cutlass_sm120", []),
        (
            64,
            "nvfp4_svdquant_gemm_cutlass_sm120_rank64",
            ["-DSVDQ_SM120_LORA_RANK=64"],
        ),
    ],
)
def test_lora_rank_specializes_module_identity(
    rank: int, name: str, rank_flags: list[str]
) -> None:
    spec = svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank=rank)

    assert isinstance(spec, core.JitSpecNvcc)
    assert spec.name == name
    assert spec.extra_cuda_cflags is not None
    assert [
        flag
        for flag in spec.extra_cuda_cflags
        if flag.startswith("-DSVDQ_SM120_LORA_RANK=")
    ] == rank_flags


@pytest.mark.usefixtures("isolated_jit_sources")
@pytest.mark.parametrize("rank", [-32, 0, 1, 31, 33, 48])
def test_invalid_rank_is_rejected_before_writing_sources(
    rank: int, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="positive multiple of 32"):
        svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank=rank)

    assert not list(tmp_path.iterdir())


@pytest.mark.usefixtures("isolated_jit_sources")
@pytest.mark.parametrize("rank", [32, 64])
def test_module_generates_every_tactic_translation_unit(rank: int) -> None:
    spec = svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant(lora_rank=rank)

    assert isinstance(spec, core.JitSpecNvcc)
    assert [source.name for source in spec.sources[:2]] == [
        "nvfp4_svdquant_gemm_cutlass_sm120.cu",
        "nvfp4_smooth_quantize_sm100.cu",
    ]
    assert len(spec.sources) == 33
    assert all(source.is_file() for source in spec.sources)
    for source, config in zip(spec.sources[2:], SVDQUANT_SM120_CONFIGS, strict=True):
        assert source.name == f"nvfp4_svdquant_gemm_cutlass_sm120_{config}.cu"
        assert f"({config})" in source.read_text(encoding="utf-8")


@pytest.mark.usefixtures("isolated_jit_sources")
def test_module_links_cublaslt() -> None:
    spec = svdquant_sm120.gen_gemm_sm120_module_cutlass_nvfp4_svdquant()

    assert isinstance(spec, core.JitSpecNvcc)
    assert spec.extra_ldflags == ["-lcublasLt"]


_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_GEMM_DIR: Final = _REPO_ROOT / "include" / "flashinfer" / "gemm"
_COLLECTIVE_PATH: Final = _GEMM_DIR / "nvfp4_svdquant_gemm_collective_sm120.h"
_TEMPLATE_PATH: Final = _GEMM_DIR / "nvfp4_svdquant_gemm_template_sm120.h"
_BINDING_PATH: Final = _REPO_ROOT / "csrc" / "nvfp4_svdquant_gemm_cutlass_sm120.cu"
_JINJA_PATH: Final = _REPO_ROOT / "csrc" / "nvfp4_svdquant_gemm_cutlass_sm120.jinja"

# Configs whose per-thread stack frame shrinks when the dead path is compiled
# out, measured with `cuobjdump -res-usage` on the campaign JIT module.
_EXPECTED_NO_INLINE_DOWN: Final = frozenset(
    {
        "Tactic128x128x128Config",
        "Tactic128x128x128SwapConfig",
        "Tactic128x128x128SkConfig",
        "Tactic128x128x128SwapSkConfig",
        "Tactic128x128x256Config",
        "Tactic128x128x256SwapConfig",
        "Tactic128x128x256SkConfig",
        "Tactic128x128x256SwapSkConfig",
        "Tactic256x128x128Config",
        "Tactic256x128x128SwapConfig",
        "Tactic128x256x128Config",
        "Tactic128x256x128SwapConfig",
        "Tactic128x64x128SwapConfig",
        "Tactic128x64x128SwapSkConfig",
        "Tactic128x64x256SwapConfig",
        "Tactic128x64x256SwapSkConfig",
        "Tactic128x32x128SwapConfig",
        "Tactic128x32x256SwapConfig",
    }
)

# Compiling the path out measurably grows these frames, so they must keep it.
_MUST_KEEP_INLINE_DOWN: Final = frozenset(
    {
        "Tactic256x128x128SkConfig",
        "Tactic256x128x128SwapSkConfig",
        "Tactic128x256x128SkConfig",
        "Tactic128x256x128SwapSkConfig",
        "Tactic256x128x128SwapStaticConfig",
    }
)


def _header_list() -> frozenset[str]:
    body = _TEMPLATE_PATH.read_text(encoding="utf-8")
    start = body.index("#define SVDQ_SM120_NO_INLINE_DOWN_CONFIG_LIST(X)")
    end = body.index(
        "template <class Config>\ninline constexpr bool kCompileInlineDownSm120", start
    )
    return frozenset(re.findall(r"X\((Tactic\w+)\)", body[start:end]))


def _jinja_list() -> frozenset[str]:
    body = _JINJA_PATH.read_text(encoding="utf-8")
    start = body.index("{% set NO_INLINE_DOWN_CONFIGS")
    end = body.index("%}", start)
    return frozenset(re.findall(r'"(Tactic\w+)"', body[start:end]))


def test_collective_still_exposes_the_compile_time_toggle() -> None:
    """The per-config toggle must remain a compile-time template parameter."""
    collective_source = _COLLECTIVE_PATH.read_text(encoding="utf-8")
    template_source = _TEMPLATE_PATH.read_text(encoding="utf-8")

    # No trailing ">" in this match: what matters is that the toggle is a
    # defaulted compile-time template parameter, not that it is the last one in
    # the list. Pinning the ">" made this fail the moment a parameter was added
    # after it, which says nothing about the toggle.
    assert "bool CompileInlineDown_ = true" in collective_source
    assert (
        "static constexpr bool CompileInlineDown = CompileInlineDown_;"
        in collective_source
    )
    # The LoRA rank reached the collective the same way, and for the same reason:
    # it has to be known at compile time for the smem overlay to be sized.
    assert "int LoRaK_ = 32" in collective_source
    assert "static constexpr int LoRaK = LoRaK_;" in collective_source
    assert "if constexpr (CollectiveMainloop::CompileInlineDown)" in collective_source
    assert "CollectiveMainloop::inline_lora_down_m537_k5120(" in collective_source
    assert "using CollectiveMainloop = CollectiveMainloopT<true>;" in template_source
    assert (
        "using CollectiveMainloopNoInlineDown = CollectiveMainloopT<false>;"
        in template_source
    )
    assert "run_tactic_no_inline_down" in template_source


def test_production_dispatch_selects_per_config_not_per_tactic_id() -> None:
    """The old `tactic == 9` dispatch covered one runtime id out of 38 affected shapes."""
    binding_source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "if (tactic == sd::kNoInlineDownRuntimeTactic)" not in binding_source
    assert "if constexpr (sd::kCompileInlineDownSm120<C>)" in binding_source
    assert "return sd::run_tactic_no_inline_down<C>(" in binding_source


def test_selection_list_matches_the_measured_configs() -> None:
    """The header list is the single source of truth for the selection."""
    assert _header_list() == _EXPECTED_NO_INLINE_DOWN


def test_jinja_instantiates_exactly_the_selected_configs() -> None:
    """Drift guard: an entry added in one place but not the other fails to link."""
    assert _jinja_list() == _header_list()


def test_regressing_configs_keep_the_inline_down_path() -> None:
    """Configs whose frame grows when the path is removed must not be selected."""
    assert _MUST_KEEP_INLINE_DOWN.isdisjoint(_header_list())


def test_inline_down_entry_still_reaches_its_only_config() -> None:
    """nvfp4_svdquant_gemm_run_inline_down needs the CompileInlineDown=true build."""
    binding_source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "using C = sd::Tactic256x128x128SwapStaticConfig;" in binding_source
    assert "Tactic256x128x128SwapStaticConfig" not in _header_list()


# Benchmark workloads and tuning precision.

H3_TEXT_LENGTHS: Final[tuple[int, ...]] = (537, 1935, 6913)
H3_OMNI_LENGTHS: Final[tuple[int, ...]] = (61056, 73984, 82752)
H3_BLOCK_NK_SHAPES: Final[tuple[tuple[int, int], ...]] = (
    (21504, 5376),
    (5376, 7168),
    (28672, 5376),
    (5376, 14336),
)


def test_model_shape_registry_is_immutable() -> None:
    # Given: the shared registry is imported by tests, benchmarks, and tools.
    # When: its public container type is inspected.
    # Then: callers cannot mutate the process-wide source of truth in place.
    assert isinstance(MODEL_SHAPE_CASES, tuple)


def test_h3_official_example_shapes_are_registered() -> None:
    # Given: all recurring Linear shapes from the three official H3 examples.
    expected_h3_shapes: set[ShapeCase] = {
        (m, n, k)
        for m in (*H3_TEXT_LENGTHS, *H3_OMNI_LENGTHS)
        for n, k in H3_BLOCK_NK_SHAPES
    }
    expected_h3_shapes.update((m, 5376, 5120) for m in H3_TEXT_LENGTHS)

    # When: the expected H3 workload is compared with the shared registry.
    missing_shapes = expected_h3_shapes.difference(MODEL_SHAPE_CASES)

    # Then: every H3 shape is available through --shape-set model.
    assert missing_shapes == set()


def test_model_shape_count_includes_h3_batch() -> None:
    # Given: the existing 44 rows and 27 new unique H3 rows.
    expected_shape_count = 71

    # When: the registry's declared and materialized sizes are inspected.
    actual_shape_counts = (MODEL_SHAPE_COUNT, len(MODEL_SHAPE_CASES))

    # Then: both expose the complete four-batch workload.
    assert actual_shape_counts == (expected_shape_count, expected_shape_count)


@pytest.fixture
def benchmark_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    return (
        importlib.import_module("svdquant_sm120_cli"),
        importlib.import_module("svdquant_sm120_benchmark"),
    )


def test_precision_defaults_preserve_operator_policy(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    cli, _ = benchmark_modules
    options = cli.parse_options([])
    assert options.tuning_repeat is None
    assert options.tuning_replays is None
    assert options.tuning_l2 == "operator"


def test_precision_options_do_not_change_cache_policy(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    cli, _ = benchmark_modules
    options = cli.parse_options(["--tuning-repeat", "64", "--tuning-replays", "5"])
    assert (options.tuning_repeat, options.tuning_replays) == (64, 5)
    assert options.tuning_l2 == "operator"


def test_benchmark_selects_rank_and_untuned_eager_mode(benchmark_modules) -> None:
    cli, _ = benchmark_modules
    options = cli.parse_options(["--rank", "64", "--untuned", "--eager"])
    assert (options.ranks, options.untuned, options.eager) == ((64,), True, True)


def test_benchmark_records_auto_latency_and_actual_rank(
    benchmark_modules, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import json

    cli, helpers = benchmark_modules
    bench = importlib.import_module("bench_svdquant_sm120_backends")
    tensor = torch.ones(2, 32)
    case = helpers.Case(*([tensor] * 8), torch.ones(32, 64), *([tensor] * 3))
    seen = []

    def make_call(case, operation, backend, enable_pdl):
        seen.append(backend)
        return helpers.Call(lambda: tensor, ())

    monkeypatch.setattr(bench, "make_call", make_call)
    monkeypatch.setattr(bench, "measure", lambda *args: [2.0, 2.0, 2.0])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    env = helpers.Environment(
        "now", "cpu", "test", (12, 0), 1, 1, "test", "test", None, "test"
    )
    output = io.StringIO()
    bench.run_operation(case, (2, 32, 32), "linear", cli.parse_options([]), env, output)
    record = json.loads(output.getvalue())
    assert seen == ["cute-dsl", "cutlass-sm120", "auto"]
    assert record["rank"] == 64
    assert record["median_us"]["auto"] == 2.0
    assert record["auto_over_best_explicit"] == 1.0
    assert record["auto_accuracy"]["exact_match"]


@pytest.mark.parametrize("flag", ["--tuning-repeat", "--tuning-replays"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_precision_counts_must_be_positive(
    benchmark_modules: tuple[ModuleType, ModuleType], flag: str, value: str
) -> None:
    cli, _ = benchmark_modules
    with pytest.raises(SystemExit) as failure:
        cli.parse_options([flag, value])
    assert failure.value.code == 2


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("policy", ["operator", "warm", "cold"])
def test_precision_restores_repeat_and_replay_context(
    benchmark_modules: tuple[ModuleType, ModuleType], fail: bool, policy: str
) -> None:
    _, helpers = benchmark_modules
    tuner = AutoTuner.get()
    original_repeat = tuner.repeat
    original_replays = tuner._override_cuda_graph_profile_replays
    original_tuning_mode = tuner.is_tuning_mode

    def run_context() -> None:
        with helpers.tuning_precision(repeat=64, replays=5):
            assert tuner.repeat == 64
            with helpers.tuning_context(helpers.tuning_policy(policy), tuning=True):
                assert tuner.is_tuning_mode
                config = tuner._apply_tuning_overrides(
                    TuningConfig(use_cuda_graph=True)
                )
                assert config.cuda_graph_profile_replays == 5
                assert tuner._get_profiling_repeat(config) == 64
            if fail:
                raise RuntimeError("benchmark failure")

    if fail:
        with pytest.raises(RuntimeError, match="benchmark failure"):
            run_context()
    else:
        run_context()
    assert tuner.repeat == original_repeat
    assert tuner._override_cuda_graph_profile_replays == original_replays
    assert tuner.is_tuning_mode == original_tuning_mode


def test_default_precision_preserves_an_outer_override(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, helpers = benchmark_modules
    tuner = AutoTuner.get()
    with helpers.tuning_precision(repeat=17, replays=3):
        with helpers.tuning_precision():
            assert tuner.repeat == 17
            assert tuner._override_cuda_graph_profile_replays == 3
        with helpers.tuning_precision(repeat=64, replays=5):
            assert tuner.repeat == 64
        assert tuner.repeat == 17
        assert tuner._override_cuda_graph_profile_replays == 3


def test_repeat_restored_if_replay_context_cannot_enter(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, helpers = benchmark_modules
    tuner = AutoTuner.get()
    original_repeat = tuner.repeat
    with (
        pytest.raises(ValueError, match="cuda_graph_profile_replays"),
        helpers.tuning_precision(repeat=64, replays=0),
    ):
        pytest.fail("invalid replay count entered the context")
    assert tuner.repeat == original_repeat


# Native dimension, alignment, empty-batch and integer guards.

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
GEMM_SOURCE: Final = REPO_ROOT / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu"
QUANT_SOURCE: Final = REPO_ROOT / "csrc/nvfp4_smooth_quantize_sm100.cu"
QUANT_HEADER: Final = (
    REPO_ROOT / "include/flashinfer/gemm/nvfp4_smooth_quantize_sm100.cuh"
)


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    closing = opening + 1
    while depth:
        depth += (source[closing] == "{") - (source[closing] == "}")
        closing += 1
    return source[start:closing]


@pytest.fixture(scope="module")
def bounds_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    source = GEMM_SOURCE.read_text()
    signatures = [
        "inline bool fits_nonnegative_int(",
        "inline void check_sm120_tactic(",
        "bool nvfp4_svdquant_gemm_can_implement(int64_t",
        "int64_t nvfp4_svdquant_gemm_fallback_tactic(int64_t",
        "int64_t nvfp4_svdquant_gemm_workspace_size(int64_t",
        "int64_t nvfp4_svdquant_gemm_shared_storage_size(int64_t",
        "int64_t nvfp4_svdquant_gemm_tactic_row(int64_t",
    ]
    # Optional helpers let the same behavioral harness run against the pre-fix source.
    functions = "\n".join(
        _function(source, signature) for signature in signatures if signature in source
    )
    quant_source = QUANT_SOURCE.read_text()
    sf_functions = []
    for name in ("nvfp4_quantize_smooth", "check_sm120_prefix_inputs"):
        body = _function(quant_source, f"void {name}(")
        initializer = re.search(r"int64_t const sfSize\s*=\s*(.*?);", body, re.DOTALL)
        assert initializer is not None
        sf_functions.append(
            f"int64_t {name}_sf(int m, int n, int k) {{ return {initializer[1]}; }}"
        )
    quant_header = QUANT_HEADER.read_text()
    padding = re.search(r"__host__\s+__device__\s+inline\s+\w+\s+padUp\(", quant_header)
    assert padding is not None
    program = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#define __host__
#define __device__
struct Check {
  explicit Check(bool condition) { if (!condition) throw std::out_of_range("FFI bounds"); }
  template <class T> Check& operator<<(T const&) { return *this; }
};
#define TVM_FFI_ICHECK(condition) Check(condition)
std::array<int64_t, 5> expected{128, 128, 128, 32, 0};
int backend_calls = 0;
namespace flashinfer::gemm {
namespace svdquant_sm120 {
constexpr int kNvfp4SvdquantGemmSm120NumTactics = 83;
struct RuntimeTactic { int kernel_id, splits, raster, swizzle; };
RuntimeTactic decode_runtime_tactic(int tactic) {
  ++backend_calls;
  if (tactic < 0 || tactic >= kNvfp4SvdquantGemmSm120NumTactics)
    throw std::out_of_range("backend tactic");
  return {tactic, 1, 0, 1};
}
bool kernel_is_streamk_sm120(int) { return false; }
}
void check_dimensions(int m, int n, int k) {
  ++backend_calls;
  assert(m == expected[0] && n == expected[1] && k == expected[2]);
}
bool nvfp4_svdquant_gemm_can_implement(int m, int n, int k, int rank, int tactic) {
  check_dimensions(m, n, k);
  assert(rank == expected[3] && tactic == expected[4]);
  return true;
}
int nvfp4_svdquant_gemm_fallback_tactic(int m, int n, int k, int rank) {
  check_dimensions(m, n, k);
  assert(rank == expected[3]);
  return 0;
}
size_t nvfp4_svdquant_gemm_workspace_size(int m, int n, int k, int tactic) {
  check_dimensions(m, n, k);
  assert(tactic == expected[4]);
  return 64;
}
size_t nvfp4_svdquant_gemm_shared_storage_size(int tactic) {
  ++backend_calls;
  assert(tactic == expected[4]);
  return 32;
}
}
namespace boundary {
"""
    program += (
        functions
        + "\n"
        + _function(quant_header, padding[0])
        + "\n"
        + "\n".join(sf_functions)
    )
    program += r"""
}
template <class F> void require_rejection(F call) {
  bool rejected = false;
  try { call(); } catch (std::out_of_range const&) { rejected = true; }
  assert(rejected);
}
int main(int argc, char** argv) {
  assert(argc == 4);
  int const scenario = std::stoi(argv[1]), axis = std::stoi(argv[2]);
  int64_t const value = std::stoll(argv[3]);
  expected[axis] = value;
  auto const [m, n, k, rank, tactic] = expected;
  using namespace boundary;
  auto can_implement = [&] { return nvfp4_svdquant_gemm_can_implement(m, n, k, rank, tactic); };
  auto workspace = [&] { return nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic); };
  switch (scenario) {
    case 0:
      assert(!can_implement());
      assert(nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank) == -1);
      if (axis != 3) require_rejection(workspace);
      assert(backend_calls == 0);
      break;
    case 1:
      require_rejection(can_implement);
      require_rejection(workspace);
      require_rejection([&] { nvfp4_svdquant_gemm_shared_storage_size(tactic); });
      require_rejection([&] { nvfp4_svdquant_gemm_tactic_row(tactic); });
      assert(backend_calls == 0);
      break;
    case 2:
      assert(can_implement());
      assert(nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank) == 0);
      assert(workspace() == 64);
      assert(nvfp4_svdquant_gemm_shared_storage_size(tactic) == 32);
      assert(nvfp4_svdquant_gemm_tactic_row(tactic) == (tactic | (1 << 8) | (1 << 24)));
      assert(backend_calls == 5);
      break;
    case 3: {
      int const rows = static_cast<int>(value);
      int64_t const padded = (value / 128 + (value % 128 != 0)) * 128;
      assert(padUp(rows, 128) == padded);
      assert(padUp(rows, 64) == (value / 64 + (value % 64 != 0)) * 64);
      assert(nvfp4_quantize_smooth_sf(rows, 128, 128) == padded * 8);
      assert(check_sm120_prefix_inputs_sf(rows, 128, 128) == padded * 8);
      break;
    }
    default: assert(false);
  }
}
"""
    directory = tmp_path_factory.mktemp("sm120_integer_bounds")
    cpp = directory / "bounds.cpp"
    binary = directory / "bounds"
    cpp.write_text(program)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def _run_case(binary: Path, arguments: tuple[int, int, int]) -> None:
    result = subprocess.run(
        [str(binary), *map(str, arguments)],
        cwd=binary.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("axis", range(4), ids=("m", "n", "k", "rank"))
@pytest.mark.parametrize("value", (-1, 2**31, 2**32 + 128, 2**32 + 32, 2**63 - 1))
def test_ffi_queries_reject_dimensions_before_narrowing(
    bounds_binary: Path, axis: int, value: int
) -> None:
    _run_case(bounds_binary, (0, axis, value))


@pytest.mark.parametrize("tactic", (-(2**32), -1, 83, 2**32, 2**32 + 1, 2**63 - 1))
def test_ffi_queries_reject_tactics_before_narrowing(
    bounds_binary: Path, tactic: int
) -> None:
    _run_case(bounds_binary, (1, 4, tactic))


@pytest.mark.parametrize("axis", range(4), ids=("m", "n", "k", "rank"))
@pytest.mark.parametrize("value", (0, 128, 2**31 - 1))
def test_ffi_queries_preserve_representable_arguments(
    bounds_binary: Path, axis: int, value: int
) -> None:
    _run_case(bounds_binary, (2, axis, value))


@pytest.mark.parametrize("rows", (0, 1, 128, 2**31 - 128, 2**31 - 127, 2**31 - 1))
def test_scale_padding_preserves_int32_boundary(bounds_binary: Path, rows: int) -> None:
    _run_case(bounds_binary, (3, 0, rows))


@pytest.mark.parametrize("entry", ("can_implement_tactic", "run_tactic_impl"))
def test_cutlass_scale_layout_uses_wide_intermediates(
    tmp_path: Path, entry: str
) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    nvcc = shutil.which("nvcc")
    cuda_root = Path(
        os.environ.get(
            "CUDA_HOME", str(Path(nvcc).parent.parent) if nvcc else "/usr/local/cuda"
        )
    )
    includes = [f"-I{cuda_root}/include", f"-I{cuda_root}/include/cccl"]
    probe = subprocess.run(
        [compiler, "-std=c++17", *includes, "-x", "c++", "-fsyntax-only", "-"],
        input="#include <cuda_runtime_api.h>\n#include <cuda/std/type_traits>\n",
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        pytest.skip("requires CUDA and CCCL development headers (no GPU needed)")
    source = (
        REPO_ROOT / "include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
    ).read_text()
    helper = _function(
        source, "template <class Layout>\nconstexpr auto narrow_sf_layout("
    )
    body = _function(source, f"{entry}(RuntimeTacticSm120")
    layout_setup = body[
        body.index("auto const sf_shape") : body.index("args.mainloop.lora_rank")
    ]
    program = r"""
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include <cassert>
#include <cstdint>
"""
    program += (
        helper
        + r"""
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<16>;
using LayoutSF = Sm1xxBlkScaledConfig::LayoutSF;
template<bool Swap> struct Configuration { static constexpr bool SwapAB = Swap; };
template<bool Swap> void check(int m, int n, int k) {
  using Config = Configuration<Swap>;
  struct { struct { LayoutSF layout_SFA, layout_SFB; } mainloop; } args;
"""
    )
    program += (
        layout_setup
        + r"""
  auto expected_size = [=](int rows) { return (rows + 127LL) / 128 * 128 * ((k + 63LL) / 64 * 4); };
  assert(cute::cosize(args.mainloop.layout_SFA) == expected_size(Swap ? n : m));
  assert(cute::cosize(args.mainloop.layout_SFB) == expected_size(Swap ? m : n));
  if (m == 131073) {
    auto result = Swap ? args.mainloop.layout_SFB : args.mainloop.layout_SFA;
    assert(result(cute::make_coord(131072, 16368, 0)) == 134348291);
  }
  if (m == 537) {
    auto original = cute::make_shape(Swap ? n : m, Swap ? m : n, k, 1);
    assert(args.mainloop.layout_SFA == Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(original));
    assert(args.mainloop.layout_SFB == Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(original));
  }
}
int main() {
  volatile int m = 131073;
  check<false>(m, 128, 16384); check<true>(m, 128, 16384);
  check<false>(537, 5376, 5120); check<true>(537, 5376, 5120);
  check<false>(1048575, 128, 32704); check<true>(1048575, 128, 32704);
}
"""
    )
    cpp, binary = tmp_path / "layout.cpp", tmp_path / "layout"
    cpp.write_text(program)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            *includes,
            f"-I{REPO_ROOT}/3rdparty/cutlass/include",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


PRODUCER_HEADER: Final = (
    REPO_ROOT / "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
)


def _run_ubsan(tmp_path: Path, source: str) -> None:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    program = tmp_path / "bounds.cpp"
    program.write_text(source)
    binary = tmp_path / "bounds"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(program),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("rank", (32, 64))
def test_dynamic_prefix_offsets_cross_int32_boundary(tmp_path: Path, rank: int) -> None:
    # Compile the production expressions so UBSan checks C++ promotion rules,
    # without allocating multi-gigabyte CUDA tensors to reach the boundary.
    source = PRODUCER_HEADER.read_text()
    small = source.split(
        "void nvfp4_smooth_quantize_lora_down_small_m_dyn_sm120_kernel(", 1
    )[1]
    small = small.split("// Runtime-M/K small-M launch.", 1)[0]
    large = source.split(
        "void nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel(", 1
    )[1]
    large = large.split("inline cudaError_t launch_large_m_dyn_kernel", 1)[0]

    def expression(body: str, pattern: str) -> str:
        match = re.search(pattern, body, re.DOTALL)
        assert match is not None, pattern
        return match[1]

    cases = [
        (
            expression(small, r"a_base\s*=\s*x\s*\+\s*(.*?);"),
            "int a_row = 131072;",
            2147483670,
        ),
        (
            expression(small, r"a_upper\s*=\s*x\s*\+\s*(.*?);"),
            "int a_row = 131064;",
            2147483670,
        ),
        (
            expression(small, r"b_base\s*=\s*l2t_smoothed\s*\+\s*(.*?);"),
            f"k_base = {2147483648 // rank}; pair_col = 0;",
            2147483655,
        ),
        (expression(small, r"down\[(.*?)\]\s*="), "", 2147483655),
        (expression(large, r"b_base\s*=\s*l2t_smoothed\s*\+\s*(.*?);"), "", 2147483655),
        (
            expression(small, r"kL2PrefetchLines\s*=\s*(.*?);"),
            "K = 33554432;",
            33554432 * rank * 2 // 128,
        ),
        (
            expression(large, r"kL2Bytes\s*=\s*(.*?);"),
            "K = 33554432;",
            33554432 * rank * 2,
        ),
        (
            expression(small, r"packed_index\s*=\s*(.*?);"),
            "k_base = 536870912;",
            2147483648,
        ),
    ]
    down_offsets = re.findall(r"store_down_pair<[^>]+>\(down\s*\+\s*(.*?),", large)
    assert down_offsets
    cases.extend((offset, "", 2147483655) for offset in down_offsets)
    checks = []
    for offset, setup, expected in cases:
        checks.append(
            "{\n"
            "int K = input, k_base = 16, pair_col = 6, b_col = 7, col = 7;\n"
            "int kDownNTiles = 1, kNFragments = 2, tile_n = 0, fragment = 0, lane = 0;\n"
            f"int row = {2147483648 // rank}, b_row = row;\n"
            "int row0 = row, row1 = row, row2 = row, row3 = row, row4 = row;\n"
            "int row5 = row, row6 = row, row7 = row, row8 = row, row9 = row;\n"
            f"{setup}\n"
            f"std::int64_t actual = {offset};\n"
            f"if (actual != {expected}LL) return __LINE__;\n"
            "}\n"
        )
    _run_ubsan(
        tmp_path,
        "#include <cstdint>\n"
        "namespace quant { using Type = std::uint16_t; }\n"
        f"constexpr int kRank = {rank}, kL2PrefetchBytes = 128;\n"
        "int main() { volatile int input = 16384;\n" + "".join(checks) + "}\n",
    )


def test_gemm_shape_bounds_precede_layout_and_workspace_construction(
    tmp_path: Path,
) -> None:
    source = (
        REPO_ROOT / "include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
    ).read_text()
    guard = source.split(
        "template <class Config>\ninline bool runtime_tactic_feasible", 1
    )[1]
    guard = (
        "template <class Config>\ninline bool runtime_tactic_feasible"
        + guard.split("// Keep CUTLASS's LayoutSF", 1)[0]
    )
    for entry in (
        "workspace_size_for_tactic",
        "can_implement_tactic",
        "run_tactic_impl",
    ):
        body = source.split(f"{entry}(RuntimeTacticSm120", 1)[1]
        assert body.index("runtime_tactic_feasible<Config>(rt, m, n, k)") < body.index(
            "using Gemm"
        )
    # Only the compile-time tile/config metadata is stubbed; the admission function is real.
    prelude = r"""
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>
struct StreamKTag {};
struct PersistentTag {};
struct RuntimeTacticSm120 { int splits, swizzle; };
namespace cute {
template<class A, class B> constexpr bool is_same_v = std::is_same_v<A, B>;
template<int I, class Shape> constexpr int size(Shape) { return I == 1 ? 64 : 128; }
}
template<bool Swap, class Scheduler> struct Config {
  struct MmaTileShape {};
  static constexpr bool SwapAB = Swap;
  using TileSchedulerTag = Scheduler;
  struct CollectiveMainloop {
    struct Sm1xxBlkScaledConfig { static constexpr int SFVecSize = 16; };
  };
};
"""
    checks = r"""
template<bool Swap> void check() {
  using P = Config<Swap, PersistentTag>;
  using S = Config<Swap, StreamKTag>;
  RuntimeTacticSm120 rt{1, 1};
  int const limit = std::numeric_limits<int>::max();
  assert(runtime_tactic_feasible<P>(rt, 131088, 128, 16384));
  assert(runtime_tactic_feasible<S>(rt, 131088, 128, 16384));
  assert(runtime_tactic_feasible<P>(rt, 1048448, 128, 32768));
  assert(!runtime_tactic_feasible<P>(rt, 1048576, 128, 32768));
  assert(!runtime_tactic_feasible<P>(rt, 128, 1048576, 32768));
  assert(!runtime_tactic_feasible<P>(rt, limit, 128, 128));
  assert(!runtime_tactic_feasible<P>(rt, 128, limit, 128));
  assert(!runtime_tactic_feasible<P>(rt, 128, 128, limit - 127));
  assert(!runtime_tactic_feasible<P>(rt, -1, 128, 128));
  assert(!runtime_tactic_feasible<P>(rt, 128, 128, 0));
  assert(!runtime_tactic_feasible<P>(rt, 8388608, 8388608, 64));
  assert(runtime_tactic_feasible<P>(rt, 1048576, 1048576, 4096));
  assert(!runtime_tactic_feasible<S>(rt, 1048576, 1048576, 4096));
  rt.splits = 32;
  assert(runtime_tactic_feasible<S>(rt, 128, 128, 4096));
  rt.splits = 33;
  assert(!runtime_tactic_feasible<S>(rt, 128, 128, 4096));
}
int main() { check<false>(); check<true>(); }
"""
    _run_ubsan(tmp_path, prelude + guard + checks)


@pytest.fixture(scope="module", params=(32, 64))
def alignment_binary(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler with UBSan")
    root = Path(__file__).resolve().parents[2]
    quantizer = (root / "csrc/nvfp4_smooth_quantize_sm100.cu").read_text()
    linear = (root / "csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu").read_text()
    functions = _function(linear, "inline bool fits_nonnegative_int(")
    # Keep the real empty/shape checks, intercepting work at the plan lookup.
    lora_down = _function(linear, "void run_lora_down_cublaslt(")
    functions += lora_down[: lora_down.index("LoraDownCublasLtState& state =")]
    functions += "++backend_calls; }\n"
    functions += "\n".join(
        _function(quantizer, f"void {name}(")
        for name in (
            "nvfp4_quantize_smooth",
            "check_sm120_prefix_inputs",
            "nvfp4_quantize_smooth_lora_down_sm120_impl",
            "nvfp4_quantize_smooth_lora_down_dyn_sm120_impl",
        )
    )
    functions += _function(
        linear, "void nvfp4_quantize_smooth_lora_down_cublaslt_sm120("
    )
    header = (
        root / "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
    ).read_text()
    large_m = _function(header, "inline cudaError_t launch_large_m_dyn_kernel(")
    launch = large_m.index("nvfp4_smooth_quantize_lora_down_large_m_dyn_sm120_kernel<")
    end = large_m.index("return cudaGetLastError();", launch)
    # Execute all real geometry checks, replacing only the CUDA launch syntax.
    large_m = large_m[:launch] + "++backend_calls;\n" + large_m[end:]
    functions += (
        "\nnamespace quant { using Type = std::uint16_t; constexpr int SF_VEC_SIZE = 16; }\n"
        "enum class DynAddress { Recompute };\n"
        "template <int BlockThreads, bool LongK, int TileM, int TileK, DynAddress Policy, int Rank>\n"
        + large_m
    )
    program = r"""
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
struct Check {
  bool passed;
  std::ostringstream message;
  explicit Check(bool condition) : passed(condition) {}
  template<class T> Check& operator<<(T const& value) {
    if (!passed) message << value;
    return *this;
  }
  ~Check() noexcept(false) { if (!passed) throw std::runtime_error(message.str()); }
};
#define TVM_FFI_ICHECK(condition) Check(condition)
#define TVM_FFI_ICHECK_EQ(a, b) Check((a) == (b))
#define TVM_FFI_ICHECK_GE(a, b) Check((a) >= (b))
#define CHECK_INPUT_AND_TYPE(...)
#define CHECK_DEVICE(...)
struct Device { int device_id = 0; };
struct TensorView {
  std::vector<int64_t> shape;
  unsigned char* pointer;
  int ndim() const { return shape.size(); }
  int64_t size(int index) const { return shape.at(index); }
  int64_t numel() const {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>{});
  }
  void* data_ptr() const { return pointer; }
  Device device() const { return {}; }
};
using cudaStream_t = int;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaErrorInvalidValue = 1;
constexpr int kSm120PrefixRank = TEST_RANK;
struct LoraDownCublasLtState { static constexpr int kRank = 32; };
int backend_calls = 0;
int getMultiProcessorCount(int) { return 1; }
int get_stream(Device) { return 0; }
const char* cudaGetErrorString(int) { return "CUDA stub"; }
int cudaGetLastError() { return cudaSuccess; }
namespace flashinfer::gemm {
template<class... Args> void nvfp4_smooth_quantize(Args...) { ++backend_calls; }
template<int Rank, class... Args> int nvfp4_smooth_quantize_lora_down_sm120(Args...) {
  ++backend_calls;
  return cudaSuccess;
}
template<int Rank, class... Args> int nvfp4_smooth_quantize_lora_down_family_sm120(Args...) {
  ++backend_calls;
  return cudaSuccess;
}
}
"""
    program += functions
    program += r"""
int main(int argc, char** argv) {
  assert(argc == 4 || argc == 5);
  bool const empty = argc == 5;
  int const entry = std::stoi(argv[1]), index = std::stoi(argv[2]), offset = std::stoi(argv[3]);
  alignas(16) std::array<unsigned char, 512> storage{};
  std::array<TensorView, 8> inputs{{
    {{64, 3072}, storage.data()}, {{3072}, storage.data() + 64},
    {{1}, storage.data() + 128}, {{3072, TEST_RANK}, storage.data() + 192},
    {{64, 1536}, storage.data() + 256}, {{128 * 192}, storage.data() + 320},
    {{64, TEST_RANK}, storage.data() + 384}, {{16}, storage.data() + 448}
  }};
  if (empty) {
    for (int i : {0, 4, 5, 6}) inputs[i].shape[0] = 0;
  }
  inputs.at(index).pointer += offset;
  auto const& [x, pqs, scale, l2t, xq, sf, down, workspace] = inputs;
  // The cuBLASLt prefix is rank-32 only; it must refuse a rank-64 module's
  // tensors before quantizing, whatever the alignment.
  if (entry == 3 && TEST_RANK != 32) {
    try {
      nvfp4_quantize_smooth_lora_down_cublaslt_sm120(x, pqs, scale, l2t, xq, sf, down, workspace);
    } catch (std::runtime_error const& error) {
      assert(std::string(error.what()).find("supports rank 32 only") != std::string::npos);
      assert(backend_calls == 0);
      return 0;
    }
    assert(false);
  }
  bool rejected = false;
  try {
    switch (entry) {
      case 0: nvfp4_quantize_smooth(x, pqs, scale, xq, sf, false); break;
      case 1: nvfp4_quantize_smooth_lora_down_sm120_impl(x, pqs, scale, l2t, xq, sf, down, false); break;
      case 2: nvfp4_quantize_smooth_lora_down_dyn_sm120_impl(x, pqs, scale, l2t, xq, sf, down, 0, 1, 1, 1, 0, false); break;
      case 3: nvfp4_quantize_smooth_lora_down_cublaslt_sm120(x, pqs, scale, l2t, xq, sf, down, workspace); break;
      case 4: {
        auto const status = launch_large_m_dyn_kernel<384, false, 32, 256, DynAddress::Recompute, TEST_RANK>(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, x.size(0), x.size(1), 0);
        assert(status == (empty ? cudaErrorInvalidValue : cudaSuccess));
        break;
      }
      default: assert(false);
    }
  } catch (std::runtime_error const& error) {
    rejected = true;
    std::string expected = std::string(index == 0 ? "x" : "pqs") + " must be 16-byte aligned";
    assert(std::string(error.what()).find(expected) != std::string::npos);
  }
  assert(rejected == (offset % 16 != 0));
  assert(backend_calls == (rejected || empty ? 0 : (entry == 3 ? 2 : 1)));
}
"""
    directory = tmp_path_factory.mktemp(f"sm120_alignment_rank{request.param}")
    cpp, binary = directory / "alignment.cpp", directory / "alignment"
    cpp.write_text(program)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            f"-DTEST_RANK={request.param}",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "entry", range(4), ids=("quantize", "fused", "dynamic", "cublaslt")
)
@pytest.mark.parametrize("index", (0, 1), ids=("x", "pqs"))
@pytest.mark.parametrize("offset", range(0, 18, 2))
def test_native_alignment_guard_before_launch(
    alignment_binary: Path, entry: int, index: int, offset: int
) -> None:
    result = subprocess.run(
        [str(alignment_binary), str(entry), str(index), str(offset)],
        cwd=alignment_binary.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "entry",
    range(5),
    ids=("quantize", "fused", "dynamic", "cublaslt", "large-m-launcher"),
)
def test_native_empty_prefix_skips_launch(alignment_binary: Path, entry: int) -> None:
    result = subprocess.run(
        [str(alignment_binary), str(entry), "0", "0", "empty"],
        cwd=alignment_binary.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_large_m_launcher_accepts_nonempty_batch(alignment_binary: Path) -> None:
    result = subprocess.run(
        [str(alignment_binary), "4", "0", "0"],
        cwd=alignment_binary.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# cuBLASLt plan-cache ownership, growth and capture guards.


def test_persisting_l2_hints_are_skipped_during_stream_capture() -> None:
    source = _BINDING_PATH.read_text(encoding="utf-8")

    assert "cudaStreamIsCapturing(stream, &capture_status)" in source

    # The cuBLASLt LoRA-down must not leave a stream window or device limit set.
    lora_down = _function(source, "void run_lora_down_cublaslt(")
    assert "persisting_l2" not in lora_down

    linear = source[source.index("void nvfp4_svdquant_linear_sm120(") :]
    linear = linear[: linear.index("\n#endif")]
    assert "configure_case7_persisting_l2 = !is_capturing" in linear
    assert linear.count("if (configure_case7_persisting_l2)") == 3


@pytest.fixture(scope="module")
def plan_cache_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = _BINDING_PATH.read_text(encoding="utf-8")
    # Compile the actual ownership and cache code; only the vendor API is faked.
    cache = source[
        source.index("struct LoraDownCublasLtState {") : source.index(
            "void run_lora_down_cublaslt("
        )
    ]
    program = r"""
#include <array>
#include <atomic>
#include <cassert>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
using cudaStream_t = void*;
using cudaError_t = int;
using cublasStatus_t = int;
using cublasLtHandle_t = int*;
using cublasLtMatmulDesc_t = int*;
using cublasLtMatrixLayout_t = int*;
using cublasLtMatmulPreference_t = int*;
constexpr int cudaSuccess = 0, CUBLAS_STATUS_SUCCESS = 0;
constexpr int CUBLAS_COMPUTE_32F = 0, CUDA_R_32F = 0, CUDA_R_16BF = 0;
constexpr int CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 0;
struct cublasLtMatmulHeuristicResult_t { int state = 0; int algo = 0; };
std::atomic<int> live_resources{0};
std::atomic<int> resource_ids{0};
thread_local int current_device = 0, api_calls = 0, fail_on_call = -1;
int cudaGetDevice(int* device) { *device = current_device; return cudaSuccess; }
int next_status() { return ++api_calls == fail_on_call ? 1 : 0; }
int create(int** resource) {
  if (int status = next_status()) return status;
  *resource = new int(++resource_ids);
  ++live_resources;
  return 0;
}
int destroy(int* resource) { delete resource; --live_resources; return 0; }
int cublasLtCreate(int** resource) { return create(resource); }
int cublasLtDestroy(int* resource) { return destroy(resource); }
template <class... Args>
int cublasLtMatmulDescCreate(int** resource, Args...) { return create(resource); }
int cublasLtMatmulDescDestroy(int* resource) { return destroy(resource); }
template <class... Args>
int cublasLtMatrixLayoutCreate(int** resource, Args...) { return create(resource); }
int cublasLtMatrixLayoutDestroy(int* resource) { return destroy(resource); }
int cublasLtMatmulPreferenceCreate(int** resource) { return create(resource); }
int cublasLtMatmulPreferenceDestroy(int* resource) { return destroy(resource); }
template <class... Args>
int cublasLtMatmulPreferenceSetAttribute(Args...) { return next_status(); }
int cublasLtMatmulAlgoGetHeuristic(int*, int*, int*, int*, int*, int*, int*,
                                 int count, cublasLtMatmulHeuristicResult_t* results,
                                 int* returned) {
  if (int status = next_status()) return status;
  *returned = count;
  for (int i = 0; i < count; ++i) results[i].algo = i;
  return 0;
}
struct Check {
  explicit Check(bool condition) { if (!condition) throw std::runtime_error("vendor failure"); }
  template <class T> Check& operator<<(T const&) { return *this; }
};
#define TVM_FFI_ICHECK(condition) Check(condition)
#define TVM_FFI_ICHECK_EQ(a, b) Check((a) == (b))
#define TVM_FFI_ICHECK_GT(a, b) Check((a) > (b))
"""
    program += cache
    program += r"""
void growth() {
  std::vector<LoraDownCublasLtState*> original;
  for (int i = 0; i < 24; ++i) {
    current_device = i % 2;
    auto& state = lora_down_cublaslt_state(current_device, 64 * (i / 2 + 1), 3072);
    assert(state.initialized && state.device_id == current_device);
    original.push_back(&state);
  }
  int const warmed_calls = api_calls;
  for (int i = 23; i >= 0; --i) {
    current_device = i % 2;
    assert(original[i] == &lora_down_cublaslt_state(current_device, 64 * (i / 2 + 1), 3072));
  }
  assert(api_calls == warmed_calls);
  auto& different_k = lora_down_cublaslt_state(current_device, 64, 6144);
  assert(&different_k != original[0]);
  assert(live_resources == 25 * 6);
}
void retry(int stage) {
  auto& warmed = lora_down_cublaslt_state(0, 64, 3072);
  int const warmed_handle = *warmed.handle;
  int const before = live_resources;
  fail_on_call = api_calls + stage;
  bool failed = false;
  try { lora_down_cublaslt_state(0, 128, 3072); }
  catch (std::runtime_error const&) { failed = true; }
  assert(failed && live_resources == before);
  assert(&warmed == &lora_down_cublaslt_state(0, 64, 3072));
  assert(*warmed.handle == warmed_handle);
  fail_on_call = -1;
  auto& recovered = lora_down_cublaslt_state(0, 128, 3072);
  assert(recovered.initialized && live_resources == before + 6);
  int const warmed_calls = api_calls;
  assert(&recovered == &lora_down_cublaslt_state(0, 128, 3072));
  assert(api_calls == warmed_calls);
}
int main(int argc, char** argv) {
  assert(argc == 2);
  int const stage = std::stoi(argv[1]);
  std::string failure;
  // Thread exit must release all plans; another thread starts with an empty cache.
  for (int repetition = 0; repetition < 2; ++repetition) {
    std::thread worker([&] {
      try { if (stage == 0) growth(); else retry(stage); }
      catch (std::runtime_error const& error) { failure = error.what(); }
    });
    worker.join();
    assert(live_resources == 0);
    if (!failure.empty()) { std::cerr << failure << '\n'; return 1; }
  }
}
"""
    directory = tmp_path_factory.mktemp("cublaslt-plan-cache")
    cpp = directory / "cache.cpp"
    binary = directory / "cache"
    cpp.write_text(program, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-pthread",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize(
    "failure_stage", range(9), ids=("growth", *(f"retry-{i}" for i in range(1, 9)))
)
def test_cublaslt_plan_cache_growth_and_failure_cleanup(
    plan_cache_binary: Path, failure_stage: int
) -> None:
    result = subprocess.run(
        [str(plan_cache_binary), str(failure_stage)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_replay_reuses_native_runner(auto_backend_case, operation):
    from flashinfer import autotune

    # Given a tuned native winner, repeated calls must reuse its loaded runner.
    _, call, profiles, builds, _ = auto_backend_case
    with autotune(True):
        call(operation)
    builds.clear()
    profiles.clear()
    # When the same operation is served from the tuning cache.
    result = call(operation)
    # Then replay executes correctly without rebuilding or profiling candidates.
    assert torch.all(result == 2)
    assert not builds
    assert not profiles


@pytest.mark.parametrize("operation", ["gemm", "linear"])
def test_auto_reused_runners_observe_tuner_cache_clear(auto_backend_case, operation):
    from flashinfer import autotune

    tuner, call, _, builds, _ = auto_backend_case
    with autotune(True):
        assert torch.all(call(operation) == 2)
    builds.clear()
    tuner.clear_cache()
    assert torch.all(call(operation) == 1)
    assert not builds


@pytest.mark.parametrize(
    "invalid",
    [
        "weight_k",
        "weight_stride",
        "sf_size",
        "sf_stride",
        "alpha_empty",
        "l1_rank",
        "l1_stride",
        "bias_size",
        "device",
    ],
)
def test_auto_linear_metadata_checks_reject_invalid_storage(auto_backend_case, invalid):
    _, call, profiles, builds, operands = auto_backend_case
    n, k_packed = operands[1].shape
    match invalid:
        case "weight_k":
            operands[1] = operands[1][:, : k_packed // 2].contiguous()
        case "weight_stride":
            operands[1] = torch.zeros(n, k_packed * 2, dtype=torch.uint8)[:, ::2]
        case "sf_size":
            operands[2] = operands[2][:-1]
        case "sf_stride":
            operands[2] = torch.zeros(operands[2].numel() * 2, dtype=torch.uint8)[::2]
        case "alpha_empty":
            operands[3] = operands[3][:0]
        case "l1_rank":
            operands[6] = operands[6][:, :16].contiguous()
        case "l1_stride":
            operands[6] = torch.ones(n, 64, dtype=torch.bfloat16)[:, ::2]
        case "bias_size":
            operands[8] = torch.ones(n + 1, dtype=torch.bfloat16)
        case "device":
            operands[3] = torch.ones(1, device="meta")
    with pytest.raises(ValueError):
        call("linear")
    assert not profiles and not builds


def test_auto_linear_allocates_scratch_only_for_execution(
    auto_backend_case, monkeypatch
):
    from flashinfer import autotune

    # Given a cached native linear winner and an allocator observer.
    _, call, _, _, operands = auto_backend_case
    with autotune(True):
        call("linear")
    allocations: list[tuple[tuple[int, ...], torch.dtype]] = []
    empty = torch.empty

    def record_empty(*args, **kwargs):
        result = empty(*args, **kwargs)
        allocations.append((tuple(result.shape), result.dtype))
        return result

    monkeypatch.setattr(torch, "empty", record_empty)
    # When serving a cached winner, metadata checks must not allocate scratch.
    result = call("linear")
    # Then each intermediate is allocated exactly once for execution.
    m, k = operands[0].shape
    rank = operands[5].shape[1]
    assert torch.all(result == 2)
    for allocation in (
        ((m, k // 2), torch.uint8),
        ((gemm_svdquant._swizzled_sf_size(m, k // 16),), torch.uint8),
        ((m, rank), torch.bfloat16),
    ):
        assert allocations.count(allocation) == 1, (allocation, allocations)
