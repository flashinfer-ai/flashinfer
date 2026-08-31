import ast
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch

from flashinfer import (
    SfLayout,
    autotune,
    mm_mxfp8_dynamic_quant,
    mxfp8_quantize,
    prepare_mxfp8_trtllm_weights,
    shuffle_matrix_a,
)
from flashinfer.autotuner import (
    AutoTuner,
    OptimizationProfile,
    StaticDim,
    TunableRunner,
    TuningConfig,
)
from flashinfer.gemm import gemm_base
from flashinfer.trace.templates.gemm import _unshuffle_trtllm_mxfp8_rows
from flashinfer.utils import BackendSupportedError, get_compute_capability


_MIN_COSINE_SIMILARITY = 0.98


def _skip_if_trtllm_dynamic_quant_unsupported() -> None:
    if not torch.cuda.is_available():
        pytest.skip("TRTLLM MXFP8 dynamic quantization requires CUDA")

    capability = get_compute_capability(torch.device("cuda"))
    if capability not in {(10, 0), (10, 3), (10, 7)}:
        pytest.skip("TRTLLM MXFP8 dynamic quantization requires SM100, SM103, or SM107")


@pytest.fixture(autouse=True)
def _isolate_autotuner() -> Generator[None, None, None]:
    AutoTuner._instance = None
    try:
        yield
    finally:
        AutoTuner._instance = None


@pytest.fixture
def blackwell_cuda() -> None:
    _skip_if_trtllm_dynamic_quant_unsupported()


def test_trtllm_dynamic_quant_buckets_match_lookup_mapping() -> None:
    buckets = gemm_base._get_trtllm_mxfp8_tuning_buckets(33)
    assert 33 not in buckets
    assert 64 in buckets
    assert gemm_base._map_to_trtllm_mxfp8_tuning_bucket(3) == 3
    assert gemm_base._map_to_trtllm_mxfp8_tuning_bucket(32) == 32
    assert gemm_base._map_to_trtllm_mxfp8_tuning_bucket(33) == 64
    dynamic_spec = gemm_base._MM_MXFP8_DYNAMIC_QUANT_TUNING_CONFIG.dynamic_tensor_specs[
        0
    ]
    assert dynamic_spec.map_to_tuning_buckets(3) == 3


def test_trtllm_dynamic_quant_profiles_like_runtime() -> None:
    tuning_config = gemm_base._MM_MXFP8_DYNAMIC_QUANT_TUNING_CONFIG
    assert tuning_config.use_cuda_graph
    assert tuning_config.use_cold_l2_cache


def test_trtllm_dynamic_quant_buckets_keep_low_m_exact() -> None:
    assert gemm_base._get_trtllm_mxfp8_tuning_buckets(64) == (
        *range(1, 33),
        64,
    )


def test_mm_mxfp8_dynamic_quant_exposes_backend_capabilities() -> None:
    assert mm_mxfp8_dynamic_quant.is_backend_supported("trtllm", 100)
    assert mm_mxfp8_dynamic_quant.is_backend_supported("trtllm", 107)
    assert not mm_mxfp8_dynamic_quant.is_backend_supported("trtllm", 90)
    assert not mm_mxfp8_dynamic_quant.is_backend_supported("cute-dsl", 100)
    assert mm_mxfp8_dynamic_quant.is_compute_capability_supported(100)
    assert not mm_mxfp8_dynamic_quant.is_compute_capability_supported(90)


def test_mm_mxfp8_dynamic_quant_exposes_trace() -> None:
    assert callable(mm_mxfp8_dynamic_quant.fi_trace)


def test_dynamic_quant_runner_selects_module_for_tensor_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_devices: list[torch.device] = []

    class FakeModule:
        def trtllm_mxfp8_gemm_runner(self, use_8x4_sf_layout: bool) -> object:
            return object()

    def get_module(device: torch.device) -> FakeModule:
        requested_devices.append(device)
        return FakeModule()

    monkeypatch.setattr(gemm_base, "get_trtllm_gemm_module", get_module)
    monkeypatch.setattr(gemm_base, "get_compute_capability", lambda device: (10, 3))

    runner = gemm_base._TrtllmDynamicQuantMxfp8Runner(torch.device("cuda:1"))

    assert requested_devices == [torch.device("cuda:1")]
    assert runner.get_cache_key_extras([]) == ((10, 3),)


def test_dynamic_quant_trace_unshuffles_trtllm_rows() -> None:
    original = torch.arange(32 * 4).reshape(32, 4)
    shuffled = shuffle_matrix_a(original, 128)
    torch.testing.assert_close(_unshuffle_trtllm_mxfp8_rows(shuffled), original)


def _prepare_trtllm_weight(
    n: int, k: int, device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_device = torch.device(device)
    with torch.cuda.device(target_device):
        weight = torch.randn((n, k), device=target_device, dtype=torch.bfloat16)
        weight_q, weight_sf = mxfp8_quantize(
            weight,
            sf_swizzle_layout=SfLayout.layout_linear,
        )
    weight_q, weight_sf = prepare_mxfp8_trtllm_weights(weight_q, weight_sf)
    return weight, weight_q, weight_sf


def test_prepare_mxfp8_trtllm_weights_rejects_cpu_tensors() -> None:
    weight = torch.empty((128, 256), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty((128 * 8,), dtype=torch.uint8)

    with pytest.raises(ValueError, match="must be CUDA tensors"):
        prepare_mxfp8_trtllm_weights(weight, weight_scale)


def test_prepare_mxfp8_trtllm_weights_pads_non_128_aligned_n(
    blackwell_cuda: None,
) -> None:
    n, k = 160, 256
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    weight_q, weight_sf = mxfp8_quantize(
        weight,
        sf_swizzle_layout=SfLayout.layout_linear,
    )

    b, b_sf = prepare_mxfp8_trtllm_weights(weight_q, weight_sf)

    assert b.shape == (k, n)
    assert b.stride() == (1, k)
    assert b_sf.shape == (256 * (k // 32),)
    a = torch.randn((3, k), device="cuda", dtype=torch.bfloat16)
    actual = mm_mxfp8_dynamic_quant(a, b, b_sf)
    assert _cosine_similarity(a @ weight.T, actual) > _MIN_COSINE_SIMILARITY


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(),
        b.float().flatten(),
        dim=0,
    ).item()


def test_mm_mxfp8_dynamic_quant_rejects_non_bf16_activation(
    blackwell_cuda: None,
) -> None:
    a = torch.empty((4, 4096), device="cuda", dtype=torch.float16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    with pytest.raises(ValueError, match="a must be a bfloat16 tensor"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_unsupported_backend(
    blackwell_cuda: None,
) -> None:
    a = torch.empty((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    with pytest.raises(
        BackendSupportedError,
        match="does not support backend 'cutlass'",
    ):
        mm_mxfp8_dynamic_quant(a, b, b_sf, backend="cutlass")


def test_mm_mxfp8_dynamic_quant_rejects_n_below_128(
    blackwell_cuda: None,
) -> None:
    a = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((256, 64), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="N >= 128"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_unshufflable_n(
    blackwell_cuda: None,
) -> None:
    a = torch.empty((4, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((256, 129), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="N divisible by 32"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_zero_k(blackwell_cuda: None) -> None:
    a = torch.empty((4, 0), device="cuda", dtype=torch.bfloat16)
    b = torch.empty((0, 128), device="cuda", dtype=torch.float8_e4m3fn)
    b_sf = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="K must be positive"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_zero_m(blackwell_cuda: None) -> None:
    a = torch.empty((0, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    with pytest.raises(ValueError, match="M must be positive"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


def test_mm_mxfp8_dynamic_quant_rejects_noncontiguous_activation(
    blackwell_cuda: None,
) -> None:
    a = torch.empty((4096, 4), device="cuda", dtype=torch.bfloat16).T
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    with pytest.raises(ValueError, match="a must be contiguous"):
        mm_mxfp8_dynamic_quant(a, b, b_sf)


@pytest.mark.parametrize(
    ("m", "auto_tuning"),
    [(3, False), (3, True), (33, True)],
)
def test_mm_mxfp8_dynamic_quant_matches_bf16(
    m: int,
    auto_tuning: bool,
    blackwell_cuda: None,
) -> None:
    torch.manual_seed(0)
    a = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16)
    weight, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    reference = a @ weight.T

    with autotune(auto_tuning):
        actual = mm_mxfp8_dynamic_quant(a, b, b_sf)

    assert _cosine_similarity(reference, actual) > _MIN_COSINE_SIMILARITY


class _RecordingTuner:
    def __init__(self, selected_use_8x4: bool) -> None:
        self.selected_use_8x4 = selected_use_8x4
        self.extras: list[tuple[Any, ...]] = []
        self.tactics: list[list[Any]] = []

    def choose_one(
        self,
        custom_op: str,
        runners: list[TunableRunner],
        tuning_config: TuningConfig,
        inputs: list[torch.Tensor],
    ) -> tuple[TunableRunner, Any]:
        profile = OptimizationProfile(
            shapes=[
                [StaticDim(dim) for dim in value.shape]
                if isinstance(value, torch.Tensor)
                else []
                for value in inputs
            ],
            tensor_initializers=[None] * len(inputs),
        )
        self.extras = [runner.get_cache_key_extras(inputs) for runner in runners]
        self.tactics = [runner.get_valid_tactics(inputs, profile) for runner in runners]
        selected_tactic = next(
            tactic for tactic in self.tactics[0] if tactic[0] == self.selected_use_8x4
        )
        return runners[0], selected_tactic


@pytest.mark.parametrize(
    "m, selected_use_8x4, selected_layout",
    [
        (4, True, SfLayout.layout_8x4),
        (33, False, SfLayout.layout_128x4),
    ],
)
def test_mm_mxfp8_dynamic_quant_offers_both_layouts(
    m: int,
    selected_use_8x4: bool,
    selected_layout: SfLayout,
    monkeypatch: pytest.MonkeyPatch,
    blackwell_cuda: None,
) -> None:
    recorder = _RecordingTuner(selected_use_8x4)
    monkeypatch.setattr(AutoTuner, "get", classmethod(lambda cls: recorder))

    a = torch.randn((m, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    quantized_layouts: list[SfLayout] = []
    real_quantize = gemm_base.mxfp8_quantize

    def recording_quantize(
        tensor: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized_layouts.append(kwargs["sf_swizzle_layout"])
        return real_quantize(tensor, **kwargs)

    monkeypatch.setattr(gemm_base, "mxfp8_quantize", recording_quantize)

    mm_mxfp8_dynamic_quant(a, b, b_sf)

    assert recorder.extras == [(get_compute_capability(a.device),)]
    assert len(recorder.tactics) == 1
    assert {use_8x4 for use_8x4, _ in recorder.tactics[0]} == {True, False}
    assert quantized_layouts == [selected_layout]


def test_mm_mxfp8_dynamic_quant_cache_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    blackwell_cuda: None,
) -> None:
    cache_path = tmp_path / "dynamic_quant.json"
    torch.manual_seed(0)
    a = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)

    with autotune(True, cache=str(cache_path)):
        tuned = mm_mxfp8_dynamic_quant(a, b, b_sf)

    AutoTuner._instance = None
    cache_hits: list[bool] = []
    real_search_cache = AutoTuner.search_cache

    def recording_search_cache(
        self: AutoTuner,
        custom_op: str,
        runners: list[TunableRunner],
        input_shapes: tuple[tuple[int, ...], ...],
        tuning_config: TuningConfig,
        inputs: list[torch.Tensor] | None = None,
    ) -> tuple[bool, int, Any, OptimizationProfile | None]:
        result = real_search_cache(
            self,
            custom_op,
            runners,
            input_shapes,
            tuning_config,
            inputs=inputs,
        )
        if custom_op == "mxfp8_dynamic_quant_gemm":
            cache_hits.append(result[0])
        return result

    monkeypatch.setattr(AutoTuner, "search_cache", recording_search_cache)
    with autotune(False, cache=str(cache_path)):
        cached = mm_mxfp8_dynamic_quant(a, b, b_sf)

    payload = json.loads(cache_path.read_text())
    dynamic_keys = [key for key in payload if "mxfp8_dynamic_quant_gemm" in key]
    parsed_keys = [ast.literal_eval(key) for key in dynamic_keys]
    assert len(parsed_keys) == 4
    assert {key[2][0][0] for key in parsed_keys} == {1, 2, 3, 4}
    assert all(key[3] == (get_compute_capability(a.device),) for key in parsed_keys)
    assert cache_hits == [True]
    torch.testing.assert_close(cached, tuned, rtol=0, atol=0)


def test_mm_mxfp8_dynamic_quant_uses_tensor_device_context(
    blackwell_cuda: None,
) -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")

    original_device = torch.cuda.current_device()
    target_device = torch.device("cuda:1" if original_device == 0 else "cuda:0")
    a = torch.randn((3, 4096), device=target_device, dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096, target_device)

    actual = mm_mxfp8_dynamic_quant(a, b, b_sf)
    torch.cuda.synchronize(target_device)

    assert actual.device == target_device
    assert torch.cuda.current_device() == original_device


def test_mm_mxfp8_dynamic_quant_cuda_graph_replay(blackwell_cuda: None) -> None:
    torch.manual_seed(0)
    static_a = torch.randn((4, 4096), device="cuda", dtype=torch.bfloat16)
    _, b, b_sf = _prepare_trtllm_weight(2688, 4096)
    static_out = torch.empty((4, 2688), device="cuda", dtype=torch.bfloat16)

    with autotune(True):
        mm_mxfp8_dynamic_quant(static_a, b, b_sf, out=static_out)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = mm_mxfp8_dynamic_quant(static_a, b, b_sf, out=static_out)

    for seed in (1, 2):
        torch.manual_seed(seed)
        next_a = torch.randn_like(static_a)
        static_a.copy_(next_a)
        graph.replay()
        replayed = graph_out.clone()
        eager = mm_mxfp8_dynamic_quant(next_a, b, b_sf)
        assert _cosine_similarity(replayed, eager) > _MIN_COSINE_SIMILARITY
