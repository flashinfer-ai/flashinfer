"""Host-side contracts for SM120 full-linear runner selection."""

import math

import torch
from types import SimpleNamespace

import pytest

from flashinfer.gemm import gemm_svdquant

from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.gemm import svdquant_sm120_routes as _sm120_routes
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig


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
