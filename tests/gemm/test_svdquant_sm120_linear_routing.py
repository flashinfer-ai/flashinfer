"""Host-side contracts for SM120 full-linear runner selection."""

import math

import torch
from types import SimpleNamespace

import pytest

from flashinfer.gemm import gemm_svdquant

from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.gemm import svdquant_sm120_routes as _sm120_routes


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
        gemm_svdquant,
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

    x = _FakeTensor((537, 5376), dtype=torch.bfloat16)
    weight = _FakeTensor((5376, 2688), dtype=torch.uint8)
    raw_l2t = _FakeTensor((5376, 32), dtype=torch.bfloat16)

    out = gemm_svdquant.svdquant_linear(
        x,
        weight,
        "weight_sf",
        "alpha",
        # A marker string everywhere it is only carried through, but the entry
        # point reads pre_quant_scale's dtype on its way to a route.
        _FakeTensor((5376,), dtype=torch.bfloat16),
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
