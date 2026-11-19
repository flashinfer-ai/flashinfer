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

Tests of the row-split Gated DeltaProduct decode / MTP-verification kernel
(:mod:`flashinfer.gdp_decode_rowsplit`) and of its routing from
:func:`flashinfer.cake_gdp.cake_gated_delta_product_mtp`.

The kernel claims the bits of the Cake program a routed call would otherwise
launch, so every comparison with Cake (``flashinfer.cake_gdp_backend``, which
the facade calls for every call it does not route) is ``torch.equal`` on the
output and on the whole state pool: the slots the call writes, the slots it
must leave alone and the padding rows' outputs.  Covered: every routed
``(num_householder, T)`` at small ``B x HV`` and at the band edge, per-token
state scatter with holes, writeback to distinct write slots with a padding row,
``disable_state_update``, a non-default scale, a pre-filled output, every CTA
tile, CUDA graph replay with changing inputs, the escape hatch and the inputs
that must stay on the Cake path.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer.cake_gdp_backend as backend
from flashinfer import gdp_decode_rowsplit as rs
from flashinfer.cake_gdp import cake_gated_delta_product_mtp, is_cake_gdp_supported
from flashinfer.utils import get_compute_capability

K = 128
V = 64
SENTINEL = 7.5
ROUTED = sorted(rs.ROUTED_BAND)
# The routed (num_householder, T) of the regular run: both lane widths, both Cake
# program families (static and runtime T), the band's T extremes and every n.  The
# other routed entries run with --full (each entry builds its own module).
REGULAR = {(3, 2), (3, 4), (3, 6), (3, 8), (3, 16), (3, 9), (3, 15), (1, 1), (1, 4)}
REGULAR |= {(2, 3), (4, 2), (4, 16)}


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and get_compute_capability(_device()) in ((10, 0), (10, 3))
        and is_cake_gdp_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="the row-split kernel matches the Cake GDP programs of SM100 / SM103",
)


def _routes_here() -> bool:
    return get_compute_capability(_device()) in rs.ROUTED_COMPUTE_CAPABILITIES


def _sm_count() -> int:
    return torch.cuda.get_device_properties(_device()).multi_processor_count


def _inputs(
    n: int,
    T: int,
    H: int,
    B: int,
    *,
    mode: str,
    seed: int,
    prefill_output: bool = False,
) -> dict:
    device = _device()
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device)

    # Read slots, written slots and a few untouched slots are all distinct: no batch
    # row reads a slot another row writes.
    pool = B * (T + 1) + 9
    perm = torch.randperm(pool, generator=gen, device=device).to(torch.int32)
    slots = perm[: B * T].view(B, T).clone()
    inputs = dict(
        q=randn(B, T, H, K).bfloat16(),
        k=randn(B, T, n, H, K).bfloat16(),
        v=randn(B, T, n, H, V).bfloat16(),
        initial_state=randn(pool, H, V, K).mul_(0.05),
        A_log=torch.rand(H, generator=gen, device=device).mul_(2.7),
        a=randn(B, T, H).mul_(2.0).bfloat16(),
        dt_bias=randn(H).mul_(0.5),
        b=randn(B, T, n, H).bfloat16(),
    )
    if mode == "scatter":
        if T > 1:
            slots[0, 0] = -1  # a hole: that token's state is not stored
        inputs.update(
            initial_state_indices=perm[B * T : B * T + B].clone(),
            ssm_state_indices=slots,
        )
    else:
        read = slots[:, 0].clone()
        if B > 1:
            read[B // 2] = -1  # a padding row: skipped entirely
        inputs.update(
            initial_state_indices=read,
            output_state_indices=slots[:, -1].contiguous(),
        )
    if prefill_output:
        inputs["output"] = torch.full(
            (B, T, H, V), SENTINEL, dtype=torch.bfloat16, device=device
        )
    return inputs


def _clone(inputs: dict) -> dict:
    return {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in inputs.items()
    }


def _assert_same_as_cake(fn, inputs: dict, **kwargs) -> None:
    ref = _clone(inputs)
    new = _clone(inputs)
    out_ref, pool_ref = backend.cake_gated_delta_product_mtp(**ref, **kwargs)
    out_new, pool_new = fn(**new, **kwargs)
    assert pool_new is new["initial_state"]
    assert out_new.dtype == out_ref.dtype and out_new.shape == out_ref.shape
    assert torch.equal(out_new, out_ref)
    assert torch.equal(pool_new, pool_ref)


def _route(inputs: dict, **kwargs):
    names = (
        "output",
        "ssm_state_indices",
        "disable_state_update",
        "use_qk_l2norm",
        "output_state_indices",
    )
    options = {name: inputs.get(name, kwargs.get(name)) for name in names}
    for name in names:
        if name in kwargs:
            options[name] = kwargs[name]
    if options["use_qk_l2norm"] is None:
        options["use_qk_l2norm"] = True
    return rs.route(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["initial_state"],
        inputs["initial_state_indices"],
        inputs["A_log"],
        inputs["a"],
        inputs["dt_bias"],
        inputs["b"],
        **options,
    )


def _skip_unless_selected(request: pytest.FixtureRequest, n: int, T: int) -> None:
    if (n, T) not in REGULAR and not request.config.getoption("--full", default=False):
        pytest.skip("routed entry outside the regular subset; run with --full")


def _shapes(n: int, T: int) -> list[tuple[int, int]]:
    """(H, B) pairs inside the band: tiny calls and the band edge at H = 10."""

    bound = int(rs.ROUTED_BAND[(n, T)][1] * _sm_count())
    shapes = [(1, 1), (10, 1), (10, 3), (40, 2)]
    if bound // 10 >= 1:
        shapes.append((10, bound // 10))
    return [(H, B) for H, B in shapes if bound >= H * B]


@pytest.mark.parametrize("n,T", ROUTED)
@pytest.mark.parametrize("mode", ["scatter", "writeback"])
def test_routed_calls_match_cake(
    request: pytest.FixtureRequest, n: int, T: int, mode: str
) -> None:
    _skip_unless_selected(request, n, T)
    for H, B in _shapes(n, T):
        inputs = _inputs(n, T, H, B, mode=mode, seed=1000 * n + 10 * T + H + B)
        if _routes_here():
            assert _route(inputs) == (
                rs.ROUTED_BAND[(n, T)][0],
                *rs.default_tile(n, T, rs.ROUTED_BAND[(n, T)][0], B * H, _sm_count()),
            )
        _assert_same_as_cake(cake_gated_delta_product_mtp, inputs)
        _assert_same_as_cake(rs.gdp_decode_rowsplit_mtp, inputs)


@pytest.mark.parametrize("n,T", ROUTED)
def test_routed_options_match_cake(
    request: pytest.FixtureRequest, n: int, T: int
) -> None:
    """disable_state_update, a non-default scale and a pre-filled output (padding rows keep it)."""
    _skip_unless_selected(request, n, T)

    inputs = _inputs(n, T, 10, 3, mode="writeback", seed=77 + T, prefill_output=True)
    _assert_same_as_cake(
        cake_gated_delta_product_mtp, inputs, disable_state_update=True
    )
    _assert_same_as_cake(cake_gated_delta_product_mtp, inputs, scale=0.375)
    out, _ = cake_gated_delta_product_mtp(**_clone(inputs))
    assert torch.all(out[1] == SENTINEL)


@pytest.mark.parametrize("n,T", ROUTED)
def test_band_edge_routing(request: pytest.FixtureRequest, n: int, T: int) -> None:
    _skip_unless_selected(request, n, T)
    if not _routes_here():
        pytest.skip("calls are routed on SM103 only")
    bound = int(rs.ROUTED_BAND[(n, T)][1] * _sm_count())
    lanes = rs.ROUTED_BAND[(n, T)][0]
    B = bound // 10 + 1  # first batch above the band at H = 10
    inputs = _inputs(n, T, 10, B, mode="scatter", seed=5)
    assert _route(inputs) is None
    _assert_same_as_cake(cake_gated_delta_product_mtp, inputs)
    if bound // 10 >= 1:
        inside = _inputs(n, T, 10, bound // 10, mode="scatter", seed=6)
        assert _route(inside) is not None and _route(inside)[0] == lanes


@pytest.mark.parametrize("n,T", [(3, 4), (3, 6), (1, 2), (4, 3)])
def test_every_tile_gives_the_same_bits(n: int, T: int) -> None:
    """The CTA tile changes which CTA owns a state row, not the row's arithmetic."""

    inputs = _inputs(n, T, 6, 5, mode="scatter", seed=11 * T + n)
    lanes = rs.cake_lanes(_device(), n, T, 30)
    assert lanes in (4, 8)
    for rows_per_cta, rows_per_lane in rs.TILES[lanes]:
        if not rs.tile_supported(n, T, lanes, rows_per_cta, rows_per_lane):
            continue
        _assert_same_as_cake(
            lambda **kw: rs.gdp_decode_rowsplit_mtp(
                **kw, rows_per_cta=rows_per_cta, rows_per_lane=rows_per_lane
            ),
            inputs,
        )


def _graph_case() -> tuple[int, int]:
    return ROUTED[0] if ROUTED else (3, 4)


def test_cuda_graph_replay_with_changing_inputs() -> None:
    n, T = _graph_case()
    H, B = 10, 2
    static = _inputs(n, T, H, B, mode="scatter", seed=21)
    pool0 = static["initial_state"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cake_gated_delta_product_mtp(**static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static, _ = cake_gated_delta_product_mtp(**static)
    for seed in (22, 23):
        fresh = _inputs(n, T, H, B, mode="scatter", seed=seed)
        for name, value in fresh.items():
            if name != "initial_state":
                static[name].copy_(value)
        static["initial_state"].copy_(pool0)
        graph.replay()
        fresh["initial_state"] = pool0.clone()
        out_ref, pool_ref = backend.cake_gated_delta_product_mtp(**fresh)
        torch.cuda.synchronize()
        assert torch.equal(out_static, out_ref)
        assert torch.equal(static["initial_state"], pool_ref)


def test_calls_that_stay_on_cake(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _routes_here() or not ROUTED:
        pytest.skip("calls are routed on SM103 only")
    n, T = ROUTED[0]
    inputs = _inputs(n, T, 4, 2, mode="scatter", seed=31)
    assert _route(inputs) is not None
    monkeypatch.setenv(rs.DISABLE_ENV, "1")
    assert _route(inputs) is None
    monkeypatch.delenv(rs.DISABLE_ENV)
    assert _route(inputs, use_qk_l2norm=False) is None
    fp16 = {
        **inputs,
        **{name: inputs[name].half() for name in ("q", "k", "v", "a", "b")},
    }
    assert _route(fp16) is None
    assert _route({**inputs, "dt_bias": inputs["dt_bias"].bfloat16()}) is None
    assert (
        _route(
            {**inputs, "initial_state_indices": inputs["initial_state_indices"].long()}
        )
        is None
    )
    B = inputs["q"].shape[0]
    wide = torch.zeros(B, T, 4, K + 2, dtype=torch.bfloat16, device=_device())
    assert _route({**inputs, "q": wide[..., 2:]}) is None  # 4-byte aligned view
    assert (
        _route({**inputs, "output": torch.zeros(B, T, 4, V, device=_device())}) is None
    )
    # The Cake path still serves them, with Cake's own results.
    _assert_same_as_cake(cake_gated_delta_product_mtp, fp16)


def test_direct_entry_rejects_other_calls() -> None:
    inputs = _inputs(3, 4, 2, 2, mode="scatter", seed=41)
    with pytest.raises(ValueError, match="use_qk_l2norm"):
        rs.gdp_decode_rowsplit_mtp(**_clone(inputs), use_qk_l2norm=False)
    with pytest.raises(ValueError, match="bfloat16"):
        rs.gdp_decode_rowsplit_mtp(**{**_clone(inputs), "q": inputs["q"].half()})
    with pytest.raises(ValueError, match="disable_state_update"):
        rs.gdp_decode_rowsplit_mtp(**_clone(inputs), disable_state_update=True)
    with pytest.raises(ValueError, match="tile"):
        rs.gdp_decode_rowsplit_mtp(**_clone(inputs), rows_per_cta=24, rows_per_lane=1)
