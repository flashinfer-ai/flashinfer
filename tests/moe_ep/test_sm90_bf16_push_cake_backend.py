"""Correctness gates for ``sm90_bf16_bf16_bf16_push_cake`` through ``MoEEpLayer``.

Native BF16 MegaMoE on Hopper: bf16 dispatch payload, Cake-generated WGMMA FC1
(fused SwiGLU) and FC2 with fp32 accumulation and a bf16 intermediate, bf16
combine wire, bf16 output.  Every output is compared elementwise against the
independent torch reference in ``_sm90_bf16_reference`` at
``atol = rtol = 1e-2`` (never loosened); the all-fp32 reference statistics are
reported alongside.

Single-GPU tests run under plain pytest; the EP2+ tests need
``torchrun --nproc_per_node=N -m pytest tests/moe_ep/test_sm90_bf16_push_cake_backend.py``.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from ._sm90_bf16_reference import compare_bf16, reference_moe_bf16, reference_moe_fp32


def _sm90_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least
        from flashinfer.utils import is_sm90a_supported

        return is_cuda_version_at_least("12.8") and is_sm90a_supported(
            torch.device("cuda")
        )
    except Exception:
        return False


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))

requires_sm90 = pytest.mark.skipif(
    not _sm90_available() or _WORLD > 1,
    reason="requires one SM90 GPU and CUDA Toolkit 12.8+ outside torchrun",
)
requires_dist = pytest.mark.skipif(
    _WORLD < 2 or not _sm90_available(),
    reason="requires torchrun with at least two SM90 GPUs and CUDA Toolkit 12.8+",
)

HIDDEN = 512
INTERMEDIATE = 768
LOCAL_EXPERTS = 4
TOP_K = 2
TOKEN_CAPACITY = 64
ATOL = 1e-2
RTOL = 1e-2
# combine wire formats of the backend (see cake_config.py): the pre-reduced wire is
# the default, the per-route wire is the round-1 format kept for A/B comparison.
WIRES = ("prereduced", "prereduced_hilo", "per_route")
# optional JSON-lines sink of every _check() statistic (precision table input)
PRECISION_LOG_ENV = "SM90_BF16_PUSH_CAKE_PRECISION_LOG"

_KEEP_ALIVE: list[object] = []


def _make_weights(
    num_experts: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (
        torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, generator=generator)
        * HIDDEN**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    w2 = (
        torch.randn(num_experts, HIDDEN, INTERMEDIATE, generator=generator)
        * INTERMEDIATE**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    return w13, w2


def _make_inputs(
    num_tokens: int,
    num_experts: int,
    seed: int,
    device: torch.device,
    *,
    mode: str = "random",
    rank: int = 0,
    world_size: int = 1,
    top_k: int = TOP_K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routing patterns: random | hot | all_remote | dup_rank | dup_remote | all_local | masked | empty_rank."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, num_experts, generator=generator)
    local_start = rank * LOCAL_EXPERTS
    if mode == "hot":
        ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    elif mode == "all_remote" and num_experts > LOCAL_EXPERTS:
        logits[:, local_start : local_start + LOCAL_EXPERTS] = float("-inf")
        ids = logits.topk(top_k, dim=1).indices.to(torch.int32)
    elif mode in ("dup_rank", "dup_remote", "all_local"):
        # every route of a token lands on ONE rank (dedup dispatch; the pre-reduced
        # combine wire folds all of them into a single row): any rank | a remote
        # rank only | this rank only
        if mode == "all_local" or world_size == 1:
            owner = torch.full((num_tokens,), rank, dtype=torch.int64)
        elif mode == "dup_remote":
            owner = torch.randint(0, world_size - 1, (num_tokens,), generator=generator)
            owner = owner + (owner >= rank).to(torch.int64)  # skip this rank
        else:
            owner = torch.randint(0, world_size, (num_tokens,), generator=generator)
        base = owner * LOCAL_EXPERTS
        offs = (
            torch.stack(
                [
                    torch.randperm(LOCAL_EXPERTS, generator=generator)[:top_k]
                    for _ in range(num_tokens)
                ]
            )
            if num_tokens
            else torch.zeros(0, top_k, dtype=torch.int64)
        )
        ids = (base.unsqueeze(1) + offs).to(torch.int32)
    elif mode == "empty_rank" and world_size > 1:
        # no token anywhere routes to the last rank's experts
        logits[:, (world_size - 1) * LOCAL_EXPERTS :] = float("-inf")
        ids = logits.topk(top_k, dim=1).indices.to(torch.int32)
    else:
        ids = logits.topk(top_k, dim=1).indices.to(torch.int32)
    if mode == "masked" and num_tokens:
        ids[::3, 0] = -1
    weights = torch.rand(num_tokens, top_k, generator=generator) + 0.1
    weights = weights / weights.sum(dim=1, keepdim=True)
    return x, ids.to(device), weights.to(device=device, dtype=torch.float32)


def _build_layer(
    world_size: int,
    rank: int,
    device: torch.device,
    *,
    dedup_dispatch: bool = True,
    capacity_factor: float = 1.0,
    clamp_limit: float | None = None,
    combine_wire: str | None = None,
    token_capacity: int = TOKEN_CAPACITY,
    seed: int = 7,
    compact_form: int | None = None,
    publish_form: int | None = None,
    top_k: int = TOP_K,
    **dispatch_kwargs,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig,
    )

    total_experts = LOCAL_EXPERTS * world_size
    w13, w2 = _make_weights(total_experts, seed, device)
    local_start = rank * LOCAL_EXPERTS
    local_end = local_start + LOCAL_EXPERTS
    process_group = None
    if world_size > 1:
        import torch.distributed as dist

        process_group = dist.group.WORLD
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size, rank=rank, process_group=process_group
        ),
        fleet_params=FleetParams(
            num_experts=total_experts,
            max_tokens_per_rank=token_capacity,
            token_hidden_size=HIDDEN,
        ),
        weights=MoEWeightPack(
            w13=w13[local_start:local_end].contiguous(),
            w2=w2[local_start:local_end].contiguous(),
        ),
        backend=MegaConfig(
            megakernel=Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=top_k,
                capacity_factor=capacity_factor,
                dedup_dispatch=dedup_dispatch,
                clamp_limit=clamp_limit,
                combine_wire=combine_wire,
                compact_form=compact_form,
                publish_form=publish_form,
                **dispatch_kwargs,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    _KEEP_ALIVE.append(layer)
    return layer, w13, w2


def _forward(
    layer, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    from flashinfer.moe_ep import MoEEpTensors

    return layer(
        MoEEpTensors(hidden_states=x, topk_ids=topk_ids, topk_weights=topk_weights)
    )


def _check(
    output: torch.Tensor,
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    clamp: float | None = None,
    label: str = "",
) -> dict[str, float]:
    assert output.shape == x.shape, (output.shape, x.shape)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all(), f"{label}: non-finite output"
    ref_bf16 = reference_moe_bf16(x, ids, weights, w13, w2, clamp=clamp)
    ref_fp32 = reference_moe_fp32(x, ids, weights, w13, w2, clamp=clamp)
    stats = compare_bf16(output, ref_bf16, atol=ATOL, rtol=RTOL)
    stats_fp32 = compare_bf16(output, ref_fp32, atol=ATOL, rtol=RTOL)
    print(
        f"[sm90_bf16_push_cake]{label} vs bf16-contract ref: {stats}; vs fp32 ref: {stats_fp32}",
        flush=True,
    )
    log_path = os.environ.get(PRECISION_LOG_ENV, "").strip()
    if log_path:
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "label": label.strip(),
                        "num_tokens": int(x.shape[0]),
                        "vs_bf16_ref": stats,
                        "vs_fp32_ref": stats_fp32,
                    }
                )
                + "\n"
            )
    assert stats["mismatches"] == 0, f"{label}: {stats} (atol={ATOL}, rtol={RTOL})"
    return stats


# --------------------------------------------------------------------------- EP1
@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("dedup_dispatch", [True, False])
def test_ep1_forward_repeated_and_deterministic(
    dedup_dispatch: bool, wire: str
) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1, 0, device, dedup_dispatch=dedup_dispatch, combine_wire=wire
    )
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 11, device)
    outputs = []
    for _ in range(3):
        outputs.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()
    _check(
        outputs[0],
        x,
        ids,
        weights,
        w13,
        w2,
        label=f" ep1 {wire} dedup={dedup_dispatch}",
    )
    # bitwise run-to-run determinism (recorded; the kernels have a fixed reduction order)
    for repeat in outputs[1:]:
        assert torch.equal(repeat, outputs[0])


# Token counts that take each path of the latency-pipelined compact on a 132-SM part
# (grid 4 x SMs = 528 blocks; m = TOP_K x tokens rows): one row per block, four rows
# per block, one warp per row.
COPY_FORM_TOKENS = (7, TOKEN_CAPACITY, 700, 2048)


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("mode", ["random", "masked", "hot", "dup_rank"])
def test_ep1_copy_forms_bitwise(mode: str, wire: str) -> None:
    """compact_form / publish_form 1 write the same bytes as the forms 0."""
    device = torch.device("cuda", 0)
    capacity = max(COPY_FORM_TOKENS)
    layers = {
        forms: _build_layer(
            1,
            0,
            device,
            combine_wire=wire,
            token_capacity=capacity,
            compact_form=forms[0],
            publish_form=forms[1],
        )
        for forms in ((0, 0), (1, 1), (1, 0), (0, 1))
    }
    _, w13, w2 = layers[(0, 0)]
    for num_tokens in COPY_FORM_TOKENS:
        x, ids, weights = _make_inputs(
            num_tokens, LOCAL_EXPERTS, 31 + num_tokens, device, mode=mode
        )
        outputs = {}
        for forms, (layer, _w13, _w2) in layers.items():
            outputs[forms] = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
        _check(
            outputs[(1, 1)],
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep1 {wire} {mode} T={num_tokens} forms=1,1",
        )
        for forms, output in outputs.items():
            assert torch.equal(output, outputs[(0, 0)]), (
                f"forms {forms} differ at T={num_tokens}"
            )


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
def test_ep1_per_route_publish_pdl(wire: str) -> None:
    """per_route_publish_pdl adds the publish edge on the per-route wire only (same bytes)."""
    device = torch.device("cuda", 0)
    ref, w13, w2 = _build_layer(1, 0, device, combine_wire=wire, pdl=6, pdl_late=True)
    layer, _w13, _w2 = _build_layer(
        1,
        0,
        device,
        combine_wire=wire,
        pdl=6,
        pdl_late=True,
        per_route_publish_pdl=True,
    )
    runner = _warm_runner(layer, device)
    assert runner.pdl == 6 and runner.per_route_publish_pdl
    assert runner._pdl_flag(8) & 1 == (1 if wire == "per_route" else 0)
    for num_tokens in (1, 8, TOKEN_CAPACITY):
        x, ids, weights = _make_inputs(
            num_tokens, LOCAL_EXPERTS, 47 + num_tokens, device
        )
        output = _forward(layer, x, ids, weights).clone()
        reference = _forward(ref, x, ids, weights).clone()
        torch.cuda.synchronize()
        assert torch.equal(output, reference), f"T={num_tokens}"
    _check(output, x, ids, weights, w13, w2, label=f" ep1 {wire} per_route_publish_pdl")


# Environment fallbacks of the backend's round switches (every one is also a config field).
ROUND_SWITCH_ENVS = (
    "FLASHINFER_SM90_CAKE_BF16_PDL",
    "FLASHINFER_SM90_CAKE_BF16_PDL_LATE",
    "FLASHINFER_SM90_CAKE_BF16_FUSED_BEGIN",
    "FLASHINFER_SM90_CAKE_BF16_FUSED_DISPATCH_MAX_TOKENS",
    "FLASHINFER_SM90_CAKE_BF16_GRID_DISPATCH_MIN_TOKENS",
    "FLASHINFER_SM90_CAKE_BF16_COMPACT_FORM",
    "FLASHINFER_SM90_CAKE_BF16_PUBLISH_FORM",
    "FLASHINFER_SM90_CAKE_BF16_PER_ROUTE_PUBLISH_PDL",
    "FLASHINFER_SM90_CAKE_BF16_FC2_EARLY_MAX_TOKENS",
    "FLASHINFER_SM90_CAKE_BF16_PREFIX_FORM",
    "FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE",
    "FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE_MIN_TOKENS",
    "FLASHINFER_SM90_CAKE_BF16_FC2_EARLY_WINDOWS",
)


@requires_sm90
def test_ep1_round_switch_defaults(monkeypatch) -> None:
    """With no config value and no environment override the runner resolves the
    shipped round switches."""
    for name in ROUND_SWITCH_ENVS + (
        "FLASHINFER_SM90_CAKE_BF16_FUSED_TAIL",
        "FLASHINFER_SM90_CAKE_BF16_FUSED_DISPATCH",
    ):
        monkeypatch.delenv(name, raising=False)
    device = torch.device("cuda", 0)
    # a token capacity no other test uses: the workspace pool is keyed on
    # (max_tokens_per_rank, config) and the switches are resolved when the pooled
    # runner is built
    layer, w13, w2 = _build_layer(1, 0, device, token_capacity=37)
    runner = _warm_runner(layer, device)
    assert runner.pdl == 6  # FC1 and FC2 edges
    assert runner.pdl_late is True
    assert runner.fused_begin == 2
    assert runner.fused_dispatch_max_tokens == 512
    assert runner.grid_dispatch_min_tokens == 513
    assert runner.compact_form == 1 and runner.publish_form == 1
    assert runner.per_route_publish_pdl is True
    assert runner.fc2_early_max_tokens == 128
    assert runner.fc2_early_windows is True
    assert runner.fc2_early_window is None  # no window for this test shape
    assert runner.prefix_form == 1
    assert runner.direct_place is False
    assert runner.direct_place_min_tokens == 513
    x, ids, weights = _make_inputs(37, LOCAL_EXPERTS, 59, device)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _check(output, x, ids, weights, w13, w2, label=" ep1 round switch defaults")


class _GemmSpy:
    """Forwards to the runner's grouped GEMM launcher and records whether each FC1 /
    FC2 launch of a round took the early-start pair (a ``sched`` buffer)."""

    def __init__(self, gemm) -> None:
        self._gemm = gemm
        self.calls: list[tuple[str, bool]] = []

    def __getattr__(self, name):
        return getattr(self._gemm, name)

    def fc1(self, *args, **kwargs):
        self.calls.append(("fc1", kwargs.get("sched") is not None))
        return self._gemm.fc1(*args, **kwargs)

    def fc2(self, *args, **kwargs):
        self.calls.append(("fc2", kwargs.get("sched") is not None))
        return self._gemm.fc2(*args, **kwargs)


# Round sizes around the FC2 early-start window (0 < local tokens <= 128 by default).
FC2_EARLY_TOKENS = (1, 8, 16, 32, 64, 128, 129, 160)


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("mode", ["random", "hot", "masked"])
def test_ep1_fc2_early_window_bitwise(mode: str, wire: str) -> None:
    """Rounds with 0 < tokens <= fc2_early_max_tokens run the early-start GEMM pair and
    write the same bytes as the regular pair; larger rounds run the regular pair."""
    device = torch.device("cuda", 0)
    capacity = max(FC2_EARLY_TOKENS)
    ref, w13, w2 = _build_layer(
        1, 0, device, combine_wire=wire, token_capacity=capacity, fc2_early_max_tokens=0
    )
    layer, _w13, _w2 = _build_layer(
        1,
        0,
        device,
        combine_wire=wire,
        token_capacity=capacity,
        fc2_early_max_tokens=128,
    )
    assert _warm_runner(ref, device).fc2_early_max_tokens == 0
    runner = _warm_runner(layer, device)
    assert runner.fc2_early_max_tokens == 128
    spy = _GemmSpy(runner.gemm)
    runner.gemm = spy
    try:
        _fc2_early_rounds(layer, ref, spy, w13, w2, mode, wire, device)
    finally:
        runner.gemm = spy._gemm


def _fc2_early_rounds(layer, ref, spy, w13, w2, mode, wire, device) -> None:
    for num_tokens in (0,) + FC2_EARLY_TOKENS:
        x, ids, weights = _make_inputs(
            num_tokens, LOCAL_EXPERTS, 61 + num_tokens, device, mode=mode
        )
        spy.calls.clear()
        output = _forward(layer, x, ids, weights).clone()
        reference = _forward(ref, x, ids, weights).clone()
        torch.cuda.synchronize()
        if num_tokens:
            early = num_tokens <= 128
            assert spy.calls == [("fc1", early), ("fc2", early)], (
                num_tokens,
                spy.calls,
            )
        else:  # an empty round never takes the early-start pair
            assert not any(e for _, e in spy.calls), spy.calls
        assert torch.equal(output, reference), f"T={num_tokens}"
        if num_tokens:
            _check(
                output,
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep1 {wire} {mode} T={num_tokens} fc2_early",
            )
    # graph replay of a window round, then a regular round on the same layer
    x, ids, weights = _make_inputs(16, LOCAL_EXPERTS, 71, device, mode=mode)
    reference = _forward(ref, x, ids, weights).clone()
    static = [t.clone() for t in (x, ids, weights)]
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        _forward(layer, *static)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, *static)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_output, reference)
    x2, ids2, weights2 = _make_inputs(160, LOCAL_EXPERTS, 73, device, mode=mode)
    output2 = _forward(layer, x2, ids2, weights2).clone()
    reference2 = _forward(ref, x2, ids2, weights2).clone()
    torch.cuda.synchronize()
    assert torch.equal(output2, reference2)


# The shipped per-shape FC2 early-start windows: (hidden, intermediate, total experts,
# top_k, ep) -> (lo, hi).
FC2_EARLY_SHIPPED_WINDOWS = {
    (7168, 2048, 256, 8, 8): (128, 4096),
    (7168, 2048, 256, 8, 4): (128, 1024),
    (7168, 3072, 384, 6, 8): (512, 1024),
}
# Round sizes around a shape window (128, 160] above the decode window (<= 128).
FC2_EARLY_SHAPE_WINDOW = (128, 160)
FC2_EARLY_SHAPE_TOKENS = (1, 128, 129, 160, 161, 192)


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("mode", ["random", "hot", "masked"])
def test_ep1_fc2_early_shape_window_bitwise(mode: str, wire: str, monkeypatch) -> None:
    """Rounds in the pipe shape's window (lo, hi] above the decode window also run the
    early-start GEMM pair and write the same bytes as the regular pair; rounds above
    the window run the regular pair."""
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import (
        cake_runner,
    )

    monkeypatch.delenv(cake_runner.FC2_EARLY_WINDOWS_ENV, raising=False)
    monkeypatch.setitem(
        cake_runner.FC2_EARLY_WINDOWS,
        (HIDDEN, INTERMEDIATE, LOCAL_EXPERTS, TOP_K, 1),
        FC2_EARLY_SHAPE_WINDOW,
    )
    device = torch.device("cuda", 0)
    # a token capacity no other test uses: the pooled runner resolves its window at build
    capacity = 193
    ref, w13, w2 = _build_layer(
        1,
        0,
        device,
        combine_wire=wire,
        token_capacity=capacity,
        fc2_early_max_tokens=0,
        fc2_early_windows=False,
    )
    layer, _w13, _w2 = _build_layer(
        1,
        0,
        device,
        combine_wire=wire,
        token_capacity=capacity,
        fc2_early_max_tokens=128,
        fc2_early_windows=True,
    )
    ref_runner = _warm_runner(ref, device)
    assert ref_runner.fc2_early_max_tokens == 0
    assert ref_runner.fc2_early_window is None
    runner = _warm_runner(layer, device)
    assert runner.fc2_early_window == FC2_EARLY_SHAPE_WINDOW
    spy = _GemmSpy(runner.gemm)
    runner.gemm = spy
    try:
        for num_tokens in (0,) + FC2_EARLY_SHAPE_TOKENS:
            x, ids, weights = _make_inputs(
                num_tokens, LOCAL_EXPERTS, 83 + num_tokens, device, mode=mode
            )
            spy.calls.clear()
            output = _forward(layer, x, ids, weights).clone()
            reference = _forward(ref, x, ids, weights).clone()
            torch.cuda.synchronize()
            early = 0 < num_tokens <= FC2_EARLY_SHAPE_WINDOW[1]
            if num_tokens:
                assert spy.calls == [("fc1", early), ("fc2", early)], (
                    num_tokens,
                    spy.calls,
                )
            else:
                assert not any(e for _, e in spy.calls), spy.calls
            assert torch.equal(output, reference), f"T={num_tokens}"
            if num_tokens:
                _check(
                    output,
                    x,
                    ids,
                    weights,
                    w13,
                    w2,
                    label=f" ep1 {wire} {mode} T={num_tokens} fc2_early window",
                )
        # graph replay of a window round, then a round above the window
        x, ids, weights = _make_inputs(150, LOCAL_EXPERTS, 89, device, mode=mode)
        reference = _forward(ref, x, ids, weights).clone()
        static = [t.clone() for t in (x, ids, weights)]
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, *static)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = _forward(layer, *static)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_output, reference)
        x2, ids2, weights2 = _make_inputs(192, LOCAL_EXPERTS, 97, device, mode=mode)
        output2 = _forward(layer, x2, ids2, weights2).clone()
        reference2 = _forward(ref, x2, ids2, weights2).clone()
        torch.cuda.synchronize()
        assert torch.equal(output2, reference2)
    finally:
        runner.gemm = spy._gemm


def test_ep1_fc2_early_windows_table() -> None:
    """The shipped windows sit above the decode window and the round predicate
    switches exactly at each window's bounds (host-only check)."""
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import (
        cake_runner,
    )

    dmax = cake_runner.FC2_EARLY_MAX_TOKENS
    assert cake_runner.FC2_EARLY_WINDOWS == FC2_EARLY_SHIPPED_WINDOWS
    for key, (lo, hi) in cake_runner.FC2_EARLY_WINDOWS.items():
        assert dmax <= lo < hi, (key, lo, hi)
        assert cake_runner.fc2_early_window_for(*key) == (lo, hi)
        win = (lo, hi)
        assert cake_runner.fc2_early_round(lo, dmax, win) == (lo <= dmax), key
        assert cake_runner.fc2_early_round(lo + 1, dmax, win), key
        assert cake_runner.fc2_early_round(hi, dmax, win), key
        assert not cake_runner.fc2_early_round(hi + 1, dmax, win), key
        assert not cake_runner.fc2_early_round(hi + 1, dmax, None), key
        assert not cake_runner.fc2_early_round(0, dmax, win), key
        # windows are per shape: a different EP size or top_k has its own entry or none
        for other in (
            (key[0], key[1], key[2], key[3], key[4] * 2),
            (key[0], key[1], key[2], key[3] + 1, key[4]),
        ):
            expected = cake_runner.FC2_EARLY_WINDOWS.get(other)
            assert cake_runner.fc2_early_window_for(*other) == expected
    assert cake_runner.fc2_early_round(dmax, dmax, None)
    assert not cake_runner.fc2_early_round(dmax + 1, dmax, None)


@requires_sm90
def test_ep1_fc2_early_switch_validation() -> None:
    """fc2_early_max_tokens rejects negative and non-int values; fc2_early_windows
    rejects non-bool values."""
    from flashinfer.moe_ep import MoEEpConfigError

    device = torch.device("cuda", 0)
    for kwargs in (
        {"fc2_early_max_tokens": -1},
        {"fc2_early_max_tokens": True},
        {"fc2_early_windows": 1},
        {"fc2_early_windows": "on"},
    ):
        with pytest.raises((MoEEpConfigError, ValueError)):
            _build_layer(1, 0, device, token_capacity=33, **kwargs)


class _DispatchSpy:
    """Forwards to the runner's fused dispatch module and records the prefix form of
    each fused dispatch launch."""

    def __init__(self, module) -> None:
        self._module = module
        self.calls: list[int] = []

    def __getattr__(self, name):
        return getattr(self._module, name)

    def sm90_cake_dispatch_fused_bf16(self, *args):
        self.calls.append(int(args[-1]))
        return self._module.sm90_cake_dispatch_fused_bf16(*args)


# Round sizes around the fused / grid-strided dispatch bound (fused up to 512 tokens,
# grid-strided from 513) and the small-route instance bound (T * top_k <= 1024).
DISPATCH_SWITCH_TOKENS = (1, 8, 64, 128, 129, 512, 513, 700)
# The dispatch paths these tests exercise (the shipped defaults), pinned so that the
# backend environment of a run cannot route the rounds elsewhere.
DISPATCH_PATHS = dict(
    fused_begin=2, fused_dispatch_max_tokens=512, grid_dispatch_min_tokens=513
)
# Direct placement of the grid-dispatch carrier rows with the compact form it needs and
# its round bound at the grid-strided bound, pinned so that the backend environment of a
# run cannot turn it off.
DIRECT_PLACE = dict(direct_place=True, direct_place_min_tokens=513, compact_form=1)
# EP1 round sizes of the direct placement that take each row form of the compact on a
# 132-SM part (grid 528 blocks; m = TOP_K x tokens rows): one row per block (m <= 1056),
# four rows per block (m <= 3168) and one warp per row.
DIRECT_PLACE_TOKENS = (513, 640, 1100, 2000)
# Per-rank round sizes of the multi-rank direct-placement tests: every rank grid-strided
# (placed directly), one rank on the fused dispatch and an idle rank (every rank takes
# the inbox path), and direct rounds after a fallback round.
DIRECT_PLACE_PLANS = (
    lambda r: 600,
    lambda r: 640 - r,
    lambda r: 512 + r % 2,
    lambda r: 0 if r == 0 else 600,
    lambda r: 600,
    lambda r: 513,
)


def _direct_expected(plan, world_size: int, bound=lambda r: 513) -> int:
    """Direct-placement flag of a round: every rank dispatches it grid-strided with at
    least its own bound of local tokens."""
    return int(all(plan(r) >= max(513, bound(r)) for r in range(world_size)))


def _duplicate_routes(ids: torch.Tensor) -> int:
    """Routes whose token already sends a route to the same rank (the rows the
    direct-placement compact copies from that token's carrier row)."""
    owners = [
        [expert // LOCAL_EXPERTS for expert in row if expert >= 0]
        for row in ids.tolist()
    ]
    return sum(len(row) - len(set(row)) for row in owners)


@requires_sm90
@pytest.mark.parametrize("mode", ["random", "masked", "hot", "dup_rank"])
def test_ep1_dispatch_switches_bitwise(mode: str) -> None:
    """prefix_form 1 writes the same bytes as prefix_form 0 in fused and grid-strided
    dispatch rounds, and a graph replay of a small round stays bitwise equal."""
    device = torch.device("cuda", 0)
    capacity = max(DISPATCH_SWITCH_TOKENS)
    layers = {
        form: _build_layer(
            1, 0, device, token_capacity=capacity, prefix_form=form, **DISPATCH_PATHS
        )
        for form in (0, 1)
    }
    _, w13, w2 = layers[0]
    for form, (layer, _w13, _w2) in layers.items():
        assert _warm_runner(layer, device).prefix_form == form
    runner = _warm_runner(layers[1][0], device)
    spy = _DispatchSpy(runner.dispatch_module)
    runner.dispatch_module = spy
    try:
        for num_tokens in DISPATCH_SWITCH_TOKENS:
            x, ids, weights = _make_inputs(
                num_tokens, LOCAL_EXPERTS, 89 + num_tokens, device, mode=mode
            )
            spy.calls.clear()
            outputs = {}
            for key, (layer, _w13, _w2) in layers.items():
                outputs[key] = _forward(layer, x, ids, weights).clone()
                torch.cuda.synchronize()
            assert spy.calls == ([1] if num_tokens <= 512 else []), (
                num_tokens,
                spy.calls,
            )
            _check(
                outputs[1],
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep1 {mode} T={num_tokens} prefix_form 1",
            )
            assert torch.equal(outputs[1], outputs[0]), f"T={num_tokens}"
        # graph replay of a small round
        layer = layers[1][0]
        x, ids, weights = _make_inputs(16, LOCAL_EXPERTS, 97, device, mode=mode)
        reference = _forward(layers[0][0], x, ids, weights).clone()
        static = [t.clone() for t in (x, ids, weights)]
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, *static)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = _forward(layer, *static)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static_output, reference)
    finally:
        runner.dispatch_module = spy._module


@requires_sm90
def test_ep1_dispatch_switch_validation() -> None:
    """prefix_form rejects values other than 0, 1 and None."""
    from flashinfer.moe_ep import MoEEpConfigError

    device = torch.device("cuda", 0)
    for kwargs in ({"prefix_form": 2}, {"prefix_form": True}):
        with pytest.raises((MoEEpConfigError, ValueError)):
            _build_layer(1, 0, device, token_capacity=33, **kwargs)


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("mode", ["random", "hot", "masked", "dup_rank"])
def test_ep1_direct_place_bitwise(mode: str, wire: str) -> None:
    """Grid-strided rounds that place the carrier rows directly in the activation matrix
    (the compact copies only the duplicate routes) write the same bytes as the inbox
    path, in every row form of the compact and with a full activation matrix (hot)."""
    device = torch.device("cuda", 0)
    capacity = max(DIRECT_PLACE_TOKENS)
    common = dict(combine_wire=wire, token_capacity=capacity, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(1, 0, device, direct_place=False, **common)
    layer, _w13, _w2 = _build_layer(1, 0, device, **DIRECT_PLACE, **common)
    assert not _warm_runner(off, device).direct_place
    runner = _warm_runner(layer, device)
    assert runner.direct_place and runner.direct_place_min_tokens == 513
    assert runner._direct_flag() == 0  # the warm-up round is fused
    duplicates = 0
    for num_tokens in DIRECT_PLACE_TOKENS:
        x, ids, weights = _make_inputs(
            num_tokens, LOCAL_EXPERTS, 211 + num_tokens, device, mode=mode
        )
        reference = _forward(off, x, ids, weights).clone()
        output = _forward(layer, x, ids, weights).clone()
        torch.cuda.synchronize()
        assert runner._direct_flag() == 1, f"T={num_tokens}"
        assert torch.equal(output, reference), f"T={num_tokens}"
        _check(
            output,
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep1 {wire} {mode} T={num_tokens} direct_place",
        )
        duplicates += _duplicate_routes(ids)
    assert duplicates > 0


@requires_sm90
def test_ep1_direct_place_switch(monkeypatch) -> None:
    """direct_place resolution: the environment fallback (0 or 1, anything else
    raises), the round bound, the pipes that cannot run it (inbox path, no error), the
    config validation and the workspace pool key."""
    from flashinfer.moe_ep import MoEEpConfigError
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim.cake_runner import (
        DIRECT_PLACE_ENV,
        DIRECT_PLACE_MIN_TOKENS_ENV,
    )

    assert DIRECT_PLACE_ENV == "FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE"
    assert (
        DIRECT_PLACE_MIN_TOKENS_ENV
        == "FLASHINFER_SM90_CAKE_BF16_DIRECT_PLACE_MIN_TOKENS"
    )
    monkeypatch.delenv(DIRECT_PLACE_ENV, raising=False)
    monkeypatch.delenv(DIRECT_PLACE_MIN_TOKENS_ENV, raising=False)
    device = torch.device("cuda", 0)
    # distinct token capacities per case (at least the round bounds, see
    # test_ep1_direct_place_small_capacity): the switches are resolved when the pooled
    # runner is built
    for raw, expected, capacity in (("1", True, 541), ("0", False, 542)):
        monkeypatch.setenv(DIRECT_PLACE_ENV, raw)
        layer, _w13, _w2 = _build_layer(
            1, 0, device, token_capacity=capacity, compact_form=1, **DISPATCH_PATHS
        )
        assert _warm_runner(layer, device).direct_place is expected, raw
    monkeypatch.setenv(DIRECT_PLACE_ENV, "2")
    with pytest.raises(Exception, match="0 or 1"):
        layer, _w13, _w2 = _build_layer(
            1, 0, device, token_capacity=545, **DISPATCH_PATHS
        )
        _warm_runner(layer, device)
    monkeypatch.delenv(DIRECT_PLACE_ENV)
    monkeypatch.setenv(DIRECT_PLACE_MIN_TOKENS_ENV, "700")
    layer, _w13, _w2 = _build_layer(
        1,
        0,
        device,
        token_capacity=747,
        direct_place=True,
        compact_form=1,
        **DISPATCH_PATHS,
    )
    runner = _warm_runner(layer, device)
    assert runner.direct_place and runner.direct_place_min_tokens == 700
    monkeypatch.delenv(DIRECT_PLACE_MIN_TOKENS_ENV)
    for capacity, kwargs in (
        (549, dict(compact_form=0)),
        (551, dict(grid_dispatch_min_tokens=0)),
        (553, dict(dedup_dispatch=False)),
    ):
        layer, _w13, _w2 = _build_layer(
            1, 0, device, token_capacity=capacity, direct_place=True, **kwargs
        )
        assert not _warm_runner(layer, device).direct_place, kwargs
    for kwargs in (
        {"direct_place": 1},
        {"direct_place": "yes"},
        {"direct_place_min_tokens": -1},
        {"direct_place_min_tokens": True},
    ):
        with pytest.raises((MoEEpConfigError, ValueError)):
            _build_layer(1, 0, device, token_capacity=33, **kwargs)
    # the pool key separates the layers that request it from the others
    on = [
        _build_layer(1, 0, device, token_capacity=55, **DIRECT_PLACE)[0]
        for _ in range(2)
    ]
    off, _w13, _w2 = _build_layer(1, 0, device, token_capacity=55, direct_place=False)
    runners = {id(_warm_runner(layer, device)) for layer in on}
    assert len(runners) == 1
    assert id(_warm_runner(off, device)) not in runners


def test_ep1_direct_place_defaults(monkeypatch) -> None:
    """The shipped per-EP default of direct placement: on at the EP sizes of
    DIRECT_PLACE_DEFAULTS with their round bound raised to the routed-element
    floor, off at the others, and the environment fallbacks override it
    (host-only check)."""
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import (
        cake_runner,
    )

    assert cake_runner.DIRECT_PLACE_DEFAULTS == {8: 513, 4: 513}
    assert cake_runner.DIRECT_PLACE_MIN_ROUTE_ELEMS == 1 << 25
    monkeypatch.delenv(cake_runner.DIRECT_PLACE_ENV, raising=False)
    monkeypatch.delenv(cake_runner.DIRECT_PLACE_MIN_TOKENS_ENV, raising=False)
    for ep, on in ((1, False), (2, False), (4, True), (8, True), (16, False)):
        assert cake_runner._direct_place_default(ep) is on, ep
    # (top_k, hidden) -> default bound at EP 8 and EP 4: the smallest round whose
    # routes carry 2**25 elements; round T - 1 stays below the floor, round T reaches it
    for (top_k, hidden), bound in (
        ((8, 7168), 586),
        ((6, 7168), 781),
        ((6, 4096), 1366),
    ):
        for ep in (8, 4):
            got = cake_runner._direct_place_min_tokens_default(ep, top_k, hidden)
            assert got == bound, (ep, top_k, hidden, got)
        for ep in (1, 2, 16):
            got = cake_runner._direct_place_min_tokens_default(ep, top_k, hidden)
            assert got == 513, (ep, top_k, hidden, got)
        assert (bound - 1) * top_k * hidden < 1 << 25 <= bound * top_k * hidden
    # the listed bound stays the minimum for wide rounds; this file's small test
    # shape gets a bound above its token capacities (direct placement off)
    assert cake_runner._direct_place_min_tokens_default(8, 8, 16384) == 513
    assert cake_runner._direct_place_min_tokens_default(4, TOP_K, HIDDEN) == 32768
    monkeypatch.setenv(cake_runner.DIRECT_PLACE_ENV, "0")
    assert cake_runner._direct_place_default(8) is False
    monkeypatch.setenv(cake_runner.DIRECT_PLACE_ENV, "1")
    assert cake_runner._direct_place_default(2) is True
    monkeypatch.setenv(cake_runner.DIRECT_PLACE_ENV, "on")
    with pytest.raises(ValueError, match="0 or 1"):
        cake_runner._direct_place_default(8)
    monkeypatch.setenv(cake_runner.DIRECT_PLACE_MIN_TOKENS_ENV, "700")
    for ep in (2, 8):
        assert cake_runner._direct_place_min_tokens_default(ep, 8, 7168) == 700


@requires_sm90
def test_ep1_direct_place_small_capacity() -> None:
    """A layer none of whose rounds can be placed directly (token capacity below the
    grid-dispatch bound or the direct-placement bound) keeps the regular activation
    matrix (no symmetric buffer) and writes the same bytes as the inbox path; at a
    capacity equal to both bounds the layer places its largest rounds directly."""
    device = torch.device("cuda", 0)
    for capacity, bounds in (
        (512, {}),
        (639, dict(direct_place_min_tokens=640)),
        (599, dict(grid_dispatch_min_tokens=600)),
    ):
        kwargs = {**DISPATCH_PATHS, **DIRECT_PLACE, **bounds}
        off_kwargs = {**kwargs, "direct_place": False}
        off, w13, w2 = _build_layer(1, 0, device, token_capacity=capacity, **off_kwargs)
        layer, _w13, _w2 = _build_layer(1, 0, device, token_capacity=capacity, **kwargs)
        runner = _warm_runner(layer, device)
        assert not runner.direct_place, (capacity, bounds)
        assert runner._a1_symm is None and runner.a1 is not None, (capacity, bounds)
        for num_tokens in (1, capacity // 2, capacity):
            x, ids, weights = _make_inputs(
                num_tokens, LOCAL_EXPERTS, 307 + num_tokens, device
            )
            reference = _forward(off, x, ids, weights).clone()
            output = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
            assert runner._direct_flag() == 0, (capacity, num_tokens)
            assert torch.equal(output, reference), (capacity, num_tokens)
        # one more token of capacity reaches both bounds: the symmetric matrix and a
        # direct round at the capacity
        layer, _w13, _w2 = _build_layer(
            1, 0, device, token_capacity=capacity + 1, **kwargs
        )
        runner = _warm_runner(layer, device)
        assert runner.direct_place and runner._a1_symm is not None, (capacity, bounds)
        x, ids, weights = _make_inputs(
            capacity + 1, LOCAL_EXPERTS, 307 + capacity, device
        )
        output = _forward(layer, x, ids, weights).clone()
        torch.cuda.synchronize()
        assert runner._direct_flag() == 1, (capacity, bounds)
        _check(
            output,
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep1 capacity {capacity + 1} direct_place",
        )


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("case", ["short", "masked", "hot", "empty", "dup_rank"])
def test_ep1_edge_routes(case: str, wire: str) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device, combine_wire=wire)
    if case == "empty":
        num_tokens = 0
    elif case == "short":
        num_tokens = 7
    else:
        num_tokens = TOKEN_CAPACITY
    mode = case if case in ("hot", "masked", "dup_rank") else "random"
    x, ids, weights = _make_inputs(num_tokens, LOCAL_EXPERTS, 21, device, mode=mode)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    assert output.shape == (num_tokens, HIDDEN)
    if num_tokens:
        _check(output, x, ids, weights, w13, w2, label=f" ep1 {wire} {case}")
    # the layer keeps working after an edge round
    x2, ids2, weights2 = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 22, device)
    output2 = _forward(layer, x2, ids2, weights2)
    torch.cuda.synchronize()
    _check(output2, x2, ids2, weights2, w13, w2, label=f" ep1 {wire} {case} recovery")


@requires_sm90
def _find_runner(root, class_name="Sm90CakeBf16MoERunner", max_depth=8):
    """Locate the runner inside the layer's object graph (bounded BFS)."""
    seen: set[int] = set()
    frontier = [(root, 0)]
    while frontier:
        obj, depth = frontier.pop(0)
        if id(obj) in seen or depth > max_depth:
            continue
        seen.add(id(obj))
        if type(obj).__name__ == class_name:
            return obj
        if isinstance(obj, dict):
            children = list(obj.values())
        elif isinstance(obj, (list, tuple, set)):
            children = list(obj)
        else:
            try:
                children = list(vars(obj).values())
            except TypeError:
                children = []
        frontier.extend((child, depth + 1) for child in children)
    return None


def _warm_runner(layer, device: torch.device):
    """One single-token round (the workspace and its runner are created lazily), then the runner."""
    x, ids, weights = _make_inputs(1, LOCAL_EXPERTS, 5, device)
    _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    runner = _find_runner(layer)
    assert runner is not None
    return runner


@requires_sm90
@pytest.mark.parametrize(
    "token_capacity, env, expected",
    [
        # distinct token capacities per case: the mega workspace pool is keyed
        # per process on (max_tokens_per_rank, config) and the wire is resolved
        # when the pooled runner is built, so a reused workspace would report
        # the wire of the case that created it
        (8, None, "per_route"),
        (9, None, "prereduced"),
        (TOKEN_CAPACITY, None, "prereduced"),
        (7, "prereduced", "prereduced"),
        (65, "per_route", "per_route"),
    ],
)
def test_ep1_combine_wire_per_shape_default(
    token_capacity: int, env: str | None, expected: str, monkeypatch
) -> None:
    """combine_wire=None: the environment override wins when set; otherwise the
    per-shape default picks per_route for max_tokens_per_rank <= 8 and
    prereduced above.  The output stays correct either way."""
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe import (
        COMBINE_WIRE_ENV,
        COMBINE_WIRE_PER_SHAPE_MAX_TOKENS,
    )

    assert COMBINE_WIRE_PER_SHAPE_MAX_TOKENS == 8
    if env is None:
        monkeypatch.delenv(COMBINE_WIRE_ENV, raising=False)
    else:
        monkeypatch.setenv(COMBINE_WIRE_ENV, env)
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1, 0, device, combine_wire=None, token_capacity=token_capacity, seed=31
    )
    x, ids, weights = _make_inputs(
        token_capacity, LOCAL_EXPERTS, 31, device, mode="random"
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _check(output, x, ids, weights, w13, w2, label=f" per-shape t_cap={token_capacity}")
    runner = _find_runner(layer)
    assert runner is not None
    assert runner.combine_wire == expected


@requires_sm90
def test_ep1_clamp_limit() -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device, clamp_limit=0.5)
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 31, device)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _check(output, x, ids, weights, w13, w2, clamp=0.5, label=" ep1 clamp=0.5")


@requires_sm90
def test_ep1_forward_validation() -> None:
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    device = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, device)
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 41, device)
    with pytest.raises(MoEEpConfigError):
        layer(
            MoEEpTensors(
                hidden_states=x, topk_ids=ids.to(torch.int64), topk_weights=weights
            )
        )
    with pytest.raises(MoEEpConfigError):
        layer(MoEEpTensors(hidden_states=x.float(), topk_ids=ids, topk_weights=weights))
    x_big, ids_big, weights_big = _make_inputs(
        TOKEN_CAPACITY + 1, LOCAL_EXPERTS, 42, device
    )
    with pytest.raises(MoEEpConfigError):
        layer(
            MoEEpTensors(
                hidden_states=x_big, topk_ids=ids_big, topk_weights=weights_big
            )
        )


@requires_sm90
def test_ep1_unsupported_geometry_raises() -> None:
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    device = torch.device("cuda", 0)
    hidden, inter = 384, 192  # not multiples of 256 / 128
    w13 = torch.randn(
        LOCAL_EXPERTS, 2 * inter, hidden, device=device, dtype=torch.bfloat16
    )
    w2 = torch.randn(LOCAL_EXPERTS, hidden, inter, device=device, dtype=torch.bfloat16)
    with pytest.raises(MoEEpConfigError):
        MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=LOCAL_EXPERTS,
                max_tokens_per_rank=TOKEN_CAPACITY,
                token_hidden_size=hidden,
            ),
            weights=MoEWeightPack(w13=w13, w2=w2),
            backend=MegaConfig(
                megakernel=Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig(
                    intermediate_size=inter, top_k=TOP_K
                )
            ),
        )


def test_grouped_gemm_guards_reject_misaligned_and_oversized_geometry() -> None:
    """CPU-only: the launcher and runner-level guards refuse what the kernels tile exactly."""
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim import cake_gemm

    def operands(rows: int, n: int, k: int, *, gated: bool) -> dict:
        # expand() keeps the storage tiny so the 32-bit bound can be exercised on a host.
        a = torch.empty(1, k, dtype=torch.bfloat16).expand(rows, k)
        out = torch.empty(1, n // 2 if gated else n, dtype=torch.bfloat16).expand(
            rows, n // 2 if gated else n
        )
        w = torch.empty(LOCAL_EXPERTS, n, k, dtype=torch.bfloat16)
        offsets = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int64)
        return dict(gated=gated, a=a, w=w, offsets=offsets, out=out)

    with pytest.raises(ValueError, match="must be contiguous"):
        cake_gemm._check_geometry(**operands(128, 512, 256, gated=True))
    with pytest.raises(ValueError, match="requires K %"):
        cake_gemm._check_geometry(**operands(128, 384, 256, gated=False))
    with pytest.raises(ValueError, match="requires K %"):
        cake_gemm._check_geometry(**operands(128, 512, 96, gated=False))
    rows = cake_gemm.MAX_OUTPUT_ELEMENTS // 256 + cake_gemm.BLOCK_M
    with pytest.raises(ValueError, match="32-bit"):
        cake_gemm._check_geometry(**operands(rows, 256, 256, gated=False))
    with pytest.raises(ValueError, match="multiple of 128"):
        cake_gemm._check_geometry(**operands(96, 512, 256, gated=False))

    assert cake_gemm.output_row_capacity(1) == cake_gemm.BLOCK_M
    assert cake_gemm.output_row_capacity(262_144) == 262_144
    cake_gemm.validate_grouped_gemm_geometry(
        hidden_size=7168,
        intermediate_size=2048,
        gate_up_group=32,
        row_capacity=cake_gemm.output_row_capacity(262_144),
    )
    with pytest.raises(MoEEpConfigError, match="32-bit"):
        cake_gemm.validate_grouped_gemm_geometry(
            hidden_size=7168,
            intermediate_size=2048,
            gate_up_group=32,
            row_capacity=cake_gemm.output_row_capacity(599_296),
        )
    with pytest.raises(MoEEpConfigError, match="multiples of 256"):
        cake_gemm.validate_grouped_gemm_geometry(
            hidden_size=384, intermediate_size=192, gate_up_group=32
        )


@requires_sm90
@pytest.mark.parametrize("wire", WIRES)
def test_ep1_graph_replay(wire: str) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device, combine_wire=wire)
    inputs = [
        _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 51 + index, device)
        for index in range(2)
    ]
    eager = []
    for x, ids, weights in inputs:
        eager.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()
    static_x = torch.empty_like(inputs[0][0]).copy_(inputs[0][0])
    static_ids = torch.empty_like(inputs[0][1]).copy_(inputs[0][1])
    static_weights = torch.empty_like(inputs[0][2]).copy_(inputs[0][2])
    for _ in range(2):
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, static_x, static_ids, static_weights)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, static_x, static_ids, static_weights)
    replayed = []
    for x, ids, weights in inputs:
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        replayed.append(static_output.clone())
    assert not torch.equal(replayed[0], replayed[1])
    for index, (x, ids, weights) in enumerate(inputs):
        assert torch.equal(replayed[index], eager[index])
        _check(
            replayed[index],
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep1 {wire} graph[{index}]",
        )


@requires_sm90
@pytest.mark.parametrize("clamp", [None, 0.5])
def test_ep1_exported_gemm_matches_in_process(clamp: float | None) -> None:
    """The shipped (generated, manifest-sealed) GEMM modules reproduce the
    generator's in-process build bit for bit on both layers."""
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim.cake_gemm import (
        DEV_LAUNCHER_ENV,
        ExportedGroupedGemm,
        create_grouped_gemm,
    )

    spec = os.environ.get(DEV_LAUNCHER_ENV, "").strip()
    if spec in ("", "0"):
        pytest.skip(f"{DEV_LAUNCHER_ENV} does not name a development launcher")
    pytest.importorskip(
        spec.partition(":")[0], reason="development launcher module not importable"
    )
    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim.cake_weights import (
        interleave_gate_up,
    )

    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(891)
    rows = 128 * 5
    counts = torch.tensor([0, 129, 7, 300], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)]).to(
        device
    )
    a1 = torch.randn(rows, HIDDEN, device=device, generator=gen).to(torch.bfloat16)
    w13 = torch.randn(
        LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device=device, generator=gen
    )
    w13 = interleave_gate_up((w13 * 0.05).to(torch.bfloat16), INTERMEDIATE).contiguous()
    w2 = (
        torch.randn(LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, device=device, generator=gen)
        * 0.05
    ).to(torch.bfloat16)
    outputs = []
    for launcher in (
        ExportedGroupedGemm(clamp=clamp),
        create_grouped_gemm(clamp=clamp),  # the development launcher named above
    ):
        h = torch.full(
            (rows, INTERMEDIATE), float("nan"), device=device, dtype=torch.bfloat16
        )
        y = torch.full(
            (rows, HIDDEN), float("nan"), device=device, dtype=torch.bfloat16
        )
        launcher.fc1(a1, w13, offsets, h)
        launcher.fc2(h, w2, offsets, y)
        torch.cuda.synchronize()
        launcher.destroy()
        valid = int(offsets[-1].item())
        assert (
            torch.isfinite(h[:valid].float()).all()
            and torch.isfinite(y[:valid].float()).all()
        )
        outputs.append((h[:valid].clone(), y[:valid].clone()))
    assert torch.equal(outputs[0][0], outputs[1][0]), "FC1 exported != in-process"
    assert torch.equal(outputs[0][1], outputs[1][1]), "FC2 exported != in-process"


# --------------------------------------------------------------------------- EP2+
def _dist_setup() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


@requires_dist
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize(
    "mode",
    [
        "random",
        "all_remote",
        "hot",
        "masked",
        "dup_rank",
        "dup_remote",
        "all_local",
        "empty_rank",
    ],
)
@pytest.mark.parametrize("dedup_dispatch", [True, False])
def test_ep_forward_modes(mode: str, dedup_dispatch: bool, wire: str) -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    layer, w13, w2 = _build_layer(
        world_size, rank, device, dedup_dispatch=dedup_dispatch, combine_wire=wire
    )
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY,
        total_experts,
        61 + rank,
        device,
        mode=mode,
        rank=rank,
        world_size=world_size,
    )
    outputs = []
    for _ in range(3):
        outputs.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()
    _check(
        outputs[0],
        x,
        ids,
        weights,
        w13,
        w2,
        label=f" ep{world_size} rank{rank} {wire} {mode} dedup={dedup_dispatch}",
    )
    for repeat in outputs[1:]:
        assert torch.equal(repeat, outputs[0])
    dist.barrier()


@requires_dist
@pytest.mark.parametrize(
    "mode", ["random", "hot", "masked", "dup_rank", "dup_remote", "empty_rank"]
)
def test_ep_grid_dispatch_matches_vendored(mode: str) -> None:
    """The grid-strided single-launch dispatch is bit-identical to the vendored kernels."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    # vendored count / reserve / store_publish (no fused small-round path, no grid path)
    vendored, w13, w2 = _build_layer(
        world_size,
        rank,
        device,
        fused_dispatch_max_tokens=1,
        grid_dispatch_min_tokens=0,
    )
    grid, _w13, _w2 = _build_layer(world_size, rank, device, grid_dispatch_min_tokens=1)
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY,
        total_experts,
        101 + rank,
        device,
        mode=mode,
        rank=rank,
        world_size=world_size,
    )
    reference = _forward(vendored, x, ids, weights).clone()
    torch.cuda.synchronize()
    for _ in range(3):
        output = _forward(grid, x, ids, weights).clone()
        torch.cuda.synchronize()
        assert torch.equal(output, reference), f"ep{world_size} rank{rank} {mode}"
    _check(
        reference,
        x,
        ids,
        weights,
        w13,
        w2,
        label=f" ep{world_size} rank{rank} {mode} grid",
    )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize("mode", ["random", "hot", "dup_remote", "empty_rank"])
def test_ep_copy_forms_bitwise(mode: str, wire: str) -> None:
    """The copy forms write the same bytes on every rank, also when ranks mix forms."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    capacity = max(COPY_FORM_TOKENS)
    mixed = rank % 2
    layers = {
        name: _build_layer(
            world_size,
            rank,
            device,
            combine_wire=wire,
            token_capacity=capacity,
            compact_form=forms[0],
            publish_form=forms[1],
        )
        for name, forms in (
            ("f0", (0, 0)),
            ("f1", (1, 1)),
            ("mixed", (mixed, 1 - mixed)),
        )
    }
    _, w13, w2 = layers["f0"]
    for num_tokens in COPY_FORM_TOKENS:
        x, ids, weights = _make_inputs(
            num_tokens,
            total_experts,
            71 + rank + num_tokens,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        outputs = {}
        for name, (layer, _w13, _w2) in layers.items():
            outputs[name] = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
        _check(
            outputs["f1"],
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep{world_size} rank{rank} {wire} {mode} T={num_tokens} forms=1,1",
        )
        for name, output in outputs.items():
            assert torch.equal(output, outputs["f0"]), (
                f"{name} differs at T={num_tokens}"
            )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "dup_remote", "empty_rank"])
def test_ep_per_route_publish_pdl_bitwise(mode: str) -> None:
    """Per-route wire: the PDL-launched publish twin writes the same bytes, also when ranks mix it with the
    vendored publish."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    common = dict(combine_wire="per_route", token_capacity=8, pdl=6, pdl_late=True)
    # own pool key for the mixed layer on every rank: the workspace pool shares a pipe
    # between layers with equal configs, so a layer whose config depends on the rank
    # would share the "off" pipe on some ranks and the "on" pipe on the others
    layers = {
        name: _build_layer(
            world_size, rank, device, per_route_publish_pdl=on, **common, **extra
        )
        for name, on, extra in (
            ("off", False, {}),
            ("on", True, {}),
            ("mixed", bool(rank % 2), {"init_timeout_s": 601.0}),
        )
    }
    _, w13, w2 = layers["off"]
    for num_tokens in (1, 5, 8):
        x, ids, weights = _make_inputs(
            num_tokens,
            total_experts,
            89 + rank + num_tokens,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        outputs = {}
        for name, (layer, _w13, _w2) in layers.items():
            outputs[name] = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
        _check(
            outputs["on"],
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep{world_size} rank{rank} {mode} T={num_tokens} per_route_publish_pdl",
        )
        for name, output in outputs.items():
            assert torch.equal(output, outputs["off"]), (
                f"{name} differs at T={num_tokens}"
            )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "masked", "empty_rank"])
def test_ep_shared_workspace_round_paths(mode: str) -> None:
    """Layers of one config share one pooled workspace; each round takes the dispatch
    path of each rank's own token count (fused up to 512 tokens, grid-strided from 513),
    which differs between ranks and rounds (across that bound, an idle rank), and every
    layer stays bitwise equal to a layer with every round switch off."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    capacity = 640
    off = dict(
        pdl=0,
        pdl_late=False,
        fused_begin=0,
        fused_dispatch_max_tokens=128,
        grid_dispatch_min_tokens=0,
        compact_form=0,
        publish_form=0,
        per_route_publish_pdl=False,
        fc2_early_max_tokens=0,
        prefix_form=0,
    )
    ref, w13, w2 = _build_layer(
        world_size, rank, device, token_capacity=capacity, **off
    )
    layers = [
        _build_layer(world_size, rank, device, token_capacity=capacity)[0]
        for _ in range(3)
    ]
    # direct placement (its own pool key): the same bytes in every round
    direct, _w13, _w2 = _build_layer(
        world_size, rank, device, token_capacity=capacity, **DIRECT_PLACE
    )
    plans = (
        lambda r: 1,
        lambda r: 16 + r % 2,
        lambda r: 124 + 8 * (r % 2),
        lambda r: 0 if r == 0 else 128,
        lambda r: 512 + r % 2,
        lambda r: 640 - r,
    )
    for index, plan in enumerate(plans):
        num_tokens = plan(rank)
        x, ids, weights = _make_inputs(
            num_tokens,
            total_experts,
            131 + 7 * index + rank,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        reference = _forward(ref, x, ids, weights).clone()
        torch.cuda.synchronize()
        for layer_index, layer in enumerate([*layers, direct]):
            output = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
            assert torch.equal(output, reference), (
                f"layer {layer_index} differs at round {index} T={num_tokens}"
            )
        if num_tokens:
            _check(
                reference,
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep{world_size} rank{rank} {mode} round {index} T={num_tokens} shared",
            )
    runners = {id(_find_runner(layer)) for layer in layers}
    assert len(runners) == 1, "layers of one config must share one pooled runner"
    assert id(_find_runner(ref)) not in runners
    assert id(_find_runner(direct)) not in runners
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "masked", "empty_rank"])
def test_ep_fc2_early_straddle_bitwise(mode: str) -> None:
    """The FC2 early start is rank-local: in one round some ranks run the early-start
    pair (<= 128 local tokens) and others the regular pair (> 128 or no tokens), also on
    the per-route wire whose publish is a PDL launch (8-token pipes); every layer stays
    bitwise equal to a layer with the early start off."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    for capacity, common, plans in (
        (
            160,
            {},
            (
                lambda r: 128 + r % 2,
                lambda r: 127 + 2 * (r % 2),
                lambda r: 0 if r == 0 else 16,
                lambda r: 1 + r,
            ),
        ),
        (
            8,
            dict(
                combine_wire="per_route",
                pdl=6,
                pdl_late=True,
                per_route_publish_pdl=True,
            ),
            (lambda r: 8, lambda r: 1 + r % 8, lambda r: 0 if r == 0 else 8),
        ),
    ):
        ref, w13, w2 = _build_layer(
            world_size,
            rank,
            device,
            token_capacity=capacity,
            fc2_early_max_tokens=0,
            **common,
        )
        layer, _w13, _w2 = _build_layer(
            world_size,
            rank,
            device,
            token_capacity=capacity,
            fc2_early_max_tokens=128,
            **common,
        )
        runner = _warm_runner(layer, device)
        assert runner.fc2_early_max_tokens == 128
        if common:
            assert runner.combine_wire == "per_route" and runner.per_route_publish_pdl
        for index, plan in enumerate(plans):
            num_tokens = plan(rank)
            x, ids, weights = _make_inputs(
                num_tokens,
                total_experts,
                151 + 11 * index + rank,
                device,
                mode=mode,
                rank=rank,
                world_size=world_size,
            )
            reference = _forward(ref, x, ids, weights).clone()
            torch.cuda.synchronize()
            output = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
            assert torch.equal(output, reference), (
                f"capacity {capacity} round {index} T={num_tokens}"
            )
            if num_tokens:
                _check(
                    output,
                    x,
                    ids,
                    weights,
                    w13,
                    w2,
                    label=f" ep{world_size} rank{rank} {mode} cap {capacity} round {index} T={num_tokens} fc2_early",
                )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "masked", "empty_rank"])
def test_ep_dispatch_switches_straddle_bitwise(mode: str) -> None:
    """The prefix form is rank-local: in one round the ranks' token counts straddle the
    small-route bound (128 tokens at top_k 8) and the fused / grid-strided bound (512),
    and ranks mix the forms; every layer stays bitwise equal to prefix_form 0."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    capacity = 640
    # own pool key for the mixed layer on every rank (see test_ep_per_route_publish_pdl_bitwise)
    layers = {
        name: _build_layer(
            world_size,
            rank,
            device,
            token_capacity=capacity,
            **DISPATCH_PATHS,
            **kwargs,
        )
        for name, kwargs in (
            ("off", dict(prefix_form=0)),
            ("on", dict(prefix_form=1)),
            ("mixed", dict(prefix_form=rank % 2, init_timeout_s=601.0)),
            ("direct", dict(prefix_form=1, **DIRECT_PLACE)),
        )
    }
    _, w13, w2 = layers["off"]
    runner = _warm_runner(layers["on"][0], device)
    assert runner.prefix_form == 1
    plans = (
        lambda r: 128 + r % 2,
        lambda r: 127 + 2 * (r % 2),
        lambda r: 0 if r == 0 else 16,
        lambda r: 512 + r % 2,
        lambda r: 1 + r,
        lambda r: 640 - 3 * r,
    )
    for index, plan in enumerate(plans):
        num_tokens = plan(rank)
        x, ids, weights = _make_inputs(
            num_tokens,
            total_experts,
            191 + 13 * index + rank,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        outputs = {}
        for name, (layer, _w13, _w2) in layers.items():
            outputs[name] = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
        if num_tokens:
            _check(
                outputs["on"],
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep{world_size} rank{rank} {mode} round {index} T={num_tokens} dispatch switches",
            )
        for name, output in outputs.items():
            assert torch.equal(output, outputs["off"]), (
                f"{name} differs at round {index} T={num_tokens}"
            )
    dist.barrier()


def _all_gather_padded(t: torch.Tensor, rows: int, fill: int) -> torch.Tensor:
    """Every rank's ``t`` (host copy, padded to ``rows`` with ``fill``), stacked by rank."""
    import torch.distributed as dist

    padded = torch.full((rows, *t.shape[1:]), fill, dtype=t.dtype)
    padded[: t.shape[0]] = t.cpu()
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    return torch.stack(gathered)


def _direct_place_rounds(
    off, layer, w13, w2, plans, *, mode, seed, label, bound=lambda r: 513, top_k=TOP_K
) -> None:
    """Every round of ``plans`` through the inbox-path layer ``off`` and the
    direct-placement ``layer``: bitwise equal outputs and the expected direct flag."""
    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    for index, plan in enumerate(plans):
        num_tokens = plan(rank)
        x, ids, weights = _make_inputs(
            num_tokens,
            LOCAL_EXPERTS * world_size,
            seed + 7 * index + rank,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
            top_k=top_k,
        )
        reference = _forward(off, x, ids, weights).clone()
        torch.cuda.synchronize()
        output = _forward(layer, x, ids, weights).clone()
        torch.cuda.synchronize()
        runner = _find_runner(layer)
        assert runner.direct_place
        expected = _direct_expected(plan, world_size, bound)
        assert runner._direct_flag() == expected, (
            f"round {index} T={num_tokens}: flag {runner._direct_flag()} != {expected}"
        )
        assert torch.equal(output, reference), f"round {index} T={num_tokens}"
        if num_tokens:
            _check(
                output,
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep{world_size} rank{rank} {label} round {index} T={num_tokens}",
            )


@requires_dist
@pytest.mark.parametrize("wire", WIRES)
@pytest.mark.parametrize(
    "mode", ["random", "hot", "masked", "dup_rank", "dup_remote", "empty_rank"]
)
def test_ep_direct_place_bitwise(mode: str, wire: str) -> None:
    """A round is placed directly on every rank or on none (every rank dispatches it
    grid-strided, else every rank takes the inbox path), and both write the same bytes
    as the inbox-path layer."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    common = dict(combine_wire=wire, token_capacity=640, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    layer, _w13, _w2 = _build_layer(world_size, rank, device, **DIRECT_PLACE, **common)
    _direct_place_rounds(
        off,
        layer,
        w13,
        w2,
        DIRECT_PLACE_PLANS,
        mode=mode,
        seed=223,
        label=f"{wire} {mode} direct_place",
    )
    dist.barrier()


@requires_dist
def test_ep_direct_place_default(monkeypatch) -> None:
    """With no config value and no environment override, direct placement is
    requested at the EP sizes of DIRECT_PLACE_DEFAULTS with the bound raised to the
    routed-element floor (above this file's token capacities, so the default layer
    keeps the inbox path and the regular activation matrix) and not requested at the
    others.  With an explicit round bound the default request places grid rounds of
    at least that bound directly.  The bytes equal the inbox path."""
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm90.cake_bf16_megamoe.shim.cake_runner import (
        DIRECT_PLACE_DEFAULTS,
        DIRECT_PLACE_ENV,
        DIRECT_PLACE_MIN_ROUTE_ELEMS,
        DIRECT_PLACE_MIN_TOKENS_ENV,
    )

    monkeypatch.delenv(DIRECT_PLACE_ENV, raising=False)
    monkeypatch.delenv(DIRECT_PLACE_MIN_TOKENS_ENV, raising=False)
    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    common = dict(token_capacity=640, compact_form=1, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    on = world_size in DIRECT_PLACE_DEFAULTS
    floor = -(-DIRECT_PLACE_MIN_ROUTE_ELEMS // (TOP_K * HIDDEN))
    bound = max(DIRECT_PLACE_DEFAULTS[world_size], floor) if on else 513
    assert bound == (32768 if on else 513)
    # own pool keys (see test_ep_per_route_publish_pdl_bitwise)
    layer, _w13, _w2 = _build_layer(
        world_size, rank, device, init_timeout_s=613.0, **common
    )
    runner = _warm_runner(layer, device)
    assert runner.direct_place_min_tokens == bound
    assert runner.direct_place is False  # token capacity 640 < bound, or not requested
    assert runner._a1_symm is None
    explicit, _w13, _w2 = _build_layer(
        world_size,
        rank,
        device,
        init_timeout_s=617.0,
        direct_place_min_tokens=513,
        **common,
    )
    explicit_runner = _warm_runner(explicit, device)
    assert explicit_runner.direct_place_min_tokens == 513
    assert explicit_runner.direct_place is on
    assert (explicit_runner._a1_symm is not None) is on
    direct_rounds = 0
    for index, num_tokens in enumerate((512, 513, 600, 640)):
        x, ids, weights = _make_inputs(
            num_tokens,
            LOCAL_EXPERTS * world_size,
            281 + 7 * index + rank,
            device,
            rank=rank,
            world_size=world_size,
        )
        reference = _forward(off, x, ids, weights).clone()
        torch.cuda.synchronize()
        for name, model, model_runner, expected in (
            ("default", layer, runner, 0),
            ("explicit 513", explicit, explicit_runner, int(on and num_tokens >= 513)),
        ):
            output = _forward(model, x, ids, weights).clone()
            torch.cuda.synchronize()
            assert model_runner._direct_flag() == expected, (name, num_tokens)
            direct_rounds += expected
            assert torch.equal(output, reference), (name, num_tokens)
            _check(
                output,
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep{world_size} rank{rank} direct_place {name} T={num_tokens}",
            )
    assert direct_rounds == (3 if on else 0)
    dist.barrier()


@requires_dist
def test_ep_direct_place_graph_replay() -> None:
    """A captured direct-placement round replays with new inputs bitwise equal to its
    eager rounds and to the inbox path."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    common = dict(token_capacity=640, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    layer, _w13, _w2 = _build_layer(world_size, rank, device, **DIRECT_PLACE, **common)
    runner = _warm_runner(layer, device)
    inputs = [
        _make_inputs(
            600,
            total_experts,
            241 + 10 * index + rank,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        for index, mode in enumerate(("random", "hot", "dup_remote"))
    ]
    eager, reference = [], []
    for x, ids, weights in inputs:
        reference.append(_forward(off, x, ids, weights).clone())
        eager.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()
        assert runner._direct_flag() == 1
    static = [t.clone() for t in inputs[0]]
    for _ in range(2):
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, *static)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, *static)
    dist.barrier()
    replayed = []
    for tensors in inputs:
        for dst, src in zip(static, tensors, strict=True):
            dst.copy_(src)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        assert runner._direct_flag() == 1
        replayed.append(static_output.clone())
        dist.barrier()
    for index, (x, ids, weights) in enumerate(inputs):
        assert torch.equal(replayed[index], eager[index]), f"input {index}"
        assert torch.equal(eager[index], reference[index]), f"input {index}"
        _check(
            replayed[index],
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep{world_size} rank{rank} direct_place graph[{index}]",
        )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "dup_remote"])
def test_ep_direct_place_mixed_request(mode: str) -> None:
    """Direct placement is a cross-rank decision: a layer that only some ranks request
    takes the inbox path on every rank, and a round in which a rank stays below its own
    round bound takes the inbox path on every rank; the bytes never change."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    common = dict(token_capacity=640, compact_form=1, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    # own pool keys on every rank (see test_ep_per_route_publish_pdl_bitwise)
    mixed, _w13, _w2 = _build_layer(
        world_size,
        rank,
        device,
        direct_place=bool(rank % 2),
        direct_place_min_tokens=513,
        init_timeout_s=611.0,
        **common,
    )
    assert not _warm_runner(mixed, device).direct_place

    def bound(r: int) -> int:
        return 513 + 64 * (r % 2)

    bounded, _w13, _w2 = _build_layer(
        world_size,
        rank,
        device,
        direct_place=True,
        direct_place_min_tokens=bound(rank),
        init_timeout_s=612.0,
        **common,
    )
    runner = _warm_runner(bounded, device)
    assert runner.direct_place and runner.direct_place_min_tokens == bound(rank)
    plans = (
        lambda r: 560,
        lambda r: 600,
        lambda r: 576 + r % 2,
        lambda r: 0 if r == 0 else 600,
        lambda r: 640,
    )
    for index, plan in enumerate(plans):
        num_tokens = plan(rank)
        x, ids, weights = _make_inputs(
            num_tokens,
            LOCAL_EXPERTS * world_size,
            263 + 7 * index + rank,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        reference = _forward(off, x, ids, weights).clone()
        torch.cuda.synchronize()
        outputs = {}
        for name, layer in (("mixed", mixed), ("bounded", bounded)):
            outputs[name] = _forward(layer, x, ids, weights).clone()
            torch.cuda.synchronize()
        assert _find_runner(mixed)._direct_flag() == 0
        expected = _direct_expected(plan, world_size, bound)
        assert runner._direct_flag() == expected, f"round {index} T={num_tokens}"
        for name, output in outputs.items():
            assert torch.equal(output, reference), f"{name} round {index}"
        if num_tokens:
            _check(
                reference,
                x,
                ids,
                weights,
                w13,
                w2,
                label=f" ep{world_size} rank{rank} {mode} round {index} T={num_tokens} mixed",
            )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("prefix_form", [0, 1])
@pytest.mark.parametrize("fused_begin", [0, 1, 2])
def test_ep_direct_place_levels(fused_begin: int, prefix_form: int) -> None:
    """Direct placement at every round-opening level and receiver prefix form."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    common = dict(
        token_capacity=640,
        fused_begin=fused_begin,
        prefix_form=prefix_form,
        fused_dispatch_max_tokens=512,
        grid_dispatch_min_tokens=513,
    )
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    layer, _w13, _w2 = _build_layer(world_size, rank, device, **DIRECT_PLACE, **common)
    plans = (lambda r: 600, lambda r: 0 if r == 0 else 600, lambda r: 513 + 7 * r)
    for mode in ("random", "dup_remote"):
        _direct_place_rounds(
            off,
            layer,
            w13,
            w2,
            plans,
            mode=mode,
            seed=281,
            label=f"{mode} fused_begin={fused_begin} prefix_form={prefix_form}",
        )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("top_k", [1, 4, 8])
def test_ep_direct_place_top_k(top_k: int) -> None:
    """Direct placement at the other top_k instances of the grid-strided dispatch."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    if top_k > LOCAL_EXPERTS * world_size:
        pytest.skip(f"top_k {top_k} needs at least {top_k} experts")
    common = dict(token_capacity=640, top_k=top_k, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    layer, _w13, _w2 = _build_layer(world_size, rank, device, **DIRECT_PLACE, **common)
    for mode in ("random", "hot", "masked"):
        _direct_place_rounds(
            off,
            layer,
            w13,
            w2,
            (lambda r: 600, lambda r: 640 - r, lambda r: 512 + r % 2),
            mode=mode,
            seed=293,
            label=f"{mode} top_k={top_k}",
            top_k=top_k,
        )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "dup_rank", "empty_rank"])
def test_ep_direct_place_activation_rows(mode: str) -> None:
    """After a round, rows [0, m) of the activation matrix hold the tokens that the
    round's row records name (source rank, token, route), in the record's expert, for
    the direct placement as for the inbox path."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    capacity = 640
    common = dict(token_capacity=capacity, **DISPATCH_PATHS)
    layers = {
        "off": _build_layer(world_size, rank, device, direct_place=False, **common)[0],
        "direct": _build_layer(world_size, rank, device, **DIRECT_PLACE, **common)[0],
    }
    plans = (lambda r: 600, lambda r: 640 - r, lambda r: 512 + r % 2)
    for index, plan in enumerate(plans):
        num_tokens = plan(rank)
        x, ids, weights = _make_inputs(
            num_tokens,
            LOCAL_EXPERTS * world_size,
            307 + 7 * index + rank,
            device,
            mode=mode,
            rank=rank,
            world_size=world_size,
        )
        # bf16 rows as int32 pairs (gloo transport), unused rows / routes -1
        rows = _all_gather_padded(x.view(torch.int32), capacity, -1)
        routes = _all_gather_padded(ids, capacity, -1)
        owner = torch.where(routes >= 0, routes // LOCAL_EXPERTS, -1)
        for name, layer in layers.items():
            _forward(layer, x, ids, weights)
            torch.cuda.synchronize()
            runner = _find_runner(layer)
            m = int(runner.pipe._m_dev.item())
            assert m == int((owner == rank).sum()), f"{name} round {index}"
            meta = runner.meta[:m].cpu().long()
            src, tok, k = meta[:, 0], meta[:, 1], meta[:, 2]
            assert torch.equal(runner.a1[:m].view(torch.int32).cpu(), rows[src, tok]), (
                f"{name} round {index} T={num_tokens}"
            )
            assert torch.equal(
                runner.row_expert[:m].cpu().long(),
                routes[src, tok, k].long() - rank * LOCAL_EXPERTS,
            ), f"{name} round {index}"
            if name == "direct":
                assert runner._direct_flag() == _direct_expected(plan, world_size)
            # no rank starts the next round before every rank read its rows
            dist.barrier()
    dist.barrier()


@requires_dist
def test_ep_direct_place_capacity_boundary() -> None:
    """Hot routing fills rank 0's activation matrix exactly (m == m_cap) in a direct
    round; the bytes equal the inbox path."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    capacity = 640
    common = dict(token_capacity=capacity, **DISPATCH_PATHS)
    off, w13, w2 = _build_layer(world_size, rank, device, direct_place=False, **common)
    layer, _w13, _w2 = _build_layer(world_size, rank, device, **DIRECT_PLACE, **common)
    x, ids, weights = _make_inputs(
        capacity,
        LOCAL_EXPERTS * world_size,
        317 + rank,
        device,
        mode="hot",
        rank=rank,
        world_size=world_size,
    )
    reference = _forward(off, x, ids, weights).clone()
    torch.cuda.synchronize()
    output = _forward(layer, x, ids, weights).clone()
    torch.cuda.synchronize()
    runner = _find_runner(layer)
    assert runner._direct_flag() == 1
    m = int(runner.pipe._m_dev.item())
    assert m == (runner.pipe.m_cap if rank == 0 else 0), (m, runner.pipe.m_cap)
    assert torch.equal(output, reference)
    _check(output, x, ids, weights, w13, w2, label=f" ep{world_size} rank{rank} m_cap")
    dist.barrier()


@requires_dist
def test_ep_combine_wire_mismatch_raises() -> None:
    """The combine wire is a cross-rank contract: a mixed pipe must fail on every rank."""
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    wire = WIRES[-1] if rank == 0 else WIRES[0]
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY, LOCAL_EXPERTS * world_size, 23 + rank, device, mode="random"
    )
    with pytest.raises(Exception, match="combine_wire"):
        # A distinct max_tokens_per_rank gives this layer a fresh process-level
        # workspace-pool key on every rank: the pipe + runner (and with them the
        # cross-rank combine_wire handshake) are created lazily by the first
        # forward, and a pooled workspace of an earlier test would be reused
        # without any handshake.  Every rank raises together (guarded phase).
        mixed, _w13, _w2 = _build_layer(
            world_size,
            rank,
            device,
            combine_wire=wire,
            token_capacity=TOKEN_CAPACITY // 2,
            seed=23,
        )
        half = TOKEN_CAPACITY // 2
        _forward(mixed, x[:half], ids[:half], weights[:half])
        torch.cuda.synchronize()
    dist.barrier()
    # the process keeps working: a consistent pipe after the rejected one
    layer, w13, w2 = _build_layer(world_size, rank, device, seed=23)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _check(
        output,
        x,
        ids,
        weights,
        w13,
        w2,
        label=f" ep{world_size} rank{rank} after-mismatch",
    )
    dist.barrier()


@requires_dist
def test_ep_init_timeout_is_restored_after_workspace_setup() -> None:
    """``init_timeout_s`` must not leak into the caller's EP process group."""
    from datetime import timedelta

    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm90.bf16_bf16_bf16_push_cake.cake_backend import (
        _process_group_backend_timeouts,
    )

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    group = dist.group.WORLD
    before = [timeout for _backend, timeout in _process_group_backend_timeouts(group)]
    if not before:
        pytest.skip("this PyTorch build exposes no process-group timeout getter")
    init_timeout = timedelta(seconds=600.0)  # the config default
    if any(timeout == init_timeout for timeout in before):
        pytest.skip(
            "the EP group already runs at init_timeout_s; restore is unobservable"
        )
    layer, w13, w2 = _build_layer(world_size, rank, device, seed=97)
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY, LOCAL_EXPERTS * world_size, 97 + rank, device, mode="random"
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    _check(
        output, x, ids, weights, w13, w2, label=f" ep{world_size} rank{rank} timeout"
    )
    after = [timeout for _backend, timeout in _process_group_backend_timeouts(group)]
    assert after == before, f"EP group timeouts changed from {before} to {after}"
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("wire", WIRES)
def test_ep_uneven_tokens_and_recovery(wire: str) -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    layer, w13, w2 = _build_layer(world_size, rank, device, combine_wire=wire)
    # rank 0 sends nothing; the others use a token count that is not a tile multiple
    num_tokens = 0 if rank == 0 else max(TOKEN_CAPACITY - 13 * rank, 1)
    x, ids, weights = _make_inputs(
        num_tokens, total_experts, 71 + rank, device, rank=rank, world_size=world_size
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    assert output.shape == (num_tokens, HIDDEN)
    if num_tokens:
        _check(
            output,
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep{world_size} rank{rank} {wire} uneven",
        )
    x2, ids2, weights2 = _make_inputs(
        TOKEN_CAPACITY,
        total_experts,
        81 + rank,
        device,
        rank=rank,
        world_size=world_size,
    )
    output2 = _forward(layer, x2, ids2, weights2)
    torch.cuda.synchronize()
    _check(
        output2,
        x2,
        ids2,
        weights2,
        w13,
        w2,
        label=f" ep{world_size} rank{rank} {wire} recovery",
    )
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("wire", WIRES)
def test_ep_graph_replay(wire: str) -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    layer, w13, w2 = _build_layer(world_size, rank, device, combine_wire=wire)
    inputs = [
        _make_inputs(
            TOKEN_CAPACITY,
            total_experts,
            91 + 10 * index + rank,
            device,
            rank=rank,
            world_size=world_size,
        )
        for index in range(2)
    ]
    eager = []
    for x, ids, weights in inputs:
        eager.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()
    static_x = torch.empty_like(inputs[0][0]).copy_(inputs[0][0])
    static_ids = torch.empty_like(inputs[0][1]).copy_(inputs[0][1])
    static_weights = torch.empty_like(inputs[0][2]).copy_(inputs[0][2])
    for _ in range(2):
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, static_x, static_ids, static_weights)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, static_x, static_ids, static_weights)
    dist.barrier()
    replayed = []
    for x, ids, weights in inputs:
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        replayed.append(static_output.clone())
        dist.barrier()
    for index, (x, ids, weights) in enumerate(inputs):
        assert torch.equal(replayed[index], eager[index])
        _check(
            replayed[index],
            x,
            ids,
            weights,
            w13,
            w2,
            label=f" ep{world_size} rank{rank} {wire} graph[{index}]",
        )
    dist.barrier()
