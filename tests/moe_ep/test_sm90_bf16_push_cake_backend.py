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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routing patterns: random | hot | all_remote | dup_rank | dup_remote | all_local | masked | empty_rank."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    logits = torch.randn(num_tokens, num_experts, generator=generator)
    local_start = rank * LOCAL_EXPERTS
    if mode == "hot":
        ids = torch.zeros(num_tokens, TOP_K, dtype=torch.int32)
    elif mode == "all_remote" and num_experts > LOCAL_EXPERTS:
        logits[:, local_start : local_start + LOCAL_EXPERTS] = float("-inf")
        ids = logits.topk(TOP_K, dim=1).indices.to(torch.int32)
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
        offs = torch.stack(
            [
                torch.randperm(LOCAL_EXPERTS, generator=generator)[:TOP_K]
                for _ in range(num_tokens)
            ]
        )
        ids = (base.unsqueeze(1) + offs).to(torch.int32)
    elif mode == "empty_rank" and world_size > 1:
        # no token anywhere routes to the last rank's experts
        logits[:, (world_size - 1) * LOCAL_EXPERTS :] = float("-inf")
        ids = logits.topk(TOP_K, dim=1).indices.to(torch.int32)
    else:
        ids = logits.topk(TOP_K, dim=1).indices.to(torch.int32)
    if mode == "masked" and num_tokens:
        ids[::3, 0] = -1
    weights = torch.rand(num_tokens, TOP_K, generator=generator) + 0.1
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
                top_k=TOP_K,
                capacity_factor=capacity_factor,
                dedup_dispatch=dedup_dispatch,
                clamp_limit=clamp_limit,
                combine_wire=combine_wire,
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
