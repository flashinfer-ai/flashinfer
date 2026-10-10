"""GPU regressions for the packed small-M producer and its mutable L2T input."""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial

import pytest
import torch

from flashinfer.autotuner import TunableRunner
from flashinfer.gemm import svdquant_sm120_cutlass as backend
from flashinfer.gemm.svdquant_sm120_routes import (
    SM120_FAMILY_SMALL_M,
    sm120_pack_tactic,
    sm120_producer_variants,
)


@pytest.fixture(autouse=True)
def sm120_device() -> Iterator[None]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    with torch.no_grad():
        yield


@dataclass(frozen=True, slots=True)
class _LinearCase:
    inputs: list[torch.Tensor]
    runner: TunableRunner


def _producer_pairs(m: int, k: int) -> list[tuple[int, int]]:
    """Require a packed counterpart for every admitted row-major tiling."""
    variants = sm120_producer_variants(m, k)
    pairs = []
    for old, (family, tiling, policy) in enumerate(variants):
        if family != SM120_FAMILY_SMALL_M:
            continue
        packed = (4, tiling, policy)
        assert packed in variants, (
            f"missing packed small-M producer for {(m, k, tiling)}"
        )
        pairs.append((old, variants.index(packed)))
    assert pairs, f"no small-M producers for {(m, k)}"
    return pairs


def _make_case(m: int, n: int, k: int, enable_pdl: bool = True) -> _LinearCase:
    """Use packed weights directly, without allocating dense weight baselines."""
    torch.manual_seed(20260914)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) / k**0.25
    pqs = (1 + 0.3 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).abs()
    global_scale = (448.0 * 6.0 / (x * pqs).float().abs().max()).reshape(1)
    l2t = (
        pqs[:, None] * torch.randn(k, 32, device="cuda", dtype=torch.bfloat16) / k**0.25
    ).contiguous()
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    weight_sf = torch.full(
        ((n + 127) // 128 * 128 * (k // 16),),
        0x38,
        device="cuda",
        dtype=torch.uint8,
    )
    alpha = (1.0 / global_scale).float()
    l1 = (
        torch.randn(n, 32, device="cuda", dtype=torch.bfloat16).float()
        / (32**0.25 * alpha)
    ).to(torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    module = backend.get_nvfp4_svdquant_sm120_module()
    assert module.nvfp4_svdquant_gemm_can_implement(m, n, k, 32, 80)
    workspace_bytes = max(
        backend.DEFAULT_WORKSPACE_SIZE,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, 80)),
    )
    inputs = [
        x,
        weight,
        weight_sf,
        alpha,
        pqs,
        l2t,
        l1,
        global_scale,
        bias,
        torch.empty(m, k // 2, device="cuda", dtype=torch.uint8),
        torch.empty(
            (m + 127) // 128 * 128 * (k // 16), device="cuda", dtype=torch.uint8
        ),
        torch.empty(m, 32, device="cuda", dtype=torch.bfloat16),
        torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
        torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8),
    ]
    return _LinearCase(inputs, backend._sm120_fused_linear_runner(enable_pdl, x.device))


def _independent_outputs(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    candidate = list(inputs)
    for index in range(9, 13):
        candidate[index] = torch.empty_like(inputs[index])
    return candidate


def _prefill(inputs: list[torch.Tensor], byte: int) -> None:
    inputs[9].fill_(byte)
    inputs[10].fill_(byte)
    inputs[11].fill_(float("nan"))
    inputs[12].fill_(float("nan"))


def _assert_outputs_match(
    reference: list[torch.Tensor], candidate: list[torch.Tensor], context: str
) -> None:
    for index, name in enumerate(("XQ", "padded SF", "LoRA-down", "linear"), start=9):
        assert torch.equal(
            reference[index].view(torch.uint8), candidate[index].view(torch.uint8)
        ), f"{context}: {name}"
    assert torch.isfinite(candidate[11]).all(), f"{context}: nonfinite LoRA-down"
    assert torch.isfinite(candidate[12]).all(), f"{context}: nonfinite linear"


@pytest.mark.parametrize(
    ("m", "n", "k"),
    (
        (17, 3072, 3072),
        (64, 3072, 3072),
        (537, 5376, 7168),
        (7800, 256, 5120),
        (1935, 256, 5376),
    ),
)
@pytest.mark.parametrize("enable_pdl", (False, True))
def test_packed_small_m_matches_every_row_major_producer(
    m: int, n: int, k: int, enable_pdl: bool
) -> None:
    pairs = _producer_pairs(m, k)
    case = _make_case(m, n, k, enable_pdl)
    reference = case.inputs
    candidate = _independent_outputs(reference)
    for old, packed in pairs:
        # Different prefills expose unwritten bytes, including all SF padding.
        _prefill(reference, 0xA5)
        _prefill(candidate, 0x5A)
        case.runner(inputs=reference, tactic=sm120_pack_tactic(80, old))
        case.runner(inputs=candidate, tactic=sm120_pack_tactic(80, packed))
        torch.cuda.synchronize()
        _assert_outputs_match(reference, candidate, f"producer {old} -> {packed}")


@pytest.mark.parametrize(
    "inference_graph", (False, True), ids=("eager", "inference-graph")
)
def test_packed_small_m_observes_l2t_mutations(inference_graph: bool) -> None:
    old, packed = _producer_pairs(64, 3072)[1]
    with torch.inference_mode(inference_graph):
        case = _make_case(64, 3072, 3072)
        reference = case.inputs
        candidate = _independent_outputs(reference)
        assert candidate[5].is_inference() == inference_graph
        packed_call = partial(
            case.runner, inputs=candidate, tactic=sm120_pack_tactic(80, packed)
        )
        old_call = partial(
            case.runner, inputs=reference, tactic=sm120_pack_tactic(80, old)
        )
        for _ in range(3):
            packed_call()
        torch.cuda.synchronize()
        replay = packed_call
        if inference_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                packed_call()
            replay = graph.replay

        previous_down = None
        for increment in (0.0, 0.015625, -0.03125):
            candidate[5].add_(increment)
            _prefill(reference, 0xA5)
            _prefill(candidate, 0x5A)
            replay()
            old_call()
            torch.cuda.synchronize()
            _assert_outputs_match(reference, candidate, f"L2T increment {increment}")
            if previous_down is not None:
                assert not torch.equal(reference[11], previous_down), (
                    "mutation had no effect"
                )
            previous_down = reference[11].clone()


def test_packed_small_m_pdl_graph_chains_two_mutable_linears() -> None:
    old, packed = _producer_pairs(64, 3072)[1]
    # Inference L2T packing becomes graph work, so replay sees fresh weights.
    with torch.inference_mode():
        first, second = (_make_case(64, 3072, 3072) for _ in range(2))
        reference = [first.inputs, second.inputs]
        candidate = [_independent_outputs(inputs) for inputs in reference]
        guards: list[torch.Tensor] = []
        for inputs in (*reference, *candidate):
            storage = torch.full(
                (3 * 64 * 3072,), -2048, device="cuda", dtype=torch.bfloat16
            )
            inputs[12] = storage[64 * 3072 : 2 * 64 * 3072].view(64, 3072)
            guards.extend((storage[: 64 * 3072], storage[2 * 64 * 3072 :]))
        reference[1][0] = reference[0][12]
        candidate[1][0] = candidate[0][12]
        assert reference[0][13].data_ptr() != reference[1][13].data_ptr()

        def run_chain(inputs: list[list[torch.Tensor]], producer: int) -> None:
            for operands in inputs:
                first.runner(inputs=operands, tactic=sm120_pack_tactic(80, producer))

        # Calibrate the second quantizer for the first linear's output range.
        first.runner(inputs=reference[0], tactic=sm120_pack_tactic(80, old))
        second.inputs[6].mul_(second.inputs[3])
        second.inputs[7].copy_(
            (
                2688.0 / (reference[0][12] * second.inputs[4]).float().abs().max()
            ).reshape(1)
        )
        second.inputs[3].copy_(second.inputs[7].reciprocal())
        second.inputs[6].div_(second.inputs[3])
        for _ in range(3):
            run_chain(candidate, packed)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_chain(candidate, packed)

        previous = None
        for replay, update_x in enumerate((True, False, True, False)):
            if update_x:
                first.inputs[0].copy_(torch.randn_like(first.inputs[0]) / 3072**0.25)
            else:
                for inputs in reference:
                    inputs[5].copy_(
                        inputs[4][:, None] * torch.randn_like(inputs[5]) / 3072**0.25
                    )
            for inputs in reference:
                _prefill(inputs, 0xA5)
            for inputs in candidate:
                _prefill(inputs, 0x5A)
            graph.replay()
            run_chain(reference, old)
            torch.cuda.synchronize()
            for stage, (expected, actual) in enumerate(
                zip(reference, candidate, strict=True)
            ):
                _assert_outputs_match(
                    expected, actual, f"replay {replay}, linear {stage}"
                )
                if previous is not None:
                    assert not torch.equal(expected[12], previous[stage]), (
                        "chain mutation had no effect"
                    )
            for guard in guards:
                assert bool((guard == -2048).all()), "linear output guard changed"
            previous = [inputs[12].clone() for inputs in reference]
