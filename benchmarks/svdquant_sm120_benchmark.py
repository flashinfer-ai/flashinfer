"""Operands and measurements for the SM120 SVDQuant backend comparison."""

import math
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import ContextManager, Final, Literal, TypeAlias, TypedDict

import torch
from typing_extensions import assert_never

import flashinfer
from flashinfer import SfLayout, mm_nvfp4_svdquant, nvfp4_quantize, svdquant_linear
from flashinfer.autotune_cache import MeasurementPolicy
from flashinfer.autotuner import AutoTuner, _tactic_to_json


Backend = Literal["cute-dsl", "cutlass-sm120"]
Operation = Literal["gemm", "linear"]
TuningL2 = Literal["operator", "warm", "cold"]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
BACKENDS: Final[tuple[Backend, ...]] = ("cute-dsl", "cutlass-sm120")
REDUCTION_ELEMENTS: Final = 1 << 20


class Winner(TypedDict):
    cache_key: str
    runner: str
    tactic: JsonValue


def tuning_policy(mode: TuningL2) -> MeasurementPolicy | None:
    match mode:
        case "operator":
            return None
        case "warm" | "cold":
            return MeasurementPolicy(
                execution_mode="cuda_graph", cold_l2=mode == "cold"
            )
        case unexpected:
            assert_never(unexpected)


def tuning_context(
    policy: MeasurementPolicy | None, *, tuning: bool
) -> ContextManager[None]:
    if policy is None:
        return flashinfer.autotune(tuning)
    return flashinfer.autotune_v2(
        mode="tune" if tuning else "replay",
        persistent_cache=False,
        measurement_policy=policy,
    )


@contextmanager
def tuning_precision(
    *, repeat: int | None = None, replays: int | None = None
) -> Iterator[None]:
    """Scope benchmark precision across tuning and replay for both backends."""
    if repeat is None and replays is None:
        yield
        return
    tuner = AutoTuner.get()
    previous_repeat = tuner.repeat
    if repeat is not None:
        tuner.repeat = repeat
    try:
        with flashinfer.autotune(False, cuda_graph_profile_replays=replays):
            yield
    finally:
        tuner.repeat = previous_repeat


def snapshot_winners() -> list[Winner]:
    """Snapshot the active partition; legacy save_configs omits v2 winners."""
    return [
        Winner(
            cache_key=key.file_key,
            runner=key.runner_class_name,
            tactic=_tactic_to_json(tactic),
        )
        for key, (tactic, _) in AutoTuner.get()._winner_cache().items()
    ]


def graph_call_count(
    shape: tuple[int, int, int], requested: int, output_budget_mib: int
) -> int:
    # Bound separate linear outputs even if capture cannot reuse their memory.
    m, n, _ = shape
    return min(requested, max(1, (output_budget_mib << 20) // (2 * m * n)))


@dataclass(frozen=True, slots=True)
class Environment:
    timestamp_utc: str
    gpu: str
    gpu_uuid: str
    capability: tuple[int, int]
    sm_count: int
    total_memory_bytes: int
    driver: str
    torch: str
    cuda: str | None
    flashinfer: str


def environment() -> Environment:
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    capability = torch.cuda.get_device_capability()
    assert capability == (12, 0), f"SM120 is required, got {capability}"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    driver = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        )
        .splitlines()[0]
        .strip()
    )
    return Environment(
        datetime.now(timezone.utc).isoformat(),
        props.name,
        str(props.uuid),
        capability,
        props.multi_processor_count,
        props.total_memory,
        driver,
        str(torch.__version__),
        torch.version.cuda,
        flashinfer.__version__,
    )


@dataclass(frozen=True, slots=True)
class Case:
    x: torch.Tensor
    pqs: torch.Tensor
    global_sf: torch.Tensor
    xq: torch.Tensor
    x_sf: torch.Tensor
    wq: torch.Tensor
    w_sf: torch.Tensor
    alpha: torch.Tensor
    l2t: torch.Tensor
    l1: torch.Tensor
    down: torch.Tensor
    bias: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class Call:
    function: Callable[..., torch.Tensor]
    operands: tuple[torch.Tensor, ...]

    def __call__(self) -> torch.Tensor:
        return self.function(*self.operands)


@dataclass(frozen=True, slots=True)
class Accuracy:
    sqnr_db: float | None
    exact_match: bool


def _global_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Find the NVFP4 scale without materializing a full FP32 matrix."""
    flat = tensor.flatten()
    peak = torch.zeros((), dtype=torch.float32, device=tensor.device)
    for start in range(0, flat.numel(), REDUCTION_ELEMENTS):
        block = flat[start : start + REDUCTION_ELEMENTS]
        peak = torch.maximum(peak, block.float().abs().amax())
    return (2688.0 / peak).reshape(1)


def _quantize(
    tensor: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    packed, sf = nvfp4_quantize(
        tensor,
        scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
        backend="cute-dsl",
    )
    return packed.view(torch.uint8), sf.view(torch.uint8).reshape(-1)


def build_case(shape: tuple[int, int, int], seed: int, bias: bool) -> Case:
    m, n, k = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def random(rows: int, columns: int) -> torch.Tensor:
        return torch.randn(
            rows, columns, dtype=torch.bfloat16, device="cuda", generator=generator
        )

    x = random(m, k).mul_(k**-0.25)
    pqs = random(1, k).flatten().mul_(0.3).add_(1.0).abs_()
    smoothed = x * pqs
    global_sf = _global_scale(smoothed)
    xq, x_sf = _quantize(smoothed, global_sf)
    del smoothed
    weight = random(n, k).mul_(k**-0.25)
    weight_scale = _global_scale(weight)
    wq, w_sf = _quantize(weight, weight_scale)
    del weight
    alpha = (global_sf * weight_scale).reciprocal()
    l2t = (pqs[:, None] * random(32, k).mul_(k**-0.25).t()).contiguous()
    l1 = (random(n, 32).mul_(32**-0.25).float() / alpha).to(torch.bfloat16)
    down = torch.mm(x, l2t)
    bias_tensor = random(1, n).flatten() if bias else None
    return Case(
        x, pqs, global_sf, xq, x_sf, wq, w_sf, alpha, l2t, l1, down, bias_tensor
    )


def make_call(
    case: Case, operation: Operation, backend: Backend, enable_pdl: bool
) -> Call:
    operands: tuple[torch.Tensor, ...]
    match operation:
        case "gemm":
            output = torch.full(
                (case.x.shape[0], case.wq.shape[0]),
                float("nan"),
                dtype=torch.bfloat16,
                device=case.x.device,
            )
            operands = (
                case.xq,
                case.wq,
                case.x_sf,
                case.w_sf,
                case.alpha,
                case.down,
                case.l1,
            )
            function = partial(
                mm_nvfp4_svdquant, out=output, backend=backend, enable_pdl=enable_pdl
            )
        case "linear":
            operands = (
                case.x,
                case.wq,
                case.w_sf,
                case.alpha,
                case.pqs,
                case.l2t,
                case.l1,
                case.global_sf,
            )
            function = partial(svdquant_linear, backend=backend, enable_pdl=enable_pdl)
        case unexpected:
            assert_never(unexpected)
    if case.bias is not None:
        operands += (case.bias,)
    return Call(function, operands)


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor) -> Accuracy:
    """Check both outputs using bounded scratch space, including large model shapes."""
    assert reference.shape == candidate.shape
    signal = torch.zeros((), dtype=torch.float64, device=reference.device)
    noise = torch.zeros_like(signal)
    finite = torch.ones((), dtype=torch.bool, device=reference.device)
    reference_flat, candidate_flat = reference.flatten(), candidate.flatten()
    for start in range(0, reference.numel(), REDUCTION_ELEMENTS):
        ref = reference_flat[start : start + REDUCTION_ELEMENTS].float()
        got = candidate_flat[start : start + REDUCTION_ELEMENTS].float()
        finite &= torch.isfinite(ref).all() & torch.isfinite(got).all()
        signal += ref.square().sum(dtype=torch.float64)
        noise += (ref - got).square().sum(dtype=torch.float64)
    assert finite.item(), "non-finite backend output"
    noise_value = noise.item()
    if noise_value == 0:
        return Accuracy(sqnr_db=None, exact_match=True)
    signal_value = signal.item()
    sqnr = 10.0 * math.log10(signal_value / noise_value) if signal_value else -math.inf
    assert sqnr > 40.0, f"backend SQNR {sqnr:.2f} dB must exceed 40 dB"
    return Accuracy(sqnr_db=sqnr, exact_match=False)
