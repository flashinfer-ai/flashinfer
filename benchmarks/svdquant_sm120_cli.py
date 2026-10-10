"""Command-line options for the SM120 SVDQuant backend benchmark."""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES, ShapeCase
from svdquant_sm120_benchmark import Operation, TuningL2


DEFAULT_SHAPES: Final[tuple[ShapeCase, ...]] = (
    (64, 3072, 3072),
    (4096, 3072, 3072),
    (537, 5376, 7168),
)


@dataclass(frozen=True, slots=True)
class Options:
    shapes: tuple[ShapeCase, ...]
    operations: tuple[Operation, ...]
    output: Path | None
    seed: int
    trials: int
    graph_calls: int
    graph_output_budget_mib: int
    warmup_ms: int
    repeat_ms: int
    cold_l2: bool
    tuning_l2: TuningL2
    tuning_repeat: int | None
    tuning_replays: int | None
    inference_mode: bool
    bias: bool
    enable_pdl: bool


def parse_shape(text: str) -> ShapeCase:
    try:
        m, n, k = (int(value) for value in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "use MxNxK, for example 64x3072x3072"
        ) from error
    if min(m, n, k) <= 0 or n % 32 or k % 32:
        raise argparse.ArgumentTypeError(
            "M/N/K must be positive; N and K must be divisible by 32"
        )
    return m, n, k


def parse_options(
    argv: Sequence[str] | None = None, *, description: str | None = None
) -> Options:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--shape", action="append", type=parse_shape, help="repeat to select a subset"
    )
    selection.add_argument(
        "--all-shapes", action="store_true", help="run the shared 71-shape registry"
    )
    parser.add_argument(
        "--operation", choices=("gemm", "linear"), action="append", help="default: both"
    )
    parser.add_argument("--output", type=Path, help="JSONL file; default: stdout")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--graph-calls",
        type=int,
        default=100,
        help="calls per warm graph; cold mode uses one",
    )
    parser.add_argument(
        "--graph-output-budget-mib",
        type=int,
        default=256,
        help="cap output storage for warm graph batches",
    )
    parser.add_argument("--warmup-ms", type=int, default=30)
    parser.add_argument("--repeat-ms", type=int, default=150)
    parser.add_argument(
        "--cold-l2",
        action="store_true",
        help="time isolated one-call graph replays, flushing 2x L2 before each start event",
    )
    parser.add_argument(
        "--tuning-l2", choices=("operator", "warm", "cold"), default="operator"
    )
    parser.add_argument(
        "--tuning-repeat",
        type=int,
        help="override the autotuner's default profiling call count for both backends",
    )
    parser.add_argument(
        "--tuning-replays",
        type=int,
        help="override CUDA-graph profiling replays for both backends",
    )
    parser.add_argument(
        "--inference-mode", action="store_true", help="default: torch.no_grad"
    )
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument(
        "--enable-pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the SM120 public API default (on); --no-enable-pdl disables it",
    )
    args = parser.parse_args(argv)
    if any(
        value is not None and value <= 0
        for value in (args.tuning_repeat, args.tuning_replays)
    ):
        parser.error("tuning repeat and replay counts must be positive")
    if (
        min(
            args.trials,
            args.graph_calls,
            args.graph_output_budget_mib,
            args.warmup_ms,
            args.repeat_ms,
        )
        <= 0
    ):
        parser.error(
            "trial counts, graph counts, memory budget and timing durations must be positive"
        )
    shapes = (
        MODEL_SHAPE_CASES
        if args.all_shapes
        else tuple(dict.fromkeys(args.shape or DEFAULT_SHAPES))
    )
    return Options(
        shapes,
        tuple(dict.fromkeys(args.operation or ("gemm", "linear"))),
        args.output,
        args.seed,
        args.trials,
        args.graph_calls,
        args.graph_output_budget_mib,
        args.warmup_ms,
        args.repeat_ms,
        args.cold_l2,
        args.tuning_l2,
        args.tuning_repeat,
        args.tuning_replays,
        args.inference_mode,
        not args.no_bias,
        args.enable_pdl,
    )
