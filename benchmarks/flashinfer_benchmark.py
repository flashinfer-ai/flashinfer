import argparse
import sys

# Only import utilities at module level - routine modules are imported lazily
# to avoid loading unnecessary dependencies (e.g., mpi4py for non-MPI benchmarks)
from routines.flashinfer_benchmark_utils import (
    benchmark_apis,
    full_output_columns,
    output_column_dict,
)


def run_test(args):
    """
    Route & run a single FlashInfer test case with test routine.

    Args:
        args: Parsed command line arguments containing test configuration
    """

    ## Depending on routine type, route to corresponding test routine
    ## Imports are done lazily to avoid loading unnecessary dependencies
    if args.routine in benchmark_apis["attention"]:
        from routines.attention import run_attention_test

        res = run_attention_test(args)
    elif args.routine in benchmark_apis["gemm"]:
        from routines.gemm import run_gemm_test

        res = run_gemm_test(args)
    elif args.routine in benchmark_apis["moe"]:
        from routines.moe import run_moe_test

        res = run_moe_test(args)
    elif args.routine in benchmark_apis["moe_comm"]:
        from routines.moe_comm import run_moe_comm_test

        res = run_moe_comm_test(args)
    elif args.routine in benchmark_apis["norm"]:
        from routines.norm import run_norm_test

        res = run_norm_test(args)
    elif args.routine in benchmark_apis["quantization"]:
        from routines.quantization import run_quantization_test

        res = run_quantization_test(args)
    elif args.routine in benchmark_apis["sampling"]:
        from routines.sampling import run_sampling_test

        res = run_sampling_test(args)
    elif args.routine in benchmark_apis["rope"]:
        from routines.rope import run_rope_test

        res = run_rope_test(args)
    elif args.routine in benchmark_apis["mamba"]:
        from routines.mamba import run_mamba_test

        res = run_mamba_test(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

    # Write results to output file if specified
    if args.output_path is not None:
        with open(args.output_path, "a") as fout:
            for cur_res in res:
                for key in output_column_dict["general"]:
                    # Only set from args if the routine hasn't already set a value
                    # This preserves routine-specific formatting while providing defaults
                    if key not in cur_res or cur_res[key] == "":
                        cur_res[key] = getattr(args, key, "")

                output_line = ",".join(
                    [str(cur_res[col]) for col in full_output_columns]
                )
                fout.write(output_line + "\n")
            fout.flush()
    return


def parse_args(line=sys.argv[1:]):
    """
    Parse command line arguments for test configuration.
    First parse shared arguments, then parse routine-specific arguments.

    Args:
        line: Command line arguments (default: sys.argv[1:])

    Returns:
        Parsed argument namespace
    """

    ## Shared arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routine",
        "-R",
        type=str,
        required=True,
        choices=list(benchmark_apis["attention"])
        + list(benchmark_apis["gemm"])
        + list(benchmark_apis["moe"])
        + list(benchmark_apis["moe_comm"])
        + list(benchmark_apis["norm"])
        + list(benchmark_apis["quantization"])
        + list(benchmark_apis["sampling"])
        + list(benchmark_apis["rope"])
        + list(benchmark_apis["mamba"]),
    )
    args, _ = parser.parse_known_args(line[:])

    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        default=False,
        help="Disable CUDA graph to execute kernels outside of the graph.",
    )
    parser.add_argument(
        "--use_cupti",
        action="store_true",
        default=False,
        help="[DEPRECATED] Use CUPTI for timing GPU kernels. This is now the default behavior.",
    )
    parser.add_argument(
        "--use_cuda_events",
        action="store_true",
        default=False,
        help="Use CUDA events for timing GPU kernels instead of CUPTI.",
    )
    parser.add_argument(
        "--refcheck",
        action="store_true",
        default=False,
        help="Run reference check that ensures outputs correct.",
    )
    parser.add_argument(
        "--allow_output_mismatch",
        action="store_true",
        default=False,
        help="Allow output mismatch between backends during reference checks. Error message will be printed but test will continue.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--verbose", "-v", action="count", help="Set verbosity level.", default=0
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Output path for results. If not specified, results will not be written to a file.",
    )
    parser.add_argument(
        "--num_iters",
        "-n",
        type=int,
        required=False,
        default=30,
        help="Number of iterations to run for measurement.",
    )
    parser.add_argument(
        "--dry_run_iters",
        "-d",
        type=int,
        required=False,
        default=5,
        help="Number of dry runs.",
    )
    parser.add_argument(
        "--case_tag",
        type=str,
        required=False,
        default=None,
        help="Optional tag for the test case for annotating output.",
    )
    parser.add_argument(
        "--generate_repro_command",
        action="store_true",
        default=False,
        help="If set, will print reproducer command and store it to output csv.",
    )
    parser.add_argument(
        "--repro_command",
        type=str,
        required=False,
        default="",
        help="Placeholder for generated reproducer command for the test case. Not to be used directly.",
    )

    ## Check routine and pass on to routine-specific argument parser
    ## Imports are done lazily to avoid loading unnecessary dependencies
    if args.routine in benchmark_apis["attention"]:
        from routines.attention import parse_attention_args

        args = parse_attention_args(line, parser)
    elif args.routine in benchmark_apis["gemm"]:
        from routines.gemm import parse_gemm_args

        args = parse_gemm_args(line, parser)
    elif args.routine in benchmark_apis["moe"]:
        from routines.moe import parse_moe_args

        args = parse_moe_args(line, parser)
    elif args.routine in benchmark_apis["moe_comm"]:
        from routines.moe_comm import parse_moe_comm_args

        args = parse_moe_comm_args(line, parser)
    elif args.routine in benchmark_apis["norm"]:
        from routines.norm import parse_norm_args

        args = parse_norm_args(line, parser)
    elif args.routine in benchmark_apis["quantization"]:
        from routines.quantization import parse_quantization_args

        args = parse_quantization_args(line, parser)
    elif args.routine in benchmark_apis["sampling"]:
        from routines.sampling import parse_sampling_args

        args = parse_sampling_args(line, parser)
    elif args.routine in benchmark_apis["rope"]:
        from routines.rope import parse_rope_args

        args = parse_rope_args(line, parser)
    elif args.routine in benchmark_apis["mamba"]:
        from routines.mamba import parse_mamba_args

        args = parse_mamba_args(line, parser)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

    if args.generate_repro_command:
        args.repro_command = "python3 flashinfer_benchmark.py " + " ".join(line)

    # Deprecation warning for use_cupti
    if args.use_cupti:
        print(
            "[WARNING] --use_cupti is deprecated and will be removed in a future release. CUPTI is now enabled by default."
        )
    # use_cupti is deprecated and will be removed in a future release. CUPTI is now enabled by default.
    # If --use_cuda_events is passed, disable use_cupti
    args.use_cupti = not args.use_cuda_events

    return args


if __name__ == "__main__":
    # Parse testlist argument first
    testlist_parser = argparse.ArgumentParser(add_help=False)
    testlist_parser.add_argument(
        "--testlist",
        type=str,
        required=False,
        default=None,
        help="Optional testlist file to run multiple cases.",
    )
    testlist_parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="Output path for results csv.",
    )
    testlist_args, _ = testlist_parser.parse_known_args()

    # Setup output file if specified
    if testlist_args.output_path is not None:
        with open(testlist_args.output_path, "w") as fout:
            fout.write(",".join(full_output_columns) + "\n")

    # Process tests either from testlist file or command line arguments
    if testlist_args.testlist is not None:
        # If testlist, run each test in the testlist
        with open(testlist_args.testlist, "r") as f:
            import shlex

            for line in f.readlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    line_args = parse_args(shlex.split(line))
                    line_args.output_path = testlist_args.output_path
                    run_test(line_args)
                except Exception as e:
                    print(f"[ERROR] Error running test: {line}")
                    print(f"[ERROR] Error: {e}")
                    continue
    else:
        # If no testlist, just run the command
        args = parse_args()
        args.output_path = testlist_args.output_path
        run_test(args)
