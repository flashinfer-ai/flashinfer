import argparse
import sys

from routines.attention import parse_attention_args, run_attention_test
from routines.flashinferTest_utils import full_output_columns, output_column_dict
from routines.gemm import parse_gemm_args, run_gemm_test
from routines.moe import parse_moe_args, run_moe_test


def run_test(args):
    """
    Route & run a single FlashInfer test case with test routine.

    Args:
        args: Parsed command line arguments containing test configuration
    """

    ## Depending on routine type, route to corresponding test routine
    if args.routine in [
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
    ]:
        res = run_attention_test(args)
    elif args.routine in [
        "gemm_fp8_nt_groupwise",
        "group_gemm_fp8_nt_groupwise",
    ]:
        res = run_gemm_test(args)
    elif args.routine in [
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp8_per_tensor_scale_moe",
    ]:
        res = run_moe_test(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

    # Write results to output file if specified
    if args.output_path is not None:
        with open(args.output_path, "a") as fout:
            for cur_res in res:
                for key in output_column_dict["general"]:
                    cur_res[key] = getattr(args, key)

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
        choices=[
            "BatchDecodeWithPagedKVCacheWrapper",
            "BatchPrefillWithPagedKVCacheWrapper",
            "BatchPrefillWithRaggedKVCacheWrapper",
            "gemm_fp8_nt_groupwise",
            "group_gemm_fp8_nt_groupwise",
            "trtllm_fp4_block_scale_moe",
            "trtllm_fp8_block_scale_moe",
            "trtllm_fp8_per_tensor_scale_moe",
        ],
    )
    args, _ = parser.parse_known_args(line[:])

    parser.add_argument(
        "--no_cuda_graph",
        action="store_true",
        default=False,
        help="Disable CUDA graph to execute kernels outside of the graph.",
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

    ## Check routine and pass on to routine-specific argument parser
    if args.routine in [
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
    ]:
        args = parse_attention_args(line, parser)
    elif args.routine in [
        "gemm_fp8_nt_groupwise",
        "group_gemm_fp8_nt_groupwise",
    ]:
        args = parse_gemm_args(line, parser)
    elif args.routine in [
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp8_per_tensor_scale_moe",
    ]:
        args = parse_moe_args(line, parser)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")

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
