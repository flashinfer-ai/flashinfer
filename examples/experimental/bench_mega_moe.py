"""Cold-L2 CUPTI timing of complete prepared MoE submission scope on SM103a."""
import argparse
from importlib.metadata import version
import json
import statistics
import warnings

from mega_moe_inputs import make_smoke, make_grouped_l2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("source", "v3", "grouped-l2"), default="source")
    parser.add_argument("--precision", choices=("fp4", "fp8"), default="fp4")
    args = parser.parse_args()
    from cupti import cupti  # Required; do not silently time with CUDA events.
    if int(version("cupti-python").split(".")[0]) < 13:
        raise RuntimeError("CUPTI timing requires cupti-python >= 13")
    from flashinfer.testing.utils import bench_gpu_time_with_cupti
    if args.family == "grouped-l2":
        if args.precision != "fp4":
            parser.error("Grouped L2 uses packed FP4 weights")
        plan = make_grouped_l2()[0]
    else:
        plan = make_smoke(args.family, args.precision)[0]
    plan.run()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*Falling back to CUDA events for benchmarking.*")
        milliseconds = bench_gpu_time_with_cupti(plan.run, cold_l2_cache=True, aggregate_op=sum)
    print(json.dumps(dict(family=args.family, precision=args.precision, median_ms=statistics.median(milliseconds),
                         timing="CUPTI cold L2", scope="complete plan.run, including reset or scale repack")))


if __name__ == "__main__":
    main()
