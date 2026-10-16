"""Compare complete fused and separate BF16 hyperconnection gate candidates.

Example: python benchmarks/experimental/b12x/benchmark_hyperconnection_gate.py
--tokens 128 512 4096 --output gate-times.json
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from b12x.gemm import hyperconnection_gate as gate
from b12x.gemm.hyperconnection_gate._tuning import TUNING
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.benchmark import measure_calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 6, 48, 128, 512, 4096]
    )
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--rank", type=int, default=320)
    parser.add_argument("--eviction-mib", type=int, default=48)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(2153)
    flush = torch.zeros(args.eviction_mib << 20, device="cuda", dtype=torch.uint8)
    records = []
    for rows in args.tokens:
        query = gate.Query(
            max_rows=rows,
            hidden_size=args.hidden_size,
            streams=args.streams,
            rank=args.rank,
        )
        options = dict(device="cuda", dtype=torch.bfloat16)
        source = torch.randn(rows, args.rank, **options) * 0.125
        weight = (
            torch.randn(args.streams * args.hidden_size, args.rank, **options) * 0.125
        )
        values = torch.randn(rows, args.streams * args.hidden_size, **options)
        output = torch.empty(rows, args.hidden_size, **options)
        workspace = torch.empty_like(values)
        reference = gate.reference(source, weight, values, streams=args.streams)
        calls, configs, requests = {}, {}, []
        with PreparationSession(
            device="cuda", autotune=False, compile_workers=0
        ) as session:
            for index, assignment in enumerate(
                TUNING.parameter_space(query, None).configurations()
            ):
                config = gate.Config(**dict(assignment))
                plan = gate.plan(query, override=config)
                name = f"candidate_{index}"
                configs[name] = asdict(config)

                def call(plan=plan):
                    return PreparedCall(
                        run=lambda: gate.run(
                            source,
                            weight,
                            values,
                            out=output,
                            workspace=workspace,
                            plan=plan,
                        ),
                        produce=lambda: None,
                        output=output,
                        owners=(source, weight, values, output, workspace, plan),
                    )

                requests.append(
                    plan.request(
                        name=name,
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(
                                source, weight, values, output, workspace
                            )
                        ),
                    )
                )
                calls[name] = call()
            session.prepare(tuple(requests))
            for call in calls.values():
                call.run()
                torch.testing.assert_close(output, reference, rtol=0.008, atol=0.002)
            measured = measure_calls(
                calls, samples=4, rounds=5, eviction=lambda: flush.add_(1)
            )
            fused = [
                name for name in calls if configs[name]["implementation"] == "fused"
            ]
            separate = [
                name for name in calls if configs[name]["implementation"] != "fused"
            ]
            best = {
                label: min(names, key=measured.latencies_us.get)
                for label, names in (("fused", fused), ("separate", separate))
            }
            row = {
                "query": asdict(query),
                "best": best,
                "configs": configs,
                "latencies_us": measured.latencies_us,
                "samples": [asdict(sample) for sample in measured.samples],
            }
            records.append(row)
            print(
                json.dumps(
                    {
                        "tokens": rows,
                        **{
                            label + "_us": measured.latencies_us[name]
                            for label, name in best.items()
                        },
                    }
                ),
                flush=True,
            )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "eviction_mib": args.eviction_mib,
                    "comparisons": records,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
