#!/usr/bin/env python3
"""Benchmark dense Trellis split-K against the whole-tile schedule.

For each (K, N, M) the script prepares two ``gemm.trellis_linear`` plans on the
same weight and input:

* ``whole_tile``: the default declaration (one CTA reduces each tile over the full K);
* ``split_k``: ``allow_split_k=True``, whose default config splits K across
  CTAs when the whole-tile launch under-fills the GPU.

With ``--sweep`` it also times explicit ``split_k_slices`` overrides at decode
M, which shows how close the default slice count is to the best one.

Both Hadamard rotations are no-ops over a pre-filled rotated buffer, so each
timed call launches only the Trellis GEMM kernel. Times are CUPTI kernel
times with a cold L2 per iteration (``bench_gpu_time``), median in µs.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from b12x.gemm import trellis_linear
from b12x.gemm.trellis_linear._tuning import TUNING, TrellisConfig
from b12x.preparation import PreparedCall, PreparationSession
from b12x.preparation.types import require_prepared
from flashinfer.testing import bench_gpu_time

# (K, N) projections from the issue's GLM-5.3 EXL3 decode profile.
DEFAULT_SHAPES = ("6144x1024", "4096x6144", "2048x4096")
DEFAULT_M = (1, 2, 4, 8, 16, 32, 64, 128, 512, 2048)
SWEEP_SLICES = (2, 3, 4, 5, 6, 8, 12, 16)
SWEEP_MAX_M = 128


def _noop_hadamard(_source, _destination, _left_scale, _right_scale, _scale):
    """Leave the rotated buffer as filled so only the GEMM kernel launches."""


def _time_us(fn, args) -> float:
    times_ms = bench_gpu_time(
        fn,
        enable_cupti=True,
        cold_l2_cache=True,
        dry_run_time_ms=args.warmup_ms,
        repeat_time_ms=args.repeat_ms,
    )
    return statistics.median(times_ms) * 1000.0


class _Case:
    def __init__(self, size_k: int, size_n: int, bits: int, device: torch.device):
        self.size_k = size_k
        self.size_n = size_n
        self.device = device
        generator = torch.Generator(device=device).manual_seed(size_k * 31 + size_n)
        trellis = torch.randint(
            -32768,
            32767,
            (size_k // 16, size_n // 16, 16 * bits),
            dtype=torch.int16,
            device=device,
            generator=generator,
        )
        self.weight = trellis_linear.prepare_weight(
            trellis,
            torch.ones(size_k, dtype=torch.float16, device=device),
            torch.ones(size_n, dtype=torch.float16, device=device),
            mcg=torch.tensor(0xCBAC1FED, dtype=torch.uint32, device=device),
            params_dtype=torch.float16,
        )
        # packed_gemm_scratch_elements caps c_tmp at SMs * 4 * block_rows * 256
        # elements; block_rows <= 64 covers every dense Trellis config.
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.c_tmp = torch.empty(
            (sms * 4 * 64 * 256,), dtype=torch.float32, device=device
        )

    def buffers(self, m: int):
        x = (torch.randn((m, self.size_k), device=self.device) * 1.0e-2).to(
            torch.float16
        )
        rotated = x.clone()
        gemm_output = torch.empty(
            (m, self.size_n), dtype=torch.float16, device=self.device
        )
        output = torch.empty_like(gemm_output)
        return x, {
            "output": output,
            "gemm_output": gemm_output,
            "rotated_f16": rotated,
        }

    def plan(self, x, kwargs, *, allow_split_k: bool, override=None):
        query = trellis_linear.query_from_weight(
            self.weight,
            max_rows=int(x.shape[0]),
            input_dtype=x.dtype,
            output_mode="provided",
            c_tmp_mode="provided",
            hadamard_128=_noop_hadamard,
            gemm_output_provided=True,
            rotated_f16_provided=True,
            allow_split_k=allow_split_k,
        )
        return trellis_linear.plan(
            query,
            weight=self.weight,
            c_tmp=self.c_tmp,
            hadamard_128=_noop_hadamard,
            override=override,
        )


def _prepare(session, plan, x, kwargs, name: str):
    request = plan.request(
        name=name,
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(x, **kwargs)),
    )
    session.prepare((request,))
    return require_prepared(plan, "gemm.trellis_linear")


def _measure(case: _Case, m: int, args) -> list[dict]:
    device = case.device
    x, kwargs = case.buffers(m)
    rows = []
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        whole_plan = case.plan(x, kwargs, allow_split_k=False)
        whole_state = _prepare(session, whole_plan, x, kwargs, "whole_tile")

        def run_whole():
            trellis_linear.run(x, plan=whole_plan, **kwargs)

        run_whole()
        torch.cuda.synchronize(device)
        whole = kwargs["gemm_output"].clone()
        t_whole = _time_us(run_whole, args)

        split_plan = case.plan(x, kwargs, allow_split_k=True)
        split_state = _prepare(session, split_plan, x, kwargs, "split_k")

        def run_split():
            trellis_linear.run(x, plan=split_plan, **kwargs)

        run_split()
        torch.cuda.synchronize(device)
        split = kwargs["gemm_output"].clone()
        run_split()
        torch.cuda.synchronize(device)
        deterministic = bool(torch.equal(split, kwargs["gemm_output"]))
        t_split = _time_us(run_split, args)
        rel_err = float(
            (split.float() - whole.float()).norm()
            / whole.float().norm().clamp_min(1e-30)
        )
        kernel = whole_state.launch.kernel
        base = {
            "K": case.size_k,
            "N": case.size_n,
            "M": m,
            "block_rows": int(kernel.moe_block_size),
            "tile_k": int(kernel.tile_k),
            "tile_n": int(kernel.tile_n),
        }
        rows.append(
            {
                **base,
                "arm": "default",
                "split_k_slices": split_state.split_k_slices,
                "t_whole_tile_us": t_whole,
                "t_split_k_us": t_split,
                "speedup": t_whole / t_split,
                "rel_err_vs_whole_tile": rel_err,
                "split_k_bit_stable": deterministic,
            }
        )

        if not args.sweep or m > SWEEP_MAX_M:
            return rows
        for slices in SWEEP_SLICES:
            # Same tile geometry as the default plans; only the K split varies.
            override = TrellisConfig(
                backend="cutedsl",
                block_rows=base["block_rows"],
                tile_k=base["tile_k"],
                tile_n=base["tile_n"],
                split_k_slices=slices,
            )
            try:
                TUNING.validate_config(split_state.query, override, None)
            except ValueError:
                continue
            plan = case.plan(x, kwargs, allow_split_k=True, override=override)
            _prepare(session, plan, x, kwargs, f"split_k_{slices}")

            def run_override(plan=plan):
                trellis_linear.run(x, plan=plan, **kwargs)

            run_override()
            torch.cuda.synchronize(device)
            rows.append(
                {
                    **base,
                    "arm": "sweep",
                    "split_k_slices": slices,
                    "t_whole_tile_us": t_whole,
                    "t_split_k_us": _time_us(run_override, args),
                    "rel_err_vs_whole_tile": float(
                        (kwargs["gemm_output"].float() - whole.float()).norm()
                        / whole.float().norm().clamp_min(1e-30)
                    ),
                }
            )
            rows[-1]["speedup"] = t_whole / rows[-1]["t_split_k_us"]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", nargs="+", default=list(DEFAULT_SHAPES))
    parser.add_argument("--m", nargs="+", type=int, default=list(DEFAULT_M))
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--output", help="write all rows as JSON to this path")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        raise SystemExit(f"SM12x required, found sm_{major}{minor}")
    props = torch.cuda.get_device_properties(device)
    print(f"device: {props.name} sm_{major}{minor} SMs={props.multi_processor_count}")

    rows = []
    header = (
        f"{'K':>5} {'N':>5} {'M':>5} {'arm':>7} {'slices':>6} "
        f"{'whole us':>9} {'split us':>9} {'speedup':>7} {'rel_err':>8}"
    )
    print(header)
    for shape in args.shapes:
        size_k, size_n = (int(v) for v in shape.lower().split("x"))
        case = _Case(size_k, size_n, args.bits, device)
        for m in args.m:
            for row in _measure(case, m, args):
                rows.append(row)
                print(
                    f"{row['K']:>5} {row['N']:>5} {row['M']:>5} {row['arm']:>7} "
                    f"{str(row['split_k_slices']):>6} {row['t_whole_tile_us']:>9.2f} "
                    f"{row['t_split_k_us']:>9.2f} {row['speedup']:>7.2f} "
                    f"{row['rel_err_vs_whole_tile']:>8.1e}",
                    flush=True,
                )
        del case
        torch.cuda.empty_cache()

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(
                {
                    "device": props.name,
                    "sm": f"{major}{minor}",
                    "sms": props.multi_processor_count,
                    "bits": args.bits,
                    "rows": rows,
                },
                handle,
                indent=2,
            )


if __name__ == "__main__":
    main()
