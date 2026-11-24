# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""ReplaySSM Markov SSU comparison with whole-course paired statistics.

Example (normal editable checkouts with CUDA13/CUPTI dependencies installed)::

    python benchmarks/bench_checkpointing_ssu_markov.py --state-dtype fp8 \
        --baseline-checkout ../flashinfer-baseline --output fp8.json

B1, H128/G8/D64/N128 and W16 are fixed to the Bolt fixture. By default,
--draft-length 7 --mean-acceptance 4.3 fits a synthetic acceptance law, including
the verifier token. Use --acceptance-csv instead for a measured histogram.
Use --units 2 --draws 8 for a runnable smoke check, not a precision claim.
"""

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from ssu_markov import (
    Metadata,
    fit_acceptance,
    load_acceptance_csv,
    stationary_workload,
    paired_pass_iqm,
)

ARMS = ("A0", "A1", "B0", "B1")
LATIN = tuple(ARMS[i:] + ARMS[:i] for i in range(4))
DTYPES = {"fp16": "float16", "fp8": "float8_e4m3fn", "int8": "int8"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_spec(args):
    from flashinfer.jit.mamba.checkpointing_ssu import gen_checkpointing_ssu_module

    kwargs = {}
    if "sr_rng" in inspect.signature(gen_checkpointing_ssu_module).parameters:
        kwargs["sr_rng"] = args.sr_rng
    elif args.sr_rng != "philox":
        raise ValueError("Reference checkout does not support the requested RNG policy")
    return gen_checkpointing_ssu_module(
        getattr(torch, DTYPES[args.state_dtype]),
        torch.bfloat16,
        getattr(torch, args.dt_dtype),
        getattr(torch, args.weight_dtype),
        torch.float32,
        torch.int32,
        None if args.state_dtype == "fp16" else torch.float32,
        64,
        128,
        args.tokens,
        16,
        16,
        8,
        args.philox_rounds,
        False,
        **kwargs,
    )


def module_record(spec, checkout):
    import flashinfer
    from flashinfer.jit import env

    require(
        Path(flashinfer.__file__).resolve().is_relative_to(checkout),
        "wrong Python source",
    )
    require(
        env.FLASHINFER_INCLUDE_DIR.samefile(checkout / "include"),
        "wrong kernel headers",
    )
    require(
        env.FLASHINFER_CSRC_DIR.samefile(checkout / "csrc"), "wrong binding sources"
    )
    sources = list(
        (checkout / "include/flashinfer/mamba").glob("*checkpointing_ssu*.cuh")
    )
    sources += [
        checkout / "include/flashinfer/mamba" / n
        for n in ("conversion.cuh", "stochastic_rounding.cuh")
    ]
    sources += list((checkout / "csrc").glob("checkpointing_ssu*"))
    sources += [
        checkout / "flashinfer/jit/mamba/checkpointing_ssu.py",
        checkout / "flashinfer/mamba/checkpointing_ssu.py",
        *spec.sources,
    ]
    sources += list(spec.sources[0].parent.glob("*.inc"))
    ninja = spec.ninja_path.read_text()
    require(
        any(
            str(p) in ninja
            for p in (env.FLASHINFER_INCLUDE_DIR, env.FLASHINFER_INCLUDE_DIR.resolve())
        ),
        "Ninja does not use reference headers",
    )
    library = spec.get_library_path()
    return dict(
        library=str(library),
        sha256=sha(library),
        name=spec.name,
        source=flashinfer.__file__,
        source_hashes={str(p): sha(p) for p in sources if p.is_file()},
        include_dirs=[str(env.FLASHINFER_INCLUDE_DIR.resolve())]
        + [str(Path(p).resolve()) for p in (spec.extra_include_dirs or [])],
        ninja_sha256=sha(spec.ninja_path),
        ninja=str(spec.ninja_path),
    )


def build_module(args):
    sys.path.insert(0, str(args.checkout.resolve()))
    global torch
    import torch

    spec = copy.copy(module_spec(args))
    spec.name += "_markov_reference"
    spec.build(need_lock=True)
    args.module_record.write_text(
        json.dumps(module_record(spec, args.checkout.resolve()), indent=2) + "\n"
    )


class Fixture:
    def __init__(self, inputs, baseline, args, public_call):
        self.inputs, self.args = inputs, args
        self.graphs = {}
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self.baseline = tvm_ffi.load_module(baseline["library"])
        self.seed = torch.empty(1, dtype=torch.int64, device=inputs.x.device)
        self.public_call = public_call
        self.output = inputs.out_incr

    def prepare(self, rows, seed):
        self.inputs.reset()
        self.inputs.ring_start.fill_(rows[0].ring_start)
        self.inputs.prev_tokens_i32.fill_(rows[0].history)
        self.seed.copy_(seed)

    def invoke(self, role):
        p = self.inputs
        seed = (
            self.seed
            if self.args.philox_rounds or self.args.sr_rng == "shared_seed"
            else None
        )
        if role == "new":
            self.public_call(
                p.state_work,
                p.x_cache_work,
                p.B_cache_work,
                p.dt_cache_work,
                p.ring_start,
                p.prev_tokens_i32,
                p.x,
                p.dt,
                p.A,
                p.B,
                p.C,
                p.out_incr,
                D=p.D,
                dt_bias=p.dt_bias,
                dt_softplus=True,
                state_scale=p.state_scale_work,
                rand_seed=seed,
                philox_rounds=self.args.philox_rounds,
                sr_rng=self.args.sr_rng,
                algorithm=self.args.algorithm,
                d_split=1,
                enable_pdl=False,
                cb_scaled=p.cb_scaled,
                cumAdt_vec=p.cumAdt_vec,
                cb_old=p.cb_old,
            )
        else:
            with tvm_ffi.use_torch_stream():
                self.baseline.checkpointing_ssu(
                    p.state_work,
                    p.x,
                    p.dt,
                    p.A,
                    p.B,
                    p.C,
                    p.out_incr,
                    p.x_cache_work,
                    p.B_cache_work,
                    p.dt_cache_work,
                    p.ring_start,
                    p.prev_tokens_i32,
                    p.D,
                    None,
                    p.dt_bias,
                    True,
                    None,
                    -1,
                    p.state_scale_work,
                    seed,
                    1,
                    None,
                    p.cb_scaled,
                    p.cumAdt_vec,
                    p.cb_old,
                    0,
                    0,
                    0,
                )

    def capture(self, name, rows, seed):
        role = "old" if name.startswith("A") else "new"
        for _ in range(3):
            self.prepare(rows, seed)
            self.invoke(role)
        self.stream.synchronize()
        self.prepare(rows, seed)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self.invoke(role)
        self.graphs[name] = graph

    def close(self):
        self.stream.synchronize()
        for graph in self.graphs.values():
            graph.reset()
        self.graphs.clear()


def summarize(units, order_index=None):
    pairs = {key: [] for key in ("AB", "AA", "BB")}
    arm_means = {arm: [] for arm in ARMS}
    for index, unit in enumerate(units):
        require(len(unit) == 4, "Each unit must contain four complete courses")
        courses = unit if order_index is None else [unit[order_index]]
        means = {arm: statistics.fmean(c[arm] for c in courses) for arm in ARMS}
        for arm in ARMS:
            arm_means[arm].append(means[arm])
        for key, left, right in (
            ("AB", ("A0", "A1"), ("B0", "B1")),
            ("AA", ("A0",), ("A1",)),
            ("BB", ("B0",), ("B1",)),
        ):
            a, b = (
                statistics.fmean(means[arm] for arm in group) for group in (left, right)
            )
            pairs[key].append(
                dict(round=index, us_saved=a - b, fractional_improvement=(a - b) / a)
            )
    effects = {key: paired_pass_iqm(value) for key, value in pairs.items()}
    controls_pass = all(
        abs(effects[key]["iqm_us_saved"]) <= 0.20
        and effects[key]["conservative_bootstrap_se_us"] <= 0.10
        for key in ("AA", "BB")
    )
    return dict(
        arm_means_us={k: statistics.fmean(v) for k, v in arm_means.items()},
        effects=effects,
        controls_pass=controls_pass,
        control_limits_us=dict(absolute_bias=0.20, standard_error=0.10),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dtype", choices=DTYPES, required=True)
    parser.add_argument(
        "--algorithm", choices=("monolith", "two-kernel"), default="monolith"
    )
    parser.add_argument("--baseline-checkout", type=Path)
    parser.add_argument(
        "--reference-workspace",
        type=Path,
        help="Reuse reference JIT builds across runs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acceptance-csv", type=Path)
    parser.add_argument("--draft-length", type=int, default=None)
    parser.add_argument("--mean-acceptance", type=float, default=None)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7601)
    parser.add_argument(
        "--rest-ms",
        type=float,
        default=0.0,
        help="Untimed rest between samples; 0 models a busy GPU",
    )
    parser.add_argument("--philox-rounds", type=int, default=5)
    parser.add_argument("--sr-rng", choices=("philox", "shared_seed"), default="philox")
    parser.add_argument("--build-module", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--module-record", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--tokens", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--dt-dtype", default="float32", help=argparse.SUPPRESS)
    parser.add_argument("--weight-dtype", default="bfloat16", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.build_module:
        build_module(args)
        return
    if args.units < 2 or args.units % 2 or args.draws < 1:
        parser.error("Use a positive even unit count and positive draw count")
    if args.state_dtype != "fp16" and args.algorithm == "two-kernel":
        parser.error("8-bit state supports only the monolithic path")
    require(not args.output.exists(), "Use a fresh output path")
    if not 0 <= args.rest_ms < float("inf"):
        parser.error("--rest-ms must be finite and nonnegative")
    if args.philox_rounds < 0:
        parser.error("--philox-rounds must be nonnegative")
    if args.acceptance_csv:
        if args.draft_length is not None or args.mean_acceptance is not None:
            parser.error(
                "Use either --acceptance-csv or the conditional-acceptance model"
            )
        acceptance = load_acceptance_csv(args.acceptance_csv)
        acceptance_model = dict(
            kind="CSV", path=str(args.acceptance_csv), sha256=sha(args.acceptance_csv)
        )
    else:
        args.draft_length = 7 if args.draft_length is None else args.draft_length
        args.mean_acceptance = (
            4.3 if args.mean_acceptance is None else args.mean_acceptance
        )
        try:
            p, acceptance = fit_acceptance(args.draft_length, args.mean_acceptance)
        except ValueError as error:
            parser.error(str(error))
        acceptance_model = dict(
            kind="synthetic constant conditional acceptance",
            draft_length=args.draft_length,
            requested_mean=args.mean_acceptance,
            conditional_p=p,
        )
    args.tokens = len(acceptance)
    require(1 <= args.tokens <= 8, "This benchmark supports 1 <= T <= 8")
    law = stationary_workload(acceptance, 16)
    directory = args.output.with_suffix(".data")
    directory.mkdir(parents=True, exist_ok=False)
    candidate = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(candidate))
    global torch, tvm_ffi
    import torch
    import tvm_ffi
    from cupti import cupti  # noqa: F401
    import bench_checkpointing_ssu as timing

    capability = torch.cuda.get_device_capability()
    inputs = timing.build_kernel_inputs(
        batch=1,
        mtp_len=args.tokens,
        max_window=16,
        state_dtype=getattr(torch, DTYPES[args.state_dtype]),
        act_dtype=torch.bfloat16,
        nheads=128,
        head_dim=64,
        d_state=128,
        ngroups=8,
        two_kernel=args.algorithm == "two-kernel",
    )
    args.dt_dtype = str(inputs.dt.dtype).removeprefix("torch.")
    args.weight_dtype = str(inputs.D.dtype).removeprefix("torch.")
    checkout = (args.baseline_checkout or candidate).resolve()
    record = directory / "reference_module.json"
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(checkout),
        CUDA_VISIBLE_DEVICES="",
        FLASHINFER_CUDA_ARCH_LIST=f"{capability[0]}.{capability[1]}a",
        FLASHINFER_WORKSPACE_BASE=str(
            (args.reference_workspace or directory / "reference").resolve()
        ),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--build-module",
        "--checkout",
        str(checkout),
        "--module-record",
        str(record.resolve()),
        "--state-dtype",
        args.state_dtype,
        "--output",
        str(args.output),
        "--sr-rng",
        args.sr_rng,
        "--philox-rounds",
        str(args.philox_rounds),
        "--tokens",
        str(args.tokens),
        "--dt-dtype",
        args.dt_dtype,
        "--weight-dtype",
        args.weight_dtype,
    ]
    subprocess.run(command, env=env, check=True)
    modules = {"old": json.loads(record.read_text())}
    result = dict(
        workload=dict(
            acceptance_model=acceptance_model,
            tokens=args.tokens,
            acceptance=acceptance,
            mean=law.mean_acceptance,
            flush_share=law.flush_share,
            balance_residual=law.balance_residual,
        ),
        benchmark_source_hashes={
            str(p): sha(p)
            for p in (
                Path(__file__).resolve(),
                Path(__file__).with_name("ssu_markov.py").resolve(),
                Path(timing.__file__).resolve(),
            )
        },
        protocol=dict(
            state_dtype=args.state_dtype,
            algorithm=args.algorithm,
            d_split=1,
            philox_rounds=args.philox_rounds,
            sr_rng=args.sr_rng,
            external_pdl=False,
            launch_environment={
                k: v for k, v in os.environ.items() if k.startswith("FLASHINFER_SSU_")
            },
            units=args.units,
            courses=4,
            draws=args.draws,
            palindrome_visits=8,
            conditioner_bytes=48 * 1024 * 1024,
            seed=args.seed,
            host_rest_ms=args.rest_ms,
            metric="enclosing captured SSU kernel span; resets/seeds/conditioner excluded",
        ),
        modules=modules,
        units=[],
        complete=False,
        device=dict(
            name=torch.cuda.get_device_name(),
            capability=capability,
            sms=torch.cuda.get_device_properties(0).multi_processor_count,
        ),
        comparison="source comparison"
        if args.baseline_checkout
        else "same-source control",
    )
    fixture = Fixture(inputs, modules["old"], args, timing.cuda_checkpointing_ssu)
    try:
        with torch.cuda.stream(fixture.stream), torch.inference_mode():
            seed = torch.tensor([0x95D65A39], dtype=torch.int64, device="cuda")
            for history in range(17):
                reference = None
                for role in ("old", "new"):
                    fixture.prepare(
                        [Metadata(history, args.tokens, 16 + args.tokens - 1)], seed
                    )
                    fixture.invoke(role)
                    fixture.stream.synchronize()
                    require(
                        bool(torch.isfinite(fixture.output).all()), "nonfinite output"
                    )
                    if reference is None:
                        reference = fixture.output.clone()
                    else:
                        torch.testing.assert_close(
                            fixture.output, reference, atol=0.002, rtol=0.02
                        )
            modules["new"] = module_record(module_spec(args), candidate)
            require(
                modules["new"]["name"] != modules["old"]["name"], "module URI collision"
            )
            result["fixture"] = dict(
                layout="standard benchmark tensor layout",
                dt_dtype=args.dt_dtype,
                weight_dtype=args.weight_dtype,
                x_stride=inputs.x.stride(),
                state_stride=inputs.state_work.stride(),
            )
            generator = torch.Generator(device="cuda").manual_seed(args.seed + 69017)
            conditioner = torch.randn(
                48 * 1024 * 1024 // 4, device="cuda", generator=generator
            )
            sink = torch.empty((), device="cuda")
            generator = torch.Generator().manual_seed(args.seed + 94013)
            seeds = torch.randint(
                0,
                2**32,
                (args.units, args.draws),
                generator=generator,
                dtype=torch.int64,
            ).cuda()
            draws = [law.draws(1, args.draws, args.seed + u) for u in range(args.units)]
            timer_args = argparse.Namespace(warmup=8, iters=args.draws, l2_flush=False)
            for u in range(args.units):
                courses = []
                for order in LATIN:
                    fixture.close()
                    for arm in order:
                        fixture.capture(arm, draws[u][0], seeds[u, :1])
                    samples = {arm: [] for arm in ARMS}

                    def prepare(index):
                        fixture.prepare(draws[u][index], seeds[u, index : index + 1])
                        torch.sum(conditioner, dim=0, out=sink)
                        fixture.stream.synchronize()
                        if args.rest_ms:
                            time.sleep(args.rest_ms / 1000)

                    orientation = order[::-1] if u % 2 else order
                    for arm in orientation + orientation[::-1]:
                        samples[arm].extend(
                            timing._time_kernel_cupti(
                                timer_args,
                                None,
                                None,
                                arm,
                                graph=fixture.graphs[arm],
                                prepare_sample=prepare,
                                return_samples=True,
                                expected_kernel_count=2
                                if args.algorithm == "two-kernel"
                                else 1,
                            )
                        )
                    courses.append(
                        {
                            arm: statistics.fmean(values)
                            for arm, values in samples.items()
                        }
                    )
                    (directory / f"unit{u:03d}_course{len(courses)}.json").write_text(
                        json.dumps(dict(order=order, samples=samples), indent=2) + "\n"
                    )
                result["units"].append(courses)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            result.update(
                summary=summarize(result["units"]),
                per_order_sensitivity=[summarize(result["units"], i) for i in range(4)],
                complete=True,
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result["summary"], indent=2))
    finally:
        fixture.close()
        timing.finalize_cupti()


if __name__ == "__main__":
    main()
