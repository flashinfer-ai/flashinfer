# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Same-input, cold-L2 CUDA-Graph comparison of native sparse MLA backends.

Example:
  python benchmarks/bench_attention_ts_sparse_mla.py --batches 1,8 --heads 16,64 \
      --contexts 4096,65536 --backends ts-auto,trtllm-gen --output results.json

No eager/hot-cache fallback. Unsupported backends are explicitly recorded.
"""

import argparse
import csv
from dataclasses import dataclass, asdict
import importlib.metadata
import itertools
import hashlib
import os
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.testing.sparse_mla import (
    sparse_mla_reference,
    sparse_mla_input_fingerprint,
)
from sparse_mla_bench_utils import ColdL2GraphBenchmark
from sparse_mla_fixtures import flashmla_fixture


@dataclass
class Fixture:
    query: torch.Tensor
    swa: torch.Tensor
    compressed: torch.Tensor
    si: torch.Tensor
    ci: torch.Tensor
    sl: torch.Tensor
    cl: torch.Tensor
    sinks: torch.Tensor
    seq_lens: torch.Tensor
    combined: torch.Tensor
    combined_lengths: torch.Tensor
    softmax_scale: float = 512**-0.5

    def fingerprint(self):
        return sparse_mla_input_fingerprint(**vars(self))

    def kwargs(self):
        return dict(
            swa_indices=self.si,
            compressed_indices=self.ci,
            swa_topk_lens=self.sl,
            compressed_topk_lens=self.cl,
            sinks=self.sinks,
            softmax_scale=self.softmax_scale,
        )


def make_fixture(
    batch,
    heads,
    queries,
    context,
    dtype,
    swa_page,
    compressed_page,
    seed,
    index_order="contiguous",
):
    if context < 128 + queries:
        raise ValueError("benchmark fixtures require a full 128-token SWA window")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def normal(shape, offset=0):
        return (
            torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
            + offset
        ).to(dtype)

    swa_rows = ((128 + queries - 1 + swa_page - 1) // swa_page) * swa_page
    max_compressed = context // 128
    comp_rows = (
        (max_compressed + compressed_page - 1) // compressed_page
    ) * compressed_page
    capacity = ((max_compressed + 3) // 4) * 4
    q = normal((batch, queries, heads, 512))
    swa = normal((batch * swa_rows // swa_page, swa_page, 512), -0.125)
    compressed = normal(
        (batch * comp_rows // compressed_page, compressed_page, 512), 0.125
    )
    b = torch.arange(batch, device="cuda", dtype=torch.int32)[:, None, None]
    qi = torch.arange(queries, device="cuda", dtype=torch.int32)[None, :, None]
    si = (
        b * swa_rows
        + (torch.arange(128, device="cuda", dtype=torch.int32)[None, None, :] + qi)
        % swa_rows
    ).contiguous()
    cl = (
        (
            (
                context
                - queries
                + torch.arange(queries, device="cuda", dtype=torch.int32)
                + 1
            )
            // 128
        )
        .expand(batch, queries)
        .contiguous()
    )
    slots = torch.arange(capacity, device="cuda", dtype=torch.int32)[None, None, :]
    ci = torch.where(slots < cl[..., None], b * comp_rows + slots, -1).contiguous()
    sl = torch.full((batch, queries), 128, device="cuda", dtype=torch.int32)
    sinks = (
        torch.randn(heads, device="cuda", dtype=torch.float32, generator=generator)
        * 0.1
    )
    seq_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    if index_order == "shuffled":
        for request in range(batch):
            for token in range(queries):
                si[request, token] = si[request, token][
                    torch.randperm(128, device="cuda", generator=generator)
                ]
                active = int(cl[request, token].item())
                ci[request, token, :active] = ci[request, token, :active][
                    torch.randperm(active, device="cuda", generator=generator)
                ]
    combined = torch.cat((si, ci), dim=-1).reshape(batch * queries, -1)
    combined_lengths = (128 + cl).reshape(-1)
    assert torch.equal(combined[:, :128].reshape_as(si), si)
    assert torch.equal(combined[:, 128:].reshape_as(ci), ci)
    return Fixture(
        q, swa, compressed, si, ci, sl, cl, sinks, seq_lens, combined, combined_lengths
    )


def prepared_metadata_fingerprint(fn):
    tensors = getattr(fn, "prepared_tensors", None)
    return None if tensors is None else sparse_mla_input_fingerprint(**tensors)


def make_backend(
    name, fixture, splits, *, prepared=True, single_source=False, tuning_override=None
):
    q = fixture.query
    batch, queries, heads, _ = q.shape
    out = torch.empty_like(q, dtype=torch.bfloat16)
    if name.startswith("ts-"):
        if name != "ts-auto":
            raise ValueError(f"unknown TS backend {name}; use ts-auto")
        wrapper = BatchSparseMLADecodePagedTSWrapper()
        wrapper._impl._tuning = tuning_override
        begin = time.perf_counter()
        from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata

        assume_valid_prefix = prepared
        plan_kwargs = dict(
            max_topk=fixture.ci.shape[-1] if single_source else fixture.si.shape[-1],
            max_extra_topk=0 if single_source else fixture.ci.shape[-1],
            max_seq_len_q=queries,
            q_data_type=q.dtype,
            has_sinks=True,
            assume_valid_prefix=assume_valid_prefix,
        )
        wrapper.plan(q.device, batch, heads, **plan_kwargs)
        plan_ms = (time.perf_counter() - begin) * 1000
        primary = fixture.compressed if single_source else fixture.swa
        extra = None if single_source else fixture.compressed
        common = dict(sinks=fixture.sinks, softmax_scale=fixture.softmax_scale)
        prep_kwargs = dict(
            extra_kv_cache=extra,
            extra_indices=None if single_source else fixture.ci,
            extra_lengths=None if single_source else fixture.cl,
            **common,
        )
        indices = fixture.ci if single_source else fixture.si
        lengths = fixture.cl if single_source else fixture.sl
        prepared_inputs = prepare_sparse_mla_metadata(
            wrapper,
            q,
            primary,
            indices,
            lengths,
            **prep_kwargs,
        )

        def fn():
            if not prepared:
                # Explicit preparation-inclusive benchmark, outside the public API.
                prepare_sparse_mla_metadata(
                    wrapper,
                    q,
                    primary,
                    indices,
                    lengths,
                    out=prepared_inputs,
                    **prep_kwargs,
                )
            return wrapper.run(
                q, primary, prepared_inputs, extra, out=out, validate=False, **common
            )

        fn.prepared_tensors = prepared_inputs._asdict()
        fn()
        metadata = dict(
            family=wrapper._impl._state["family"],
            assume_valid_prefix=assume_valid_prefix,
            tuning=asdict(wrapper._impl._state["tuning"]),
            selection_reason=wrapper._impl._state["selection_reason"],
            tile_q=wrapper._impl._state["tile"],
            split_count=wrapper._impl._state["splits"],
            execution_mode=wrapper._impl._state["last_execution_mode"],
            fused_epilogue=wrapper._impl._state["last_fused_epilogue"],
            direct_inputs=wrapper._impl._state["last_direct_inputs"],
            static_scales=wrapper._impl._state["last_static_scales"],
            workspace_bytes=wrapper.workspace_size_bytes,
            gather_issue_warps=(
                4
                if wrapper._impl._state["family"] == "2cta"
                and q.dtype == torch.float8_e4m3fn
                else wrapper._impl._state["tuning"].gather_issue_warps
            ),
            offset_cache=(
                "quad"
                if wrapper._impl._state["family"] == "2cta"
                and q.dtype == torch.float8_e4m3fn
                else wrapper._impl._state["tuning"].offset_cache
            ),
            head_dim_ctas=wrapper._impl._state["head_dim_ctas"],
            plan_ms=plan_ms,
            prepared_input=prepared,
            forced_profile=tuning_override is not None,
            source_mode="single" if single_source else "two",
            prepared_metadata_sha256=prepared_metadata_fingerprint(fn),
        )
        return fn, out, metadata
    if name == "trtllm-gen":
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
        from flashinfer.artifacts import ArtifactPath, CheckSumHash

        workspace = torch.empty(128 * 1024 * 1024, device=q.device, dtype=torch.uint8)
        s = fixture.swa.unsqueeze(1)
        c = fixture.compressed.unsqueeze(1)
        combined, combined_lengths = fixture.combined, fixture.combined_lengths
        if prepared:
            from flashinfer.testing.sparse_mla_metadata import map_sparse_indices

            si, sl = map_sparse_indices(
                fixture.si.view(batch * queries, -1),
                fixture.sl.view(-1),
                page_size=fixture.swa.shape[1],
                page_stride_rows=fixture.swa.stride(0) // 512,
            )
            ci, cl = map_sparse_indices(
                fixture.ci.view(batch * queries, -1),
                fixture.cl.view(-1),
                page_size=fixture.compressed.shape[1],
                page_stride_rows=fixture.compressed.stride(0) // 512,
            )
            # The shipped DSV4 comparator has a fixed 128-column SWA segment.
            # Single-source trials leave it invalid; it selects no extra KV.
            combined = torch.cat((si, ci), dim=-1)
            combined_lengths = 128 + cl

        def fn():
            return trtllm_batch_decode_sparse_mla_dsv4(
                q,
                s,
                workspace,
                sparse_indices=combined,
                compressed_kv_cache=c,
                sparse_topk_lens=combined_lengths,
                seq_lens=fixture.seq_lens,
                sinks=fixture.sinks,
                out=out,
                bmm1_scale=fixture.softmax_scale,
                bmm2_scale=1.0,
                kv_layout="HND",
                backend="trtllm-gen",
                sparse_indices_are_storage_offsets=prepared,
            )

        if prepared:
            fn.prepared_tensors = dict(indices=combined, lengths=combined_lengths)

        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        metadata = dict(
            artifact=str(ArtifactPath.TRTLLM_GEN_FMHA),
            artifact_hash=str(CheckSumHash.TRTLLM_GEN_FMHA),
            plan_ms=(time.perf_counter() - begin) * 1000,
            workspace_bytes=workspace.numel(),
            prepared_input=prepared,
            source_mode="single" if single_source else "two",
            fixed_swa_columns=128,
            prepared_metadata_sha256=prepared_metadata_fingerprint(fn),
        )
        return fn, out, metadata
    raise ValueError(f"unknown backend {name}")


def accuracy(out, expected, bound):
    actual = out.double()
    error = actual - expected
    if not torch.isfinite(actual).all().item():
        raise AssertionError("nonfinite backend output")
    if not (error.abs() <= bound).all().item():
        raise AssertionError(
            f"forward-error bound exceeded by {(error.abs() - bound).max().item()}"
        )
    a = actual.flatten()
    r = expected.flatten()
    return dict(
        max_abs_error=error.abs().max().item(),
        normalized_l2_error=(error.norm() / expected.norm().clamp_min(1e-12)).item(),
        cosine_error=(
            1 - torch.dot(a, r) / (a.norm() * r.norm()).clamp_min(1e-12)
        ).item(),
    )


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--heads", default="16,64")
    parser.add_argument("--contexts", default="4096,65536")
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fixture", choices=("flashmla", "hca"), default="flashmla")
    parser.add_argument("--swa-cache-tokens", type=int, default=16384)
    parser.add_argument("--compressed-cache-tokens", type=int, default=16384)
    parser.add_argument("--compressed-topk", type=int, default=None)
    parser.add_argument("--fixed-cache-lengths", action="store_true")
    parser.add_argument("--variable-topk-lengths", action="store_true")
    parser.add_argument(
        "--valid-prefix-lengths",
        action="store_true",
        help="Trim trailing invalid slots from TS live lengths for every TS backend; "
        "preserve TRT's fixed 128-slot SWA boundary and identical selected KV.",
    )
    parser.add_argument(
        "--index-order", choices=("contiguous", "shuffled"), default="contiguous"
    )
    parser.add_argument("--dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--swa-page", type=int, default=256)
    parser.add_argument("--compressed-page", type=int, default=1)
    parser.add_argument("--splits", type=int, default=1)
    parser.add_argument("--backends", default="ts-auto,trtllm-gen")
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--samples-per-replay", type=int, default=8)
    parser.add_argument("--eviction-multiplier", type=int, default=4)
    parser.add_argument("--profile-names", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-preparation",
        action="store_true",
        help="Include external preparation in a separate timing protocol",
    )
    args = parser.parse_args()
    if args.replays < 1 or args.eviction_multiplier < 4:
        parser.error("positive replays and eviction multiplier >=4 are required")
    torch.backends.cuda.matmul.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    source_hash = hashlib.sha256()
    source_names = (
        subprocess.check_output(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "flashinfer/attention/prims_ts",
                "flashinfer/prims_ts",
                "flashinfer/testing/sparse_mla.py",
            ],
        )
        .decode()
        .split("\0")
    )
    for name in sorted(set(n for n in source_names if n and n.endswith(".py"))):
        source_hash.update(name.encode() + b"\0")
        source_hash.update(Path(name).read_bytes())
    report = dict(
        protocol=(
            "identical-input/external-preparation/cold-L2/CUDA-Graph"
            if args.include_preparation
            else "identical-input/prepared-input/cold-L2/CUDA-Graph"
        ),
        source_tree_sha256=source_hash.hexdigest(),
        benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_generator_sha256=hashlib.sha256(
            Path(__file__).with_name("sparse_mla_fixtures.py").read_bytes()
        ).hexdigest(),
        flashmla_fixture_reference="ba89a3466e9470ad08ab39738d4e7bb66989e1e7/tests/lib.py",
        timer_sha256=hashlib.sha256(
            Path(__file__).with_name("sparse_mla_bench_utils.py").read_bytes()
        ).hexdigest(),
        eviction_method="read_only_reduction",
        eviction_bytes=args.eviction_multiplier * properties.L2_cache_size,
        replays=args.replays,
        samples_per_replay=args.samples_per_replay,
        seed=args.seed,
        job_id=os.getenv("SLURM_JOB_ID"),
        node=os.getenv("SLURM_JOB_NODELIST"),
        compute_capability=torch.cuda.get_device_capability(),
        driver=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                "0",
            ],
            text=True,
        ).strip(),
        source_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--", "flashinfer"])
        ).hexdigest(),
        timing_scope="external-preparation+attention"
        if args.include_preparation
        else "prepared-attention",
        cuda_graph=True,
        cold_l2=True,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        gpu=properties.name,
        sm_count=properties.multi_processor_count,
        l2_bytes=properties.L2_cache_size,
        cuda=torch.version.cuda,
        torch=torch.__version__,
        dsl=importlib.metadata.version("nvidia-cutlass-dsl"),
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    backends = args.backends.split(",")
    sizes = itertools.product(
        *(
            list(map(int, x.split(",")))
            for x in (args.batches, args.heads, args.contexts)
        )
    )
    for batch, heads, context in sizes:
        dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        if args.fixture == "hca":
            fixture = make_fixture(
                batch,
                heads,
                args.queries,
                context,
                dtype,
                args.swa_page,
                args.compressed_page,
                args.seed,
                args.index_order,
            )
        else:
            fixture = Fixture(
                **flashmla_fixture(
                    batch,
                    heads,
                    args.queries,
                    dtype,
                    args.swa_page,
                    args.compressed_page,
                    args.seed,
                    compressed_topk=(
                        context // 128
                        if args.compressed_topk is None
                        else args.compressed_topk
                    ),
                    swa_cache_tokens=args.swa_cache_tokens,
                    compressed_cache_tokens=args.compressed_cache_tokens,
                    variable_cache_lengths=not args.fixed_cache_lengths,
                    variable_topk_lengths=args.variable_topk_lengths,
                )
            )
        if args.valid_prefix_lengths:
            # FlashMLA's sorted score top-k puts invalid slots at the end.
            # TRT retains its fixed 128-slot SWA region and masks startup
            # padding using seq_lens. TS describes that same set explicitly.
            # Apply this encoding to ALL TS backends, before the reference
            # and fingerprint, with no route reordering or extra GPU work
            # inside the timed region.
            for indices, lengths in (
                (fixture.si, fixture.sl),
                (fixture.ci, fixture.cl),
            ):
                positions = torch.arange(indices.shape[-1], device=indices.device)
                valid = (indices >= 0) & (positions < lengths[..., None])
                counts = valid.sum(-1).to(torch.int32)
                if not torch.equal(valid, positions < counts[..., None]):
                    raise ValueError(
                        "fixture has interior holes; cannot trim to a prefix"
                    )
                lengths.copy_(counts)
        fingerprint = fixture.fingerprint()
        expected, _, bound = sparse_mla_reference(
            fixture.query,
            fixture.swa,
            fixture.compressed,
            fixture.si,
            fixture.ci,
            swa_topk_lens=fixture.sl,
            compressed_topk_lens=fixture.cl,
            sinks=fixture.sinks,
            softmax_scale=fixture.softmax_scale,
            return_fp8_error_bound=True,
        )
        if args.dtype == "bf16":
            bound = expected.abs() * 0.02 + 8e-4
        case = dict(
            batch=batch,
            heads=heads,
            queries=args.queries,
            raw_context=context,
            dtype=args.dtype,
            fixture=args.fixture,
            valid_prefix_lengths=args.valid_prefix_lengths,
            index_order=(
                "random-score-topk/random-pages"
                if args.fixture == "flashmla"
                else args.index_order
            ),
            swa_cache_tokens=(
                args.swa_cache_tokens if args.fixture == "flashmla" else None
            ),
            compressed_cache_tokens=(
                args.compressed_cache_tokens if args.fixture == "flashmla" else None
            ),
            variable_cache_lengths=args.fixture == "flashmla"
            and not args.fixed_cache_lengths,
            variable_topk_lengths=args.fixture == "flashmla"
            and args.variable_topk_lengths,
            selected_capacity=128 + fixture.ci.shape[-1],
            softmax_scale=fixture.softmax_scale,
            pool_rows=[
                fixture.swa.shape[0] * fixture.swa.shape[1],
                fixture.compressed.shape[0] * fixture.compressed.shape[1],
            ],
            swa_page=args.swa_page,
            compressed_page=args.compressed_page,
            fixture_id=fingerprint,
            valid_slots=sum(
                int(
                    (
                        (indices >= 0)
                        & (
                            torch.arange(indices.shape[-1], device=indices.device)[
                                None, None, :
                            ]
                            < lengths[..., None]
                        )
                    )
                    .sum()
                    .item()
                )
                for indices, lengths in (
                    (fixture.si, fixture.sl),
                    (fixture.ci, fixture.cl),
                )
            ),
            backends={},
        )
        report["cases"].append(case)
        runners = {}
        outputs = {}
        for name in backends:
            try:
                fn, out, metadata = make_backend(
                    name, fixture, args.splits, prepared=not args.include_preparation
                )
                torch.cuda.synchronize()
                metrics = accuracy(out, expected, bound)
                runner = ColdL2GraphBenchmark(
                    fn,
                    device=fixture.query.device,
                    eviction_bytes=args.eviction_multiplier * properties.L2_cache_size,
                    samples_per_replay=args.samples_per_replay,
                )
                runner.sample()
                accuracy(out, expected, bound)
                runners[name] = runner
                outputs[name] = out
                case["backends"][name] = dict(
                    status="ok", **metadata, **metrics, times_us=[]
                )
            except Exception as error:
                case["backends"][name] = dict(
                    status="unsupported_or_failed",
                    error=f"{type(error).__name__}: {error}",
                )
                print(f"{name}: {error}", flush=True)
                if name.startswith("ts-"):
                    save()
                    raise
        if not runners:
            save()
            raise RuntimeError("no backend satisfied the benchmark protocol")
        # Alternate backend order per replay; every invocation has its own
        # captured eviction and external timing event pair.
        for iteration in range(args.replays):
            order = list(runners)
            if iteration % 2:
                order.reverse()
            for name in order:
                case["backends"][name]["times_us"].extend(runners[name].sample())
        for name, runner in runners.items():
            entry = case["backends"][name]
            times = entry["times_us"]
            entry.update(
                median_us=statistics.median(times),
                p10_us=percentile(times, 0.1),
                p90_us=percentile(times, 0.9),
            )
            entry.update(accuracy(outputs[name], expected, bound))
            entry["useful_tflops"] = (2 * heads * case["valid_slots"] * 1024) / (
                entry["median_us"] * 1e6
            )
            if args.profile_names:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profiler:
                    runner.sample()
                trace_path = (
                    args.output.parent / f"{fingerprint[:12]}-{name}-trace.json"
                )
                profiler.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text())
                entry["kernel_names"] = sorted(
                    {
                        e["name"]
                        for e in trace["traceEvents"]
                        if e.get("cat") == "kernel"
                    }
                )
            print(
                f"B={batch} H={heads} L={context} {name}: {entry['median_us']:.3f} us",
                flush=True,
            )
        if fixture.fingerprint() != fingerprint:
            raise AssertionError("a backend mutated the shared input fixture")
        save()
    rows = []
    for case in report["cases"]:
        for name, result in case["backends"].items():
            rows.append(
                {
                    k: v
                    for k, v in {
                        **{k: v for k, v in case.items() if k != "backends"},
                        "backend": name,
                        **result,
                    }.items()
                    if not isinstance(v, (dict, list))
                }
            )
    with args.output.with_suffix(".csv").open("w") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=sorted({k for row in rows for k in row}),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
