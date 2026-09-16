# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Model-shaped synthetic H32/D128 MXFP4 candidate scoring benchmark.

Run: python benchmarks/bench_deepseek_v41_indexer.py --output results.json
CUPTI: add --cupti (requires cupti-python>=13). Default timings are complete
CUDA Graph calls including candidate metadata, masking and scoring. CUPTI
reports the GPU activity span of the same graph separately; neither is serving E2E.
The optional cute_prepared benchmark arm times the public snapshot consumer
after explicit publication, which is excluded from that arm's timing.

DeepSeek V4.1 config dba1be0a40aa45a94ad051997016db3960a90277:
index_n_heads=32, index_head_dim=128, top2048 block8 candidates. Candidate
layers have compression ratio1. No model weights or serving traces are used.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import traceback

import numpy as np
import torch
import triton as tr

from flashinfer.deepseek_v41 import deepseek_v41_index_scores_fp32


def decode(data, scales):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
        dtype=torch.float64,
    )
    codes = torch.stack((data.long() & 15, data.long() >> 4), -1).flatten(-2)
    return lut[codes] * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)


def capture(fn, count=1):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            out = fn()
    return graph, out


def time_graph(graph, calls=10):
    samples = []
    for _ in range(20):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls)
    return samples


def quantize_normal(rows):
    x = torch.randn((rows, 4, 32), device="cuda")
    exponent = torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-8) / 6))
    y = x * torch.exp2(-exponent[..., None])
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], device="cuda")
    codes = torch.bucketize(y.abs().contiguous(), thresholds).to(torch.uint8)
    codes |= (y < 0).to(torch.uint8) * 8
    codes = codes.flatten(1)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), (exponent + 127).to(torch.uint8)


class Fixture:
    def __init__(self, batch, context, page=64, pattern="spread", mixed=False):
        self.batch, self.context, self.page = batch, context, page
        self.pattern, self.mixed = pattern, mixed
        self.count = min(2048, (context + 7) // 8)
        self.width = self.count * 8
        self.pages = tr.cdiv(context, page)
        self.phys = batch * self.pages
        stride = tr.cdiv(page * 68, 512) * 512
        self.storage = torch.full(
            (self.phys, stride), 71, dtype=torch.uint8, device="cuda"
        )
        self.cache = self.storage.as_strided(
            (self.phys, page, 1, 68), (stride, 68, 68, 1)
        )
        self.kd = self.storage[:, : page * 64].view(self.phys, page, 64)
        self.ks = self.storage[:, page * 64 : page * 68].view(self.phys, page, 4)
        torch.manual_seed(202609150 + batch + context + page)
        for first in range(0, self.phys, 1024):
            n = min(1024, self.phys - first)
            d, s = quantize_normal(n * page)
            self.kd[first : first + n].copy_(d.view(n, page, 64))
            self.ks[first : first + n].copy_(s.view(n, page, 4))
        qd, qs = quantize_normal(batch * 32)
        self.qd, self.qs = qd.view(batch, 32, 64), qs.view(batch, 32, 4)
        self.weights = (
            torch.randn(batch, 32, device="cuda") / math.sqrt(32 * 128)
        ).bfloat16()
        self.table = (
            torch.randperm(self.phys, device="cuda").int().view(batch, self.pages)
        )
        self.lengths = torch.empty(batch, device="cuda", dtype=torch.int32)
        self.candidates = torch.empty(
            (batch, self.count), device="cuda", dtype=torch.int32
        )
        self.regenerate_candidates(0)

    def regenerate_candidates(self, repeat):
        rng = np.random.default_rng(199711 + self.batch + self.context + repeat)
        lengths, candidates = [], []
        for b in range(self.batch):
            fraction = [1, 0.125, 0.5, 0.75][b % 4] if self.mixed else 1
            length = max(1, int(self.context * fraction) - (b * 13 + repeat * 7) % 8)
            blocks = tr.cdiv(length, 8)
            count = min(self.count, blocks)
            if self.pattern == "clustered" and blocks > count:
                recent = min(count // 2, 512)
                ids = np.concatenate(
                    [
                        rng.choice(blocks - recent, count - recent, replace=False),
                        np.arange(blocks - recent, blocks),
                    ]
                )
            else:
                ids = rng.choice(blocks, count, replace=False)
            # The source pins its newest partially visible block.
            if blocks - 1 not in ids:
                ids[0] = blocks - 1
            ids.sort()
            if self.pattern == "unordered":
                rng.shuffle(ids)
            row = np.full(self.count, -1, dtype=np.int32)
            row[:count] = ids
            candidates.append(row)
            lengths.append(length)
        self.candidates.copy_(torch.tensor(np.stack(candidates), device="cuda"))
        self.lengths.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))

    def output(self):
        stride = tr.cdiv(self.width, 512) * 512 + 512
        storage = torch.full(
            (self.batch + 2, stride), 17, device="cuda", dtype=torch.bfloat16
        )
        return storage[1:-1, : self.width], storage

    def reference(self):
        result = []
        q = decode(self.qd, self.qs)
        for b in range(self.batch):
            token = (
                self.candidates[b].long()[:, None] * 8 + torch.arange(8, device="cuda")
            ).flatten()
            valid = (token >= 0) & (token < self.lengths[b]) & (token < self.context)
            safe = token.clamp(0, self.context - 1)
            p = self.table[b, safe // self.page].long()
            valid &= (p >= 0) & (p < self.phys)
            p = p.clamp(0, self.phys - 1)
            k = decode(self.kd[p, safe % self.page], self.ks[p, safe % self.page])
            score = ((q[b] @ k.T).relu() * self.weights[b].double()[:, None]).sum(0)
            result.append(score.masked_fill(~valid, -float("inf")))
        return torch.stack(result)

    def mutate(self, repeat):
        self.qd.bitwise_xor_(0x88)
        self.weights.copy_((torch.randn_like(self.weights.float()) / 64).bfloat16())
        self.table.copy_(self.table.flip(1))
        self.kd.bitwise_xor_(0x11)
        self.regenerate_candidates(repeat)

    def args(self, out):
        return (
            self.qd,
            self.qs,
            self.cache,
            self.weights,
            self.lengths,
            self.table,
            self.candidates,
            self.context,
            out,
        )


def check(out, storage, ref):
    valid = torch.isfinite(ref)
    assert torch.isneginf(out[~valid]).all(), "invalid mask mismatch"
    assert torch.isfinite(out[valid]).all(), "unwritten/nonfinite score"
    diff = out[valid].double() - ref[valid]
    rel = float(diff.norm() / ref[valid].norm().clamp_min(1e-20))
    maximum = float(diff.abs().max() / ref[valid].abs().max().clamp_min(1e-20))
    assert rel < 0.0021 and maximum < 0.0042, (rel, maximum)
    assert (storage[0] == 17).all() and (storage[-1] == 17).all()
    assert (storage[1:-1, out.shape[1] :] == 17).all(), "padding modified"
    return dict(relative_l2=rel, normalized_max=maximum)


CASES = {
    **{
        f"b{b}_{c // 1024}k_small": (b, c, 64, "spread", False)
        for b in (1, 2, 4, 8, 16)
        for c in (8192, 32768, 131072)
    },
    "b1_128k": (1, 131072, 64, "spread", False),
    "b8_8k": (8, 8192, 64, "spread", False),
    "b32_32k": (32, 32768, 64, "spread", False),
    "b64_64k": (64, 65536, 64, "spread", False),
    "b128_128k": (128, 131072, 64, "spread", False),
    "b256_64k": (256, 65536, 64, "spread", False),
    "b512_64k": (512, 65536, 64, "spread", False),
    "mixed128_64k": (128, 65536, 64, "clustered", True),
    "page32_b64": (64, 65536, 32, "clustered", True),
    "page128_b64": (64, 65536, 128, "spread", False),
    "unordered_b64": (64, 65536, 64, "unordered", True),
}


def prepare(f, out, backend):
    if backend == "cute_prepared":
        from flashinfer.deepseek_v41 import (
            deepseek_v41_candidate_scores_fp32,
            prepare_deepseek_v41_candidate_metadata,
        )

        config = dict(
            page_size=f.page,
            num_physical_pages=f.phys,
            max_context_len=f.context,
        )
        metadata = prepare_deepseek_v41_candidate_metadata(
            f.lengths, f.table, f.candidates, **config
        )

        def publish():
            prepare_deepseek_v41_candidate_metadata(
                f.lengths, f.table, f.candidates, out=metadata, **config
            )

        def consume():
            return deepseek_v41_candidate_scores_fp32(
                f.qd, f.qs, f.cache, f.weights, metadata, out=out
            )

        return consume, publish
    if backend == "triton_fixed":
        # Reproduce the previous draft's candidate tile policy using the
        # same native MXFP4 math. This is a benchmark baseline, not an API.
        from flashinfer.experimental.deepseek_v41.indexer_fp32 import _scores

        tile = 128 if f.width <= 2048 else 256

        def run_fixed():
            _scores[(f.batch, tr.cdiv(f.width, tile))](
                f.qd,
                f.qs,
                f.cache,
                f.weights,
                f.lengths,
                f.table,
                f.candidates,
                out,
                f.page,
                f.cache.stride(0),
                f.phys,
                f.pages,
                f.context,
                f.width,
                out.stride(0),
                True,
                BLOCK_N=tile,
                num_warps=4,
                num_stages=2,
                enable_fp_fusion=False,
            )
            return out

        return run_fixed, None
    # Explicit scratch ownership; no cached tensors shared across calls.
    workspace = None
    if backend == "cute_dsl":
        from flashinfer.experimental.deepseek_v41.indexer_cute import workspace_size

        size = sum(workspace_size(f.batch, f.count))
        workspace = torch.empty(size, device="cuda", dtype=torch.uint8)

    def run():
        return deepseek_v41_index_scores_fp32(
            f.qd,
            f.qs,
            f.cache,
            f.weights,
            f.lengths,
            f.table,
            max_context_len=f.context,
            candidates=f.candidates,
            out=out,
            backend=backend,
            workspace=workspace,
        )

    return run, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="b1_128k,b8_8k,b32_32k,b64_64k,b128_128k,b256_64k,mixed128_64k,page32_b64,page128_b64,unordered_b64",
    )
    parser.add_argument("--backends", default="triton_fixed,triton,cute_dsl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cupti", action="store_true")
    args = parser.parse_args()
    if args.cupti:
        # Refuse silent event fallback under a CUPTI result label.
        from cupti import cupti  # noqa: F401

        if int(importlib.metadata.version("cupti-python").split(".")[0]) < 13:
            parser.error("--cupti requires cupti-python>=13")
    if args.output.exists():
        parser.error("output already exists; preserve previous measurements")
    prop = torch.cuda.get_device_properties(0)
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__),
        root / "flashinfer/deepseek_v41.py",
        root / "flashinfer/experimental/deepseek_v41/indexer_fp32.py",
        root / "flashinfer/experimental/deepseek_v41/indexer_cute.py",
        root / "flashinfer/experimental/deepseek_v41/candidate_metadata.py",
        root / "flashinfer/attn_scores/kernels/fp4_paged_mqa_logits.py",
    ]
    report = {
        "status": "running",
        "gpu": prop.name,
        "sm": torch.cuda.get_device_capability(),
        "sm_count": prop.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cutlass": importlib.metadata.version("nvidia-cutlass-dsl"),
        "scope": "synthetic single-pool candidate scorer; graph call, allocation/JIT excluded; cute_prepared excludes explicit metadata publication",
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "cases": [],
    }

    def save():
        temp = args.output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(args.output)

    try:
        with torch.inference_mode():
            for case in args.cases.split(","):
                f = Fixture(*CASES[case])
                row = {
                    "case": case,
                    "shape": CASES[case],
                    "arms": {},
                    "status": "running",
                }
                report["cases"].append(row)
                arms = {}
                for backend in args.backends.split(","):
                    out, guard = f.output()
                    run, publish = prepare(f, out, backend)
                    graph, _ = capture(run)
                    arms[backend] = (run, out, guard, graph, publish)
                    row["arms"][backend] = {
                        "correctness": [],
                        "metadata_publication": "excluded; explicit snapshot"
                        if publish
                        else "included in scorer call",
                    }
                for repeat in range(2):
                    f.mutate(repeat)
                    ref = f.reference()
                    for backend, (_, out, guard, graph, publish) in arms.items():
                        if publish is not None:
                            publish()
                        out.fill_(float("nan"))
                        graph.replay()
                        row["arms"][backend]["correctness"].append(
                            check(out, guard, ref)
                        )
                # A captured no-op/stale result must not pass the same gate.
                out.fill_(float("nan"))
                try:
                    check(out, guard, ref)
                except AssertionError:
                    pass
                else:
                    raise AssertionError("negative control escaped")
                timed = {name: capture(v[0], 10)[0] for name, v in arms.items()}
                names = list(timed)
                samples = {name: [] for name in names}
                for order in (names, names[::-1], names[::-1], names):
                    for name in order:
                        samples[name].extend(time_graph(timed[name]))
                for name, values in samples.items():
                    row["arms"][name].update(
                        median_us=statistics.median(values), samples_us=values
                    )
                if args.cupti:
                    from flashinfer.testing import bench_gpu_time

                    for name, graph in timed.items():
                        values = bench_gpu_time(
                            graph.replay,
                            dry_run_iters=5,
                            repeat_iters=30,
                            enable_cupti=True,
                            cold_l2_cache=False,
                        )
                        row["arms"][name]["cupti_graph_span_median_us"] = (
                            statistics.median(values) * 1000 / 10
                        )
                row["status"] = "pass"
                print(
                    case,
                    {name: round(row["arms"][name]["median_us"], 3) for name in names},
                    flush=True,
                )
                save()
                del arms, timed, f
            report["status"] = "pass"
    except Exception:
        report["status"] = "fail"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
