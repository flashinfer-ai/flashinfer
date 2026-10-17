# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SM120 GLM DSA NVFP4 versus FP8 cache microbenchmark.

Run: python benchmarks/bench_sparse_mla_glm_nvfp4_sm120.py
Synthetic H=8, QK=576, V=512, top-k=2048, page size=64 inputs.
Native FlashInfer autotuning, CUDA Graph, and a 3x-L2 flush before every
CUPTI profile (outside the measured span). Excludes cache writes.
"""

import argparse
import importlib
import json
import math
import statistics
import tempfile

import torch

from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from flashinfer.jit.mla import (
    gen_sparse_mla_glm_nvfp4_sm120_module,
    gen_sparse_mla_sm120_module,
)


def make_cache(rows):
    packed = torch.randint(256, (rows, 256), device="cuda", dtype=torch.uint8)
    codes = torch.tensor([16, 18, 20, 24, 26, 28], device="cuda", dtype=torch.uint8)
    sf = codes[torch.randint(len(codes), (rows, 32), device="cuda")]
    rope = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    nv = torch.cat((packed, sf, rope.view(torch.uint8)), -1).view(-1, 64, 1, 416)
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    nibbles = torch.stack((packed & 15, packed >> 4), -1).reshape(rows, 32, 16)
    latent = (
        lut[nibbles.long()] * sf.view(torch.float8_e4m3fn).float()[..., None]
    ).reshape(rows, 512)
    blocks = latent.reshape(rows, 4, 128)
    scales = blocks.abs().amax(-1).clamp_min(1e-8) / 448
    values = (blocks / scales[..., None]).to(torch.float8_e4m3fn)
    fp8 = torch.cat(
        (
            values.reshape(rows, 512).view(torch.uint8),
            scales.view(torch.uint8),
            rope.view(torch.uint8),
        ),
        -1,
    ).view(-1, 64, 1, 656)
    decoded = {
        "nvfp4": torch.cat((latent, rope.float()), -1),
        "fp8": torch.cat(
            ((values.float() * scales[..., None]).reshape(rows, 512), rope.float()), -1
        ),
    }
    return {"nvfp4": nv, "fp8": fp8}, decoded


class Runner(TunableRunner):
    def __init__(self, mode, module, q, cache, indices, decoded):
        self.inputs = [q, cache, indices]
        self.calls, self.outputs = {}, {}
        t, h, _ = q.shape
        topk = indices.shape[-1]
        self.identity = (mode, t, h, topk, "glm_nvfp4_cold_graph_v1")
        self.tactics = [0] + (
            list(range(1, topk // (32 if mode == "nvfp4" else 64) + 1))
            if t <= 64
            else []
        )
        scale = torch.ones(1, device=q.device)
        sm_scale = 576**-0.5
        sample = decoded[indices[:4].long()]
        scores = torch.einsum("thd,tkd->thk", q[:4].float(), sample) * sm_scale
        expected = torch.einsum("thk,tkd->thd", scores.softmax(-1), sample[..., :512])
        expected_lse = scores.logsumexp(-1) / math.log(2)
        for cpb in self.tactics:
            out = torch.empty(t, h, 512, device=q.device, dtype=q.dtype)
            lse = torch.empty(t, h, device=q.device)
            if mode == "nvfp4":
                splits = math.ceil(topk / 32 / cpb) if cpb else 0
                mid = torch.empty(t, h, splits, 512, device=q.device) if cpb else None
                mlse = torch.empty(t, h, splits, device=q.device) if cpb else None
                fn = module.sparse_mla_glm_nvfp4
                args = (
                    q,
                    cache,
                    indices,
                    out,
                    lse,
                    scale,
                    sm_scale,
                    cpb,
                    None,
                    None,
                    mid,
                    mlse,
                )
            elif cpb:
                splits = topk // 64
                mid = torch.empty(t, h, splits, 512, device=q.device, dtype=q.dtype)
                mlse = torch.empty(t, h, splits, device=q.device)
                fn = module.sparse_mla_sm120_decode_dsv3_2
                args = (
                    q,
                    cache,
                    indices,
                    mid,
                    mlse,
                    out,
                    lse,
                    splits,
                    sm_scale,
                    None,
                    None,
                    2,
                    cpb,
                    1.0,
                )
            else:
                fn = module.sparse_mla_sm120_paged_attention
                args = (
                    q,
                    cache,
                    indices,
                    out,
                    lse,
                    sm_scale,
                    2,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                    False,
                )
            self.calls[cpb], self.outputs[cpb] = (fn, args), (out, lse)
            fn(*args)
            # Check the first four queries of every candidate against FP32.
            torch.testing.assert_close(out[:4].float(), expected, atol=0.05, rtol=0.05)
            torch.testing.assert_close(lse[:4], expected_lse, atol=0.05, rtol=0.05)
        self.calls[-1] = self.calls[0]

    def __hash__(self):
        return hash(self.identity)

    def get_cache_key_extras(self, inputs):
        return self.identity

    def get_valid_tactics(self, inputs, profile):
        return self.tactics

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if not do_preparation:
            fn, args = self.calls[tactic]
            fn(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 2048])
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--rounds", type=int, default=42)
    args = parser.parse_args()
    assert args.rows > 0 and args.rows % 64 == 0 and args.rounds > 0
    assert torch.cuda.get_device_capability()[0] == 12
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    at = importlib.import_module("flashinfer.autotuner.autotuner")
    assert at._load_cupti() is not None, "CUPTI required; no timing fallback"
    modules = {
        "nvfp4": gen_sparse_mla_glm_nvfp4_sm120_module().build_and_load(),
        "fp8": gen_sparse_mla_sm120_module().build_and_load(),
    }
    tuner = AutoTuner.get()
    policy = MeasurementPolicy(
        execution_mode="cuda_graph", cold_l2=True, _timer="cupti"
    )
    caches, decoded = make_cache(args.rows)
    flush = torch.empty(
        3 * tuner._get_l2_cache_size_in_bytes(), device="cuda", dtype=torch.uint8
    )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "rows": args.rows,
                "heads": 8,
                "topk": 2048,
                "flush_bytes": flush.numel(),
                "cuda_graph": True,
                "autotune": True,
                "cold_l2": True,
                "timer": "CUPTI",
            }
        ),
        flush=True,
    )
    for tokens in args.tokens:
        q = torch.randn(tokens, 8, 576, device="cuda", dtype=torch.bfloat16)
        indices = torch.randint(
            args.rows, (tokens, 2048), device="cuda", dtype=torch.int32
        )
        runners, selected, graphs, snapshots = {}, {}, {}, {}
        for mode, module in modules.items():
            runner = Runner(mode, module, q, caches[mode], indices, decoded[mode])
            runners[mode] = runner
            config = TuningConfig(
                use_cuda_graph=True,
                use_cold_l2_cache=True,
                profiling_repeat=7,
                inputs_pre_hook=lambda _, r=runner: r.inputs,
            )
            with (
                tempfile.TemporaryDirectory() as cache_dir,
                autotune_v2(
                    mode="tune", cache_root=cache_dir, measurement_policy=policy
                ),
            ):
                _, tactic = tuner.choose_one(
                    "glm_dsa_nvfp4_benchmark", [runner], config, runner.inputs
                )
            assert not tuner._cupti_disabled
            selected[mode] = tactic
            runner(runner.inputs, tactic=tactic)
            snapshots[mode] = tuple(x.clone() for x in runner.outputs[tactic])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner(runner.inputs, tactic=tactic)
            graphs[mode] = graph
        modes = list(modules)
        for _ in range(30):
            for mode in modes:
                flush.zero_()
                graphs[mode].replay()
        torch.cuda.synchronize()
        schedule = [
            m
            for r in range(args.rounds)
            for m in (modes if r % 2 == 0 else modes[::-1])
        ]
        it = iter(schedule)
        spans = at._cupti_measure_spans(
            lambda: graphs[next(it)].replay(), len(schedule), prologue=flush.zero_
        )
        samples = {m: [] for m in modes}
        for mode, ms in zip(schedule, spans, strict=True):
            samples[mode].append(ms * 1000)
        for mode in modes:
            for actual, expected in zip(
                runners[mode].outputs[selected[mode]], snapshots[mode], strict=True
            ):
                assert torch.equal(actual, expected), "graph replay changed output"
        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "selected_cpb": selected,
                    "median_us": {m: statistics.median(samples[m]) for m in modes},
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
