# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Compare Frost WOA with cuBLAS, public TGV, and every native TGV tactic.

Run on SM100 with ``python benchmarks/bench_deepseek_v41_woa.py``.
Five synthetic weight sets use the model's single-token geometry. This is a
component benchmark; it does not measure a complete decode request or model.
TGV internals are used only to enumerate native tactics for this benchmark.
"""

import argparse
import json
from pathlib import Path
import statistics
import traceback

import torch

import flashinfer
from flashinfer.autotuner import autotune
from flashinfer.deepseek_v41 import deepseek_v41_woa, deepseek_v41_woa_plan
from flashinfer.gemm import gemm_base
from flashinfer.gemm.kernels.tgv_gemm_cute_ext import _TGV_CUTE_EXT_TACTIC_CONFIGS


def check(actual, reference):
    delta = actual.double() - reference
    assert torch.isfinite(actual).all()
    assert delta.norm() <= 0.003 * reference.norm()
    assert delta.abs().max() <= 0.006 * reference.abs().max()
    return (
        (delta.norm() / reference.norm().clamp_min(1e-30)).item(),
        (delta.abs().max() / reference.abs().max().clamp_min(1e-30)).item(),
    )


@torch.no_grad()
def benchmark(seed):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark requires an SM100 GPU")
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    validation = dict(outputs=0, worst_relative_l2=0.0, worst_max_scaled=0.0)

    def checked(actual, reference):
        relative, maximum = check(actual, reference)
        validation["outputs"] += 1
        validation["worst_relative_l2"] = max(validation["worst_relative_l2"], relative)
        validation["worst_max_scaled"] = max(validation["worst_max_scaled"], maximum)

    inputs, weights, plans = [], [], []
    immutable = []
    for _ in range(5):
        x = torch.randn((1, 8, 4096), device="cuda", dtype=torch.bfloat16)
        stored = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        scales = torch.randint(122, 131, (256, 128), device="cuda", dtype=torch.uint8)
        plan = deepseek_v41_woa_plan(stored, scales, backend="frost")
        # Independent scale decoding and the source BF16 weight boundary.
        decoded = (
            (
                stored.double().view(256, 32, 128, 32)
                * torch.exp2(scales.double() - 127)[:, None, :, None]
            )
            .bfloat16()
            .reshape(8, 1024, 4096)
        )
        inputs.append(x)
        weights.append(decoded.transpose(1, 2))
        plans.append(plan)
        immutable.extend((t, t.clone()) for t in (stored, scales, decoded))

    runner = gemm_base._tgv_gemm_runner(
        torch.bfloat16, gemm_base.is_sm100f_supported(inputs[0].device)
    )
    workspace = gemm_base._get_cache_buf(
        "bmm_bf16_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, inputs[0].device
    )
    inventory_out = torch.empty((8, 1, 1024), device="cuda", dtype=torch.bfloat16)
    tactics = runner.get_valid_tactics(
        [inputs[0].transpose(0, 1), weights[0], None, False, inventory_out, workspace],
        None,
    )
    assert tactics == list(range(len(_TGV_CUTE_EXT_TACTIC_CONFIGS)))
    roles = ["cublas", "tgv", *[f"tgv_t{t:02d}" for t in tactics], "frost"]

    def run(role, index):
        x, weight = inputs[index], weights[index]
        if role == "frost":
            return deepseek_v41_woa(x, plans[index])
        a = x.transpose(0, 1)
        if role == "cublas":
            return torch.bmm(a, weight).transpose(0, 1)
        if role == "tgv":
            return flashinfer.bmm_bf16(a, weight, backend="tgv").transpose(0, 1)
        out = torch.empty((8, 1, 1024), device=x.device, dtype=x.dtype)
        runner(
            inputs=[a, weight, None, False, out, workspace],
            tactic=int(role.removeprefix("tgv_t")),
        )
        return out.transpose(0, 1)

    with autotune(tune_mode=True):
        run("tgv", 0)
    originals = [x.clone() for x in inputs]
    references = [
        torch.bmm(x.transpose(0, 1).double(), w.double()).transpose(0, 1)
        for x, w in zip(inputs, weights, strict=True)
    ]
    graphs, outputs = {}, {}
    for role in roles:
        for index in range(5):
            checked(run(role, index), references[index])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            values = [run(role, index % 5) for index in range(30)]
        graphs[role], outputs[role] = graph, values

    # Changed inputs plus poisoned outputs reject stale or uncaptured work.
    for variant in ("zero", "negation", "roll", "restore"):
        for x, original in zip(inputs, originals, strict=True):
            changed = {
                "zero": lambda: torch.zeros_like(original),
                "negation": lambda: -original,
                "roll": lambda: original.roll(17, dims=-1),
                "restore": lambda: original,
            }[variant]()
            x.copy_(changed)
        expected = [
            torch.bmm(x.transpose(0, 1).double(), w.double()).transpose(0, 1)
            for x, w in zip(inputs, weights, strict=True)
        ]
        for role in roles:
            for value in outputs[role]:
                value.fill_(torch.nan)
            graphs[role].replay()
            for index, value in enumerate(outputs[role]):
                checked(value, expected[index % 5])

    samples = {role: [] for role in roles}
    for block in range(8):
        offset = (7 * block) % len(roles)
        order = roles[offset:] + roles[:offset]
        if block % 2:
            order = order[::-1]
        for role in order:
            batch = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(4):
                    graphs[role].replay()
                end.record()
                end.synchronize()
                batch.append(start.elapsed_time(end) * 1000 / 120)
            samples[role].append(batch)
    for role in roles:
        for index, value in enumerate(outputs[role]):
            checked(value, references[index % 5])
    for actual, original in [*immutable, *zip(inputs, originals, strict=True)]:
        assert torch.equal(actual.view(torch.uint8), original.view(torch.uint8))
    medians = {
        role: statistics.median(map(statistics.median, samples[role])) for role in roles
    }
    best_native = min(roles[:-1], key=medians.__getitem__)
    return dict(
        status="passed",
        gpu=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        flashinfer_version=flashinfer.__version__,
        seed=seed,
        input_shape=[1, 8, 4096],
        stored_weight_shape=[8192, 4096],
        scale_shape=[256, 128],
        weight_sets=5,
        validation=validation,
        native_tactic_configurations=_TGV_CUTE_EXT_TACTIC_CONFIGS,
        median_us=medians,
        samples_us=samples,
        best_tested_native=best_native,
        lower_latency_percent=100 * (1 - medians["frost"] / medians[best_native]),
        paired_blocks_faster=sum(
            statistics.median(a) < statistics.median(b)
            for a, b in zip(samples["frost"], samples[best_native], strict=True)
        ),
        scope="synthetic model-shaped WOA component; 30-call Graph; cuBLAS/public TGV/all native TGV tactics",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=419227)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    try:
        result = benchmark(args.seed)
    except BaseException:
        result = dict(status="fail", timing=None, error=traceback.format_exc())
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        with args.output.open("x") as f:
            f.write(text)
    print(text, end="", flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
