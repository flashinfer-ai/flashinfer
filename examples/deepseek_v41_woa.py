# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Run experimental WOA and compare graph time with native BF16 BMM on SM100.

python examples/deepseek_v41_woa.py

Uses five synthetic weight sets with the model's single-token geometry.
Reports component GPU time; this does not measure a complete decode request.
"""

import json
import statistics

import torch

from flashinfer import bmm_bf16
from flashinfer.autotuner import autotune
from flashinfer.deepseek_v41 import deepseek_v41_woa, deepseek_v41_woa_plan


def check(actual, expected):
    delta = actual.double() - expected
    assert torch.isfinite(actual).all()
    assert delta.norm() <= 0.003 * expected.norm()
    assert delta.abs().max() <= 0.006 * expected.abs().max()


@torch.no_grad()
def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This example requires an SM100 GPU")
    torch.manual_seed(419223)
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs, weights, plans, references = [], [], [], []
    for _ in range(5):
        x = torch.randn((1, 8, 4096), device="cuda", dtype=torch.bfloat16)
        stored = torch.randn((8192, 4096), device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        scales = torch.randint(122, 131, (256, 128), device="cuda", dtype=torch.uint8)
        plan = deepseek_v41_woa_plan(stored, scales, backend="frost")
        decoded = (
            (
                stored.double().view(256, 32, 128, 32)
                * torch.exp2(scales.double() - 127)[:, None, :, None]
            )
            .bfloat16()
            .reshape(8, 1024, 4096)
        )
        reference = torch.bmm(
            x.transpose(0, 1).double(), decoded.transpose(1, 2).double()
        ).transpose(0, 1)
        check(deepseek_v41_woa(x, plan), reference)
        inputs.append(x)
        weights.append(decoded.transpose(1, 2))
        plans.append(plan)
        references.append(reference)

    def run(role, index):
        if role == "frost":
            return deepseek_v41_woa(inputs[index], plans[index])
        return bmm_bf16(
            inputs[index].transpose(0, 1), weights[index], backend="tgv"
        ).transpose(0, 1)

    with autotune(tune_mode=True):
        run("tgv", 0)
    roles = ("tgv", "frost")
    graphs, outputs = {}, {}
    originals = [x.clone() for x in inputs]
    for role in roles:
        for index in range(5):
            check(run(role, index), references[index])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            values = [run(role, index % 5) for index in range(30)]
        for x in inputs:
            x.zero_()
        for value in values:
            value.fill_(torch.nan)
        graph.replay()
        for value in values:
            assert torch.count_nonzero(value) == 0
        for x, original in zip(inputs, originals, strict=True):
            x.copy_(original)
        for value in values:
            value.fill_(torch.nan)
        graph.replay()
        for index, value in enumerate(values):
            check(value, references[index % 5])
        graphs[role], outputs[role] = graph, values

    samples = {role: [] for role in roles}
    for block in range(8):
        order = roles if block % 4 in (0, 3) else roles[::-1]
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
            check(value, references[index % 5])
    medians = {
        role: statistics.median(map(statistics.median, samples[role])) for role in roles
    }
    print(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                input_shape=[1, 8, 4096],
                stored_weight_shape=[8192, 4096],
                scale_shape=[256, 128],
                weight_sets=5,
                median_us=medians,
                samples_us=samples,
                lower_latency_percent=100 * (1 - medians["frost"] / medians["tgv"]),
                scope="synthetic model-shaped WOA component, 30-call CUDA Graph",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
