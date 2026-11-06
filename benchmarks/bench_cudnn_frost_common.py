# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Shared timing and correctness helpers for cuDNN Frost MoE benchmarks."""

import inspect
import json
import os
import subprocess

import torch

from flashinfer.fused_moe.backends.cudnn_frost.activations import (
    ACTIVATIONS,
)


def check_idle_gpu(gpu_uuid):
    """Fail instead of reporting timings contaminated by another GPU process."""
    uuid = str(gpu_uuid).removeprefix("GPU-").lower()
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    peers = []
    for line in output.splitlines():
        device, pid = (value.strip() for value in line.split(",", 1))
        if device.removeprefix("GPU-").lower() == uuid and int(pid) != os.getpid():
            peers.append(int(pid))
    if peers:
        raise RuntimeError(
            f"Other compute processes on benchmark GPU: {peers}. "
            "Aborting: any timing from this interrupted run is not an isolated "
            "performance result. Retry when this GPU is idle; do not stop others' jobs."
        )


def capture(fn, batch, warmup=3):
    for _ in range(warmup):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(batch):
            fn()
    return graph


def measure(graph, iterations, batch, *, warmup=True):
    if warmup:
        graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (iterations * batch)


def error(out, ref):
    value = (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-20)
    ).item()
    if not torch.isfinite(out).all().item() or not value < 0.01:
        raise RuntimeError(f"cuDNN Frost numerical check failed: relative L2={value}")
    return value


def activation_reference(values, name):
    import torch.nn.functional as F

    if ACTIVATIONS[name]().is_gated:
        up, gate = values.chunk(2, dim=-1)
        if name == "swiglu":
            return F.silu(gate) * up
        if name == "swiglu_step":
            return F.silu(gate).clamp(max=7.0) * up.clamp(-7.0, 7.0)
        if name == "situ":
            return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
                25.0 * torch.tanh(up / 25.0)
            )
        return F.gelu(gate, approximate="tanh" if name == "geglu_tanh" else "none") * up
    return {
        "relu2": lambda x: F.relu(x).square(),
        "relu": F.relu,
        "gelu": F.gelu,
        "silu": F.silu,
        "identity": lambda x: x,
    }[name](values)


def emit(file, record, *, quiet=False):
    file.write(json.dumps(record) + "\n")
    if quiet:
        return
    file.flush()
    print(
        json.dumps({k: v for k, v in record.items() if k not in ("ranked", "samples")}),
        flush=True,
    )


# Original backends whose runners take BF16 hidden states and quantize them in
# kernel. The pre-quantized activation pack that the block-scale Frost runners
# consume cannot reach them in the same MoELayer call, so the block-scale
# comparisons exclude them and record the exclusion in their audit.
BF16_INPUT_BACKENDS = frozenset(
    (
        "b12x_nvfp4",
        "cutile_mxfp4",
        "cutile_mxfp8",
        "cutile_nvfp4",
        "sm12x_mxfp8_mxfp4",
    )
)


def prepare_original_view(cfg_cls, w1, w2, *, quant, **kwargs):
    """Prepare an original backend's weight view from canonical BF16 weights."""
    parameters = inspect.signature(cfg_cls.prepare_weights).parameters
    if list(parameters)[:2] != ["w1_bf16", "w2_bf16"]:
        raise RuntimeError(
            f"Add benchmark weight preparation for {cfg_cls.__name__}, which does "
            "not prepare canonical BF16 weights"
        )
    if "quant" in parameters:
        kwargs["quant"] = quant
    return cfg_cls.prepare_weights(w1, w2, **kwargs)
