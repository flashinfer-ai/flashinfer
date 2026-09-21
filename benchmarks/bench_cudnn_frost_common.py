# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Shared timing and correctness helpers for cuDNN Frost MoE benchmarks."""

import json
import os
import subprocess

import torch

from flashinfer.experimental.cudnn_frost_selected_kernels.activations import ACTIVATIONS


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
