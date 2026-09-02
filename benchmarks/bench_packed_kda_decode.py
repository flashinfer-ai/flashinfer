# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark the packed-input CuTe KDA T=1 decode kernel on B200."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from flashinfer.kda_kernels.packed_kda_decode_cute import _select_tile_v
from flashinfer.kda_kernels.packed_kda_decode_cute import (
    run_packed_kda_decode_cute,
)
from flashinfer.testing import bench_gpu_time


HEADS = 12
HEAD_DIM = 128
MIXED_WIDTH = 3 * HEADS * HEAD_DIM
GATE_WIDTH = HEADS * HEAD_DIM
MIXED_STRIDE = 6144
STATE_ELEMENTS = HEADS * HEAD_DIM * HEAD_DIM
STATE_PADDING = 256
DEFAULT_BATCHES = (1, 8, 16, 31, 32, 64, 128, 256, 512)


def _state_view(storage, slots, slot_stride):
    return storage.as_strided(
        (slots, HEADS, HEAD_DIM, HEAD_DIM),
        (slot_stride, HEAD_DIM * HEAD_DIM, HEAD_DIM, 1),
    )


def _make_case(batch, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    mixed_storage = torch.randn(
        batch,
        MIXED_STRIDE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    raw_gate = torch.randn(
        batch,
        GATE_WIDTH,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.25)
    raw_beta = torch.randn(
        batch,
        HEADS,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    A_log = torch.empty(HEADS, dtype=torch.float32, device=device)
    A_log.uniform_(-2.0, -0.1, generator=generator)
    dt_bias = torch.randn(
        GATE_WIDTH,
        dtype=torch.float32,
        device=device,
        generator=generator,
    ).mul_(0.1)

    slots = batch + 1
    slot_stride = STATE_ELEMENTS + STATE_PADDING
    state_storage = torch.randn(
        slots * slot_stride,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.02)
    state = _state_view(state_storage, slots, slot_stride)
    state_indices = torch.arange(
        batch,
        0,
        -1,
        dtype=torch.int32,
        device=device,
    )
    output = torch.empty(
        batch,
        1,
        HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    return {
        "batch": batch,
        "mixed_qkv": mixed_storage[:, :MIXED_WIDTH],
        "raw_gate": raw_gate,
        "raw_beta": raw_beta,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_storage": state_storage,
        "state": state,
        "state_indices": state_indices,
        "output": output,
        "initial_state_storage": state_storage.clone(),
        "slots": slots,
        "slot_stride": slot_stride,
    }


def _run(case, tile_v):
    return run_packed_kda_decode_cute(
        case["mixed_qkv"],
        case["raw_gate"],
        case["raw_beta"],
        case["A_log"],
        case["dt_bias"],
        case["state"],
        case["state_indices"],
        output=case["output"],
        tile_v=tile_v,
    )


def _restore(case):
    case["state_storage"].copy_(case["initial_state_storage"])
    case["output"].zero_()


def _reference(case):
    batch = case["batch"]
    packed = case["mixed_qkv"].float().reshape(batch, 3, HEADS, HEAD_DIM)
    q_raw = packed[:, 0]
    k_raw = packed[:, 1]
    q = (
        q_raw
        * torch.rsqrt((q_raw * q_raw).sum(dim=-1, keepdim=True) + 1.0e-6)
        * (HEAD_DIM**-0.5)
    )
    k = k_raw * torch.rsqrt((k_raw * k_raw).sum(dim=-1, keepdim=True) + 1.0e-6)
    value = packed[:, 2]
    gate = case["raw_gate"].float().reshape(batch, HEADS, HEAD_DIM)
    gate = gate + case["dt_bias"].reshape(HEADS, HEAD_DIM)
    decay = torch.exp(
        -5.0 * torch.sigmoid(torch.exp(case["A_log"])[None, :, None] * gate)
    )
    beta = torch.sigmoid(case["raw_beta"].float())

    indices = case["state_indices"].long()
    reference_storage = case["initial_state_storage"].clone()
    reference_state = _state_view(
        reference_storage,
        case["slots"],
        case["slot_stride"],
    )
    selected = reference_state.index_select(0, indices).float()
    decayed = selected * decay[:, :, None, :]
    prediction = torch.einsum("bhvk,bhk->bhv", decayed, k)
    delta = (value - prediction) * beta[:, :, None]
    updated = decayed + delta[:, :, :, None] * k[:, :, None, :]
    projected = torch.einsum("bhvk,bhk->bhv", updated, q)
    reference_state.index_copy_(0, indices, updated.to(torch.bfloat16))
    return projected.to(torch.bfloat16).unsqueeze(1), reference_state


def _check(case, tile_v):
    expected_output, expected_state = _reference(case)
    _restore(case)
    result = _run(case, tile_v)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        result,
        expected_output,
        atol=1.0e-2,
        rtol=1.0e-2,
        check_dtype=False,
    )
    torch.testing.assert_close(
        case["state"],
        expected_state,
        atol=1.0e-2,
        rtol=1.0e-2,
        check_dtype=False,
    )
    output_error = float((result.float() - expected_output.float()).abs().max())
    state_error = float((case["state"].float() - expected_state.float()).abs().max())
    _restore(case)
    torch.cuda.synchronize()
    return output_error, state_error


def _capture(case, tile_v):
    _restore(case)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream(device=case["state"].device)
    stream.wait_stream(torch.cuda.current_stream(case["state"].device))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        _run(case, tile_v)
    torch.cuda.synchronize()
    _restore(case)
    torch.cuda.synchronize()
    return graph


def _logical_bytes(batch):
    bf16 = 2
    state = batch * STATE_ELEMENTS * bf16 * 2
    inputs = batch * (MIXED_WIDTH + GATE_WIDTH + HEADS) * bf16
    parameters = (HEADS + GATE_WIDTH) * 4
    output = batch * HEADS * HEAD_DIM * bf16
    indices = batch * 4
    return state + inputs + parameters + output + indices


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCHES),
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "cuda_graph", "both"),
        default="both",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--tile-v", type=int, choices=(8, 16, 32, 64, 128))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if any(batch <= 0 or batch > 65535 for batch in args.batch_size):
        parser.error("batch sizes must be in [1, 65535]")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    return args


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (10, 0):
        raise RuntimeError("packed-input CuTe KDA decode requires exact CC 10.0")

    modes = ("direct", "cuda_graph") if args.mode == "both" else (args.mode,)
    rows = []
    print(f"{'mode':<10} {'B':>5} {'tile':>6} {'median_us':>12} {'logical_TB/s':>14}")
    for ordinal, batch in enumerate(args.batch_size):
        case = _make_case(batch, device, args.seed + ordinal)
        # tile_v=None lets the per-batch policy pick the kernel shape;
        # _select_tile_v is what it will choose (display only).
        tile_v = args.tile_v
        display_tile = tile_v or _select_tile_v(batch)
        output_error, state_error = _check(case, tile_v)
        for mode in modes:
            run = (
                (lambda: _run(case, tile_v))
                if mode == "direct"
                else _capture(case, tile_v).replay
            )
            samples_ms = bench_gpu_time(
                run,
                enable_cupti=True,
                cold_l2_cache=args.cold_l2,
                use_cuda_graph=False,
                dry_run_iters=args.warmup,
                repeat_iters=args.iterations,
            )
            median_ms = float(statistics.median(samples_ms))
            logical_tbps = _logical_bytes(batch) / median_ms / 1.0e9
            row = {
                "mode": mode,
                "batch_size": batch,
                "tile_v": display_tile,
                "median_us": median_ms * 1000.0,
                "logical_tb_per_second": logical_tbps,
                "output_max_abs": output_error,
                "state_max_abs": state_error,
                "samples_ms": [float(value) for value in samples_ms],
            }
            rows.append(row)
            print(
                f"{mode:<10} {batch:>5} {display_tile:>6} "
                f"{row['median_us']:>12.4f} {logical_tbps:>14.4f}"
            )
        torch.cuda.empty_cache()

    if args.json is not None:
        report = {
            "device": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "cold_l2": args.cold_l2,
            "rows": rows,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
