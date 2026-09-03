"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Measure fused gated activation plus MXFP8 quantization against the decomposed
FlashInfer path. Timings use CUPTI GPU activities with a cold L2 cache.
"""

import argparse
import json
import statistics
import warnings
from importlib.metadata import version
from pathlib import Path

import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.testing import bench_gpu_time


SHAPES = (
    (128, 128),
    (4096, 2048),
    (4096, 7168),
    (16384, 7168),
    (131072, 8192),
)

MODES = (
    ("fwd-row", "forward", True, False),
    ("fwd-col", "forward", False, True),
    ("fwd-both", "forward", True, True),
    ("bwd-row", "backward", True, False),
    ("bwd-col", "backward", False, True),
    ("bwd-both", "backward", True, True),
)


@torch.compile(fullgraph=True, dynamic=False)
def _forward_activation(gated_input: torch.Tensor, grad_output: torch.Tensor):
    del grad_output
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    return (torch.nn.functional.silu(gate) * up).bfloat16()


@torch.compile(fullgraph=True, dynamic=False)
def _backward_activation(gated_input: torch.Tensor, grad_output: torch.Tensor):
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    grad = grad_output.float()
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    dact = silu_gate * (1.0 - sigmoid_gate) + sigmoid_gate
    return torch.cat(
        (((dact * grad) * up).bfloat16(), (silu_gate * grad).bfloat16()), dim=1
    )


def _decomposed(
    direction: str,
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    rowwise: bool,
    colwise: bool,
):
    logical = (
        _forward_activation(gated_input, grad_output)
        if direction == "forward"
        else _backward_activation(gated_input, grad_output)
    )
    if rowwise:
        flashinfer.mxfp8_quantize(logical, sf_swizzle_layout=SfLayout.layout_128x4)
    if colwise:
        flashinfer.mxfp8_quantize(
            logical.T.contiguous(), sf_swizzle_layout=SfLayout.layout_128x4
        )


def _fused(
    direction: str,
    gated_input: torch.Tensor,
    grad_output: torch.Tensor,
    rowwise: bool,
    colwise: bool,
):
    if direction == "forward":
        return flashinfer.silu_and_mul_mxfp8_quantize(
            gated_input, rowwise=rowwise, colwise=colwise
        )
    return flashinfer.silu_and_mul_mxfp8_quantize_backward(
        gated_input, grad_output, rowwise=rowwise, colwise=colwise
    )


def _median_ms(fn) -> float:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        measurements = bench_gpu_time(
            fn,
            enable_cupti=True,
            use_cuda_graph=False,
            cold_l2_cache=True,
            dry_run_iters=10,
            repeat_iters=100,
        )
    fallback = [item for item in caught if "Falling back" in str(item.message)]
    if fallback:
        raise RuntimeError(str(fallback[0].message))
    return float(statistics.median(measurements))


def _parse_shape(value: str) -> tuple[int, int]:
    try:
        m_text, k_text = value.lower().split("x", 1)
        shape = int(m_text), int(k_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must use MxK syntax") from error
    if shape not in SHAPES:
        raise argparse.ArgumentTypeError(f"shape must be one of {SHAPES}")
    return shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise RuntimeError("this benchmark requires SM100 or SM103")
    cupti_version = version("cupti-python")
    if int(cupti_version.split(".", 1)[0]) < 13:
        raise RuntimeError(f"cupti-python 13 or newer is required, got {cupti_version}")

    shapes = tuple(args.shape) if args.shape else SHAPES
    rows = []
    for m, k in shapes:
        generator = torch.Generator(device="cuda").manual_seed(20260820 + m + k)
        gated_input = torch.randn(
            (m, 2 * k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        grad_output = torch.randn(
            (m, k), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        for mode, direction, rowwise, colwise in MODES:
            baseline = lambda: _decomposed(
                direction, gated_input, grad_output, rowwise, colwise
            )
            candidate = lambda: _fused(
                direction, gated_input, grad_output, rowwise, colwise
            )
            baseline()
            candidate()
            torch.cuda.synchronize()
            baseline_ms = _median_ms(baseline)
            candidate_ms = _median_ms(candidate)
            rows.append(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "architecture": f"sm_{capability[0]}{capability[1]}",
                    "timing": "CUPTI cold-L2",
                    "M": m,
                    "K": k,
                    "mode": mode,
                    "candidate_ms": candidate_ms,
                    "baseline_ms": baseline_ms,
                    "speedup": baseline_ms / candidate_ms,
                }
            )

    print("| M | K | Mode | Fused (ms) | Decomposed (ms) | Speedup |")
    print("|---:|---:|:---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['M']} | {row['K']} | {row['mode']} | "
            f"{row['candidate_ms']:.6f} | {row['baseline_ms']:.6f} | "
            f"{row['speedup']:.4f}x |"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
